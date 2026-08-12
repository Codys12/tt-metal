# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 text generation demo.

Simple prefill + decode loop following gpt-oss text_demo.py pattern.

Usage:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600

    # With fewer layers for testing:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600 -k "test_demo"

Op-level profiling:
    # Easiest: use the wrapper (sets env vars, runs tracy, dumps CSV)
    python3 tools/tracy/profile_this.py \\
        -c "pytest models/demos/gemma4/demo/text_demo.py::test_demo -v"

    # Or set env vars manually
    export TT_METAL_DEVICE_PROFILER=1
    pytest models/demos/gemma4/demo/text_demo.py::test_demo -v
    # CSV lands under generated/profiler/reports/<timestamp>/
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_cody.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_cody.tt.common import create_tt_model
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig

# Op-level device profiling — enabled by setting TT_METAL_DEVICE_PROFILER=1
# (see module docstring for invocation). When the env var is unset, this is a no-op.
_DEVICE_PROFILE = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"


# === DEBUG: trace sharding failures (set GEMMA4_DEBUG_SHARDING=1 to enable) ===
def _install_sharding_debug_hooks():
    if os.environ.get("GEMMA4_DEBUG_SHARDING") != "1":
        return
    import traceback as _tb

    def _wrap(name, fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except RuntimeError as e:
                logger.error(f"=== {name} FAILED ===")
                try:
                    t = args[0] if args else kwargs.get("input_tensor") or kwargs.get("tensor")
                    if t is not None and hasattr(t, "shape"):
                        logger.error(
                            f"input: shape={t.shape}, layout={getattr(t, 'layout', '?')}, "
                            f"dtype={getattr(t, 'dtype', '?')}, "
                            f"mem={getattr(t, 'memory_config', lambda: '?')() if callable(getattr(t, 'memory_config', None)) else getattr(t, 'memory_config', '?')}"
                        )
                except Exception as ie:
                    logger.error(f"(could not introspect input: {ie})")
                logger.error(f"args[1:]={args[1:]}, kwargs={kwargs}")
                logger.error("Python stack (most-recent-last):")
                for line in _tb.format_stack()[:-1]:
                    for sub in line.rstrip().split("\n"):
                        logger.error(sub)
                logger.error(f"Original error: {e}")
                raise

        wrapper.__name__ = f"_debug_{name}"
        return wrapper

    for name in ("to_memory_config", "interleaved_to_sharded", "sharded_to_interleaved"):
        fn = getattr(ttnn, name, None)
        if fn is not None:
            setattr(ttnn, name, _wrap(name, fn))
            logger.info(f"[DEBUG] Installed wrapper for ttnn.{name}")


_install_sharding_debug_hooks()


def _resolve_demo_env():
    """Resolve GEMMA4_NUM_LAYERS / GEMMA4_MAX_NEW_TOKENS.

    Defaults: all layers (None) and 128 generated tokens.
    Set GEMMA4_NUM_LAYERS=N to cap layers, or =0 / unset for all.
    Set GEMMA4_MAX_NEW_TOKENS=N to cap tokens (set 1 for fast profiling).
    """
    num_layers = int(os.environ.get("GEMMA4_NUM_LAYERS", "0")) or None
    max_new_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", "128"))
    return num_layers, max_new_tokens


def _read_device_profiler(mesh_device):
    if _DEVICE_PROFILE:
        ttnn.ReadDeviceProfiler(mesh_device)


def run_generation(
    mesh_device,
    model_path,
    prompts,
    max_new_tokens=32,
    num_layers=None,
    max_seq_len=4096,
    page_params=None,
    enable_decode_trace=True,
):
    """
    Run text generation with Gemma4.

    Args:
        mesh_device: TT device
        model_path: Path to model weights
        prompts: List of prompt strings
        max_new_tokens: Number of tokens to generate per prompt
        num_layers: Override layer count (for quick testing)
        max_seq_len: Maximum sequence length (determines KV cache size)
        page_params: Paged attention params dict with "page_block_size" and "page_max_num_blocks"

    Returns:
        List of generated text strings
    """
    from transformers import AutoTokenizer

    is_ci_env = os.environ.get("CI") == "true"
    batch_size = 1  # Gemma4 demo is single-user
    # tt-metal decode QKV-split tile-quantizes the batch dim to 32. Q's shard grid,
    # K/V dim_1, page_table dim_0 and SDPA's cur_pos all need to be 32 to stay
    # consistent. We use 1 real user in slot 0 and leave slots 1-31 dormant (-1 in
    # page_table, -1 in update_idxs, zero-pad embeddings).
    decode_batch_pad = 32

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Load tokenizer
    profiler.start("loading_inputs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"Tokenizer loaded from {model_path}")
    profiler.end("loading_inputs")

    # Paged attention config
    if page_params is None:
        page_params = {"page_block_size": 64, "page_max_num_blocks": max_seq_len // 64}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    # Page table: identity mapping for the real user (slot 0); -1 marks unused slots.
    # Prefill uses the single-user view; decode uses the batch-padded view.
    page_table_prefill = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        batch_size, paged_attention_config.max_num_blocks
    )
    page_table_decode = torch.full((decode_batch_pad, paged_attention_config.max_num_blocks), -1, dtype=torch.int32)
    page_table_decode[0] = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32)

    # Create model
    logger.info(f"Creating model with {num_layers or 'all'} layers, max_seq_len={max_seq_len}...")
    t0 = time.time()
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=decode_batch_pad,  # KV cache sized for 32 user-slots; only slot 0 is real
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )
    logger.info(f"Model created in {time.time() - t0:.1f}s")

    # The fused matmul_reduce_scatter_async path (attention o_proj /
    # shared_mlp down_proj) is still buggy at batch=32 on this fork — it
    # trips a NOC-usage TT_FATAL during the fused kernel. Production
    # server.py disables it the same way (_disable_fused_reduce_scatter_buffers).
    # Drop the persistent buffers so both fall back to the proven unfused
    # linear + all_reduce path.
    freed = 0
    for layer in model.layers:
        for module in (getattr(layer, "self_attn", None), getattr(layer, "shared_mlp", None)):
            if module is None:
                continue
            for attr in ("_fused_intermediate", "_fused_output"):
                buf = getattr(module, attr, None)
                if buf is not None:
                    try:
                        buf.deallocate(True)
                    except Exception:
                        pass
                    setattr(module, attr, None)
                    freed += 1
    if freed:
        logger.info(f"Disabled fused reduce-scatter path: freed {freed} persistent buffers")

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # Page tables on device — prefill uses single-user view, decode uses 32-row padded view.
    page_table_prefill_tt = ttnn.from_torch(
        page_table_prefill,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )
    page_table_decode_tt = ttnn.from_torch(
        page_table_decode,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    generated_texts = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {prompt_idx}: {prompt}")

        # Tokenize using chat template for instruct models
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            chat_result = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            input_ids = chat_result["input_ids"].squeeze(0)  # [seq_len]
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

        prompt_len = input_ids.shape[0]
        # Pad to standard prefill lengths (matches tt_transformers/gpt_oss pattern)
        if prompt_len <= 128:
            padded_len = 128
        elif prompt_len <= 1024:
            padded_len = 1024
        else:
            padded_len = 2 ** (prompt_len - 1).bit_length()
        input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
        logger.info(f"Prompt tokens: {prompt_len} (padded to {padded_len})")

        # Prefill
        logger.info("Prefilling...")
        profiler.start(f"compile_prefill", iteration=prompt_idx)

        import traceback as tb

        tokens_tt = ttnn.from_torch(
            input_ids_padded.unsqueeze(0).to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

        needs_per_layer_inputs = bool(model.hidden_size_per_layer_input and model._per_layer_input_weight_keys)
        embeds_torch = None
        if needs_per_layer_inputs:
            # Read only the embedding rows for this prompt's tokens. Materializing the full
            # [vocab, hidden] matrix here would burn ~2.7 GB of host RAM per prompt.
            flat_ids = input_ids_padded.long().reshape(-1)
            embed_rows = None
            for embed_key in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"):
                if embed_key in state_dict:
                    embed_rows = state_dict.get_tensor_rows(embed_key, flat_ids)
                    break
            if embed_rows is None:
                embed_rows = torch.zeros(flat_ids.numel(), model_args.hidden_size, dtype=torch.bfloat16)
            embeds_torch = (embed_rows.view(1, padded_len, -1) * model.embed_scale).float()
            del embed_rows

        # Get last token tile for first decode token
        get_last_token = ((prompt_len - 1) // 32) * 32
        try:
            logits = model.ttnn_prefill_forward(
                embeds,
                page_table=page_table_prefill_tt,
                kv_cache=tt_kv_cache,
                get_last_token=get_last_token,
                input_ids_torch=input_ids_padded.unsqueeze(0),
                embeds_torch=embeds_torch,
            )
        except Exception as e:
            logger.error(f"Prefill failed: {e}")
            tb.print_exc()
            raise

        # Sample first token (argmax from last position)
        if is_mesh:
            logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            logits_cpu = ttnn.to_torch(logits)
        logits.deallocate(True)
        _read_device_profiler(mesh_device)

        # Get logits at the actual last prompt position within the tile
        pos_in_tile = (prompt_len - 1) - get_last_token
        next_token = logits_cpu[0, 0, pos_in_tile, :].argmax().item()

        profiler.end(f"compile_prefill", iteration=prompt_idx)

        # Also record as inference_prefill (compile_prefill includes first-run compile cost)
        profiler.start(f"inference_prefill", iteration=prompt_idx)
        profiler.end(f"inference_prefill", iteration=prompt_idx)

        logger.info(
            f"Prefill done in {profiler.get_duration('compile_prefill', iteration=prompt_idx):.2f}s, "
            f"first token: {next_token} = '{tokenizer.decode([next_token])}'"
        )

        # Decode loop
        generated_tokens = [next_token]
        current_pos = prompt_len
        iteration = 0
        trace_id = None
        trace_output = None
        trace_device_inputs = None

        # ── Decode helpers ─────────────────────────────────────────────────
        # Token embedding is always computed on device. PLI models still compute
        # the per-layer input tensor on host because PLI needs CPU projection weights.
        # Sampling: SamplingGenerator for TP >= 2, host torch.argmax for TP = 1.
        on_device_sampling = model.sampling is not None

        def _make_decode_inputs(tok, pos):
            """Create host tensors for one decode iteration.

            Real user occupies slot 0; slots 1..31 are dormant (token=0, update_idx=-1).
            """
            tokens_padded = torch.zeros((1, decode_batch_pad), dtype=torch.int32)
            tokens_padded[0, 0] = tok
            pos_padded = torch.nn.functional.pad(
                torch.tensor([pos], dtype=torch.int32).reshape(1, 1),
                (0, decode_batch_pad - 1),
                "constant",
                0,
            )
            pos_int32_padded = torch.full((decode_batch_pad,), -1, dtype=torch.int32)
            pos_int32_padded[0] = pos
            inputs = {
                "tokens": ttnn.from_torch(
                    tokens_padded,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    mesh_mapper=replicate,
                ),
                "position": ttnn.from_torch(
                    pos_padded,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    mesh_mapper=replicate,
                ),
                "position_int32": ttnn.from_torch(
                    pos_int32_padded,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.int32,
                    mesh_mapper=replicate,
                ),
            }
            if needs_per_layer_inputs:
                _, pli_torch = model.compute_host_embeddings(tok)
                inputs["pli"] = ttnn.from_torch(
                    pli_torch,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=replicate,
                )
            return inputs

        def _fwd(device_inputs):
            decode_embeds = model.embed_tokens(device_inputs["tokens"])
            decode_embeds = ttnn.reshape(decode_embeds, (1, 1, decode_batch_pad, model_args.hidden_size))
            return model.ttnn_decode_forward(
                x=decode_embeds,
                current_pos=device_inputs["position"],
                rot_mat_idxs=device_inputs["position_int32"],  # pos_int32 passed as rot_mat_idxs
                page_table=page_table_decode_tt,
                kv_cache=tt_kv_cache,
                sampling_on_device=on_device_sampling,
                precomputed_pli=device_inputs.get("pli"),
            )

        def _inputs_to_device(inputs):
            return {k: ttnn.to_device(v, device=mesh_device) for k, v in inputs.items() if v is not None}

        def _copy_inputs_to_trace(host_inputs):
            for k, v in host_inputs.items():
                if v is not None and k in trace_device_inputs:
                    ttnn.copy_host_to_device_tensor(v, trace_device_inputs[k])

        def _extract_token(decode_output):
            """Extract next token from model output (token IDs or logits).

            Output is batch-padded to decode_batch_pad; the real user is in slot 0.
            """
            output_cpu = (
                ttnn.to_torch(ttnn.get_device_tensors(decode_output)[0]) if is_mesh else ttnn.to_torch(decode_output)
            )
            if on_device_sampling:
                # output: sampled token IDs at batch_pad positions; slot 0 is the real user
                return output_cpu.reshape(-1)[0].item()
            else:
                # output: logits [1, 1, batch_pad, vocab]; argmax over slot 0
                return output_cpu.reshape(-1, output_cpu.shape[-1])[0].argmax().item()

        sample_mode = "device" if on_device_sampling else "host"
        logger.info(
            f"Decoding (trace={'ON' if enable_decode_trace else 'OFF'}, "
            f"embedding=device, sampling={sample_mode})..."
        )
        profiler.start(f"inference_decode", iteration=prompt_idx)

        # ── Main decode loop (mode-agnostic) ──────────────────────────────
        for step in range(max_new_tokens - 1):
            if iteration == 0:
                profiler.start(f"compile_decode", iteration=prompt_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=prompt_idx)

            inputs_h = _make_decode_inputs(next_token, current_pos)

            if enable_decode_trace and trace_id is not None:
                # ── Traced execution: copy inputs and replay ──
                _copy_inputs_to_trace(inputs_h)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                decode_logits = trace_output

            elif enable_decode_trace and iteration == 0:
                # ── Iteration 0: compile run + trace capture ──
                # 1. Compile run (un-traced)
                inputs_d = _inputs_to_device(inputs_h)
                decode_logits, _ = _fwd(inputs_d)
                next_token = _extract_token(decode_logits)
                generated_tokens.append(next_token)
                current_pos += 1
                profiler.end(f"compile_decode", iteration=prompt_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=prompt_idx)
                logger.debug(
                    f"Iteration {iteration} (compile): {1000*decode_iteration_time:.0f}ms @ "
                    f"{1/decode_iteration_time:.1f} tok/s/user"
                )
                iteration += 1

                # 2. Capture trace with fresh device buffers
                logger.info("Capturing decode trace...")
                inputs_h2 = _make_decode_inputs(next_token, current_pos)
                trace_device_inputs = _inputs_to_device(inputs_h2)

                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                trace_output, _ = _fwd(trace_device_inputs)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                logger.info("Decode trace captured")

                # 3. Execute trace for current iteration
                profiler.start(f"inference_decode_time_{iteration}", iteration=prompt_idx)
                _copy_inputs_to_trace(inputs_h2)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                decode_logits = trace_output

            else:
                # ── No tracing: straightforward forward ──
                inputs_d = _inputs_to_device(inputs_h)
                decode_logits, _ = _fwd(inputs_d)

            next_token = _extract_token(decode_logits)
            generated_tokens.append(next_token)
            current_pos += 1

            if iteration == 0:
                profiler.end(f"compile_decode", iteration=prompt_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=prompt_idx)
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=prompt_idx)
                decode_iteration_time = profiler.get_duration(
                    f"inference_decode_time_{iteration}", iteration=prompt_idx
                )

            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.debug(
                f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ "
                f"{tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            iteration += 1

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

        # Release trace
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)

        profiler.end(f"inference_decode", iteration=prompt_idx)
        _read_device_profiler(mesh_device)

        # Final output
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text
        generated_texts.append(full_text)

        short_prompt = (
            (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:]) if len(prompt) > 200 else prompt
        )
        logger.info(f"\n==PROMPT {prompt_idx}\n{short_prompt}\n==OUTPUT {prompt_idx}\n{generated_text.strip()}\n")

    num_tokens_generated_decode = iteration  # from last prompt

    profiler.end("run")

    if num_tokens_generated_decode == 0:
        return generated_texts

    # ── Performance metrics ──────────────────────────────────────────────
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")

    # inference_prefill is a zero-duration marker (prefill compile+run are not separated yet)
    total_inference_prefill_time = compile_prefill_time

    total_inference_decode_time = 0
    for i in range(1, num_tokens_generated_decode):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    avg_time_to_first_token = total_inference_prefill_time / batch_size
    avg_decode_iteration_time = (
        total_inference_decode_time / (num_tokens_generated_decode - 1) if num_tokens_generated_decode > 1 else 0
    )

    prefill_tok_s = prompt_len / total_inference_prefill_time * batch_size if total_inference_prefill_time > 0 else 0
    decode_tok_s_user = (
        (num_tokens_generated_decode - 1) / total_inference_decode_time
        if num_tokens_generated_decode > 1 and total_inference_decode_time > 0
        else 0
    )
    decode_tok_s = decode_tok_s_user * batch_size

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,
        "decode_t/s/u": decode_tok_s_user,
        "decode_t/s": decode_tok_s,
        # Optional measurements
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Decode performance at specific token milestones
    tok_1_perf = profiler.get_duration("inference_decode_time_1") if 1 < num_tokens_generated_decode else 0
    tok_128_perf = profiler.get_duration("inference_decode_time_127") if 127 < num_tokens_generated_decode else 0

    logger.info("")
    logger.info("=== Performance metrics ===")
    if tok_1_perf > 0:
        logger.info(
            f"1st token decode time: {tok_1_perf * 1000:.2f}ms "
            f"[{round(1 / tok_1_perf, 2)} t/s/u, {round((1 / tok_1_perf) * batch_size, 2)} t/s]"
        )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf * 1000:.2f}ms "
            f"[{round(1 / tok_128_perf, 2)} t/s/u, {round((1 / tok_128_perf) * batch_size, 2)} t/s]"
        )
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ "
        f"{round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Generated {num_tokens_generated_decode} tokens")
    logger.info(f"Full demo runtime: {round(profiler.get_duration('run'), 2)}s")

    # Save benchmark data for CI dashboard
    if is_ci_env:
        targets = {}  # No perf targets for Gemma4 yet
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting
        for i in range(1, num_tokens_generated_decode):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )

        # Average decode performance for first 128 iterations (excluding compile)
        num_iterations_for_avg = min(128, num_tokens_generated_decode)
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, num_iterations_for_avg)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / max(1, num_iterations_for_avg - 1),
            step_warm_up_num_iterations=None,
            target=None,
        )

        model_name = "Gemma4"
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name=model_name,
            ml_model_type="llm",
            num_layers=num_layers or model_args.num_hidden_layers,
            batch_size=batch_size,
            config_params={},
            input_sequence_length=prompt_len,
            output_sequence_length=num_tokens_generated_decode,
        )

    return generated_texts


# ── Pytest entry points ──────────────────────────────────────────────────


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )


def test_demo_single_layer(device, model_path):
    """Single-device demo. Defaults: all layers, 128 tokens.

    Override with env vars:
        GEMMA4_NUM_LAYERS — layer count (default: all; set 1 for the original smoke test)
        GEMMA4_MAX_NEW_TOKENS — generated token count (default: 128; set 1 for profiling)
    """
    num_layers, max_new_tokens = _resolve_demo_env()
    prompts = ["The capital of France is"]
    results = run_generation(
        mesh_device=device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        num_layers=num_layers,
    )
    assert len(results) == 1
    assert len(results[0]) > len(prompts[0])


@parametrize_mesh_with_fabric()
def test_demo(mesh_device, model_path):
    """Full model demo — runs on any multi-device mesh.

    Filter by mesh shape:
        pytest -k "1x2"   # N300 / TP=2
        pytest -k "1x8"   # T3K  / TP=8

    Override with env vars:
        GEMMA4_NUM_LAYERS — layer count (default: all; set 2 for fast on-device profiling)
        GEMMA4_MAX_NEW_TOKENS — generated token count (default: 128; set 1 for profiling)
    """
    num_layers, max_new_tokens = _resolve_demo_env()

    prompts = ["Explain quantum computing in simple terms."]
    results = run_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        max_seq_len=4 * 1024,
        enable_decode_trace=True,
        num_layers=num_layers,
    )
    assert len(results) == 1
    logger.info(f"Full model output: {results[0]}")


# @parametrize_mesh_with_fabric()
# def test_demo_no_trace(mesh_device, model_path):
#     """Debug test: no trace, fewer tokens."""
#     prompts = ["Explain quantum computing in simple terms."]
#     results = run_generation(
#         mesh_device=mesh_device,
#         model_path=model_path,
#         prompts=prompts,
#         max_new_tokens=16,
#         max_seq_len=4 * 1024,
#         enable_decode_trace=False,
#     )
#     assert len(results) == 1
#     logger.info(f"No-trace output: {results[0]}")
