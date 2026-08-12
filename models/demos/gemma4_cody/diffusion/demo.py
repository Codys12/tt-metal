# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DiffusionGemma block-diffusion text demo on TT hardware.

Usage (tokenfactory host, 4 chips of an 8xP150 visible):
    export HF_MODEL=/mnt/nas/gemma-diff TT_CACHE_PATH=/mnt/nas/gemma_cache
    export MESH_DEVICE=8xP150 TT_VISIBLE_DEVICES=0,2,3,5
    pytest models/demos/gemma4_cody/diffusion/demo.py -v --timeout=3600

Env overrides:
    DIFF_NUM_LAYERS      cap layers (debug)
    DIFF_MAX_NEW_TOKENS  generated token budget (default 256 = 1 canvas)
    DIFF_MAX_STEPS       denoising step cap (default 48)
    DIFF_PROMPT          prompt override
    DIFF_SEED            torch seed (default 0)
"""

import os
import time

import pytest
import torch
from loguru import logger

from models.demos.gemma4_cody.diffusion.sampler import (
    DiffusionSamplerParams,
    EntropyBoundSampler,
    finalize_canvas,
    temperature_at,
)
from models.demos.gemma4_cody.diffusion.tt_model import TTDiffusionGemma
from models.demos.gemma4_cody.tests.test_factory import parametrize_mesh_with_fabric


def run_diffusion_generation(
    mesh_device,
    model_path,
    prompt,
    max_new_tokens=256,
    max_steps=48,
    num_layers=None,
    seed=0,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.chat_template:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = (out["input_ids"] if not isinstance(out, torch.Tensor) else out).squeeze(0)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
    logger.info(f"Prompt tokens: {input_ids.numel()}")

    t0 = time.time()
    model = TTDiffusionGemma(mesh_device, model_path, num_layers=num_layers)
    logger.info(f"Model built in {time.time() - t0:.1f}s")

    params = DiffusionSamplerParams(max_denoising_steps=max_steps)
    gen = torch.Generator().manual_seed(seed)
    sampler = EntropyBoundSampler(params, model.canvas_length, model.args.vocab_size, generator=gen)

    sequence = input_ids.clone()
    max_canvases = -(-max_new_tokens // model.canvas_length)
    total_fwd = 0

    for canvas_idx in range(max_canvases):
        t_enc = time.time()
        model.encode_prefix(sequence)
        model.decode_prepare()  # persistent rope/masks/buffers for this prefix
        logger.info(f"[canvas {canvas_idx}] prefix {sequence.numel()} encoded in {time.time() - t_enc:.1f}s")

        canvas = sampler.initialize_canvas()
        sampler.reset()
        prev_temp = 1.0  # temperature used to scale the *previous* step's logits in SC
        argmax_canvas = canvas

        # step counts down N..1; step_index counts up 1..N (mode: eager warmup
        # for 1-2, captured-trace replay for 3+).
        for step_index, step in enumerate(range(params.max_denoising_steps, 0, -1), start=1):
            temp = temperature_at(params, step)  # current-step temp (sampler entropy)
            t_step = time.time()
            # device returns only the [S] reductions (argmax + entropy); the full
            # [S, V] logits stay on-chip (self-conditioning feeds back internally).
            argmax_canvas, ent = model.decode_step(canvas, 1.0 / prev_temp, 1.0 / temp, step_index)
            t_fwd = time.time() - t_step  # device forward + reductions + small read
            total_fwd += 1

            prev_temp = temp
            canvas, argmax_canvas, finished = sampler.step_from_reductions(canvas, ent, argmax_canvas)
            n_acc = int(sampler.accepted_token_mask.sum())
            mode = "eager" if step_index <= 2 else "trace"
            logger.info(
                f"[canvas {canvas_idx}] step {step_index} ({mode}): "
                f"T={temp:.2f} accepted={n_acc}/{model.canvas_length} "
                f"fwd={t_fwd:.2f}s total={time.time() - t_step:.2f}s"
            )
            if finished:
                logger.info(f"[canvas {canvas_idx}] adaptive stop")
                break

        model.decode_finish()

        final_canvas, hit_eos = finalize_canvas(argmax_canvas, params)
        sequence = torch.cat([sequence, final_canvas])
        text_so_far = tokenizer.decode(sequence[input_ids.numel() :].tolist(), skip_special_tokens=True)
        logger.info(f"[canvas {canvas_idx}] text so far:\n{text_so_far}")
        if hit_eos:
            break

    new_tokens = sequence[input_ids.numel() :]
    text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
    logger.info(f"\n=== PROMPT ===\n{prompt}\n=== OUTPUT ({new_tokens.numel()} tokens, {total_fwd} fwd) ===\n{text}")
    return text


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL", "/mnt/nas/gemma-diff")


@parametrize_mesh_with_fabric()
def test_diffusion_demo(mesh_device, model_path):
    num_layers = int(os.environ.get("DIFF_NUM_LAYERS", "0")) or None
    max_new_tokens = int(os.environ.get("DIFF_MAX_NEW_TOKENS", "256"))
    max_steps = int(os.environ.get("DIFF_MAX_STEPS", "48"))
    prompt = os.environ.get("DIFF_PROMPT", "Why is the sky blue? Answer in two sentences.")
    seed = int(os.environ.get("DIFF_SEED", "0"))

    text = run_diffusion_generation(
        mesh_device,
        model_path,
        prompt,
        max_new_tokens=max_new_tokens,
        max_steps=max_steps,
        num_layers=num_layers,
        seed=seed,
    )
    if num_layers is None:
        assert len(text.strip()) > 0  # layer-truncated debug runs may emit whitespace
