# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight OpenAI-format server for DiffusionGemma.

Same request shape as ``gemma4_cody/server/server.py`` (a subset):
    POST /v1/chat/completions   {"messages":[...], "max_tokens":N, "seed":S}
    POST /v1/completions        {"prompt":"...", "max_tokens":N, "seed":S}
    GET  /v1/models
The response is OpenAI-shaped and additionally carries ``tokens_per_second``,
``forwards`` (denoising steps run) and ``generation_seconds``.

It is launched through pytest so it reuses the proven 1x4 mesh + FABRIC_1D device
setup (trace region size, fabric, dispatch core) instead of re-deriving it:

    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch2 \
           HF_MODEL=/mnt/nas/gemma-diff TT_CACHE_PATH=/mnt/nas/gemma_cache \
           MESH_DEVICE=8xP150 TT_VISIBLE_DEVICES=0,2,3,5
    python -u -m pytest models/demos/gemma4_cody/diffusion/server.py -k 1x4 -q -s --timeout=0

Env knobs: DIFF_SERVER_HOST (0.0.0.0), DIFF_SERVER_PORT (8000),
DIFF_MAX_STEPS (48 denoising steps cap).
"""

import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from models.demos.gemma4_cody.diffusion.sampler import (
    DiffusionSamplerParams,
    EntropyBoundSampler,
    finalize_canvas,
    temperature_at,
)
from models.demos.gemma4_cody.diffusion.tt_model import TTDiffusionGemma
from models.demos.gemma4_cody.tests.test_factory import parametrize_mesh_with_fabric

_MAX_STEPS = int(os.getenv("DIFF_MAX_STEPS", "48"))


# ── OpenAI-format request schemas (subset matching server.py) ───────────────


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: float = 1.0  # accepted; the diffusion sampler uses its own entropy schedule
    seed: Optional[int] = None
    stream: bool = False  # streaming not supported (block-diffusion emits a whole canvas)


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    seed: Optional[int] = None
    stream: bool = False


def _flatten(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


# ── Engine ──────────────────────────────────────────────────────────────────


class DiffEngine:
    """Single-batch DiffusionGemma wrapper. One request at a time (device state
    is canvas-scoped), so the HTTP layer serializes via ``self.lock``."""

    def __init__(self, mesh_device, model_path: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        t0 = time.time()
        self.model = TTDiffusionGemma(mesh_device, model_path)
        logger.info(f"DiffusionGemma loaded in {time.time() - t0:.1f}s")
        self.lock = threading.Lock()

    def ids_from_messages(self, messages: List[ChatMessage]) -> torch.Tensor:
        if self.tokenizer.chat_template:
            out = self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": _flatten(m.content)} for m in messages],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            ids = (out["input_ids"] if not isinstance(out, torch.Tensor) else out).squeeze(0)
        else:
            text = "\n".join(_flatten(m.content) for m in messages)
            ids = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
        return ids

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, seed: Optional[int]):
        """Run the block-diffusion denoising loop (mirrors demo.run_diffusion_generation).

        Returns (text, completion_tokens, forwards, gen_seconds, warm_step_s).
        """
        params = DiffusionSamplerParams(max_denoising_steps=_MAX_STEPS)
        gen = torch.Generator().manual_seed(seed if seed is not None else 0)
        sampler = EntropyBoundSampler(params, self.model.canvas_length, self.model.args.vocab_size, generator=gen)

        sequence = input_ids.clone()
        prompt_len = int(input_ids.numel())
        max_canvases = max(1, -(-max_new_tokens // self.model.canvas_length))
        n_fwd = 0
        warm_step_times: List[float] = []
        t_start = time.time()

        for _canvas_idx in range(max_canvases):
            self.model.encode_prefix(sequence)
            self.model.decode_prepare()
            canvas = sampler.initialize_canvas()
            sampler.reset()
            prev_temp = 1.0
            argmax_canvas = canvas
            for step_index, step in enumerate(range(params.max_denoising_steps, 0, -1), start=1):
                temp = temperature_at(params, step)
                t_step = time.time()
                argmax_canvas, ent = self.model.decode_step(canvas, 1.0 / prev_temp, 1.0 / temp, step_index)
                # steps 1-2 are eager (compile), step 3 captures the trace; only
                # step 4+ are pure warm trace replays — average those for warm_step.
                if step_index > 3:
                    warm_step_times.append(time.time() - t_step)
                n_fwd += 1
                prev_temp = temp
                canvas, argmax_canvas, finished = sampler.step_from_reductions(canvas, ent, argmax_canvas)
                if finished:
                    break
            self.model.decode_finish()
            final_canvas, hit_eos = finalize_canvas(argmax_canvas, params)
            sequence = torch.cat([sequence, final_canvas])
            if hit_eos:
                break

        gen_seconds = time.time() - t_start
        new_tokens = sequence[prompt_len:]
        text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
        n_tok = int(new_tokens.numel())
        warm_step_s = sum(warm_step_times) / len(warm_step_times) if warm_step_times else 0.0
        return text, n_tok, n_fwd, gen_seconds, warm_step_s


# ── HTTP app ────────────────────────────────────────────────────────────────

app = FastAPI(title="DiffusionGemma server")


def _usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "diffusion-gemma", "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    eng: DiffEngine = app.state.engine
    ids = eng.ids_from_messages(req.messages)
    max_new = req.max_tokens or eng.model.canvas_length
    with eng.lock:
        text, n_tok, n_fwd, secs, warm_step_s = eng.generate(ids, max_new, req.seed)
    tok_s = n_tok / secs if secs > 0 else 0.0
    logger.info(
        f"[chat] {n_tok} tok in {secs:.2f}s = {tok_s:.1f} tok/s ({n_fwd} fwd, warm step {warm_step_s*1000:.0f}ms)"
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "diffusion-gemma",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": _usage(int(ids.numel()), n_tok),
        "tokens_per_second": round(tok_s, 2),
        "forwards": n_fwd,
        "generation_seconds": round(secs, 3),
        "warm_step_ms": round(warm_step_s * 1000, 1),
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    eng: DiffEngine = app.state.engine
    ids = eng.tokenizer.encode(req.prompt, return_tensors="pt").squeeze(0)
    max_new = req.max_tokens or eng.model.canvas_length
    with eng.lock:
        text, n_tok, n_fwd, secs, warm_step_s = eng.generate(ids, max_new, req.seed)
    tok_s = n_tok / secs if secs > 0 else 0.0
    logger.info(
        f"[cmpl] {n_tok} tok in {secs:.2f}s = {tok_s:.1f} tok/s ({n_fwd} fwd, warm step {warm_step_s*1000:.0f}ms)"
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or "diffusion-gemma",
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": _usage(int(ids.numel()), n_tok),
        "tokens_per_second": round(tok_s, 2),
        "forwards": n_fwd,
        "generation_seconds": round(secs, 3),
        "warm_step_ms": round(warm_step_s * 1000, 1),
    }


# ── pytest entry: opens the 1x4 mesh via the fixture, then serves forever ─────


@parametrize_mesh_with_fabric([(1, 4)])
def test_serve(mesh_device):
    model_path = os.getenv("HF_MODEL", "/mnt/nas/gemma-diff")
    host = os.getenv("DIFF_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("DIFF_SERVER_PORT", "8000"))

    engine = DiffEngine(mesh_device, model_path)
    app.state.engine = engine

    # Warm the kernels (first forward JIT-compiles every op) so the first real
    # request reports a representative tok/s.
    logger.info("Warming up (one short generation)...")
    t0 = time.time()
    _t, n_tok, n_fwd, secs, warm = engine.generate(
        engine.tokenizer.encode("Hello", return_tensors="pt").squeeze(0), engine.model.canvas_length, 0
    )
    logger.info(f"Warmup done in {time.time()-t0:.1f}s ({n_fwd} fwd, warm step {warm*1000:.0f}ms)")

    logger.info(f"Serving DiffusionGemma on http://{host}:{port}  (POST /v1/chat/completions | /v1/completions)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
