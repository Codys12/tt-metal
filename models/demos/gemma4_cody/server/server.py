# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI-compatible FastAPI server for the Gemma4 demo.

Trace setup:
  * One prefill trace per power-of-two bucket from 1024 up to 131072 tokens. A
    request is dispatched to the smallest bucket whose length is >= prompt_len.
  * One decode trace — batch of 32 users, page-table dim sized to the largest
    bucket so a single slot can hold up to 131072 tokens.

KV-cache pages are managed as a global pool. Each request reserves
``ceil((prompt_len + max_new_tokens) / block_size)`` pages on admission
(capped at the largest bucket's block count). Idle / past-end entries in
each per-slot row of the decode page table point at a shared scratch row,
so users with shorter contexts don't pay for empty cache.

Run (full T3K 1x8):
    python -m models.demos.gemma4_cody.server.server \\
        --model-path /mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it \\
        --host 0.0.0.0 --port 8000 --max-seq-len 4096

Run (1x4 board / half-T3K / mock 1x4):
    python -m models.demos.gemma4_cody.server.server \\
        --model-path /mnt/nas/gemma --mesh-shape 1x4 \\
        --host 0.0.0.0 --port 8000 --max-seq-len 4096
    # equivalent: --mesh-shape P150x4  (or N150x4)

If neither --mesh-shape nor $MESH_DEVICE is set, the server defaults to a
single-row mesh sized to ``ttnn.get_num_devices()``, so on a 4-device system
you implicitly get ``1x4``.

Endpoints:
    GET  /v1/models
    POST /v1/completions          (stream / non-stream)
    POST /v1/chat/completions     (stream / non-stream)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel

import ttnn
from models.demos.gemma4_cody.tt.ccl import ccl_allgather
from models.demos.gemma4_cody.tt.common import create_tt_model
from models.tt_transformers.tt.common import PagedAttentionConfig

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_env_file(path: Path) -> None:
    """Minimal KEY=VALUE .env loader; existing env vars take precedence."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class _PatchBody(BaseModel):
    code: str


DECODE_BATCH = 32
DEFAULT_BLOCK_SIZE = 64
# Sliding-window layers physically need at most `sliding_window` tokens of cache
# per user (must be a multiple of block_size for the ring buffer math). Matches
# Gemma4's sliding_window so we don't depend on the HF config at import time.
DEFAULT_SLIDING_CACHE_LEN = 1024
# Prefill bucket lengths in tokens — powers of two from 1024 up to 131072. A
# request is dispatched to the smallest bucket whose length is >= prompt_len.
# The largest bucket is also the per-slot KV cap (max_user_seq_len). Chunked
# prefill will replace this with a single chunk-sized trace later.
PREFILL_BUCKET_LENS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)

_KV_CACHE_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}

_DECODE_PROFILE = os.environ.get("GEMMA4_DECODE_PROFILE") == "1"
_PROF_PRINT_EVERY = int(os.environ.get("GEMMA4_DECODE_PROFILE_EVERY", "16"))
_PROF_WARMUP_STEPS = int(os.environ.get("GEMMA4_DECODE_PROFILE_WARMUP", "8"))

# Speculative decoding (MTP drafter). Opt-in. Defaults to OFF — when the env
# var is unset cody behaves exactly as before. When enabled, ``Engine`` loads
# the drafter at init and records the per-layer-type "deepest layer" indices
# so the drafter can read the target's KV. The actual propose/verify hook
# requires also exposing the target's last-layer hidden state through the
# decode trace — that piece is NOT yet wired (see server/INTEGRATION.md).
_SPECULATIVE_DECODE = os.environ.get("GEMMA4_SPECULATIVE_DECODE") == "1"
# Which drafter to run: "mtp" (the Gemma-4 MTP assistant, default) or "dflash"
# (the block-diffusion DFlash drafter — owns its KV, taps 5 target layers,
# drafts block_size tokens/step). DFlash reuses the packed-verify path with
# num_drafts = block_size-1; see server/dflash_spec.py.
_DRAFTER_KIND = os.environ.get("GEMMA4_DRAFTER_KIND", "mtp").lower()
# Per-phase hang-localizing checkpoints for the dflash decode step. Each
# synchronizes the device then prints, so the LAST printed line names the phase
# whose ops hung. Enable with GEMMA4_DFLASH_DEBUG=1.
_DFLASH_DEBUG = os.environ.get("GEMMA4_DFLASH_DEBUG") == "1"
# DFlash checkpoint + converted-cache locations (only used when KIND=dflash).
# The default cache dir is checkpoint-specific so multiple DFlash variants (e.g.
# the RedHatAI block-8 speculator and the z-lab block-16 drafter) never collide
# on the same tensorbin cache. The legacy ``/mnt/nas/dflash-gemma`` default keeps
# its original ``tensor_cache_dflash_bf16`` name for back-compat; any other path
# gets ``tensor_cache_dflash_<basename>_bf16``. If the resolved cache dir does not
# exist, the drafter streams weights straight from the checkpoint safetensors
# (see DflashSpeculativeDecoder.__init__), so a fresh checkpoint needs no
# separate convert_weights pass.
_DFLASH_PATH = os.environ.get("DFLASH_PATH", "/mnt/nas/dflash-gemma")


def _default_dflash_cache(dflash_path: str) -> str | None:
    tt_cache = os.environ.get("TT_CACHE_PATH")
    if not tt_cache:
        return None
    if os.path.abspath(dflash_path) == os.path.abspath("/mnt/nas/dflash-gemma"):
        name = "tensor_cache_dflash_bf16"
    else:
        name = f"tensor_cache_dflash_{os.path.basename(os.path.normpath(dflash_path))}_bf16"
    return os.path.join(tt_cache, name)


_DFLASH_CACHE = os.environ.get("DFLASH_CACHE") or _default_dflash_cache(_DFLASH_PATH)
# The drafter must match the target: its backbone_hidden_size has to equal
# the target's hidden_size. gemma-31b-assistant has backbone_hidden_size=5376
# (matches gemma-4-26B-A4B); gemma-assistant (2816) is for a smaller target
# and will fail the drafter pre_projection matmul.
_SPEC_DRAFTER_PATH = os.environ.get("GEMMA4_DRAFTER_PATH", "/mnt/nas/gemma-31b-assistant")
_SPEC_DRAFTER_CACHE = os.environ.get("GEMMA4_DRAFTER_CACHE_DIR") or (
    os.path.join(os.environ["TT_CACHE_PATH"], "tensor_cache_assistant_31b_bf16")
    if os.environ.get("TT_CACHE_PATH")
    else None
)
_SPEC_NUM_DRAFTS = int(os.environ.get("GEMMA4_NUM_DRAFTS", "1"))
# Packed-verify full-attention key-extent cap (tokens). The packed verify's
# attn mask is [B, 1, H_local*P, S_k] — rebuilt and copied to device every
# step — and the packed SDPA iterates all S_k keys (no cur_pos early-exit).
# Capping S_k bounds both the host copy and the verify SDPA compute; slots
# whose cur_pos nears the cap fall back to the single-token decode path.
# Must be a multiple of block_size.
_PV_SK_CAP = int(os.environ.get("GEMMA4_PV_SK_CAP", "4096"))
# Packed-verify occupancy buckets. The verify trace is captured at each of
# these batch sizes; per step the smallest bucket >= the active greedy-slot
# count is replayed, so a lightly-loaded server could run B_v*P rows instead
# of 32*P.
#
# LIMITATION: only B_v == the decode batch (32) currently works. The decode
# QKV-heads split (`nlp_create_qkv_heads_decode`, via `split_qkv_heads_decode`)
# produces a height-sharded layout hardwired to 32 shards; at B_v<32 the
# per-position prep in `packed_decode_forward` fails with
# `TT_FATAL: Number of shards along height 32 must not exceed cores ...`.
# Smaller buckets need that kernel to support batch<32. Until then the default
# is the single 32 bucket; the bucket machinery is kept for when it does.
_PV_BUCKETS = os.environ.get("GEMMA4_PV_BUCKETS", "32")


# ── OpenAI-compatible request/response schemas ──────────────────────────────


class ToolCallFunction(BaseModel):
    name: str
    # OpenAI specifies `arguments` as a JSON-encoded string. We accept either
    # a string or an already-decoded dict so we can replay our own non-string
    # responses back through the message log.
    arguments: Union[str, Dict[str, Any]] = ""


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    # OpenAI accepts either a plain string or a list of content parts (each
    # `{"type": "text", "text": ...}` / `{"type": "image_url", ...}` etc.).
    # We accept both shapes and flatten parts to a string for the template —
    # this server is text-only, so non-text parts are noted as placeholders.
    # Content is also optional because assistant messages with tool_calls
    # (and tool-role messages whose payload is captured in `name`) commonly
    # omit it.
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    # Assistant: the tool calls the model previously emitted.
    tool_calls: Optional[List[ToolCall]] = None
    # Tool role: id of the assistant tool_call this response is for. We use it
    # to look up the function name when the message itself doesn't carry one.
    tool_call_id: Optional[str] = None
    # Tool role: function name. Optional in OpenAI; we fall back to looking up
    # the matching tool_call_id in the prior assistant message.
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 ⇒ disabled (server defaults to a wide top-k window)
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    # Thinking control. The Gemma 4 chat template branches on
    # `enable_thinking`: when truthy it injects `<|think|>` into the system
    # block; when falsy it prefills `<|channel>thought\n<channel|>` in the
    # assistant prefix, suppressing the thought channel. We expose this as
    # `verbosity` — thinking is on by default and only disabled when
    # `verbosity == "low"`.
    verbosity: Optional[str] = None
    # OpenAI tool-calling parameters.
    #   tools: function definitions; rendered into the system block as
    #          `<|tool>{declaration}<tool|>` entries.
    #   tool_choice: "none" (drop tools), "auto" / None (default — model
    #          decides), "required", or {"type":"function","function":
    #          {"name":...}} (forced). The Gemma 4 template doesn't gate on
    #          this kwarg, so we honor "none" by suppressing tools and
    #          otherwise forward tools unchanged.
    #   parallel_tool_calls: accepted for OpenAI compatibility but not
    #          strictly enforced — the model can already emit one or many
    #          `<|tool_call>...<tool_call|>` blocks.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    stop: Optional[List[str]] = None


# ── Internal types ──────────────────────────────────────────────────────────


@dataclass
class _Request:
    """A request awaiting service. The worker emits events into an asyncio.Queue
    owned by the request handler, marshalled across threads via call_soon_threadsafe.
    """

    rid: str
    prompt_text: str
    input_ids: torch.Tensor  # int32, [prompt_len]
    max_new_tokens: int
    is_chat: bool
    created: float
    output_queue: "asyncio.Queue[Optional[dict]]"
    output_loop: asyncio.AbstractEventLoop
    # Sampling params (mapped to TTSampling's per-row tensors at prefill time)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 / unset ⇒ wide window (server default)
    seed: Optional[int] = None  # None ⇒ device's diversified RNG state
    # Set by the HTTP handler in `finally`; the worker's cancel-sweep frees
    # the slot so we stop generating tokens nobody is reading.
    cancelled: bool = False
    # When the prompt was rendered with enable_thinking=True, the assistant
    # prefix ends at `<|turn>model\n` and the model is expected to emit
    # `<|channel>...thoughts...<channel|>...reply...`. When False, the prefix
    # already contains `<|channel>thought\n<channel|>` so all model output is
    # plain content. The worker uses this to decide whether the initial token
    # stream should be classified as content or remain unset until the model
    # opens a channel.
    enable_thinking: bool = False
    # Free-form debug payload. Patched ``_step_decode`` / ``_prefill_request``
    # implementations can append anything here; the API surfaces it on the
    # response so custom logic can return per-request diagnostics.
    debug: List[Any] = field(default_factory=list)

    def emit(self, event: Optional[dict]) -> None:
        self.output_loop.call_soon_threadsafe(self.output_queue.put_nowait, event)


@dataclass
class _Slot:
    """Per-batch-slot state during decode."""

    rid: Optional[str] = None
    prompt_len: int = 0
    cur_pos: int = 0
    next_token: int = 0
    generated: int = 0
    max_new_tokens: int = 0
    request: Optional[_Request] = None
    finished: bool = False
    eos_set: set = field(default_factory=set)
    all_tokens: List[int] = field(default_factory=list)
    cum_text: str = ""
    # Reasoning/content split for `<channel|>`-delimited Gemma 4 output.
    # When the prompt was rendered with enable_thinking=True the model is
    # expected to emit `<|channel>...thoughts...<channel|>...reply...`; we
    # route per-token text into either the reasoning or the content buffer
    # based on whether we've seen `<channel|>` yet. With enable_thinking=False
    # the assistant prefix already includes `<channel|>`, so `in_thinking`
    # starts False and all output goes to content.
    in_thinking: bool = False
    reasoning_tokens: List[int] = field(default_factory=list)
    content_tokens: List[int] = field(default_factory=list)
    cum_reasoning: str = ""
    # Tool-call capture. While `in_tool_call`, the decode loop accumulates
    # tokens into `tool_call_buffer` instead of emitting them as content/
    # reasoning. On close, the buffer is decoded, parsed via
    # `_parse_gemma_tool_call`, and pushed to `tool_calls` for finish-reason
    # bookkeeping. We don't stream argument deltas mid-call because the
    # model emits the call as a single contiguous run.
    in_tool_call: bool = False
    tool_call_buffer: List[int] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Physical KV-cache pages this slot owns; drawn from the engine's free
    # pool on prefill and returned on free. Decoupling slot index from
    # pages lets compaction be a host-side rebind with no device copy.
    full_pages: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    sliding_pages: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    # Speculative decode: True once this slot has a valid `spec_hidden`
    # (the target hidden that produced `next_token`) in the engine's
    # `_spec_hidden_host` buffer. A freshly prefilled slot starts False and
    # spends its first decode step on the single-token path to bootstrap the
    # hidden; from then on it is eligible for the packed multi-token verify.
    # On the slot so it travels through `_compact_slots`.
    spec_hidden_valid: bool = False


# ── Engine ─────────────────────────────────────────────────────────────────


class Engine:
    """Continuous-batching engine with two traces (prefill + batch_32 decode).

    Owns the device, model, KV cache, traces, and a worker thread that services
    a request queue. API handlers call `submit()` and stream tokens out of the
    request's per-call output queue.
    """

    def __init__(
        self,
        mesh_device,
        model_path: str,
        max_seq_len: int = 4096,
        block_size: int = DEFAULT_BLOCK_SIZE,
        num_layers: Optional[int] = None,
        sliding_cache_len: int = DEFAULT_SLIDING_CACHE_LEN,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        max_prefill_bucket: Optional[int] = None,
        batch: int = DECODE_BATCH,
    ):
        from transformers import AutoTokenizer

        assert max_seq_len % block_size == 0, "max_seq_len must be a multiple of block_size"
        assert sliding_cache_len % block_size == 0, "sliding_cache_len must be a multiple of block_size"

        self.mesh_device = mesh_device
        self.model_path = model_path
        self.max_seq_len = max_seq_len  # legacy field; prefill buckets are now fixed powers of two
        self.block_size = block_size
        # Number of concurrent decode slots. Defaults to DECODE_BATCH (32, the
        # hardware decode width). Smaller values run "true" batch<32 decode —
        # see the known sub-32 constraints in the QKV-heads decode split.
        self.batch = batch
        # The single-token decode and drafter traces run the QKV-heads decode
        # split (`nlp_create_qkv_heads_decode`), which is hardwired to 32 user
        # shards. Those traces therefore run at this padded width — `self.batch`
        # active rows (0..batch-1) plus idle rows (batch..decode_width-1) that
        # carry -1 skip sentinels + scratch page-table entries; outputs slice
        # back to `self.batch`. The packed-verify path does NOT use this — it
        # runs at the true `B_v=self.batch` via the (user,position) KV-write
        # repack (B_v*P fills the 32-shard requirement with no padding).
        self.decode_width = max(DECODE_BATCH, ((self.batch + DECODE_BATCH - 1) // DECODE_BATCH) * DECODE_BATCH)

        # Bucket lengths (multiples of block_size, deduplicated, sorted ascending).
        # The largest bucket doubles as max_user_seq_len — the per-slot context cap.
        # ``max_prefill_bucket`` (if set) drops every bucket strictly larger
        # than that cap, so e.g. ``--max-prefill-bucket 65536`` skips the
        # 131072 trace capture (which alone takes ~minutes at startup).
        bucket_lens = []
        cap = int(max_prefill_bucket) if max_prefill_bucket is not None else None
        for bl in PREFILL_BUCKET_LENS:
            assert bl % block_size == 0, f"bucket {bl} must be a multiple of block_size {block_size}"
            if cap is not None and bl > cap:
                continue
            bucket_lens.append(bl)
        self.bucket_lens = sorted(set(bucket_lens))
        assert self.bucket_lens, (
            f"no valid buckets configured (max_prefill_bucket={max_prefill_bucket}); "
            f"need at least one bucket <= cap"
        )
        self.max_user_seq_len = self.bucket_lens[-1]

        # Cap sliding cache at max_user_seq_len — no point allocating more ring
        # slots than the largest bucket has logical positions.
        self.sliding_cache_len = min(sliding_cache_len, self.max_user_seq_len)

        # Full-attention paging: per-slot allocation is dynamic. The page-table
        # dim is sized to the per-slot cap (max_user_seq_len) so a single user
        # can grow up to the largest bucket, but the shared block pool is sized
        # to ``batch * max_seq_len`` — i.e. the cumulative budget across all
        # users. A single oversize request can therefore consume more than its
        # "fair share" as long as the total cumulative usage stays in budget.
        # One extra scratch page (id = full_pool_blocks) is reserved as the
        # single shared backing for every idle slot row, the suffix of an
        # active row past its allocated block count, and prefill-padding
        # positions. vllm-style "null block" semantics: every "this slot
        # doesn't need K/V here" page-table entry resolves to the same physical
        # block, so idle→active row mutations don't perturb BPU_max distinct
        # page ids on the kernel side.
        self.blocks_per_user_max = self.max_user_seq_len // block_size
        self.full_pool_blocks = self.batch * (max_seq_len // block_size)
        assert self.full_pool_blocks >= self.blocks_per_user_max, (
            f"max_seq_len={max_seq_len} too small for cumulative pool: "
            f"batch*max_seq_len ({self.batch * max_seq_len}) must be >= "
            f"max_user_seq_len ({self.max_user_seq_len}) so a single max-size "
            f"request fits in the pool."
        )
        self.scratch_full_id = self.full_pool_blocks
        self.total_blocks = self.full_pool_blocks + 1

        # Sliding paging: each slot is a ring buffer of `sliding_cache_len` tokens.
        # Same +1 scratch convention. Per-user budget here is fixed (the ring
        # never grows), so only the prefill page-table dim follows the bucket.
        self.blocks_per_user_sliding = self.sliding_cache_len // block_size
        self.total_blocks_sliding = (self.batch + 1) * self.blocks_per_user_sliding

        page_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=self.total_blocks)
        page_cfg_sliding = PagedAttentionConfig(block_size=block_size, max_num_blocks=self.total_blocks_sliding)

        logger.info(
            f"Loading Gemma4 model: max_seq_len={max_seq_len}, batch={self.batch}, "
            f"block_size={block_size}, buckets={self.bucket_lens}, "
            f"max_user_seq_len={self.max_user_seq_len}, "
            f"blocks_per_user_max={self.blocks_per_user_max}, "
            f"full_pool_blocks={self.full_pool_blocks}, scratch_full_id={self.scratch_full_id}, "
            f"total_blocks={self.total_blocks}, sliding_cache_len={self.sliding_cache_len}, "
            f"blocks_per_user_sliding={self.blocks_per_user_sliding}, "
            f"total_blocks_sliding={self.total_blocks_sliding}"
        )
        t0 = time.time()
        # Pass max_user_seq_len so RoPE caches cover the largest bucket.
        self.model_args, self.model, self.tt_kv_cache, self.state_dict = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=self.batch,
            max_seq_len=self.max_user_seq_len,
            num_layers=num_layers,
            paged_attention_config=page_cfg,
            paged_attention_config_sliding=page_cfg_sliding,
            model_path=model_path,
            create_kv_cache=True,
            kv_cache_dtype=kv_cache_dtype,
        )
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

        # PLI requires per-user host computation that doesn't fit the static
        # batch_32 trace. Block the server on PLI variants instead of silently
        # producing wrong outputs.
        if getattr(self.model, "_per_layer_input_weight_keys", None):
            raise RuntimeError(
                "Server does not support Gemma4 variants with per-layer input embeddings (E2B/E4B). "
                "Use the A4B / non-PLI variant."
            )

        # Force the unfused o_proj/down_proj path. The fused
        # matmul_reduce_scatter_async kernel currently asserts at batch=32 with
        # "bad optional access"; text_demo never hits it because batch=1 trips
        # the shape-mismatch fallback inside ccl_matmul_reduce_scatter_allgather.
        # Both attention and shared_mlp check `_fused_intermediate is None` to
        # decide which path to take.
        self._disable_fused_reduce_scatter_buffers()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.eos_token_ids = set()
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(int(self.tokenizer.eos_token_id))
        # The model's generation_config.json is the source of truth for stop
        # tokens. For Gemma 4 it lists [<eos>=1, <turn|>=106, <|tool_response>=50]
        # — without honoring <turn|> the worker would keep generating past every
        # assistant turn until it happened to emit <eos> or hit max_tokens.
        try:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig.from_pretrained(model_path)
            eos = gen_cfg.eos_token_id
            if eos is not None:
                for tid in eos if isinstance(eos, (list, tuple)) else [eos]:
                    self.eos_token_ids.add(int(tid))
        except Exception:
            pass
        # Robustness fallback: look up known close tokens by name. Covers
        # Gemma 4 (<turn|>) and Gemma 1/2/3 (<end_of_turn>) without depending
        # on which token format the loaded tokenizer exposes.
        for name in ("<turn|>", "<end_of_turn>", "<eos>"):
            tid = self.tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                self.eos_token_ids.add(tid)

        # Channel-separator id (`<channel|>`, id=101 on Gemma 4): the model
        # uses it to delimit thought-channel content from the actual reply.
        # The streaming path needs this to split reasoning_content from
        # content. -1 means "not present in this tokenizer".
        def _lookup_special(name: str) -> int:
            tid = self.tokenizer.convert_tokens_to_ids(name)
            return int(tid) if isinstance(tid, int) and tid >= 0 else -1

        self.channel_close_id: int = _lookup_special("<channel|>")
        self.channel_open_id: int = _lookup_special("<|channel>")
        # Tool-call delimiter ids. The decode loop captures the tokens
        # between open and close, then parses `call:NAME{ARGS}` into an
        # OpenAI tool_call. -1 disables the tool-call branch entirely on
        # tokenizers that don't define these.
        self.tool_call_open_id: int = _lookup_special("<|tool_call>")
        self.tool_call_close_id: int = _lookup_special("<tool_call|>")

        is_mesh = hasattr(mesh_device, "shape")
        self._is_mesh = is_mesh
        self._replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        self.on_device_sampling = self.model.sampling is not None

        # Free-page pool: slots draw a per-request count on prefill and return
        # them on free. Page id ``scratch_full_id`` (= full_pool_blocks) is
        # reserved as the single shared scratch backing and excluded from the
        # pool. Scratch entries appear in:
        #  (a) idle slot rows of the decode page table (full row of scratch);
        #  (b) the suffix of an active slot's row past its allocated block count;
        #  (c) the prefill page table past `ceil(prompt_len / block_size)`.
        # SDPA never reads past `cur_pos`, so scratch entries are never touched.
        # Every "unused" entry resolves to the same physical block, mirroring
        # vllm's null-block invariant.
        Bs = self.blocks_per_user_sliding
        BPU_max = self.blocks_per_user_max
        self._free_full_pages: Deque[int] = deque(range(self.full_pool_blocks))
        self._free_sliding_pages: Deque[int] = deque(range(self.batch * Bs))

        self._idle_blocks = torch.full((BPU_max,), self.scratch_full_id, dtype=torch.int32)
        self._idle_blocks_sliding = torch.arange(self.batch * Bs, (self.batch + 1) * Bs, dtype=torch.int32)

        # Live slot state (mutated only on the worker thread)
        self.slots: List[_Slot] = [_Slot() for _ in range(self.batch)]

        # Internal admission queue. API handlers push to ``request_queue``;
        # the worker drains into ``_waiting`` (FIFO). A request leaves
        # ``_waiting`` when both a free slot and enough free pages are available.
        self._waiting: List[_Request] = []

        # Per-slot sampling params (length batch). Pushed to the device's
        # k/p/temp tensors on prefill — the only point at which a slot's
        # sampling configuration changes during its lifetime.
        # Defaults: top_k=50, top_p=1.0 (no filter), temp=1.0.
        self._sampling_top_k: List[int] = [50] * self.batch
        self._sampling_top_p: List[float] = [1.0] * self.batch
        self._sampling_temp: List[float] = [1.0] * self.batch

        # Per-slot host torch.Generator used by the TP=1 / no-on-device-sampling
        # fallback. Reseeded at prefill from the request's `seed`. On TP>1 the
        # device's SeedManager handles per-slot RNG state instead.
        import secrets as _secrets

        self._slot_torch_rngs: List[torch.Generator] = []
        for _ in range(self.batch):
            g = torch.Generator()
            g.manual_seed(_secrets.randbits(63))
            self._slot_torch_rngs.append(g)

        # Request intake: API handlers push here, worker drains it.
        self.request_queue: "Queue[_Request]" = Queue()

        # Push initial sampling params so the persistent k/p/temp tensors hold
        # sane values when the decode trace is captured. After this point, the
        # device tensors are only updated on prefill (one slot at a time).
        self._push_sampling_params()

        # ── Speculative-decode drafter (opt-in via env var) ─────────────────
        # When enabled, load the MTP drafter alongside the target. _step_decode
        # runs the full propose -> verify -> accept loop and exposes acceptance
        # via /admin/spec-stats. Wall-clock speedup (skipping target steps via
        # a packed multi-token verify trace) is the documented follow-on; see
        # server/INTEGRATION.md.
        self._spec = None
        self._drafter_kind = _DRAFTER_KIND
        self._drafter_trace = None
        self._dflash_trace = None
        self._dflash_append_trace = None
        self._packed_verify_traces = None
        self._last_layer_of_type: Dict[str, int] = {}
        # Number of successfully committed packed-verify steps.  This state is
        # consumed by the shared decode loop even when speculative decode is
        # disabled (and after an optional drafter fails to initialize), so it
        # must exist independently of the opt-in setup branch below.
        self._spec_steps_called = 0
        if _SPECULATIVE_DECODE:
            # Identify last-of-each-type layer index on the target. The drafter
            # reads the target's KV at these indices for shared_kv. Clamp to
            # the actually-loaded layer count — `model_args.layer_types` is the
            # full (60-layer) HF config, but `--num-layers` can load fewer, so
            # the index must stay within `self.model.layers` / `tt_kv_cache`.
            n_actual = len(self.model.layers)
            for i, lt in enumerate(self.model_args.layer_types):
                if i < n_actual:
                    self._last_layer_of_type[lt] = i
            logger.info(
                f"Speculative decode ENABLED: kind={self._drafter_kind}, "
                f"last layer of each type = {self._last_layer_of_type}, num_drafts={_SPEC_NUM_DRAFTS}"
            )
            try:
                t0 = time.time()
                if self._drafter_kind == "dflash":
                    # DFlash: owns its KV, taps 5 target layers, drafts
                    # block_size tokens/step. num_drafts = block_size-1 so the
                    # packed verify (P=block_size) is reused unchanged. Propose
                    # is eager (no MTP drafter trace / doubled-input buffers), so
                    # _build_drafter_rope_caches_host + _preallocate_spec_buffers
                    # are skipped; only the packed-verify buffers are needed.
                    from models.demos.gemma4_cody.server.dflash_spec import DflashSpeculativeDecoder

                    self._spec = DflashSpeculativeDecoder(
                        mesh_device=mesh_device,
                        model=self.model,
                        dflash_path=_DFLASH_PATH,
                        dflash_cache_dir=_DFLASH_CACHE,
                        num_slots=self.batch,
                        mesh_config=self.model.mesh_config,
                        ccl_manager=self.model.ccl_manager,
                    )
                    # Tell the target which layer outputs to expose as aux taps;
                    # the packed-verify forward emits them (return_aux_hidden).
                    self.model.configure_aux_taps(self._spec.aux_hidden_layers)
                    logger.info(
                        f"DFlash drafter loaded in {time.time() - t0:.1f}s "
                        f"(block_size={self._spec.block_size}, aux_layers={self._spec.aux_hidden_layers})"
                    )
                else:
                    from models.demos.gemma4_cody.server.speculative import SpeculativeDecoder

                    self._spec = SpeculativeDecoder(
                        mesh_device=mesh_device,
                        num_slots=self.batch,
                        num_drafts=_SPEC_NUM_DRAFTS,
                        assistant_path=_SPEC_DRAFTER_PATH,
                        drafter_cache_dir=_SPEC_DRAFTER_CACHE,
                        # Reuse the target's CCL manager + mesh config so the
                        # drafter's column-parallel attention all-gather is
                        # trace-safe (pre-allocated semaphores).
                        mesh_config=self.model.mesh_config,
                        ccl_manager=self.model.ccl_manager,
                    )
                    logger.info(f"Drafter loaded in {time.time() - t0:.1f}s")

                    # Build the drafter's own RoPE caches (different config from
                    # target — see drafter config.json's rope_parameters dict).
                    # Host-side torch caches; per-step we slice rows and ship
                    # to device via copy_host_to_device_tensor to avoid mid-trace
                    # device allocations.
                    self._build_drafter_rope_caches_host()
                # Pre-allocate ALL propose-path device buffers BEFORE trace
                # capture. Allocations made between trace replays are unsafe
                # and corrupt the trace's static memory layout.
                #
                # Force ttnn op validation ON for these one-time setup
                # allocations. The server otherwise runs with
                # enable_fast_runtime_mode=true, which SKIPS validation — so an
                # oversized/invalid buffer here (notably the packed-verify
                # attn_mask, which scales with P=num_drafts+1: [B,1,H_local*P,S_k])
                # would crash the whole process with a bare "Illegal instruction
                # (core dumped)" instead of a catchable error. Preallocation is
                # one-time, so the validation cost is irrelevant. (Restore in
                # finally; same toggle pattern ttnn uses in graph.py.)
                _prev_fast_rt = ttnn.CONFIG.enable_fast_runtime_mode
                ttnn.CONFIG.enable_fast_runtime_mode = False
                try:
                    if self._drafter_kind != "dflash":
                        # MTP-only: doubled-input + drafter-RoPE device buffers
                        # for the chained drafter trace. DFlash propose is eager.
                        self._preallocate_spec_buffers()
                    # Packed multi-token verify buffers (Item 1.2). Same rule:
                    # everything allocated before the first begin_trace_capture.
                    self._preallocate_packed_verify_buffers()
                except Exception as alloc_err:
                    P = self._spec.num_drafts + 1
                    n_layers = getattr(self.model_args, "num_hidden_layers", "?")
                    raise RuntimeError(
                        f"Speculative-decode buffer preallocation failed "
                        f"(P={P}, batch={self.batch}, layers={n_layers}). These buffers scale "
                        f"with P — chiefly the [B, 1, H_local*P, S_k] packed-verify attn_mask — so "
                        f"at large P on top of a full multi-layer KV cache they can exhaust device "
                        f"memory. Lower GEMMA4_NUM_DRAFTS (current P-1) or the KV-cache footprint. "
                        f"Root cause: {alloc_err}"
                    ) from alloc_err
                finally:
                    ttnn.CONFIG.enable_fast_runtime_mode = _prev_fast_rt
            except Exception as e:
                logger.error(f"Failed to load speculative-decode drafter (continuing without): {e}")
                import traceback

                logger.error(traceback.format_exc())
                self._spec = None
        else:
            logger.info("Speculative decode disabled (set GEMMA4_SPECULATIVE_DECODE=1 to enable)")

        # Capture both traces back-to-back. Buffer allocations happen BEFORE
        # the first begin_trace_capture: tt-metal forbids host-side allocations
        # while any trace is live, including the gap between two traces.
        # ``_prefill_traces`` is a list of bucket dicts (one per ``self.bucket_lens``);
        # ``_decode_trace`` is the single batch_32 decode dict.
        self._prefill_traces, self._decode_trace = self._capture_traces()

        # Background worker
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name="gemma4-engine", daemon=True)
        self._worker.start()
        logger.info("Engine ready.")

    # ── Sampling helpers ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_sampling(req_temp: float, req_top_p: float, req_top_k: int) -> tuple[float, float, int]:
        """Translate OpenAI-style sampling params into TTSampling's expected form.

        - temperature == 0 (OpenAI greedy convention) → top_k=1, temp=1.0 deterministic.
        - top_k <= 0 means "disabled"; the sampling op needs a real positive int
          window, so we cap to a wide default (50). This matches HF's defaults.
        - top_p == 0 from a caller is unusual and would zero out the entire
          distribution, so we floor it to a small epsilon.
        """
        if req_temp <= 0.0:
            return 1.0, 1.0, 1
        top_k = req_top_k if req_top_k and req_top_k > 0 else 50
        top_p = max(req_top_p, 1e-3)
        return req_temp, top_p, top_k

    def _reseed_slot(self, slot_idx: int, seed: Optional[int]) -> None:
        """Update slot ``slot_idx``'s RNG state from the request's seed.

        On TP>1 (on-device sampling) this routes through ``SeedManager.reset_seed``
        which re-seeds that slot's host RNG and marks the FSM dirty so the next
        decode step pushes fresh per-user values to the device.

        On TP=1 (host fallback) we own a per-slot ``torch.Generator``; reseed it
        directly. ``seed=None`` falls back to a fresh random seed.
        """
        import secrets as _secrets

        if self.on_device_sampling:
            sm = self.model.sampling.seed_manager
            sm.reset_seed(seeds=[seed], user_ids=[slot_idx])
        else:
            self._slot_torch_rngs[slot_idx].manual_seed(seed if seed is not None else _secrets.randbits(63))

    def _advance_seeds(self) -> None:
        """Push fresh seed values to the device's seeds tensor when needed.

        ``SeedManager.get_new_values`` is a near-no-op once the FSM has settled
        into its steady state with no explicitly-seeded slots. With seeded slots
        it pushes one host→device copy per step.
        """
        if not self.on_device_sampling:
            return
        self.model.sampling.seed_manager.get_new_values()

    def _push_sampling_params(self) -> None:
        """Copy `self._sampling_*` arrays into the device's k/p/temp tensors.

        Only call this on prefill (per-slot params don't change during decode).
        Bypasses ``SamplingGenerator.reset_sampling_params`` to avoid touching
        penalties / logprobs state.
        """
        if not self.on_device_sampling:
            return
        s = self.model.sampling.tt_sampling
        # The device k/p/temp tensors are sized to the sampling module's padded
        # max_batch_size (batch rounded up to a 32-tile, × sampling_dp), which
        # is larger than self.batch when batch < 32. copy_host_to_device needs
        # matching element counts, so pad the per-slot lists up to that size.
        # The extra rows are inactive (their sampled tokens are discarded by the
        # [:self.batch] reads), so default greedy params (k=1, p=1.0, temp=1.0)
        # are harmless.
        n = s.max_batch_size * getattr(s, "_sampling_dp", 1)
        pad = max(0, n - self.batch)
        s.reset_params(
            k=list(self._sampling_top_k) + [1] * pad,
            p=list(self._sampling_top_p) + [1.0] * pad,
            temp=list(self._sampling_temp) + [1.0] * pad,
        )

    def _host_sample(self, logits: torch.Tensor) -> List[int]:
        """Per-row top-k + top-p + temperature sampling on host (TP=1 fallback).

        ``logits`` is shape [batch, vocab].
        """
        out: List[int] = []
        for i in range(logits.shape[0]):
            row = logits[i].float()
            top_k = self._sampling_top_k[i]
            top_p = self._sampling_top_p[i]
            temp = self._sampling_temp[i]

            # temperature == 1.0 + top_k == 1 ≡ greedy (the OpenAI temp=0 case).
            if top_k == 1 and temp == 1.0:
                out.append(int(row.argmax().item()))
                continue

            row = row / max(temp, 1e-5)
            if top_k > 0 and top_k < row.shape[-1]:
                kth = torch.topk(row, top_k).values[-1]
                row = torch.where(row < kth, torch.full_like(row, float("-inf")), row)
            probs = torch.softmax(row, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cum = sorted_probs.cumsum(-1)
                # Keep the smallest prefix whose cumulative prob ≥ top_p (always
                # keep the top-1).
                mask = cum > top_p
                mask[0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                probs = probs / probs.sum()
            gen = self._slot_torch_rngs[i]
            out.append(int(torch.multinomial(probs, 1, generator=gen).item()))
        return out

    # ── Trace capture helpers ───────────────────────────────────────────────

    def _disable_fused_reduce_scatter_buffers(self) -> None:
        """Drop the persistent buffers used by the fused matmul-reduce-scatter
        path so attention.o_proj and shared_mlp.down_proj fall back to the
        unfused linear + all_reduce path that text_demo exercises.
        """
        freed = 0
        for layer in self.model.layers:
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

    def _build_sliding_prefill_page_table(
        self, sliding_pages: torch.Tensor, prompt_len: int, bucket_len: int
    ) -> torch.Tensor:
        """Build a per-request page table for sliding-window prefill cache fills.

        Routes each logical block of the padded prompt to either the slot's
        sliding ring page (if the block overlaps the last W tokens of the
        prompt) or a per-block scratch row (out-of-window / padding).
        In-window blocks that share a ring page write in order; ring-buffer
        last-write-wins gives the right semantics.

        ``bucket_len`` is the prefill trace's padded sequence length — the
        page table must have exactly ``bucket_len // block_size`` entries to
        match the trace's input shape.
        """
        Bs = self.blocks_per_user_sliding
        blocks_per_seq = bucket_len // self.block_size
        scratch_base = self.batch * Bs

        W = self.sliding_cache_len
        window_start = max(0, prompt_len - W)

        pt = torch.empty(blocks_per_seq, dtype=torch.int32)
        for i in range(blocks_per_seq):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            if block_end <= window_start or block_start >= prompt_len:
                pt[i] = scratch_base + (i % Bs)
            else:
                pt[i] = int(sliding_pages[i % Bs].item())
        return pt.reshape(1, -1)

    def _alloc_device_tensor(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        """Allocate a fresh device buffer initialized from a torch tensor.

        Use ONLY for trace input buffers, allocated once before any trace is
        captured. Allocating new device buffers while a trace is live is unsafe.
        """
        return ttnn.from_torch(t, device=self.mesh_device, layout=layout, dtype=dtype, mesh_mapper=self._replicate)

    def _host_tensor(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        """Build a host-only ttnn tensor (no device allocation).

        Pair with ``copy_host_to_device_tensor`` to load values into a
        pre-allocated trace input buffer.
        """
        return ttnn.from_torch(t, layout=layout, dtype=dtype, mesh_mapper=self._replicate)

    def _build_prefill_fwd(self, bucket_len: int):
        """Allocate prefill input device buffers and return (buffers, fwd-fn).

        One trace per bucket — tokens / page tables / get_last_token are sized
        to ``bucket_len`` (one of ``self.bucket_lens``).

        Inputs swapped per request (via copy_host_to_device_tensor):
          * tokens [1, bucket_len] uint32
          * page_table [1, bucket_len/block_size] int32              (full layers)
          * page_table_sliding [1, bucket_len/block_size] int32      (sliding layers,
            same logical-block count as page_table; values point at the smaller
            sliding ring or scratch row.)
        """
        L = bucket_len
        bucket_blocks = L // self.block_size
        hidden = self.model_args.hidden_size
        tokens_dev = self._alloc_device_tensor(torch.zeros(1, L, dtype=torch.int32), ttnn.uint32)
        # Dummy initial values; _prefill_request overwrites them per request.
        page_table_dev = self._alloc_device_tensor(
            torch.arange(bucket_blocks, dtype=torch.int32).reshape(1, -1), ttnn.int32
        )
        Bs = self.blocks_per_user_sliding
        init_sliding_pt = torch.arange(bucket_blocks, dtype=torch.int32) % Bs
        page_table_sliding_dev = self._alloc_device_tensor(init_sliding_pt.reshape(1, -1), ttnn.int32)
        get_last_token = ((L - 1) // 32) * 32

        def fwd():
            embeds = self.model.embed_tokens(tokens_dev)
            embeds = ttnn.reshape(embeds, (1, 1, L, hidden))
            embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
            return self.model.ttnn_prefill_forward(
                embeds,
                page_table=page_table_dev,
                kv_cache=self.tt_kv_cache,
                get_last_token=get_last_token,
                page_table_sliding=page_table_sliding_dev,
                # Server discards prefill logits — the first decode iteration
                # regenerates the next-token logits from the cached prompt's
                # last token. Skipping lm_head + softcap + all-gather collapses
                # the peak intermediate from O(seq * vocab) to O(seq * hidden).
                skip_logits=True,
            )

        return (
            {
                "tokens": tokens_dev,
                "page_table": page_table_dev,
                "page_table_sliding": page_table_sliding_dev,
                "bucket_len": L,
                "bucket_blocks": bucket_blocks,
            },
            fwd,
        )

    def _build_decode_fwd(self):
        """Allocate decode input device buffers and return (buffers, fwd-fn).

        Inputs are pre-allocated once before trace capture and refreshed on
        the host every step via ``copy_host_to_device_tensor`` from
        ``_step_decode`` — no host writes ever land inside the trace. The
        ``fwd`` function only runs the model forward pass.

        Inputs swapped per step:
          * tokens             [1, 32] uint32
          * position           [1, 32] uint32  — RoPE embedding lookup index
          * position_int32     [32]    int32   — KV cache update + SDPA cur_pos
          * pos_sliding_write  [32]    int32   — cur_pos % sliding_cache_len
          * pos_sliding_sdpa   [32]    int32   — min(cur_pos, sliding_cache_len-1)
          * page_table         [32, BPU_max] int32 — full-layer page table
          * page_table_sliding [32, BPU_S]   int32 — sliding-layer page table

        ``BPU_max`` (= ``blocks_per_user_max``) is sized to the largest bucket
        so any slot can hold up to ``max_user_seq_len`` tokens. Active slots
        whose actual allocation is smaller fill the trailing entries with
        scratch pages — SDPA never reads past ``cur_pos``.
        """
        # Single-token decode runs the 32-hardwired QKV-heads decode split, so
        # its input buffers run at the padded width; rows >= self.batch are idle.
        B = self.decode_width
        hidden = self.model_args.hidden_size

        # All slots idle at allocation time: tokens=0, position=0, scratch row.
        tokens_init = torch.zeros(1, B, dtype=torch.int32)
        position_init = torch.zeros(1, B, dtype=torch.int32)
        pos_int32_init = torch.zeros(B, dtype=torch.int32)
        pos_sliding_write_init = torch.zeros(B, dtype=torch.int32)
        pos_sliding_sdpa_init = torch.zeros(B, dtype=torch.int32)
        pt_idle = torch.stack([self._idle_blocks for _ in range(B)], dim=0).clone()
        pt_sliding_idle = torch.stack([self._idle_blocks_sliding for _ in range(B)], dim=0).clone()

        tokens_dev = self._alloc_device_tensor(tokens_init, ttnn.uint32)
        position_dev = self._alloc_device_tensor(position_init, ttnn.uint32)
        position_int32_dev = self._alloc_device_tensor(pos_int32_init, ttnn.int32)
        pos_sliding_write_dev = self._alloc_device_tensor(pos_sliding_write_init, ttnn.int32)
        pos_sliding_sdpa_dev = self._alloc_device_tensor(pos_sliding_sdpa_init, ttnn.int32)
        page_table_dev = self._alloc_device_tensor(pt_idle, ttnn.int32)
        page_table_sliding_dev = self._alloc_device_tensor(pt_sliding_idle, ttnn.int32)

        # When spec decode is enabled, the trace also exposes the target's
        # last-layer hidden state — the drafter consumes it as input. With
        # spec off the existing single-output trace is preserved exactly.
        spec_enabled = self._spec is not None

        def fwd():
            embeds = self.model.embed_tokens(tokens_dev)
            embeds = ttnn.reshape(embeds, (1, 1, B, hidden))
            if spec_enabled:
                out, _, final_hidden = self.model.ttnn_decode_forward(
                    x=embeds,
                    current_pos=position_dev,
                    rot_mat_idxs=position_int32_dev,
                    page_table=page_table_dev,
                    kv_cache=self.tt_kv_cache,
                    sampling_on_device=self.on_device_sampling,
                    page_table_sliding=page_table_sliding_dev,
                    position_idx_cache_sliding_write=pos_sliding_write_dev,
                    position_idx_cache_sliding_sdpa=pos_sliding_sdpa_dev,
                    return_hidden_state=True,
                )
                return (out, final_hidden)
            else:
                out, _ = self.model.ttnn_decode_forward(
                    x=embeds,
                    current_pos=position_dev,
                    rot_mat_idxs=position_int32_dev,
                    page_table=page_table_dev,
                    kv_cache=self.tt_kv_cache,
                    sampling_on_device=self.on_device_sampling,
                    page_table_sliding=page_table_sliding_dev,
                    position_idx_cache_sliding_write=pos_sliding_write_dev,
                    position_idx_cache_sliding_sdpa=pos_sliding_sdpa_dev,
                )
                return out

        return (
            {
                "tokens": tokens_dev,
                "position": position_dev,
                "position_int32": position_int32_dev,
                "pos_sliding_write": pos_sliding_write_dev,
                "pos_sliding_sdpa": pos_sliding_sdpa_dev,
                "page_table": page_table_dev,
                "page_table_sliding": page_table_sliding_dev,
            },
            fwd,
        )

    def _build_drafter_rope_caches_host(self):
        """Build 2D cos/sin caches for the drafter's RoPE, per layer_type.

        Builds host caches for sizing/diagnostics and uploads the same
        ``[max_seq_len, head_dim]`` tables to device as ``_drafter_rope_dev_cache``
        so the chained drafter trace can gather per-slot rows via
        ``ttnn.embedding`` instead of paying the per-step host gather +
        H2D refresh of ``_spec_rope_dev``. Matches the target's
        ``use_embedding_rope`` pattern in ``tt/attention/decode.py``.
        """
        from transformers import AutoConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        hf_cfg = AutoConfig.from_pretrained(_SPEC_DRAFTER_PATH, trust_remote_code=True)
        text_cfg = hf_cfg.get_text_config()
        max_seq_len = self.max_user_seq_len
        position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        rope_module = Gemma4TextRotaryEmbedding(text_cfg)
        dummy = torch.zeros(1, 1, text_cfg.hidden_size, dtype=torch.float32)

        self._drafter_rope_host: Dict[str, tuple] = {}
        self._drafter_head_dim: Dict[str, int] = {}
        self._drafter_rope_dev_cache: Dict[str, tuple] = {}
        for lt in dict.fromkeys(text_cfg.layer_types):
            cos, sin = rope_module(dummy, position_ids, layer_type=lt)
            # cos/sin shape: [1, max_seq, head_dim_for_lt]; squeeze leading B.
            cos_2d = cos[0].to(torch.bfloat16)  # [max_seq_len, head_dim]
            sin_2d = sin[0].to(torch.bfloat16)
            self._drafter_rope_host[lt] = (cos_2d, sin_2d)
            self._drafter_head_dim[lt] = cos.shape[-1]
            # Persistent device caches: replicated, ROW_MAJOR so ttnn.embedding
            # can index them directly. Allocated once here, before trace capture.
            cos_dev_2d = self._alloc_device_tensor(cos_2d, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            sin_dev_2d = self._alloc_device_tensor(sin_2d, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._drafter_rope_dev_cache[lt] = (cos_dev_2d, sin_dev_2d)
        logger.info(
            f"Drafter RoPE caches built for layer_types={list(self._drafter_rope_host.keys())} "
            f"(host + device [max_seq_len={max_seq_len}, head_dims={list(self._drafter_head_dim.values())}])"
        )

        # Stash some often-used dims for the propose path.
        self._drafter_backbone_hidden = self.model_args.hidden_size  # target's hidden_size (matches drafter backbone)

    def _preallocate_spec_buffers(self):
        """Pre-allocate all propose-path device buffers BEFORE trace capture.

        Per-step refresh via copy_host_to_device_tensor only — never allocates
        new device buffers during the hot loop.
        """
        # The drafter trace cross-attends the target KV through the shared decode
        # page tables (which run at decode_width), so its inputs run at the same
        # padded width; rows >= self.batch are idle (see decode_width).
        B = self.decode_width
        H_backbone = self._drafter_backbone_hidden

        # Doubled hidden state input to drafter.pre_projection.
        # Shape: [1, 1, B, 2 * backbone_hidden], TILE layout, bf16.
        self._spec_input_dev = self._alloc_device_tensor(
            torch.zeros(1, 1, B, 2 * H_backbone, dtype=torch.bfloat16),
            ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # SDPA cur_pos tensor — full-attention layer (raw cur_pos = KV extent).
        self._spec_cur_pos_dev = self._alloc_device_tensor(
            torch.zeros(B, dtype=torch.int32),
            ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # SDPA cur_pos tensor — sliding layers. The target's sliding KV is a
        # ring buffer of W tokens; SDPA reads use min(cur_pos, W-1) so the
        # kernel walks the whole ring. The drafter must match (raw cur_pos
        # reads wrong ring slots once cur_pos >= W).
        self._spec_cur_pos_sliding_dev = self._alloc_device_tensor(
            torch.zeros(B, dtype=torch.int32),
            ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # uint32 [1, B] mirrors of cur_pos used as ``ttnn.embedding`` indices
        # into the 2D RoPE caches inside the drafter trace. Same per-step
        # data as the int32 SDPA cur_pos, just shaped/dtyped for the
        # embedding kernel (matches the target decode trace's ``position``
        # buffer).
        self._spec_position_2d_dev = self._alloc_device_tensor(
            torch.zeros(1, B, dtype=torch.int32),
            ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._spec_position_2d_sliding_dev = self._alloc_device_tensor(
            torch.zeros(1, B, dtype=torch.int32),
            ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # Per-layer-type RoPE cos/sin output buffers — KEPT for back-compat
        # but no longer refreshed per step (the chained drafter trace now
        # gathers cos/sin on-device via ``ttnn.embedding`` against
        # ``_drafter_rope_dev_cache``).
        self._spec_rope_dev: Dict[str, tuple] = {}
        for lt, hd in self._drafter_head_dim.items():
            cos_dev = self._alloc_device_tensor(
                torch.zeros(1, 1, B, hd, dtype=torch.bfloat16),
                ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            sin_dev = self._alloc_device_tensor(
                torch.zeros(1, 1, B, hd, dtype=torch.bfloat16),
                ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self._spec_rope_dev[lt] = (cos_dev, sin_dev)
        # Cache the drafter's per-shard vocab size for on-device cross-shard
        # topk reconstruction inside the chained drafter trace. It's a single
        # scalar (vocab // tp) — same across all devices — so we don't need a
        # pre-allocated per-device constant tensor (which previously SIGILL'd
        # the host on TILE-layout creation with shape [1,1,1,tp] sharded).
        tp = self.model.mesh_config.tp if (self._is_mesh and self.model.mesh_config) else 1
        drafter_vocab = self._spec.drafter.config.vocab_size
        assert drafter_vocab % tp == 0, f"drafter vocab {drafter_vocab} not divisible by tp {tp}"
        self._spec_drafter_shard_vocab = drafter_vocab // tp
        # Persistent host-side scratch for the doubled hidden — avoids
        # reallocating a torch tensor each step.
        self._spec_input_host_scratch = torch.zeros(1, 1, B, 2 * H_backbone, dtype=torch.bfloat16)
        logger.info(
            f"Spec-decode pre-allocated buffers: input=[1,1,{B},{2*H_backbone}], "
            f"rope per type: {[(lt, self._drafter_head_dim[lt]) for lt in self._drafter_head_dim]}"
        )

    def _refresh_spec_inputs(self, positions_torch, hidden_dev, embed_token_ids, draft_step=0, hidden_host=None):
        """Update pre-allocated propose-path device buffers from host data.

        ``positions_torch`` is a torch [B] tensor of the target's cur_pos this
        step (the position it just decoded).
        ``hidden_dev`` is the device hidden tensor paired into the drafter input.
        ``hidden_host`` is an alternative host torch source for the paired
        hidden ([B, hidden] or broadcastable) — used by the packed-verify
        propose loop, whose draft-0 hidden lives in ``_spec_hidden_host``
        rather than on device. Exactly one of ``hidden_dev`` / ``hidden_host``
        is used (``hidden_host`` takes precedence when not None).
        ``embed_token_ids`` is the list of B token IDs to embed for the first
        input half.
        ``draft_step`` is the index of the draft within this step's T-draft loop.

        The MTP drafter input is ``[embed(token) ‖ hidden]`` — see HF
        ``Gemma4AssistantCandidateGenerator``: ``cat([last_token_embedding,
        last_hidden_state])``, with the *unscaled* embedding table.

        - draft_step 0: ``embed_token_ids`` = the tokens the target consumed
          this step; ``hidden_dev`` = the target's last-layer hidden. The
          drafter predicts the token at cur_pos+1.
        - draft_step k>=1 (multi-draft): ``embed_token_ids`` = the previous
          draft's argmax tokens; ``hidden_dev`` = the drafter's own previous
          ``out_hidden`` (post_projection output). The drafter predicts the
          token at cur_pos+1+k. KV extent stays cur_pos (stale-KV mode 2a — the
          drafter does not see its own speculative positions; the autoregressive
          signal flows through the input pairing).

        The drafter's RoPE position is a CONSTANT cur_pos for every draft in the
        round (see below). Its cross-attention over the target KV uses cur_pos
        for the full layer and min(cur_pos, W-1) for the sliding layers (the
        target's sliding KV is a W-token ring).
        """
        H_backbone = self._drafter_backbone_hidden
        # First half = the Gemma-SCALED embedding of the token. HF's MTP
        # candidate generator passes `target_model_input_embeddings =
        # target.get_input_embeddings()`, which is the target's
        # `Gemma4TextScaledWordEmbedding` — its forward multiplies the lookup
        # by `embed_scale = sqrt(hidden_size)`. So HF feeds the drafter the
        # *scaled* embedding. The target's `embed_tokens` here applies the same
        # sqrt(hidden) scale — use it directly, do NOT divide it back out.
        # (Earlier code unscaled it, feeding the drafter a first input half
        # ~sqrt(5376)≈73× too small — a major drafter-acceptance bug.)
        _emb_t0 = time.perf_counter() if _DECODE_PROFILE else 0.0
        # Drafter input runs at the padded decode_width; embed_token_ids and
        # positions_torch are decode_width-long (rows >= self.batch are idle).
        DW = self.decode_width
        tok_t = torch.tensor(embed_token_ids, dtype=torch.int32).reshape(1, DW)
        tok_host = self._host_tensor(tok_t, ttnn.uint32)
        tok_dev = ttnn.to_device(tok_host, device=self.mesh_device)
        emb = self.model.embed_tokens(tok_dev)  # [1, 1, DW, hidden], sqrt(H)-scaled
        emb_torch = (ttnn.to_torch(ttnn.get_device_tensors(emb)[0]) if self._is_mesh else ttnn.to_torch(emb)).to(
            torch.float32
        )
        self._spec_input_host_scratch[..., :H_backbone].copy_(
            emb_torch.reshape(1, 1, DW, H_backbone).to(torch.bfloat16)
        )
        ttnn.deallocate(emb)
        ttnn.deallocate(tok_dev)
        if _DECODE_PROFILE:
            self._pt_pr_embed = getattr(self, "_pt_pr_embed", 0.0) + (time.perf_counter() - _emb_t0)
        # Second half = the paired hidden (target last-hidden for draft 0, the
        # drafter's own previous out_hidden for draft k>=1). When sourced from
        # device it is [1, 1, B, H_backbone]; read BEFORE the next trace replay
        # (which reuses the drafter's out_hidden buffer) — this read happens
        # here, ahead of execute_trace, so that ordering holds. The packed
        # verify propose loop instead passes its draft-0 hidden as a host
        # tensor (``hidden_host``) carried from the prior step.
        if hidden_host is not None:
            fh_torch = hidden_host.to(torch.bfloat16)
        elif self._is_mesh:
            fh_torch = ttnn.to_torch(ttnn.get_device_tensors(hidden_dev)[0]).to(torch.bfloat16)
        else:
            fh_torch = ttnn.to_torch(hidden_dev).to(torch.bfloat16)
        self._spec_input_host_scratch[..., H_backbone:].copy_(
            fh_torch.reshape(self._spec_input_host_scratch[..., H_backbone:].shape)
        )

        host_input = self._host_tensor(self._spec_input_host_scratch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_input, self._spec_input_dev)

        # Drafter SDPA cur_pos. Full layer: raw cur_pos (KV extent 0..cur_pos).
        # Sliding layers: min(cur_pos, W-1) so the kernel walks the whole ring.
        # Also push uint32 [1, B] mirrors used by the on-device RoPE gather
        # inside the chained drafter trace (replaces the previous per-step
        # host gather + 4 H2Ds of [1,1,B,head_dim] cos/sin tensors).
        pos_int32 = positions_torch.to(torch.int32)
        host_cur_pos = self._host_tensor(pos_int32, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_cur_pos, self._spec_cur_pos_dev)
        W = self.sliding_cache_len
        pos_sliding = torch.clamp(pos_int32, max=W - 1)
        host_cur_pos_sliding = self._host_tensor(pos_sliding, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_cur_pos_sliding, self._spec_cur_pos_sliding_dev)
        # uint32 [1, B] mirrors for ttnn.embedding (different dtype/shape than
        # the int32 [B] SDPA cur_pos — embedding requires uint32 indices and
        # the gather output's batch dim follows index dim 1).
        host_pos_2d = self._host_tensor(pos_int32.reshape(1, -1), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_pos_2d, self._spec_position_2d_dev)
        host_pos_2d_sliding = self._host_tensor(pos_sliding.reshape(1, -1), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_pos_2d_sliding, self._spec_position_2d_sliding_dev)

    def _build_drafter_fwd(self, decode_buffers):
        """Closure that runs T sequential drafter forwards in ONE trace.

        Per-step the host only refreshes draft-0's inputs (``_spec_input_dev``,
        ``_spec_cur_pos_*``, ``_spec_position_2d_*_dev``). RoPE cos/sin are
        gathered ON-DEVICE inside the trace from the persistent
        ``_drafter_rope_dev_cache`` via ``ttnn.embedding`` — same pattern as
        the target's decode trace (``tt/attention/decode.py`` use_embedding_rope
        path). All inter-draft state lives on device:

          for k in 0..T-1:
              out_hidden_k, logits_k = drafter.forward(target_last_hidden=spec_input)
              logits_full = all_gather(logits_k, dim=vocab)        # TP gather
              draft_k = argmax(logits_full, dim=vocab, keepdim=1)   # [1,1,B,1] uint32
              if k < T-1:
                  emb_k = target.embed_tokens(reshape(draft_k, (1,B)))   # [1,1,B,H]
                  spec_input = concat([emb_k, out_hidden_k], dim=hidden)  # [1,1,B,2H]

        RoPE, cur_pos, and shared KV are CONSTANT across all T iterations —
        the HF MTP candidate generator locks the drafter to position cur_pos
        for every draft in the round and cross-attends the same shared KV.
        Only the doubled hidden input changes between drafts, and it's built
        on device from the previous iteration's argmax + post-projection out.

        Outputs: tuple of T tensors ``draft_k`` (each [1, 1, B, 1] uint32 in
        ROW_MAJOR, replicated across the mesh — every device computed the same
        global argmax because the all-gather happens before argmax). The host
        reads T × ~128 B per step instead of T × (topk shards + spec_input
        H2D + per-draft argmax/embed work).

        Net change vs. the prior single-forward trace replayed T times:
          * H2D per step: 1 × spec_input (~340 KB on this shape), not T × that
          * D2H per step: T × 128 B (the T argmax results), not T × (topk_val
            + topk_idx) + the per-replay control-flow round-trips
          * Trace dispatch: 1 execute, not T
        """
        # Persistent device-side 2D RoPE caches. The chained trace below
        # gathers per-slot cos/sin rows once at the top via ``ttnn.embedding``
        # and reuses them across all T drafts (positions are CONSTANT across
        # the round). Replaces the previous per-step host-gather + 4 H2Ds
        # into ``_spec_rope_dev`` from ``_refresh_spec_inputs``.
        rope_full_cos_cache, rope_full_sin_cache = self._drafter_rope_dev_cache["full_attention"]
        rope_swa_cos_cache, rope_swa_sin_cache = self._drafter_rope_dev_cache["sliding_attention"]
        shared_kv = {
            "sliding_attention": self.tt_kv_cache[self._last_layer_of_type["sliding_attention"]],
            "full_attention": self.tt_kv_cache[self._last_layer_of_type["full_attention"]],
        }
        T = self._spec.num_drafts
        # Drafter trace runs at the padded decode_width (rows >= self.batch idle).
        B = self.decode_width
        H_backbone = self._drafter_backbone_hidden
        mesh_config = self.model.mesh_config
        ccl_manager = self.model.ccl_manager
        has_tp = mesh_config is not None and mesh_config.tp > 1
        # ``fwd_const_kwargs`` is populated inside ``fwd()`` after the on-device
        # RoPE gather (the gathered cos/sin tensors are trace-internal — they
        # don't exist outside the closure).
        page_table_full = decode_buffers["page_table"]
        page_table_sliding = decode_buffers["page_table_sliding"]

        shard_vocab = self._spec_drafter_shard_vocab

        def fwd():
            # On-device RoPE gather. Each ttnn.embedding pulls one row per
            # slot from the 2D RoPE cache using ``_spec_position_2d_*_dev``
            # (uint32 [1, B]) → [1, B, head_dim] TILE, then unsqueeze_to_4D
            # → [1, 1, B, head_dim] which is the shape the drafter forward
            # consumes. Replaces 4 H2Ds per step (cos+sin × full+sliding)
            # plus the host-side row gather. Replayed once per execute_trace
            # since positions are constant across the round.
            cos_full = ttnn.unsqueeze_to_4D(
                ttnn.embedding(self._spec_position_2d_dev, rope_full_cos_cache, layout=ttnn.TILE_LAYOUT)
            )
            sin_full = ttnn.unsqueeze_to_4D(
                ttnn.embedding(self._spec_position_2d_dev, rope_full_sin_cache, layout=ttnn.TILE_LAYOUT)
            )
            cos_swa = ttnn.unsqueeze_to_4D(
                ttnn.embedding(self._spec_position_2d_sliding_dev, rope_swa_cos_cache, layout=ttnn.TILE_LAYOUT)
            )
            sin_swa = ttnn.unsqueeze_to_4D(
                ttnn.embedding(self._spec_position_2d_sliding_dev, rope_swa_sin_cache, layout=ttnn.TILE_LAYOUT)
            )
            fwd_const_kwargs = dict(
                shared_kv=shared_kv,
                cos_pos_full=cos_full,
                sin_pos_full=sin_full,
                cos_pos_sliding=cos_swa,
                sin_pos_sliding=sin_swa,
                cur_pos_tensor=self._spec_cur_pos_dev,
                cur_pos_tensor_sliding=self._spec_cur_pos_sliding_dev,
                page_table_full=page_table_full,
                page_table_sliding=page_table_sliding,
            )

            spec_input = self._spec_input_dev  # external buffer — never deallocate
            spec_input_owned = False  # True once we replace it with a concat result
            drafts: List = []

            for k in range(T):
                out_hidden, logits = self._spec.drafter.forward(
                    target_last_hidden=spec_input,
                    **fwd_const_kwargs,
                )
                # Per-shard top-1. We CANNOT all-gather the full logits
                # (16 MB × T per step) — that alone took ~600+ ms in profiling
                # of an earlier revision. Instead reconstruct the global
                # argmax via tiny tensors:
                #   topk_val [1,1,B,1] bf16 + topk_idx [1,1,B,1] uint16/32
                #   → add per-device shard_offset to idx (now a GLOBAL idx)
                #   → all_gather both along TP axis (each gather is ~128 B)
                #   → argmax over [1,1,B,tp] picks winning shard per batch row
                #   → gather corresponding global idx for the winner
                topk_val, topk_idx = ttnn.topk(logits, k=1, dim=-1)
                # logits is large but no longer needed — release before any
                # further compute to free L1.
                ttnn.deallocate(logits)

                # Reconstruct GLOBAL token IDs as TILE int32, then convert
                # once to ROW_MAJOR uint32 (the dtype/layout embed_tokens
                # consumes — same as _refresh_spec_inputs at server.py:1079).
                if has_tp:
                    # All-gather the per-shard top-1 results (TINY: each is
                    # [1,1,B,1] → [1,1,B,tp], ~128 B). Layout preserved (TILE).
                    val_g = ccl_allgather(topk_val, mesh_config, ccl_manager, dim=3)
                    idx_g_local = ccl_allgather(topk_idx, mesh_config, ccl_manager, dim=3)
                    # win_shard[b] ∈ {0..tp-1} = winning shard for batch row b.
                    # argmax returns ROW_MAJOR uint32; ttnn.gather requires
                    # TILE for both input AND index
                    # (gather_device_operation.cpp:78).
                    win_shard_rm = ttnn.argmax(val_g, dim=-1, keepdim=True)
                    ttnn.deallocate(val_g)
                    win_shard = ttnn.to_layout(win_shard_rm, ttnn.TILE_LAYOUT)
                    ttnn.deallocate(win_shard_rm)
                    # Pick the winning shard's LOCAL top-1 idx.
                    local_idx_tile = ttnn.gather(idx_g_local, dim=-1, index=win_shard)
                    ttnn.deallocate(idx_g_local)
                    # Promote to int32 so the scalar-multiply + add yield exact
                    # ints (uint16 would overflow at shard_vocab × 3 for vocab
                    # > 65k / tp).
                    local_idx_i32 = ttnn.typecast(local_idx_tile, ttnn.int32)
                    ttnn.deallocate(local_idx_tile)
                    win_i32 = ttnn.typecast(win_shard, ttnn.int32)
                    ttnn.deallocate(win_shard)
                    # global_idx = local_idx + win_shard * shard_vocab.
                    # shard_vocab is a Python scalar — ttnn.mul broadcasts it
                    # across the int32 tile.
                    shard_offset = ttnn.mul(win_i32, shard_vocab)
                    ttnn.deallocate(win_i32)
                    next_tokens_i32_tile = ttnn.add(local_idx_i32, shard_offset)
                    ttnn.deallocate(local_idx_i32)
                    ttnn.deallocate(shard_offset)
                    ttnn.deallocate(topk_idx)
                else:
                    # TP=1: local topk_idx IS the global idx.
                    next_tokens_i32_tile = ttnn.typecast(topk_idx, ttnn.int32)
                    ttnn.deallocate(topk_idx)
                    ttnn.deallocate(topk_val)
                # Cast to uint32 (still TILE), then layout-convert to
                # ROW_MAJOR for embed_tokens + host read.
                next_tokens_u32_tile = ttnn.typecast(next_tokens_i32_tile, ttnn.uint32)
                ttnn.deallocate(next_tokens_i32_tile)
                next_tokens = ttnn.to_layout(next_tokens_u32_tile, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(next_tokens_u32_tile)
                drafts.append(next_tokens)

                if k < T - 1:
                    # Reshape [1,1,B,1] → [1,B] for embed_tokens (view of
                    # next_tokens — do NOT deallocate; it's a trace output).
                    tokens_2d = ttnn.reshape(next_tokens, (1, B))
                    # target embed_tokens: column-parallel embed + scale + AG →
                    # [1, 1, B, H_backbone] bf16 ROW_MAJOR (or post-AG layout).
                    next_emb = self.model.embed_tokens(tokens_2d)
                    next_emb = ttnn.reshape(next_emb, (1, 1, B, H_backbone))
                    next_emb = ttnn.to_layout(next_emb, ttnn.TILE_LAYOUT)
                    # spec_input = [embed_of_prev_draft ‖ prev_out_hidden]
                    # along the hidden dim — same packing _refresh_spec_inputs
                    # uses for the host build (see ~line 1084: first half =
                    # embed, second half = paired hidden).
                    spec_input_new = ttnn.concat([next_emb, out_hidden], dim=3)
                    ttnn.deallocate(next_emb)
                    ttnn.deallocate(out_hidden)
                    if spec_input_owned:
                        ttnn.deallocate(spec_input)
                    spec_input = spec_input_new
                    spec_input_owned = True
                else:
                    ttnn.deallocate(out_hidden)
                    if spec_input_owned:
                        ttnn.deallocate(spec_input)

            return tuple(drafts)

        return fwd

    def _read_drafter_chained_drafts(self) -> List[List[int]]:
        """Read the T draft token tensors emitted by the chained drafter trace.

        Each draft tensor is shape [1, 1, B, 1] UINT32 ROW_MAJOR. Because the
        chained trace all-gathers the TP-sharded vocab BEFORE the per-iter
        argmax, every device in the mesh computes the same global indices —
        the result is replicated across the mesh, so reading device 0 yields
        the correct global token IDs. Returns a list of T lists of B ints.
        """
        T = self._spec.num_drafts
        out: List[List[int]] = []
        for k in range(T):
            draft_t = self._drafter_trace["drafts"][k]
            src = ttnn.get_device_tensors(draft_t)[0] if self._is_mesh else draft_t
            out.append(ttnn.to_torch(src).reshape(-1)[: self.batch].to(torch.int64).tolist())
        return out

    # ── Packed multi-token verify (Item 1.2) ────────────────────────────────

    def _preallocate_packed_verify_buffers(self):
        """Pre-allocate every packed-verify device buffer BEFORE trace capture.

        The packed verify runs P = T+1 query positions per slot through one
        target forward (``tt/attention/decode.py:packed_decode_forward``). To
        keep a lightly-loaded server fast, the verify trace is captured at
        several **occupancy buckets** (``_pv_bucket_list``) and per step the
        smallest bucket >= the active greedy-slot count is replayed. Each
        bucket gets its own buffer set; ``_alloc_pv_bucket`` builds one.

        All inputs are refreshed per step via ``copy_host_to_device_tensor``
        from ``_refresh_packed_verify_inputs`` — no device allocation in the
        hot loop. The verify path uses its own page tables; the full-attention
        key extent is capped at ``_pv_sk_cap``.
        """
        B = self.batch
        T = self._spec.num_drafts
        P = T + 1
        self._pv_p = P
        # Hot-block slots per verify row for the loop-free KV write: a P-token
        # tail (P <= block_size) straddles at most 2 cache pages, so each row
        # reserves 2 slots (slot 0 = the page holding cur_pos, slot 1 = spill).
        self._pv_blk = 2
        # Loop-free KV write: one paged_fill_cache instead of the per-p
        # paged_update_cache loop. Merge op is ttnn.embedding row-gather over a
        # per-head-flattened index — transpose-free, no large on-device index
        # materialization, ~0.7 ms/layer in trace (the dim-2 ttnn.gather
        # baseline is ~272 ms/layer at S2=4096).
        # See decode.py::_packed_fill_kv_loopfree*.
        # Per-device local KV-head count per layer type, read from the built
        # staging — needed to size/lay-out the embedding-merge per-head index
        # (nkv differs: full vs sliding). None if staging absent.
        self._pv_nkv_full = None
        self._pv_nkv_sliding = None
        if getattr(self.model, "tt_kv_staging", None):
            _ltypes = self.model.hf_config.layer_types
            for _li, _stg in enumerate(self.model.tt_kv_staging):
                if _stg is None:
                    continue
                _nkv = int(_stg[0].shape[1])
                if _ltypes[_li] == "full_attention" and self._pv_nkv_full is None:
                    self._pv_nkv_full = _nkv
                elif _ltypes[_li] == "sliding_attention" and self._pv_nkv_sliding is None:
                    self._pv_nkv_sliding = _nkv
        # Per-slot block index that the loop-free staging currently reflects
        # (-1 = unseeded ⇒ next verify reseeds from the cache). Set in
        # _refresh_loopfree_write_idx; reset to -1 on (re)assignment / compaction.
        self._pv_a_prev = [-1] * self.batch
        tp = self.model.mesh_config.tp if self.model.mesh_config else 1
        self._pv_h_local = self.model_args.num_attention_heads // tp

        # Full-attention key-extent cap, snapped down to a block multiple.
        cap = min(_PV_SK_CAP, self.max_user_seq_len)
        cap = (cap // self.block_size) * self.block_size
        self._pv_sk_cap = cap
        self._pv_nblocks_full = cap // self.block_size

        # Occupancy bucket list: from the env, filtered to [1, B], B always in.
        # At --batch < 32 this is [batch] (e.g. [8]) — the whole packed-verify
        # forward (QKV / SDPA / MLP / lm-head) then runs at B_v*P rows (8*4=32),
        # the 4x compute win vs the 32-user path. The only piece that needs 32
        # is the KV write, handled hazard-free by the (u,p) masked path below
        # (see up_kv_write) — no padding of the forward.
        buckets = sorted({int(x) for x in _PV_BUCKETS.split(",") if x.strip()} | {B})
        self._pv_bucket_list = [b for b in buckets if 1 <= b <= B]
        # Use the (u,p) masked KV-write path when the B_v*P packed rows fit the
        # decode split's DECODE_BATCH-user limit (i.e. --batch < 32). Otherwise
        # (batch == 32: B_v*P = 128) use the classic per-p loop at B_v=32.
        self._pv_up_kv_write = (max(self._pv_bucket_list) * P) <= DECODE_BATCH

        # Pre-baked causal mask tables. Row r encodes the mask for a query at
        # absolute position r: ``mask_table[r, k] = 0 if k <= r else NEG``.
        # The packed-verify trace gathers rows ``cur_pos + p`` via
        # ``ttnn.embedding`` instead of host-building and H2D-ing a
        # [B, 1, H_local*P, S_k] mask every step — on an underpowered host
        # the H2D is ~220 ms/step at production shapes (12.5 MB across 4
        # devices on PCIe Gen1 ×1; see test_mask_construction_profile in
        # tests/unit/test_paged_update_cache_preamble.py). The device gather
        # is ~1.8 ms — a ~120× win that lets speculative decode actually be
        # faster than the bare decode trace on this hardware.
        #
        # Speculator eligibility (_spec_eligible_at) guarantees cur_pos + T
        # stays below both cap and W, so the gathered rows stay in-table.
        W = self.sliding_cache_len
        NEG = -1e9
        mask_table_full_torch = torch.full((cap, cap), NEG, dtype=torch.bfloat16)
        for r in range(cap):
            mask_table_full_torch[r, : r + 1] = 0.0
        self._pv_mask_table_full_dev = self._alloc_device_tensor(
            mask_table_full_torch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        mask_table_slide_torch = torch.full((W, W), NEG, dtype=torch.bfloat16)
        for r in range(W):
            mask_table_slide_torch[r, : r + 1] = 0.0
        self._pv_mask_table_sliding_dev = self._alloc_device_tensor(
            mask_table_slide_torch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        # Baked per-position tables for the on-device index builds (int32 —
        # values exceed bf16's 8-bit mantissa; int eltwise has add/sub/mul):
        # local offset within a slot's staging region, identity, plus the
        # P-stride tables for position/kv-write rows.
        bs_lf = self.block_size
        blkt_lf = self._pv_blk * bs_lf
        S2_lf = self.batch * blkt_lf
        self._lf_local_dev = self._alloc_device_tensor(torch.arange(S2_lf, dtype=torch.int32) % blkt_lf, ttnn.int32)
        self._lf_iota_dev = self._alloc_device_tensor(torch.arange(S2_lf, dtype=torch.int32), ttnn.int32)

        self._pv_buckets = {bv: self._alloc_pv_bucket(bv) for bv in self._pv_bucket_list}

        # Per-slot carried spec state (slot-indexed, bucket-independent): the
        # target hidden that produced the slot's `next_token`. Draft-0 of the
        # next propose round pairs it with embed(next_token). Rows are remapped
        # by `_compact_slots`. Allocated at decode_width so it reshapes cleanly
        # into the width-DW drafter input scratch in `_refresh_spec_inputs`;
        # rows >= self.batch stay zero (idle padding).
        self._spec_hidden_host = torch.zeros(self.decode_width, self.model_args.hidden_size, dtype=torch.bfloat16)
        logger.info(
            f"Packed-verify buffers pre-allocated: P={P}, H_local={self._pv_h_local}, "
            f"sk_cap={cap} ({self._pv_nblocks_full} blocks), W={W}, "
            f"occupancy buckets={self._pv_bucket_list}, "
            f"mask_table_full={mask_table_full_torch.shape}, "
            f"mask_table_sliding={mask_table_slide_torch.shape}"
        )

    def _alloc_pv_bucket(self, B_v: int) -> dict:
        """Allocate the device buffers + persistent host scratch for one
        occupancy bucket — ``B_v`` slots, ``B_v*P`` packed rows. Row u*P+p is
        verify-row u's p-th packed token (verify rows are a compaction of the
        step's active verify slots; the slot↔row map is the step's slot list).
        """
        P = self._pv_p
        H_local = self._pv_h_local
        cap = self._pv_sk_cap
        W = self.sliding_cache_len
        # Loop-free KV-write index sizing. Staging is slot-indexed over ALL
        # decode slots (prefill seeds per slot), so size by self.batch — NOT
        # B_v. n_slots_all = batch * BLK hot blocks; S2 = staging seq length.
        n_slots_all = self.batch * self._pv_blk
        S2 = n_slots_all * self.block_size
        pt_full_idle = torch.full((B_v, self._pv_nblocks_full), self.scratch_full_id, dtype=torch.int32)
        pt_sliding_idle = torch.stack([self._idle_blocks_sliding for _ in range(B_v)], dim=0).clone()
        dev = {
            "tokens": self._alloc_device_tensor(torch.zeros(1, B_v * P, dtype=torch.int32), ttnn.uint32),
            "position_idx": self._alloc_device_tensor(torch.zeros(1, B_v * P, dtype=torch.int32), ttnn.uint32),
            "kv_write_idxs": [
                self._alloc_device_tensor(torch.full((B_v,), -1, dtype=torch.int32), ttnn.int32) for _ in range(P)
            ],
            "kv_write_idxs_sliding": [
                self._alloc_device_tensor(torch.full((B_v,), -1, dtype=torch.int32), ttnn.int32) for _ in range(P)
            ],
            "page_table": self._alloc_device_tensor(pt_full_idle, ttnn.int32),
            "page_table_sliding": self._alloc_device_tensor(pt_sliding_idle, ttnn.int32),
            # (user,position) KV write — used when B_v*P <= 32 (the decode
            # split's user limit), i.e. --batch < 32. The B_v*P (u,p) rows are
            # resharded once to a B_v*P-user decode spec, then written with P
            # SEQUENTIAL paged_update_cache calls, one per packed position. Each
            # call uses a masked write-idx (real only on the rows for that
            # position, -1 elsewhere) so only distinct users / distinct cache
            # tiles are touched per call — avoiding the tile read-modify-write
            # race a single all-positions call would hit. Row u*P+p ← verify-row
            # u, packed pos p (u-major/p-minor, matching ``position_idx``).
            "kv_write_up_masked_full": [
                self._alloc_device_tensor(torch.full((B_v * P,), -1, dtype=torch.int32), ttnn.int32) for _ in range(P)
            ],
            "kv_write_up_masked_sliding": [
                self._alloc_device_tensor(torch.full((B_v * P,), -1, dtype=torch.int32), ttnn.int32) for _ in range(P)
            ],
            "page_table_up_full": self._alloc_device_tensor(
                torch.full((B_v * P, self._pv_nblocks_full), self.scratch_full_id, dtype=torch.int32), ttnn.int32
            ),
            "page_table_up_sliding": self._alloc_device_tensor(
                torch.stack([self._idle_blocks_sliding for _ in range(B_v * P)], dim=0).clone(), ttnn.int32
            ),
            # ── Loop-free KV write via persistent staging ────────────────────
            # merge_idx (slot-indexed over ALL decode slots; layer-independent —
            # addresses staging positions, not pages): per staging position, the
            # source index into concat([staging, new_seq]) ( < S2 ⇒ copy staging,
            # >= S2 ⇒ a new token). hot_pt / hot_pt_sliding: physical block to
            # WRITE per hot-slot ( -1 = skip). Full-shape gather index is rebuilt
            # on device via ttnn.repeat, so only these small tensors are H2D'd.
            "merge_idx": self._alloc_device_tensor(torch.arange(S2, dtype=torch.int32), ttnn.uint32),
            # On-device per-step index build inputs — the ONLY loop-free
            # tensors pushed per step: cur_pos, rollover flag, verify-row base
            # (r*P, -1 idle). merge_idx/embed_idx/position_idx/kv_write_idxs
            # all derive from these in-trace (see _build_packed_verify_fwd).
            "lf_cur": self._alloc_device_tensor(torch.zeros(1, self.batch, dtype=torch.int32), ttnn.int32),
            "lf_roll": self._alloc_device_tensor(torch.zeros(1, self.batch, dtype=torch.int32), ttnn.int32),
            "lf_rmap": self._alloc_device_tensor(torch.full((1, self.batch), -1, dtype=torch.int32), ttnn.int32),
            # Baked iota tables at this bucket's row count.
            "lf_iota_p": self._alloc_device_tensor(
                (torch.arange(B_v * P, dtype=torch.int32) % P).reshape(1, B_v * P), ttnn.int32
            ),
            "lf_iota_p_major": self._alloc_device_tensor(
                (torch.arange(B_v * P, dtype=torch.int32) // B_v).reshape(1, B_v * P), ttnn.int32
            ),
            "hot_pt": self._alloc_device_tensor(torch.full((1, n_slots_all), -1, dtype=torch.int32), ttnn.int32),
            "hot_pt_sliding": self._alloc_device_tensor(
                torch.full((1, n_slots_all), -1, dtype=torch.int32), ttnn.int32
            ),
            "attn_mask": {
                "full_attention": self._alloc_device_tensor(
                    torch.zeros(B_v, 1, H_local * P, cap, dtype=torch.bfloat16),
                    ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "sliding_attention": self._alloc_device_tensor(
                    torch.zeros(B_v, 1, H_local * P, W, dtype=torch.bfloat16),
                    ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
            },
        }
        pt_up_full_idle = torch.full((B_v * P, self._pv_nblocks_full), self.scratch_full_id, dtype=torch.int32)
        pt_up_sliding_idle = torch.stack([self._idle_blocks_sliding for _ in range(B_v * P)], dim=0).clone()
        host = {
            "tokens": torch.zeros(1, B_v * P, dtype=torch.int32),
            "page_table": pt_full_idle.clone(),
            "page_table_sliding": pt_sliding_idle.clone(),
            # (u,p) masked write-idx mirrors (one [B_v*P] per packed position)
            # and the (u,p) page tables. See dev above.
            "kv_write_up_masked_full": [torch.full((B_v * P,), -1, dtype=torch.int32) for _ in range(P)],
            "kv_write_up_masked_sliding": [torch.full((B_v * P,), -1, dtype=torch.int32) for _ in range(P)],
            "page_table_up_full": pt_up_full_idle.clone(),
            "page_table_up_sliding": pt_up_sliding_idle.clone(),
            "_pt_up_full_idle": pt_up_full_idle,
            "_pt_up_sliding_idle": pt_up_sliding_idle,
            # ``mask_full`` / ``mask_slide`` are no longer host-built. The
            # packed-verify trace gathers the masks on device from
            # ``_pv_mask_table_*_dev``; see _build_packed_verify_fwd and the
            # commented-out block in _refresh_packed_verify_inputs.
            # Loop-free KV-write host scratch (merge/embed idxs build in-trace).
            "hot_pt": torch.full((1, n_slots_all), -1, dtype=torch.int32),
            "hot_pt_sliding": torch.full((1, n_slots_all), -1, dtype=torch.int32),
        }
        # Embedding-merge per-head-flattened gather index. For a flattened
        # [nkv*(S2+B_v*P), hd] concat view, row h*S2+j gathers concat position
        # h*(S2+B_v*P)+merge_idx[j]; baked here as the identity
        # (merge_idx==arange) and rebuilt in-trace each step.
        # nkv differs full vs sliding ⇒ two indices.
        if self._pv_nkv_sliding:
            src_seq = S2 + B_v * P

            def _embed_identity(nkv):
                off = (torch.arange(nkv, dtype=torch.int32) * src_seq).unsqueeze(1)  # [nkv,1]
                return (torch.arange(S2, dtype=torch.int32).unsqueeze(0) + off).reshape(1, nkv * S2)

            ef = _embed_identity(self._pv_nkv_full or 1)
            es = _embed_identity(self._pv_nkv_sliding)
            dev["embed_idx_full"] = self._alloc_device_tensor(ef, ttnn.uint32)
            dev["embed_idx_sliding"] = self._alloc_device_tensor(es, ttnn.uint32)

            # Per-head row offsets (h * src_seq), baked: the in-trace index
            # build broadcasts merge_idx across heads and adds these.
            def _hoff(nkv):
                return (
                    (torch.arange(nkv, dtype=torch.int32) * src_seq)
                    .unsqueeze(1)
                    .expand(nkv, S2)
                    .reshape(1, nkv * S2)
                    .contiguous()
                )

            dev["hoff_full"] = self._alloc_device_tensor(_hoff(self._pv_nkv_full or 1), ttnn.int32)
            dev["hoff_sliding"] = self._alloc_device_tensor(_hoff(self._pv_nkv_sliding), ttnn.int32)
        return {"B_v": B_v, "dev": dev, "host": host}

    def _pick_pv_bucket(self, n: int) -> int:
        """Smallest occupancy bucket >= n active verify slots."""
        for bv in self._pv_bucket_list:
            if bv >= n:
                return bv
        return self._pv_bucket_list[-1]

    def _build_packed_verify_fwd(self, B_v: int):
        """Closure that runs one packed multi-token verify forward for the
        ``B_v`` occupancy bucket — B_v*P query rows in a single target forward
        over that bucket's pre-allocated buffers. Captured as the bucket's
        trace, replayed when the step's active verify-slot count picks it.

        The argmax over the vocab is done **on device** (``ttnn.topk`` k=1) so
        the hot path reads back only the per-row (value, index) pair, not the
        full ``[B_v*P, vocab]`` logits. The lm-head logits are TP-sharded over
        the vocab, so topk runs per shard and the global token is
        reconstructed host-side in ``_read_packed_verify``.

        Returns ``(topk_val, topk_idx, hidden)``.
        """
        P = self._pv_p
        H_local = self._pv_h_local
        cap = self._pv_sk_cap
        W = self.sliding_cache_len
        hidden = self.model_args.hidden_size
        dev = self._pv_buckets[B_v]["dev"]
        spec = {
            "p": P,
            "position_idx": dev["position_idx"],
            "kv_write_idxs": dev["kv_write_idxs"],
            "kv_write_idxs_sliding": dev["kv_write_idxs_sliding"],
            "attn_mask": dev["attn_mask"],
            # (user,position) masked KV write — runs the forward at B_v (8*4=32
            # rows) but writes the cache with P sequential masked calls over the
            # B_v*P (u,p) rows resharded to a B_v*P-user spec. Each call writes
            # one position's rows only (distinct users → distinct cache tiles),
            # so it is hazard-free under paged_update_cache's tile RMW. Enabled
            # when B_v*P <= DECODE_BATCH (--batch < 32); else the per-p loop runs.
            "up_kv_write": self._pv_up_kv_write,
            "kv_write_up_masked_full": dev["kv_write_up_masked_full"],
            "kv_write_up_masked_sliding": dev["kv_write_up_masked_sliding"],
            "page_table_up_full": dev["page_table_up_full"],
            "page_table_up_sliding": dev["page_table_up_sliding"],
            # Loop-free KV write via persistent staging
            # (decode.py::_packed_fill_kv_loopfree_embed). merge_idx is shared
            # across all layers (it addresses staging positions, not pages); only
            # hot_pt (fill destination) differs full vs sliding. The per-layer
            # staging buffers ride on the attention layer (self.kv_staging), not
            # the spec.
            "merge_idx": dev["merge_idx"],
            "hot_pt": dev["hot_pt"],
            "hot_pt_sliding": dev["hot_pt_sliding"],
            "kv_merge": "embedding",
            "embed_idx_full": dev.get("embed_idx_full"),
            "embed_idx_sliding": dev.get("embed_idx_sliding"),
        }

        def fwd():
            # ── On-device per-step index build ───────────────────────────────
            # Inputs: 3 tiny [1, batch] int tensors (cur_pos, rollover, verify-
            # row base / -1 idle). Derives in-trace: position_idx, per-p KV
            # write rows (full + sliding), loop-free merge_idx and embed
            # indices. Replaces the host loops + ~40 H2D copies per step.
            # NOTE: ttnn.repeat_interleave on int32 silently round-trips values
            # through bfloat16 (307→308, 45518→45568 — bisected 2026-06-05,
            # minimal repro on tf4 device 0; broke acceptance 6→3 once values
            # crossed 256). `_ri()` is the bit-exact equivalent built from
            # ttnn.repeat (proven exact): [1,N] → [N,1] → repeat k cols → [1,N·k].
            def _ri(x, k):
                n = x.shape[-1]
                return ttnn.reshape(ttnn.repeat(ttnn.reshape(x, [n, 1]), [1, k]), [1, n * k])

            bs = self.block_size
            blkt = self._pv_blk * bs
            S2 = self.batch * blkt
            cur_b = ttnn.slice(dev["lf_cur"], [0, 0], [1, B_v])
            rmap_b = ttnn.slice(dev["lf_rmap"], [0, 0], [1, B_v])
            # position_idx [1, B_v*P] = cur + p (idle rows harmless).
            pos = ttnn.add(_ri(cur_b, P), dev["lf_iota_p"])
            ttnn.assign(ttnn.typecast(pos, ttnn.uint32), dev["position_idx"])
            # Per-p KV write rows (position-major): cur + p, skips (< 0)
            # for idle slots; sliding mirrors mod W (W is a power of 2).
            idle_pen = ttnn.multiply(ttnn.ltz(ttnn.repeat(rmap_b, [1, P])), 1 << 20)
            kv_all = ttnn.add(ttnn.repeat(cur_b, [1, P]), dev["lf_iota_p_major"])
            kv_full = ttnn.subtract(kv_all, idle_pen)
            kv_slide = ttnn.subtract(ttnn.bitwise_and(kv_all, W - 1), idle_pen)
            for p in range(P):
                ttnn.assign(
                    ttnn.reshape(ttnn.slice(kv_full, [0, p * B_v], [1, (p + 1) * B_v]), [B_v]),
                    dev["kv_write_idxs"][p],
                )
                ttnn.assign(
                    ttnn.reshape(ttnn.slice(kv_slide, [0, p * B_v], [1, (p + 1) * B_v]), [B_v]),
                    dev["kv_write_idxs_sliding"][p],
                )
            # merge_idx [S2]: identity, +bs on rollover prefix, S2+r*P+rel
            # for the P fresh tokens of each verify row.
            cur = _ri(dev["lf_cur"], blkt)  # [1, S2]
            roll = _ri(dev["lf_roll"], blkt)
            rmap = _ri(dev["lf_rmap"], blkt)
            off = ttnn.bitwise_and(cur, bs - 1)
            rel = ttnn.subtract(self._lf_local_dev, ttnn.reshape(off, [S2]))
            is_new = ttnn.eqz(ttnn.bitwise_and(rel, ~(P - 1)))  # 0 <= rel < P (P pow2)
            is_new = ttnn.multiply(is_new, ttnn.gez(ttnn.reshape(rmap, [S2])))
            m_new = ttnn.add(ttnn.add(ttnn.reshape(rmap, [S2]), rel), S2)
            spill = ttnn.multiply(ttnn.gtz(ttnn.reshape(roll, [S2])), ttnn.ltz(rel))
            m_old = ttnn.add(self._lf_iota_dev, ttnn.multiply(spill, bs))
            m_i32 = ttnn.add(m_old, ttnn.multiply(is_new, ttnn.subtract(m_new, m_old)))
            ttnn.assign(ttnn.typecast(m_i32, ttnn.uint32), dev["merge_idx"])
            if "embed_idx_full" in dev:
                m_row = ttnn.reshape(m_i32, [1, S2])
                for key, hoff, nkv in (
                    ("embed_idx_full", dev["hoff_full"], self._pv_nkv_full or 1),
                    ("embed_idx_sliding", dev["hoff_sliding"], self._pv_nkv_sliding),
                ):
                    rep = m_row if nkv == 1 else ttnn.repeat(m_row, [1, nkv])
                    ttnn.assign(ttnn.typecast(ttnn.add(rep, hoff), ttnn.uint32), dev[key])

            # On-device mask gather. Replaces the per-step host build +
            # ~220 ms H2D of the [B_v, 1, H_local*P, S_k] mask. position_idx
            # was already pushed by _refresh_packed_verify_inputs; we gather
            # its rows from the pre-baked [cap, cap] / [W, W] mask tables,
            # reshape, and repeat across heads. Result is bit-identical to
            # the host build for active rows — proven in
            # tests/unit/test_paged_update_cache_preamble.py
            # :test_mask_construction_profile. Idle rows' mask diverges
            # harmlessly (kv_write_idxs=-1 skips their KV writes and their
            # SDPA output is discarded; see the same test's docstring).
            #
            # ttnn.repeat([1,1,H_local,1]) follows torch.tile semantics —
            # the P input rows are stacked H_local times consecutively,
            # giving the head-major [h*P + p] layout packed_decode_forward
            # expects.
            mask_full_rows = ttnn.embedding(dev["position_idx"], self._pv_mask_table_full_dev, layout=ttnn.TILE_LAYOUT)
            mask_full = ttnn.reshape(mask_full_rows, (B_v, 1, P, cap))
            mask_full = ttnn.repeat(mask_full, [1, 1, H_local, 1])
            ttnn.deallocate(mask_full_rows)
            ttnn.assign(mask_full, dev["attn_mask"]["full_attention"])
            ttnn.deallocate(mask_full)

            mask_slide_rows = ttnn.embedding(
                dev["position_idx"], self._pv_mask_table_sliding_dev, layout=ttnn.TILE_LAYOUT
            )
            mask_slide = ttnn.reshape(mask_slide_rows, (B_v, 1, P, W))
            mask_slide = ttnn.repeat(mask_slide, [1, 1, H_local, 1])
            ttnn.deallocate(mask_slide_rows)
            ttnn.assign(mask_slide, dev["attn_mask"]["sliding_attention"])
            ttnn.deallocate(mask_slide)

            embeds = self.model.embed_tokens(dev["tokens"])
            embeds = ttnn.reshape(embeds, (1, 1, B_v * P, hidden))
            embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
            # DFlash also needs the 5 aux-hidden taps at every packed position
            # so committed positions can be turned into anchors on commit. The
            # taps are returned flattened into the tuple (so the trace warmup's
            # `for t in fwd(): t.deallocate()` and the capture unpack both work).
            want_aux = self._drafter_kind == "dflash"
            out = self.model(
                hidden_states=embeds,
                position_idx=None,
                page_table=dev["page_table"],
                page_table_sliding=dev["page_table_sliding"],
                kv_caches=self.tt_kv_cache,
                is_decode=True,
                packed=spec,
                return_hidden_state=True,
                return_aux_hidden=want_aux,
            )
            if want_aux:
                logits, hidden_t, aux_taps = out
            else:
                logits, hidden_t = out
            topk_val, topk_idx = ttnn.topk(logits, k=1, dim=-1)
            if want_aux:
                return (topk_val, topk_idx, hidden_t, *aux_taps)
            return (topk_val, topk_idx, hidden_t)

        return fwd

    def _refresh_packed_verify_inputs(self, B_v: int, verify_slots, drafts):
        """Host→device refresh of bucket ``B_v``'s packed-verify buffers.

        ``verify_slots`` (length <= B_v) is compacted onto verify rows
        ``0 .. len-1``: row ``r`` carries slot ``verify_slots[r]``. Its P
        packed rows ``r*P .. r*P+T`` carry tokens ``[next_token, d0, ..,
        d_{T-1}]`` at absolute positions ``c .. c+T``. Unused rows
        ``len .. B_v-1`` keep the ``-1`` skip sentinel so ``paged_update_cache``
        leaves their cache untouched.

        ``drafts[i]`` is slot i's list of T draft token IDs (slot-indexed).
        """
        P = self._pv_p
        W = self.sliding_cache_len
        bk = self._pv_buckets[B_v]
        dev, host = bk["dev"], bk["host"]

        host["tokens"].zero_()

        # Vectorized host fill — one [R, P] computation (was a B·P Python loop).
        # position_idx / kv_write idxs / merge_idx / embed_idx all build ON
        # DEVICE inside the trace; tokens + page tables are this step's only
        # per-verify-slot H2D pushes (drafts are real data; tables can shrink).
        R = len(verify_slots)
        if R:
            slots = [self.slots[i] for i in verify_slots]
            toks = torch.stack(
                [
                    torch.tensor([int(s.next_token)] + [int(t) for t in drafts[i]][: P - 1], dtype=torch.int32)
                    for s, i in zip(slots, verify_slots)
                ]
            )  # [R, P]
            host["tokens"][0, : R * P] = toks.reshape(-1)

            pt = torch.full((R, self._pv_nblocks_full), self.scratch_full_id, dtype=torch.int32)
            for r, s in enumerate(slots):
                m = min(int(s.full_pages.shape[0]), self._pv_nblocks_full)
                pt[r, :m] = s.full_pages[:m]
            host["page_table"][:R] = pt
            host["page_table_sliding"][:R] = torch.stack([s.sliding_pages for s in slots])

        if self._pv_up_kv_write:
            # (u,p) masked write-idx mirrors + per-row page tables (batch < 32).
            for p in range(P):
                host["kv_write_up_masked_full"][p].fill_(-1)
                host["kv_write_up_masked_sliding"][p].fill_(-1)
            host["page_table_up_full"].copy_(host["_pt_up_full_idle"])
            host["page_table_up_sliding"].copy_(host["_pt_up_sliding_idle"])
            rows = torch.arange(R, dtype=torch.int64) * P
            cur = torch.tensor([s.cur_pos for s in slots], dtype=torch.int32)
            for p in range(P):
                host["kv_write_up_masked_full"][p][rows + p] = cur + p
                host["kv_write_up_masked_sliding"][p][rows + p] = (cur + p) % W
            host["page_table_up_full"][: R * P] = pt.repeat_interleave(P, dim=0)
            host["page_table_up_sliding"][: R * P] = host["page_table_sliding"][:R].repeat_interleave(P, dim=0)

        ttnn.copy_host_to_device_tensor(self._host_tensor(host["tokens"], ttnn.uint32), dev["tokens"])
        ttnn.copy_host_to_device_tensor(self._host_tensor(host["page_table"], ttnn.int32), dev["page_table"])
        ttnn.copy_host_to_device_tensor(
            self._host_tensor(host["page_table_sliding"], ttnn.int32), dev["page_table_sliding"]
        )
        # (u,p) masked write-idx + page tables (consumed only when up_kv_write).
        if self._pv_up_kv_write:
            for p in range(P):
                ttnn.copy_host_to_device_tensor(
                    self._host_tensor(host["kv_write_up_masked_full"][p].contiguous(), ttnn.int32),
                    dev["kv_write_up_masked_full"][p],
                )
                ttnn.copy_host_to_device_tensor(
                    self._host_tensor(host["kv_write_up_masked_sliding"][p].contiguous(), ttnn.int32),
                    dev["kv_write_up_masked_sliding"][p],
                )
            ttnn.copy_host_to_device_tensor(
                self._host_tensor(host["page_table_up_full"], ttnn.int32), dev["page_table_up_full"]
            )
            ttnn.copy_host_to_device_tensor(
                self._host_tensor(host["page_table_up_sliding"], ttnn.int32), dev["page_table_up_sliding"]
            )

        # Loop-free packed KV-write indices (consumed by the staging write path
        # in packed_decode_forward).
        self._refresh_loopfree_write_idx(B_v, verify_slots)

    def _seed_staging(self, i):
        """One-time seed of slot ``i``'s staging slot-0 with the committed
        content of its current hot block, read from the cache (eager, off the
        hot path — once per request at the first verify, and after a compaction
        move). Sets ``_pv_a_prev[i]``. ``ttnn.slice`` extracts one block (no
        transpose); this is the ONLY committed-cache read in the design and it
        is amortized over the whole generation.
        """
        s = self.slots[i]
        c = s.cur_pos
        bs = self.block_size
        BLK = self._pv_blk
        a = c // bs
        self._pv_a_prev[i] = a
        if c % bs == 0:
            return  # hot block is fresh (empty committed prefix) — zeros are fine
        W = self.sliding_cache_len
        a_slide = (c % W) // bs
        dst = i * BLK * bs  # staging position of slot i, block-slot 0 (block-aligned)
        layer_types = self.model.hf_config.layer_types
        for li, staging in enumerate(self.model.tt_kv_staging):
            if staging is None:
                continue
            cache = self.model.tt_kv_cache[li]
            is_sliding = layer_types[li] == "sliding_attention"
            pages = s.sliding_pages if is_sliding else s.full_pages
            ablk = a_slide if is_sliding else a
            npages = int(pages.shape[0])
            if ablk >= npages:
                continue
            blk_phys = int(pages[ablk])
            for kv in (0, 1):
                cch, stg = cache[kv], staging[kv]
                nkv, S2, hd = stg.shape[1], stg.shape[2], stg.shape[3]
                block = ttnn.slice(cch, [blk_phys, 0, 0, 0], [blk_phys + 1, nkv, bs, hd])  # [1,nkv,bs,hd]
                parts = []
                if dst > 0:
                    parts.append(ttnn.slice(stg, [0, 0, 0, 0], [1, nkv, dst, hd]))
                parts.append(block)
                if dst + bs < S2:
                    parts.append(ttnn.slice(stg, [0, 0, dst + bs, 0], [1, nkv, S2, hd]))
                rebuilt = ttnn.concat(parts, dim=2)
                ttnn.assign(rebuilt, stg)
                ttnn.deallocate(rebuilt)
                for pt in parts:
                    ttnn.deallocate(pt)

    def _refresh_loopfree_write_idx(self, B_v, verify_slots):
        """Build + push the indices for the loop-free packed KV write
        (decode.py::_packed_fill_kv_loopfree).

        Staging is SLOT-indexed over all decode slots: slot ``i`` owns staging
        positions ``[i*BLK*bs, (i+1)*BLK*bs)`` — block-slot 0 = the cur_pos
        block, block-slot 1 = the spill block. ``merge_idx`` (shared full/
        sliding) is the per-staging-position source into ``concat([staging,
        new_seq])``: committed positions copy from staging (identity, or shifted
        one block on a rollover), the P new positions pull from ``new_seq``
        (concat index ``S2 + r*P + p`` for verify row ``r``). ``hot_pt`` /
        ``hot_pt_sliding`` name the physical page to fill each slot's block(s)
        into ( -1 = skip ⇒ idle slots / unused spill block untouched).

        cur_pos advances by ``n_acc+1 <= P < bs`` per step, so the block index
        advances by 0 or 1; a +1 (rollover) means the new hot block's committed
        prefix is last step's spill block (staging block-slot 1) — pulled via
        the shifted source. In the spec-eligible regime ``cur_pos < W`` so the
        sliding ring never wraps (``cur_pos % W == cur_pos``).
        """
        P = self._pv_p
        bs = self.block_size
        W = self.sliding_cache_len
        BLK = self._pv_blk
        bk = self._pv_buckets[B_v]
        dev, host = bk["dev"], bk["host"]

        hf = host["hot_pt"]
        hf.fill_(-1)
        hsl = host["hot_pt_sliding"]
        hsl.fill_(-1)
        # merge_idx / embed_idx build ON DEVICE inside the verify trace from
        # cur/roll/rmap (see fwd preamble); host pushes only [1, batch] tensors.
        lf_cur = torch.zeros(1, self.batch, dtype=torch.int32)
        lf_roll = torch.zeros(1, self.batch, dtype=torch.int32)
        lf_rmap = torch.full((1, self.batch), -1, dtype=torch.int32)

        for r, i in enumerate(verify_slots):
            if self._pv_a_prev[i] < 0:
                self._seed_staging(i)  # first verify (or post-compaction): seed from cache
            s = self.slots[i]
            c = s.cur_pos
            a = c // bs
            off = c % bs
            a_prev = self._pv_a_prev[i]
            lf_cur[0, i] = c
            lf_roll[0, i] = 1 if a == a_prev + 1 else 0
            lf_rmap[0, i] = r * P
            # fill destinations (physical pages) for this slot's block(s):
            n_full = int(s.full_pages.shape[0])
            n_slide = int(s.sliding_pages.shape[0])
            a_slide = (c % W) // bs
            if a < n_full:
                hf[0, i * BLK] = int(s.full_pages[a])
            if a_slide < n_slide:
                hsl[0, i * BLK] = int(s.sliding_pages[a_slide])
            if off + P > bs:  # tail spills into the next page
                if a + 1 < n_full:
                    hf[0, i * BLK + 1] = int(s.full_pages[a + 1])
                if a_slide + 1 < n_slide:
                    hsl[0, i * BLK + 1] = int(s.sliding_pages[a_slide + 1])
            self._pv_a_prev[i] = a  # staging block-slot 0 now holds block a

        ttnn.copy_host_to_device_tensor(self._host_tensor(hf.contiguous(), ttnn.int32), dev["hot_pt"])
        ttnn.copy_host_to_device_tensor(self._host_tensor(hsl.contiguous(), ttnn.int32), dev["hot_pt_sliding"])
        ttnn.copy_host_to_device_tensor(self._host_tensor(lf_cur, ttnn.int32), dev["lf_cur"])
        ttnn.copy_host_to_device_tensor(self._host_tensor(lf_roll, ttnn.int32), dev["lf_roll"])
        ttnn.copy_host_to_device_tensor(self._host_tensor(lf_rmap, ttnn.int32), dev["lf_rmap"])

    def _read_packed_verify(self, B_v: int, want_aux: bool = False):
        """Read bucket ``B_v``'s packed-verify trace outputs to host.

        Returns ``(tgt, hidden)`` — ``tgt`` is a length-``B_v*P`` list of the
        target's per-position argmax token IDs; ``hidden`` is the per-position
        post-norm hidden ``[B_v*P, hidden]`` bf16. Both are verify-row-indexed
        (row r ↔ the step's verify_slots[r]).

        ``want_aux`` (the DFlash path) additionally returns the K aux-hidden
        taps and **skips the hidden D2H** (``hidden`` is None): DFlash conditions
        its next anchors on the aux taps, not the post-norm hidden, so reading
        the ``[B_v*P, hidden]`` hidden back every step is pure waste. MTP
        (``want_aux=False``) still reads it for its carried ``spec_hidden``.

        The argmax was computed on device (``ttnn.topk`` k=1) so only the
        per-row (value, index) pair is read back, not the full logits. The
        lm-head logits are TP-sharded over the vocab, so topk ran per shard:
        the global token is ``local_idx + shard * (vocab // tp)`` for the
        shard with the largest value.
        """
        tr = self._packed_verify_traces[B_v]
        val_t = tr["topk_val"]
        idx_t = tr["topk_idx"]
        ht = tr["hidden"]
        if self._is_mesh and self.on_device_sampling:
            tp = self.model.mesh_config.tp if self.model.mesh_config else 1
            shard_vocab = self.model_args.vocab_size // tp
            vals = torch.stack(
                [ttnn.to_torch(ttnn.get_device_tensors(val_t)[d]).float().reshape(-1) for d in range(tp)], dim=0
            )  # [tp, B*P]
            idxs = torch.stack(
                [ttnn.to_torch(ttnn.get_device_tensors(idx_t)[d]).reshape(-1).to(torch.int64) for d in range(tp)],
                dim=0,
            )  # [tp, B*P]
            win = vals.argmax(dim=0)  # [B*P] — winning shard per row
            rows = torch.arange(win.shape[0])
            tgt = (idxs[win, rows] + win.to(torch.int64) * shard_vocab).tolist()
            # DFlash (want_aux) ignores hidden — skip its D2H entirely.
            hidden = None if want_aux else ttnn.to_torch(ttnn.get_device_tensors(ht)[0]).to(torch.bfloat16)
        else:
            # TP=1 / no on-device sampling: logits are full-vocab, topk gave
            # the global argmax directly.
            src = ttnn.get_device_tensors(idx_t)[0] if self._is_mesh else idx_t
            tgt = ttnn.to_torch(src).reshape(-1).to(torch.int64).tolist()
            hidden = (
                None
                if want_aux
                else ttnn.to_torch(ttnn.get_device_tensors(ht)[0] if self._is_mesh else ht).to(torch.bfloat16)
            )
        if hidden is not None:
            hidden = hidden.reshape(-1, self.model_args.hidden_size)
        if want_aux:
            # Phase 2a: keep the K aux taps ON DEVICE (each [1,1,B_v*P,hidden],
            # replicated across TP). The committed positions are gathered from
            # them into `_aux_dev` on-device in `append_committed_ondevice`,
            # eliminating the ~K*[B_v*P,hidden] D2H (the largest per-step
            # transfer) and the matching `_aux_dev` H2D. Return the device list.
            return tgt, hidden, self._packed_verify_traces[B_v]["aux"]
        return tgt, hidden

    def _capture_traces(self):
        """Allocate buffers, run un-traced compile/warmup for every pass, then
        capture all traces back-to-back.

        Captures one prefill trace per bucket length plus one decode trace.
        The compile run serves as trace warmup: it forces every kernel — most
        critically the CCL fabric ops — to allocate their persistent state
        (program cache, semaphores, etc.) outside the trace. With the CCL
        helpers passing pre-allocated barrier / RS / AG semaphores, the actual
        capture then enqueues no host writes.

        tt-metal disallows host-side buffer allocations while any trace is
        live (including the gap between traces), so all allocations and all
        compile runs must happen before the first ``begin_trace_capture``.
        """
        # Drain any host→device writes still in flight from model load (weights,
        # KV caches, etc.). A stray write that lands inside trace capture trips
        # `Writes are not supported during trace capture` in fd_mesh_command_queue.
        ttnn.synchronize_device(self.mesh_device)

        # Build all buffers + fwd closures up front. Any host→device allocation
        # has to land before any trace capture begins.
        prefill_pairs = [(bl, *self._build_prefill_fwd(bl)) for bl in self.bucket_lens]
        decode_buffers, decode_fwd = self._build_decode_fwd()

        # Drafter fwd closure. The drafter trace is T-agnostic: it operates on
        # fixed pre-allocated input buffers, so for multi-draft (T>1) the same
        # trace is replayed T times with host-mediated argmax+embed between
        # replays (see _step_decode's propose loop).
        drafter_fwd = None
        dflash_fwd = None
        dflash_append_fwd = None
        packed_verify_fwds = {}  # B_v -> fwd closure
        if self._spec is not None:
            # MTP captures a chained T-draft trace; DFlash captures a single
            # batched propose trace + a single batched append trace (the
            # commit-side anchor write — kept off the hot path otherwise it
            # allocates while traces are active and corrupts state).
            if self._drafter_kind == "dflash":
                dflash_fwd = self._spec.build_propose_fwd()
                dflash_append_fwd = self._spec.build_append_fwd()
            else:
                drafter_fwd = self._build_drafter_fwd(decode_buffers)
            # Packed multi-token verify forward, one per occupancy bucket.
            # Own page-table / mask / token buffers — independent of the
            # decode trace.
            packed_verify_fwds = {bv: self._build_packed_verify_fwd(bv) for bv in self._pv_bucket_list}

        # Warmup: compile each kernel against its bucket-specific shape.
        for bl, _, fwd in prefill_pairs:
            logger.info(f"Compiling prefill bucket={bl} (trace warmup)...")
            out = fwd()
            out.deallocate(True)
            ttnn.synchronize_device(self.mesh_device)

        logger.info("Compiling decode (trace warmup)...")
        out = decode_fwd()
        # decode_fwd returns either a single tensor (default) or a 2-tuple
        # (output, final_hidden) when spec decode is enabled. Deallocate both.
        if isinstance(out, tuple):
            for t in out:
                if t is not None:
                    t.deallocate(True)
        else:
            out.deallocate(True)
        ttnn.synchronize_device(self.mesh_device)

        if drafter_fwd is not None:
            logger.info("Compiling drafter (trace warmup)...")
            for t in drafter_fwd():
                t.deallocate(True)
            ttnn.synchronize_device(self.mesh_device)

        if dflash_fwd is not None:
            logger.info("Compiling dflash propose (trace warmup)...")
            # fwd() returns (topk_val, topk_idx) — on-device argmax of the draft
            # logits (see DflashSpeculativeDecoder.build_propose_fwd).
            for t in dflash_fwd():
                t.deallocate(True)
            ttnn.synchronize_device(self.mesh_device)
            # Op-section profile (GEMMA4_DFLASH_OPPROF=1): a SECOND eager pass,
            # post-compile, so the section times reflect steady-state compute (not
            # the compile pass). The syncs run only here, never in capture/replay
            # (mirrors PV_OPPROF). At warmup the noise write_idxs are all -1 ⇒
            # paged_update_cache skips, so this pass leaves the cache untouched.
            from models.demos.gemma4_cody.tt.dflash.model import _DFLASH_OPPROF
            from models.demos.gemma4_cody.tt.dflash.model import DFLASH_OPPROF as _df_opprof

            if _DFLASH_OPPROF:
                for k in _df_opprof:
                    if k != "active":
                        _df_opprof[k] = 0.0
                _df_opprof["active"] = True
                _t_df = time.perf_counter()
                for t in dflash_fwd():
                    t.deallocate(True)
                ttnn.synchronize_device(self.mesh_device)
                _df_opprof["active"] = False
                _df_tot = time.perf_counter() - _t_df
                _df_secs = sum(v for k, v in _df_opprof.items() if k != "active")
                logger.info(
                    f"DFLASH-OPPROF (propose compute, B={self._spec.batch}, MA={self._spec.max_anchors}, "
                    f"eager post-compile): total={_df_tot * 1e3:.0f}ms | "
                    f"qkv={_df_opprof['qkv'] * 1e3:.0f} kvwrite={_df_opprof['kvwrite'] * 1e3:.0f} "
                    f"rope_sdpa={_df_opprof['rope_sdpa'] * 1e3:.0f} ccl_oproj={_df_opprof['ccl_oproj'] * 1e3:.0f} "
                    f"mlp={_df_opprof['mlp'] * 1e3:.0f} head={_df_opprof['head'] * 1e3:.0f} "
                    f"(rope_sdpa+kvwrite scale with MA ⇒ cache-len bucketing; qkv/ccl_oproj/mlp scale with B ⇒ batch bucketing)"
                )

        if dflash_append_fwd is not None:
            # Append warmup MUST follow propose warmup — propose populates
            # `_q_sharded_mem_B`, which `write_anchors_packed` consumes. At
            # warmup the widx tensors are all -1 ⇒ paged_update_cache skips
            # every slot, so the anchor cache stays zero (no spurious writes).
            logger.info("Compiling dflash append (trace warmup)...")
            dflash_append_fwd()  # write_anchors_packed returns None
            ttnn.synchronize_device(self.mesh_device)

        from models.demos.gemma4_cody.tt.attention.decode import PV_OPPROF as _pv_opprof

        for bv, pv_fwd in packed_verify_fwds.items():
            logger.info(f"Compiling packed verify bucket B_v={bv} (trace warmup)...")
            # Op-profile this (untraced) warmup: PV_OPPROF["active"] gates the
            # section timers in packed_decode_forward / layer.py — they run
            # only here, never during capture/replay.
            _pv_opprof["attn_prep"] = _pv_opprof["sdpa"] = _pv_opprof["mlp"] = 0.0
            _pv_opprof["active"] = True
            _t_op = time.perf_counter()
            for t in pv_fwd():
                t.deallocate(True)
            ttnn.synchronize_device(self.mesh_device)
            _pv_opprof["active"] = False
            _op_total = time.perf_counter() - _t_op
            _op_other = _op_total - _pv_opprof["attn_prep"] - _pv_opprof["sdpa"] - _pv_opprof["mlp"]
            logger.info(
                f"PV-OPPROF B_v={bv}: warmup_total={_op_total * 1e3:.0f}ms | "
                f"attn_prep(P-loop)={_pv_opprof['attn_prep'] * 1e3:.0f}ms "
                f"sdpa={_pv_opprof['sdpa'] * 1e3:.0f}ms "
                f"mlp={_pv_opprof['mlp'] * 1e3:.0f}ms "
                f"other(lm_head/topk/embed/norms/CCL/dispatch)={_op_other * 1e3:.0f}ms"
            )

        # Capture all traces back-to-back. Order doesn't matter; we just keep
        # them in bucket-ascending order to match self.bucket_lens.
        prefill_traces: List[dict] = []
        for bl, buffers, fwd in prefill_pairs:
            logger.info(f"Capturing prefill trace bucket={bl}...")
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            output = fwd()
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            prefill_traces.append({"trace_id": tid, "output": output, **buffers})
        logger.info(f"Captured {len(prefill_traces)} prefill traces")

        logger.info("Capturing decode trace...")
        decode_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        decode_output = decode_fwd()
        ttnn.end_trace_capture(self.mesh_device, decode_trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Decode trace captured")

        # Unpack the optional spec-decode "final_hidden" trace output.
        if isinstance(decode_output, tuple):
            decode_output, decode_final_hidden = decode_output
        else:
            decode_final_hidden = None

        decode_trace_dict = {
            "trace_id": decode_trace_id,
            "output": decode_output,
            "final_hidden": decode_final_hidden,  # None unless spec decode enabled
            **decode_buffers,
        }

        # Drafter trace: independent of the decode trace, captures the drafter's
        # T chained forwards in ONE trace. Replayed once per _step_decode after
        # the target finishes; emits T draft tokens with no inter-iter host
        # round-trip. The drafter's internal ops allocate their output buffers
        # at capture time only; replay reuses those buffers, so no mid-trace
        # host allocations corrupt other traces.
        if drafter_fwd is not None:
            logger.info("Capturing drafter trace (chained T-draft)...")
            drafter_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            drafter_outputs = drafter_fwd()  # tuple of T tensors, each [1, 1, B, 1] uint32
            ttnn.end_trace_capture(self.mesh_device, drafter_trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            logger.info(f"Drafter trace captured (T={len(drafter_outputs)})")
            self._drafter_trace = {
                "trace_id": drafter_trace_id,
                "drafts": drafter_outputs,
            }
        else:
            self._drafter_trace = None

        # DFlash propose trace: one batched forward (own anchor KV cache). The
        # argmax is on-device (ttnn.topk) so the trace emits only the per-row
        # (value, index) pair — the full draft-vocab logits stay on device.
        if dflash_fwd is not None:
            logger.info("Capturing dflash propose trace...")
            dflash_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            dflash_val, dflash_idx = dflash_fwd()  # each [1,1,B*(block-1),1] per vocab shard
            ttnn.end_trace_capture(self.mesh_device, dflash_trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            logger.info("DFlash propose trace captured.")
            self._dflash_trace = {"trace_id": dflash_trace_id, "topk_val": dflash_val, "topk_idx": dflash_idx}
        else:
            self._dflash_trace = None

        # DFlash append trace: writes the just-committed positions' anchors
        # into the cache. Replayed at commit time (after a host refresh of
        # `_aux_dev` + `_anchor_widx_devs`) — keeping the append in a trace is
        # what removes the "Allocating device buffers... while a trace is
        # active" warning that was corrupting state across steps.
        if dflash_append_fwd is not None:
            logger.info("Capturing dflash append trace...")
            dflash_append_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            dflash_append_fwd()
            ttnn.end_trace_capture(self.mesh_device, dflash_append_trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            logger.info("DFlash append trace captured.")
            self._dflash_append_trace = {"trace_id": dflash_append_trace_id}
        else:
            self._dflash_append_trace = None

        # Packed multi-token verify traces — one per occupancy bucket.
        # _step_decode replays the bucket matching the active verify-slot count.
        if packed_verify_fwds:
            self._packed_verify_traces = {}
            for bv, pv_fwd in packed_verify_fwds.items():
                logger.info(f"Capturing packed verify trace B_v={bv}...")
                pv_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                pv_out = pv_fwd()
                ttnn.end_trace_capture(self.mesh_device, pv_trace_id, cq_id=0)
                ttnn.synchronize_device(self.mesh_device)
                # DFlash appends 5 aux taps after (topk_val, topk_idx, hidden);
                # MTP returns exactly the 3 (pv_out[3:] is empty).
                self._packed_verify_traces[bv] = {
                    "trace_id": pv_trace_id,
                    "topk_val": pv_out[0],
                    "topk_idx": pv_out[1],
                    "hidden": pv_out[2],
                    "aux": list(pv_out[3:]),
                }
            logger.info(f"Packed verify traces captured: buckets={list(self._packed_verify_traces)}")
        else:
            self._packed_verify_traces = None

        return (prefill_traces, decode_trace_dict)

    # ── Worker loop ─────────────────────────────────────────────────────────

    def submit(self, req: _Request) -> None:
        self.request_queue.put(req)

    def stop(self) -> None:
        self._stop.set()
        self.request_queue.put(None)  # wake the worker

    def _free_slot_index(self) -> Optional[int]:
        for i, s in enumerate(self.slots):
            if s.rid is None:
                return i
        return None

    def _has_active_slot(self) -> bool:
        return any(s.rid is not None for s in self.slots)

    def _release_slot_pages(self, slot: _Slot) -> None:
        """Return ``slot``'s physical KV-cache pages to the free pool."""
        if slot.full_pages.numel() > 0:
            self._free_full_pages.extend(int(p) for p in slot.full_pages.tolist())
        if slot.sliding_pages.numel() > 0:
            self._free_sliding_pages.extend(int(p) for p in slot.sliding_pages.tolist())

    def _compact_slots(self) -> None:
        """Slide active slots into the lowest free indices.

        Pages live on the slot (not at a fixed index), so moving a slot is
        a host-side rebind — no device K/V copy. Sampling host state and
        on-device RNG seeds are remapped in lock-step. This avoids the
        slot-0-frees / slot-1-stays-alive kernel transition that empirically
        corrupts the surviving slot's K/V.
        """
        write_idx = 0
        moves: List[tuple[int, int]] = []  # (dst, src)
        for read_idx in range(self.batch):
            s = self.slots[read_idx]
            if s.rid is None:
                continue
            if write_idx != read_idx:
                self.slots[write_idx] = s
                self.slots[read_idx] = _Slot()
                self._sampling_top_k[write_idx] = self._sampling_top_k[read_idx]
                self._sampling_top_p[write_idx] = self._sampling_top_p[read_idx]
                self._sampling_temp[write_idx] = self._sampling_temp[read_idx]
                self._slot_torch_rngs[write_idx] = self._slot_torch_rngs[read_idx]
                # Speculative-decode per-slot state moves with the slot.
                if self._spec is not None:
                    if self._drafter_kind == "dflash":
                        self._spec.move_slot(write_idx, read_idx)
                    else:
                        self._spec_hidden_host[write_idx] = self._spec_hidden_host[read_idx]
                # Loop-free staging is slot-indexed and not paged, so it cannot
                # follow a slot for free. Invalidate both ends → the moved user
                # reseeds its staging from the cache on its next verify.
                if getattr(self, "_pv_a_prev", None) is not None:
                    self._pv_a_prev[write_idx] = -1
                    self._pv_a_prev[read_idx] = -1
                moves.append((write_idx, read_idx))
            write_idx += 1

        if not moves:
            return

        logger.info(f"Compacted slots (write_idx ← read_idx): {moves}")
        self._push_sampling_params()
        if self.on_device_sampling:
            sm = self.model.sampling.seed_manager
            remap = list(range(self.batch))
            for write_idx, read_idx in moves:
                remap[write_idx] = read_idx
            sm.apply_slot_remap(torch.tensor(remap, dtype=torch.int32))

    def _worker_loop(self) -> None:
        try:
            while not self._stop.is_set():
                # Pull as many new requests as fit (slot + page budget).
                self._intake_new_requests()

                if not self._has_active_slot():
                    # If admission stalled with nothing in flight, all pages
                    # are already free — by construction _intake_new_requests
                    # would have admitted at least one waiter. So _waiting is
                    # empty here. Block on the external queue for new work.
                    try:
                        req = self.request_queue.get(timeout=0.1)
                    except Empty:
                        continue
                    if req is None:
                        return
                    self._intake_new_requests(initial=req)
                    continue

                self._step_decode()
        except Exception:
            logger.exception("Engine worker crashed")
            # Flush any in-flight requests with an error sentinel so callers don't hang.
            for s in self.slots:
                if s.request is not None:
                    s.request.emit({"error": "engine crashed"})
                    s.request.emit(None)

    def _intake_new_requests(self, initial: Optional[_Request] = None) -> None:
        """Drain the request queue into ``_waiting`` and admit FIFO.

        Admission requires both a free slot AND enough free pages to cover
        the request's worst-case allocation. If the head of ``_waiting``
        can't be admitted yet, the worker blocks (head-of-line). Requests
        cancelled while waiting are dropped without ever paying for prefill.
        """
        # Drain external queue into the internal waiting list.
        if initial is not None:
            self._waiting.append(initial)
        while True:
            try:
                req = self.request_queue.get_nowait()
            except Empty:
                break
            if req is None:
                self._stop.set()
                return
            self._waiting.append(req)

        # Drop cancelled before they get scheduled.
        survivors: List[_Request] = []
        for r in self._waiting:
            if r.cancelled:
                r.emit(None)
            else:
                survivors.append(r)
        self._waiting = survivors

        # Admit FIFO. Stop on first head-of-line miss — the next decode step
        # might free a slot and pages.
        while self._waiting:
            req = self._waiting[0]
            free = self._free_slot_index()
            if free is None:
                return
            prompt_len = int(req.input_ids.shape[0])
            # Cap the prompt length used for the page-budget check at the
            # post-truncation length so admission matches what _prefill_request
            # will actually consume.
            effective_prompt_len = min(prompt_len, self.max_user_seq_len - 1)
            needed_full = self._pages_needed(effective_prompt_len, req.max_new_tokens)
            if len(self._free_full_pages) < needed_full:
                return
            if len(self._free_sliding_pages) < self.blocks_per_user_sliding:
                return
            self._waiting.pop(0)
            self._prefill_request(free, req)

    def _pick_bucket(self, prompt_len: int) -> int:
        """Smallest bucket index whose length >= prompt_len.

        ``prompt_len`` is assumed already capped at ``max_user_seq_len - 1``
        (the truncation happens in ``_prefill_request``), so the largest
        bucket always satisfies the request.
        """
        for i, bl in enumerate(self.bucket_lens):
            if bl >= prompt_len:
                return i
        return len(self.bucket_lens) - 1

    def _pages_needed(self, prompt_len: int, max_new_tokens: int) -> int:
        """Number of full-attention pages a request will consume over its
        lifetime, capped at the per-slot maximum. Sliding allocation is fixed
        at ``blocks_per_user_sliding`` regardless of prompt length.
        """
        n = (prompt_len + max_new_tokens + self.block_size - 1) // self.block_size
        return min(n, self.blocks_per_user_max)

    def _prefill_request(self, slot_idx: int, req: _Request) -> None:
        """Run the bucketed prefill trace for a single user into slot `slot_idx`.

        Routes to the smallest prefill bucket whose length covers the prompt,
        reserves ``ceil((prompt_len + max_new_tokens) / block_size)`` full-
        attention pages from the global pool (capped at blocks_per_user_max),
        and writes per-slot sampling params. The caller (``_intake_new_requests``)
        guarantees enough pages are free before invoking this.
        """
        prompt_len = int(req.input_ids.shape[0])
        if prompt_len >= self.max_user_seq_len:
            # We need at least one position left for decode.
            req.input_ids = req.input_ids[: self.max_user_seq_len - 1]
            prompt_len = int(req.input_ids.shape[0])

        bucket_idx = self._pick_bucket(prompt_len)
        trace = self._prefill_traces[bucket_idx]
        bucket_len = trace["bucket_len"]
        bucket_blocks = trace["bucket_blocks"]

        # Pad prompt to bucket length. Padding tokens write garbage into KV
        # positions past prompt_len, which decode never reads (cur_pos is
        # bounded by SDPA).
        padded = torch.zeros(bucket_len, dtype=torch.int32)
        padded[:prompt_len] = req.input_ids.to(torch.int32)

        # Reserve pages from the global pool. Full-attention allocation is
        # sized to cover prompt + generation; sliding is per-user fixed.
        needed_full = self._pages_needed(prompt_len, req.max_new_tokens)
        full_pages = torch.tensor(
            [self._free_full_pages.popleft() for _ in range(needed_full)],
            dtype=torch.int32,
        )
        sliding_pages = torch.tensor(
            [self._free_sliding_pages.popleft() for _ in range(self.blocks_per_user_sliding)],
            dtype=torch.int32,
        )

        # Build the prefill full-page-table: real pages cover the prompt's
        # blocks, the rest of the bucket points at scratch (prefill writes
        # zeros into those positions but decode never reads them — cur_pos is
        # bounded by prompt_len + generated).
        used_blocks = (prompt_len + self.block_size - 1) // self.block_size
        full_pt = torch.empty(bucket_blocks, dtype=torch.int32)
        full_pt[:used_blocks] = full_pages[:used_blocks]
        if bucket_blocks > used_blocks:
            full_pt[used_blocks:] = self._idle_blocks[: bucket_blocks - used_blocks]

        host_tokens = self._host_tensor(padded.unsqueeze(0), ttnn.uint32)
        host_page_table = self._host_tensor(full_pt.reshape(1, -1), ttnn.int32)
        # Per-request sliding page table — depends on prompt_len because the
        # window position [prompt_len - W, prompt_len) shifts with the prompt.
        sliding_pt = self._build_sliding_prefill_page_table(sliding_pages, prompt_len, bucket_len)
        host_page_table_sliding = self._host_tensor(sliding_pt, ttnn.int32)

        ttnn.copy_host_to_device_tensor(host_tokens, trace["tokens"])
        ttnn.copy_host_to_device_tensor(host_page_table, trace["page_table"])
        ttnn.copy_host_to_device_tensor(host_page_table_sliding, trace["page_table_sliding"])

        # Refresh this slot's sampling row before the prefill kicks off so the
        # very first decode step (which uses prefill's last token) samples with
        # the new request's parameters.
        temp, top_p, top_k = self._resolve_sampling(req.temperature, req.top_p, req.top_k)
        self._sampling_top_k[slot_idx] = int(top_k)
        self._sampling_top_p[slot_idx] = float(top_p)
        self._sampling_temp[slot_idx] = float(temp)
        self._push_sampling_params()
        self._reseed_slot(slot_idx, req.seed)

        ttnn.execute_trace(self.mesh_device, trace["trace_id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)

        # Skip prefill logits — first decode iteration produces the first generated
        # token by re-running attention at position prompt_len-1 with the last
        # prompt token. The KV write at that position is idempotent.
        last_token = int(req.input_ids[-1].item())

        slot = self.slots[slot_idx]
        slot.rid = req.rid
        slot.prompt_len = prompt_len
        slot.cur_pos = prompt_len - 1
        slot.next_token = last_token
        # No spec hidden yet — the first decode step runs on the single-token
        # path to bootstrap it before this slot becomes packed-verify eligible.
        slot.spec_hidden_valid = False
        if self._spec is not None and self._drafter_kind == "dflash":
            # Fresh generation → empty anchor cache for this slot.
            self._spec.reset_slot(slot_idx)
        slot.generated = 0
        slot.max_new_tokens = req.max_new_tokens
        slot.request = req
        slot.finished = False
        slot.eos_set = self.eos_token_ids
        slot.all_tokens = []
        slot.cum_text = ""
        # If the prompt was rendered with enable_thinking=True, the assistant
        # prefix ends at `<|turn>model\n` — the model is expected to open its
        # own thought channel before producing the user-visible reply, so
        # everything before the first `<channel|>` is reasoning_content.
        slot.in_thinking = bool(req.enable_thinking)
        slot.reasoning_tokens = []
        slot.content_tokens = []
        slot.cum_reasoning = ""
        slot.in_tool_call = False
        slot.tool_call_buffer = []
        slot.tool_calls = []
        slot.full_pages = full_pages
        slot.sliding_pages = sliding_pages
        # New request → loop-free staging is stale; force a reseed from the
        # cache on this slot's first verify step.
        if getattr(self, "_pv_a_prev", None) is not None:
            self._pv_a_prev[slot_idx] = -1

        logger.info(
            f"Prefilled slot {slot_idx} (rid={req.rid[:8]}, prompt_len={prompt_len}, "
            f"bucket={bucket_len}, full_pages={needed_full}/{self.blocks_per_user_max}, "
            f"free_pool={len(self._free_full_pages)})"
        )

    def _slot_uses_packed_verify(self, slot: _Slot, slot_idx: int) -> bool:
        """Whether ``slot`` is verified via the packed multi-token path this
        step (vs. the single-token decode trace).

        It is iff: speculative decode is live; the slot is bootstrapped (has a
        carried ``spec_hidden``); the request is greedy; and ``cur_pos`` clears
        the sliding-ring-wrap and full-mask-cap fallback margins.
        """
        if not slot.spec_hidden_valid:
            return False
        return self._spec_eligible_at(slot_idx, slot.cur_pos)

    def _spec_eligible_at(self, slot_idx: int, cur_pos: int) -> bool:
        """Whether a slot at ``cur_pos`` qualifies for packed verify, ignoring
        ``spec_hidden_valid`` — used to decide if a single-token decode step
        should capture ``spec_hidden`` for the next step.

        Greedy-only: ``_resolve_sampling`` maps temperature<=0 to
        ``(temp=1.0, top_k=1)``; any sampled request has top_k!=1 or
        temp!=1.0 and stays on the exact single-token path until speculative
        rejection sampling lands.
        """
        if self._packed_verify_traces is None or self._spec is None:
            return False
        if not (self._sampling_top_k[slot_idx] == 1 and self._sampling_temp[slot_idx] == 1.0):
            return False
        T = self._spec.num_drafts
        # Sliding ring: keep every packed position (cur_pos .. cur_pos+T) < W
        # so no rejected speculative write can clobber an in-window entry —
        # there is no active sliding-ring rollback (HANDOFF Item 1.3).
        if cur_pos >= self.sliding_cache_len - T:
            return False
        # Full-attention verify mask / SDPA key extent is capped at _pv_sk_cap.
        if cur_pos >= self._pv_sk_cap - T:
            return False
        # DFlash's fixed anchor cache fills toward max_anchors; keep
        # anchor_len + block <= max_anchors so cache writes never run off the end
        # (v1 has no ring — mirrors the sliding-window W-T gate).
        if self._drafter_kind == "dflash" and self._spec is not None:
            if self._spec.anchor_len_at(slot_idx) >= self._spec.max_anchors - T:
                return False
        return True

    def _route_token(self, slot: _Slot, tok: int, req: "_Request") -> None:
        """Special-token routing + incremental detokenization for one token.

        Order matters: tool-call delimiters take precedence over channel
        delimiters because the model can emit ``<|tool_call>`` while a thought
        channel is technically still open in our state machine.
        """
        if tok == self.tool_call_open_id and self.tool_call_open_id >= 0:
            slot.in_tool_call = True
            slot.tool_call_buffer = []
        elif tok == self.tool_call_close_id and self.tool_call_close_id >= 0:
            slot.in_tool_call = False
            raw = self.tokenizer.decode(slot.tool_call_buffer, skip_special_tokens=True)
            parsed = _parse_gemma_tool_call(raw)
            if parsed is not None:
                idx = len(slot.tool_calls)
                slot.tool_calls.append(parsed)
                req.emit({"kind": "tool_call", "index": idx, "tool_call": parsed})
            slot.tool_call_buffer = []
        elif slot.in_tool_call:
            slot.tool_call_buffer.append(tok)
        elif tok == self.channel_open_id and self.channel_open_id >= 0:
            slot.in_thinking = True
        elif tok == self.channel_close_id and self.channel_close_id >= 0:
            slot.in_thinking = False
        else:
            # Incremental detokenization on the *active* buffer so we never
            # glue reasoning text to content text. Decode a sliding suffix
            # window (not the cumulative buffer — O(n²) per stream and the
            # dominant commit cost at long generations): the delta is the
            # extra text vs the same window without its newest token.
            if slot.in_thinking:
                slot.reasoning_tokens.append(tok)
                delta = self._delta_decode(slot.reasoning_tokens)
                slot.cum_reasoning += delta
                if delta:
                    req.emit({"token_id": tok, "kind": "reasoning", "text": delta})
            else:
                slot.content_tokens.append(tok)
                delta = self._delta_decode(slot.content_tokens)
                slot.cum_text += delta
                if delta:
                    req.emit({"token_id": tok, "kind": "content", "text": delta})

    _DETOK_WINDOW = 8

    def _delta_decode(self, tokens: List[int]) -> str:
        """Delta text contributed by the newest token of ``tokens``.

        Decode the last K tokens with and without the newest one; the suffix
        beyond the prefix's text is the new text. K=8 covers sentencepiece
        merges/multibyte sequences; falls back to "" mid-grapheme (the next
        token flushes it).
        """
        window = tokens[-self._DETOK_WINDOW :]
        prefix = self.tokenizer.decode(window[:-1], skip_special_tokens=True)
        text = self.tokenizer.decode(window, skip_special_tokens=True)
        return text[len(prefix) :]

    def _emit_tokens(self, i: int, emitted: List[int]) -> None:
        """Stream slot ``i``'s tokens for this step through the routing FSM.

        A packed-verify step emits up to T+1 tokens. Generation stops — and
        the slot is freed — at the first EOS / length / context finisher; any
        tokens past the finisher are dropped (the target never intended
        output past EOS, and overshoot past ``max_new_tokens`` is trimmed to
        the exact budget). ``cur_pos`` has already been advanced by the
        caller; this only advances the per-token ``generated`` count.
        """
        slot = self.slots[i]
        req = slot.request
        for tok in emitted:
            tok = int(tok)
            slot.generated += 1
            slot.all_tokens.append(tok)
            self._route_token(slot, tok, req)
            self._tokens_this_step += 1
            hit_eos = tok in slot.eos_set
            hit_max = slot.generated >= slot.max_new_tokens
            hit_ctx = slot.cur_pos + 1 >= self.max_user_seq_len
            if hit_eos or hit_max or hit_ctx:
                finish_reason = "stop" if hit_eos else "length"
                req.emit({"finish_reason": finish_reason})
                req.emit(None)
                self._release_slot_pages(slot)
                if self._spec is not None and self._drafter_kind == "dflash":
                    self._spec.reset_slot(i)
                self.slots[i] = _Slot()
                return

    def _refresh_decode_page_tables(self, active_indices: List[int]) -> None:
        """Refresh the decode trace's page tables for all active slots.

        Shared by the single-token decode trace AND the drafter trace (the
        drafter cross-attends the target KV through these page tables), so
        they must cover every active slot — including packed-verify slots —
        even on steps where the decode trace itself does not run.
        """
        B = self.batch
        DW = self.decode_width  # decode/drafter page tables run at the padded width
        BPU_max = self.blocks_per_user_max
        active = set(active_indices)
        page_table = torch.empty((DW, BPU_max), dtype=torch.int32)
        page_table_sliding = torch.empty((DW, self.blocks_per_user_sliding), dtype=torch.int32)
        for i in range(DW):
            # Padding rows (i >= B) have no slot; treat them like idle slots.
            if i >= B or i not in active:
                page_table[i] = self._idle_blocks
                page_table_sliding[i] = self._idle_blocks_sliding
            else:
                s = self.slots[i]
                # Real pages first, scratch padding after. SDPA reads through
                # block index `cur_pos // block_size` which is < n_pages by
                # construction (admission caps `prompt_len + max_new_tokens`
                # at `n_pages * block_size`), so the padding is never read.
                n_pages = int(s.full_pages.shape[0])
                page_table[i, :n_pages] = s.full_pages
                if n_pages < BPU_max:
                    page_table[i, n_pages:] = self._idle_blocks[: BPU_max - n_pages]
                page_table_sliding[i] = s.sliding_pages
        ttnn.copy_host_to_device_tensor(self._host_tensor(page_table, ttnn.int32), self._decode_trace["page_table"])
        ttnn.copy_host_to_device_tensor(
            self._host_tensor(page_table_sliding, ttnn.int32), self._decode_trace["page_table_sliding"]
        )

    def _run_decode_trace(self, decode_slots: List[int]):
        """Replay the single-token decode trace for ``decode_slots``.

        The page tables are refreshed separately (``_refresh_decode_page_tables``)
        for all active slots. Here only the per-row token / position tensors
        are written: every non-decode row (idle slots AND packed-verify slots)
        gets the -1 skip sentinel in the int32 position tensors so the trace
        neither updates their KV nor reads stale positions.

        Returns ``(sampled, final_hidden)`` — ``sampled`` is a length-B list
        of token IDs (only ``decode_slots`` entries are meaningful);
        ``final_hidden`` is the target's ``[1,1,B,hidden]`` post-norm hidden
        torch tensor, or None when spec decode is disabled.
        """
        B = self.batch
        DW = self.decode_width  # device-tensor width (32-padded); rows >= B are idle
        W = self.sliding_cache_len
        decode_set = set(decode_slots)

        tokens = torch.zeros(DW, dtype=torch.int32)
        positions = torch.zeros(DW, dtype=torch.int32)
        pos_int32 = torch.full((DW,), -1, dtype=torch.int32)
        pos_sliding_write = torch.full((DW,), -1, dtype=torch.int32)
        pos_sliding_sdpa = torch.full((DW,), -1, dtype=torch.int32)

        for i in decode_set:
            s = self.slots[i]
            tokens[i] = s.next_token
            positions[i] = s.cur_pos
            pos_int32[i] = s.cur_pos
            pos_sliding_write[i] = s.cur_pos % W
            pos_sliding_sdpa[i] = min(s.cur_pos, W - 1)

        ttnn.copy_host_to_device_tensor(
            self._host_tensor(tokens.reshape(1, DW), ttnn.uint32), self._decode_trace["tokens"]
        )
        ttnn.copy_host_to_device_tensor(
            self._host_tensor(positions.reshape(1, DW), ttnn.uint32), self._decode_trace["position"]
        )
        ttnn.copy_host_to_device_tensor(self._host_tensor(pos_int32, ttnn.int32), self._decode_trace["position_int32"])
        ttnn.copy_host_to_device_tensor(
            self._host_tensor(pos_sliding_write, ttnn.int32), self._decode_trace["pos_sliding_write"]
        )
        ttnn.copy_host_to_device_tensor(
            self._host_tensor(pos_sliding_sdpa, ttnn.int32), self._decode_trace["pos_sliding_sdpa"]
        )

        # Cycle the on-device sampling RNG. No-op once every slot has settled
        # at its default seed; the FSM only enqueues a host write on change.
        self._advance_seeds()
        ttnn.execute_trace(self.mesh_device, self._decode_trace["trace_id"], cq_id=0, blocking=False)

        out = self._decode_trace["output"]
        if self._is_mesh:
            out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        else:
            out_cpu = ttnn.to_torch(out)
        if self.on_device_sampling:
            sampled = out_cpu.reshape(-1)[:B].to(torch.int64).tolist()
        else:
            # TP=1: model returns full logits [1, 1, 32, vocab]. Host sampling.
            sampled = self._host_sample(out_cpu[0, 0, :B, :])

        final_hidden = None
        fh = self._decode_trace.get("final_hidden")
        if fh is not None:
            if self._is_mesh:
                final_hidden = ttnn.to_torch(ttnn.get_device_tensors(fh)[0]).to(torch.bfloat16)
            else:
                final_hidden = ttnn.to_torch(fh).to(torch.bfloat16)
        return sampled, final_hidden

    def _propose_drafts(self, verify_slots: List[int]) -> List[List[int]]:
        """Generate T drafts/slot via a single chained drafter-trace replay.

        Draft-0 inputs (``_spec_input_dev`` = [embed(slot.next_token) ‖
        slot.spec_hidden], RoPE, cur_pos) are refreshed ONCE here. The trace
        itself chains T drafter forwards: after each iteration its on-device
        ``argmax(all_gather(logits))`` produces global next-token IDs, the
        target ``embed_tokens`` looks them up, and a ``concat`` with the
        drafter's ``out_hidden`` builds the next iteration's ``spec_input`` —
        all without leaving the device. (See ``_build_drafter_fwd`` for the
        chained closure and per-iter dependencies.)

        Returns a length-B list of per-slot draft lists (T tokens each;
        non-verify rows carry whatever the trace produced for the idle rows
        and are discarded by the caller). The host only does:
          * 1× ``_refresh_spec_inputs`` (small H2D writes for cur_pos/RoPE +
            doubled-hidden scratch) per step,
          * 1× ``ttnn.execute_trace``,
          * T × small uint32 D2H reads (one per draft, each [1, B]).
        """
        B = self.batch
        DW = self.decode_width  # drafter trace width (32-padded); rows >= B idle
        T = self._spec.num_drafts
        drafts: List[List[int]] = [[] for _ in range(B)]  # slot-indexed; only verify_slots populated

        positions = torch.zeros(DW, dtype=torch.int32)
        prev_tokens = [0] * DW
        for i in verify_slots:
            positions[i] = self.slots[i].cur_pos
            prev_tokens[i] = int(self.slots[i].next_token)

        if _DECODE_PROFILE:
            self._pt_pr_refresh = 0.0
            self._pt_pr_exec = 0.0
            self._pt_pr_embed = 0.0
            _r0 = time.perf_counter()

        # Single host→device refresh for draft 0; drafts 1..T-1 are built
        # on-device inside the trace.
        self._refresh_spec_inputs(positions, None, prev_tokens, draft_step=0, hidden_host=self._spec_hidden_host)

        if _DECODE_PROFILE:
            _r1 = time.perf_counter()
            self._pt_pr_refresh = _r1 - _r0

        ttnn.execute_trace(self.mesh_device, self._drafter_trace["trace_id"], cq_id=0, blocking=False)
        per_draft_tokens = self._read_drafter_chained_drafts()  # list of T lists of B ints

        if _DECODE_PROFILE:
            self._pt_pr_exec = time.perf_counter() - _r1

        for k in range(T):
            row = per_draft_tokens[k]
            for i in verify_slots:
                drafts[i].append(int(row[i]))
        return drafts

    def _dflash_ckpt(self, msg: str, sync: bool = True) -> None:
        """Hang-localizing checkpoint (no-op unless GEMMA4_DFLASH_DEBUG=1).

        Synchronizes the device (drains all enqueued ops) THEN prints, so the
        last line printed before a hang names the phase that deadlocked.
        """
        if not _DFLASH_DEBUG:
            return
        if sync:
            ttnn.synchronize_device(self.mesh_device)
        print(f"[dflash-ckpt] {msg}", flush=True)

    def _step_decode(self) -> None:
        """One batch_32 decode iteration.

        Each active slot is routed to one of two paths this step:
          * packed multi-token verify — the target verifies the drafter's T
            drafts in one packed forward and emits ``n_accepted+1`` tokens
            (greedy requests, bootstrapped, away from the ring wrap / cap);
          * single-token decode — the existing one-token-per-step trace, for
            sampled requests, every slot's first (bootstrap) step, slots near
            the sliding-ring wrap / mask cap, and whenever spec is disabled.
        A step may replay either trace, both, or neither.

        Idle / cross-path rows pass -1 in the int32 position tensors so
        ``paged_update_cache`` skips them — stray K/V writes would otherwise
        race on a shared scratch cell and corrupt active layers' caches.
        """
        if _DECODE_PROFILE:
            self._prof_step = getattr(self, "_prof_step", 0) + 1
            _t_total_start = time.perf_counter()
            # Per-section timers for the packed-verify path (all 0 on a
            # decode-only step). _pt_exec is isolated from _pt_read by an
            # explicit synchronize after execute_trace (profile-only).
            _pt_propose = _pt_refresh = _pt_exec = _pt_read = _pt_commit = 0.0
            _pt_decode = 0.0
        self._tokens_this_step = 0

        # Free zombie slots (HTTP handler gone) before building this step's
        # inputs so they appear idle and get skipped via the -1 sentinel.
        for i, s in enumerate(self.slots):
            if s.rid is not None and s.request is not None and s.request.cancelled:
                s.request.emit(None)
                self._release_slot_pages(s)
                if self._spec is not None and self._drafter_kind == "dflash":
                    self._spec.reset_slot(i)
                self.slots[i] = _Slot()

        # Compact actives into the lowest indices so the freed-slot transition
        # is always at the highest active index — avoids the asymmetric kernel
        # bug on the slot-0-frees / slot-1-stays-alive transition. Pages move
        # with the slot reference (host-side), no device K/V copy.
        self._compact_slots()

        active_indices = [i for i in range(self.batch) if self.slots[i].rid is not None]
        verify_slots = [i for i in active_indices if self._slot_uses_packed_verify(self.slots[i], i)]
        verify_set = set(verify_slots)
        decode_slots = [i for i in active_indices if i not in verify_set]
        self._dflash_ckpt(
            f"=== step {self._spec_steps_called + 1}: active={active_indices} verify={verify_slots} decode={decode_slots} ===",
            sync=False,
        )

        # Refresh the decode trace's page tables for all active slots — used
        # by the single-token decode trace and (for verify slots) by the
        # drafter trace, which cross-attends the target KV through them.
        if active_indices:
            self._refresh_decode_page_tables(active_indices)

        # ── single-token decode path (bootstrap / sampled / near-wrap) ──────
        if decode_slots:
            if _DECODE_PROFILE:
                _t0 = time.perf_counter()
            self._dflash_ckpt(f"decode path: _run_decode_trace ({len(decode_slots)} slots)")
            sampled, final_hidden = self._run_decode_trace(decode_slots)
            self._dflash_ckpt("decode path: _run_decode_trace done")
            if _DECODE_PROFILE:
                ttnn.synchronize_device(self.mesh_device)
                _pt_decode = time.perf_counter() - _t0
            for i in list(decode_slots):
                slot = self.slots[i]
                tok = int(sampled[i])
                slot.cur_pos += 1
                slot.next_token = tok
                # Bootstrap / refresh the carried spec_hidden so this slot can
                # join the packed-verify path next step (if eligible there).
                if self._drafter_kind == "dflash":
                    # DFlash needs no carried hidden — its anchors start empty
                    # and are seeded from the first packed-verify commit's aux
                    # taps. Just mark the slot eligible.
                    if self._spec is not None and self._spec_eligible_at(i, slot.cur_pos):
                        slot.spec_hidden_valid = True
                elif final_hidden is not None and self._spec_eligible_at(i, slot.cur_pos):
                    self._spec_hidden_host[i] = final_hidden[0, 0, i]
                    slot.spec_hidden_valid = True
                self._emit_tokens(i, [tok])

        # ── packed multi-token verify path ─────────────────────────────────
        if verify_slots:
            try:
                if _DECODE_PROFILE:
                    _t0 = time.perf_counter()
                self._dflash_ckpt("verify: propose start")
                if self._drafter_kind == "dflash":
                    # Traced dflash propose: refresh buffers, replay the trace,
                    # synchronize (so its CCL drains before the packed-verify
                    # trace runs — no shared-fabric contention), then read drafts.
                    self._spec.refresh_propose(verify_slots, self.slots)
                    self._dflash_ckpt("verify: dflash refresh done")
                    ttnn.execute_trace(self.mesh_device, self._dflash_trace["trace_id"], cq_id=0, blocking=False)
                    ttnn.synchronize_device(self.mesh_device)
                    self._dflash_ckpt("verify: dflash trace done")
                    drafts = self._spec.read_drafts(
                        self._dflash_trace["topk_val"], self._dflash_trace["topk_idx"], verify_slots
                    )
                else:
                    drafts = self._propose_drafts(verify_slots)
                self._dflash_ckpt("verify: propose done")
                if _DECODE_PROFILE:
                    ttnn.synchronize_device(self.mesh_device)
                    _t1 = time.perf_counter()
                    _pt_propose = _t1 - _t0
                # Pick the smallest occupancy bucket >= the verify-slot count;
                # the packed forward then runs B_v*P rows instead of 32*P.
                B_v = self._pick_pv_bucket(len(verify_slots))
                self._refresh_packed_verify_inputs(B_v, verify_slots, drafts)
                self._dflash_ckpt(f"verify: refresh_packed_verify_inputs done (B_v={B_v})")
                if _DECODE_PROFILE:
                    _t2 = time.perf_counter()
                    _pt_refresh = _t2 - _t1
                ttnn.execute_trace(
                    self.mesh_device, self._packed_verify_traces[B_v]["trace_id"], cq_id=0, blocking=False
                )
                self._dflash_ckpt("verify: execute_trace(packed_verify) done")
                if _DECODE_PROFILE:
                    ttnn.synchronize_device(self.mesh_device)
                    _t3 = time.perf_counter()
                    _pt_exec = _t3 - _t2
                if self._drafter_kind == "dflash":
                    pv_tgt, pv_hidden, pv_aux = self._read_packed_verify(B_v, want_aux=True)
                else:
                    pv_tgt, pv_hidden = self._read_packed_verify(B_v)
                    pv_aux = None
                self._dflash_ckpt("verify: read_packed_verify done")
                if _DECODE_PROFILE:
                    _t4 = time.perf_counter()
                    _pt_read = _t4 - _t3
                self._commit_packed_verify(B_v, verify_slots, drafts, pv_tgt, pv_hidden, pv_aux=pv_aux)
                self._dflash_ckpt("verify: commit_packed_verify done")
                if _DECODE_PROFILE:
                    _pt_commit = time.perf_counter() - _t4
                    self._pt_bucket = B_v
                self._spec_steps_called += 1
                if self._spec_steps_called <= 3 or self._spec_steps_called % 50 == 0:
                    agg = self._spec.aggregate_stats()
                    logger.info(
                        f"Spec-decode step={self._spec_steps_called}: "
                        f"verify_slots={len(verify_slots)} "
                        f"mean_accepted/step={agg['mean_accepted_per_step']:.2f} "
                        f"mean_tokens/step={agg.get('mean_tokens_per_step', 0.0):.2f} "
                        f"windowed={agg['windowed_acceptance'] * 100:.1f}%"
                    )
            except Exception as e:
                if not getattr(self, "_spec_error_logged", False):
                    logger.error(f"Packed verify failed (continuing without spec): {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    self._spec_error_logged = True
                # Disable spec; affected slots resume on the decode path next
                # step (they neither advanced nor emitted this step).
                self._spec = None
                self._packed_verify_traces = None

        if _DECODE_PROFILE:
            _prof_total = time.perf_counter() - _t_total_start
            if self._prof_step > _PROF_WARMUP_STEPS and (self._prof_step - _PROF_WARMUP_STEPS) % _PROF_PRINT_EVERY == 0:
                tok = self._tokens_this_step
                print(
                    f"DECODE_PROF step={self._prof_step} "
                    f"verify={len(verify_slots)} bucket={getattr(self, '_pt_bucket', 0)} "
                    f"decode={len(decode_slots)} "
                    f"tokens={tok} total={_prof_total * 1e3:.1f}ms "
                    f"tok/s={tok / _prof_total if _prof_total > 0 else 0.0:.1f} | "
                    f"decode_path={_pt_decode * 1e3:.1f} "
                    f"propose={_pt_propose * 1e3:.1f}"
                    f"(refresh={getattr(self, '_pt_pr_refresh', 0.0) * 1e3:.1f}"
                    f"[embed={getattr(self, '_pt_pr_embed', 0.0) * 1e3:.1f}],"
                    f"draftexec={getattr(self, '_pt_pr_exec', 0.0) * 1e3:.1f}) "
                    f"refresh={_pt_refresh * 1e3:.1f} "
                    f"exec={_pt_exec * 1e3:.1f} "
                    f"read={_pt_read * 1e3:.1f} "
                    f"commit={_pt_commit * 1e3:.1f}",
                    flush=True,
                )

    def _commit_packed_verify(self, B_v, verify_slots, drafts, pv_tgt, pv_hidden, pv_aux=None) -> None:
        """Greedy-verify the packed forward against the drafts, commit the
        accepted prefix + bonus per slot, and emit.

        ``pv_tgt`` / ``pv_hidden`` are verify-row-indexed: verify row ``r``
        (rows ``r*P .. r*P+T``) is the step's ``verify_slots[r]``. For that
        slot at cur_pos ``c``, row ``r*P+p`` predicts the token at absolute
        position ``c+p+1``. ``n_accepted`` is the longest prefix of drafts
        matching the target argmax; the bonus is the target's own token at the
        first mismatch (always correct, so the emitted stream is identical to
        non-spec greedy decode). The next drafter round consumes
        ``hidden[r*P+n_accepted]`` — the per-position verify hidden that
        produced the bonus.
        """
        P = self._pv_p
        T = self._spec.num_drafts
        # DFlash: record each commit as (slot, verify-row r, n_acc); the aux taps
        # stay ON DEVICE and are gathered into `_aux_dev` in ONE on-device
        # row-gather after the loop (append_committed_ondevice), then the append
        # trace writes them as anchors. (Phase 2a — no aux D2H/H2D round-trip.)
        dflash_commits = [] if pv_aux is not None else None
        for r, i in enumerate(verify_slots):  # verify row r ← slot i
            slot = self.slots[i]
            c = slot.cur_pos
            base = r * P
            tgt = [int(pv_tgt[base + p]) for p in range(P)]  # P = T+1 entries
            drafts_i = drafts[i]
            n_acc = 0
            for k in range(T):
                if tgt[k] == drafts_i[k]:
                    n_acc += 1
                else:
                    break
            bonus = tgt[n_acc]
            emitted = drafts_i[:n_acc] + [bonus]
            # Commit: advance past the n_acc accepted drafts + the bonus.
            slot.cur_pos = c + n_acc + 1
            slot.next_token = bonus
            if pv_aux is not None:
                # The n_acc+1 committed INPUT positions (verify rows base+0..base+n_acc
                # = next_token + accepted drafts) become anchors. The bonus is
                # anchored next step, so anchors added == the cur_pos advance.
                # Aux taps stay on device — record (slot, verify-row r, n_acc).
                dflash_commits.append((i, r, n_acc))
            else:
                self._spec_hidden_host[i] = pv_hidden[base + n_acc]
            # Stats.
            sp = self._spec.slots[i]
            sp.n_proposed += T
            sp.n_steps_with_drafts += 1
            sp.pending_drafts = list(drafts_i)
            self._spec.verify(i, drafts_i, tgt[:T])
            self._spec.commit(i, n_acc, bonus)
            self._emit_tokens(i, emitted)
        if dflash_commits:
            # `append_committed_ondevice` only REFRESHES the pre-allocated aux +
            # widx buffers (zero device allocs); the actual K/V writes are in
            # the captured `_dflash_append_trace` we replay here. Synchronize
            # after to keep the next step's propose / packed-verify trace
            # safely sequenced (separate CCL semaphores on the drafter mgr,
            # but the propose reads from the same caches we just wrote).
            did_refresh = self._spec.append_committed_ondevice(dflash_commits, B_v, pv_aux)
            self._dflash_ckpt("commit: append refresh done")
            if did_refresh and self._dflash_append_trace is not None:
                ttnn.execute_trace(self.mesh_device, self._dflash_append_trace["trace_id"], cq_id=0, blocking=False)
                ttnn.synchronize_device(self.mesh_device)
                self._dflash_ckpt("commit: append trace done")


# ── FastAPI app + endpoints ─────────────────────────────────────────────────


def _resolve_enable_thinking(verbosity: Optional[str]) -> bool:
    """`verbosity` → Gemma 4 `enable_thinking`.

    Thinking is on by default and only disabled when `verbosity == "low"`.
    Any other value (including missing) enables thinking.
    """
    if verbosity is None:
        return True
    return verbosity.strip().lower() != "low"


def _flatten_content(content: Any) -> str:
    """Reduce OpenAI multi-part content to a plain string.

    Accepts:
      - ``None`` (assistant tool-only messages) → ``""``.
      - ``str`` (the common case) → returned as-is.
      - ``list`` of part dicts → text parts concatenated; image/audio/video
        parts replaced with a short placeholder so the model still sees that
        a non-text input was attached. This server doesn't run a vision/audio
        processor, so placeholders are the best we can do.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text":
                parts.append(str(item.get("text", "")))
            elif kind in ("image_url", "image"):
                parts.append("[image]")
            elif kind == "input_audio" or kind == "audio":
                parts.append("[audio]")
            elif kind == "video":
                parts.append("[video]")
            # else: silently drop unknown parts
        return "".join(parts)
    return str(content)


def _coerce_tool_args(arguments: Union[str, Dict[str, Any]]) -> Any:
    """Best-effort: hand the chat template a mapping when possible.

    The Gemma 4 template's tool-call branch accepts either a mapping (which it
    formats key:value-style) or a string (emitted verbatim). OpenAI sends
    arguments as a JSON-encoded string; if it parses to an object, prefer the
    mapping path so the rendered call matches the model's training format.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return arguments
    s = arguments.strip()
    if not s:
        return {}
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return arguments  # raw string fallback


def _convert_openai_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Translate OpenAI-style messages into the Gemma 4 chat-template shape.

    Specifically:
      - assistant messages with `tool_calls` → `tool_calls=[{function:{name,
        arguments}}]` (arguments coerced to a mapping when it parses).
      - tool-role messages → `tool_responses=[{name, response}]`. OpenAI emits
        these as separate `{role: "tool", tool_call_id, content}` messages;
        the template expects them inlined on a non-assistant message and a
        function name attached. We fall back to looking up the function name
        in the most recent assistant `tool_calls` when `name` is omitted.
        Consecutive tool-role messages are coalesced so the template emits
        them as a single grouped response block.
    """
    # Walk OpenAI tool messages → Gemma `tool_responses`. Build a lookup so
    # tool messages without an explicit `name` can find theirs from the
    # preceding assistant call.
    tool_call_id_to_name: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id and tc.function and tc.function.name:
                    tool_call_id_to_name[tc.id] = tc.function.name
            entry: Dict[str, Any] = {"role": "assistant", "content": _flatten_content(msg.content)}
            entry["tool_calls"] = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": _coerce_tool_args(tc.function.arguments),
                    }
                }
                for tc in msg.tool_calls
            ]
            out.append(entry)
            continue

        if msg.role == "tool":
            name = msg.name or tool_call_id_to_name.get(msg.tool_call_id or "", "unknown")
            response: Any = _flatten_content(msg.content)
            if isinstance(response, str):
                stripped = response.strip()
                if stripped:
                    try:
                        parsed = json.loads(stripped)
                        response = parsed
                    except Exception:
                        pass  # leave as raw string; template wraps in {value:...}
            tool_resp = {"name": name, "response": response if response != "" else ""}
            # Coalesce with the immediately preceding tool-response message so
            # the template emits one grouped response block (matches the
            # Gemma 4 inline-response convention).
            if out and out[-1].get("role") == "tool" and out[-1].get("tool_responses") is not None:
                out[-1]["tool_responses"].append(tool_resp)
            else:
                out.append({"role": "tool", "content": "", "tool_responses": [tool_resp]})
            continue

        out.append({"role": msg.role, "content": _flatten_content(msg.content)})
    return out


def _format_chat_prompt(
    tokenizer,
    messages: List[ChatMessage],
    *,
    enable_thinking: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> torch.Tensor:
    if tokenizer.chat_template:
        msg_dicts = _convert_openai_messages(messages)
        chat = tokenizer.apply_chat_template(
            msg_dicts,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        return chat["input_ids"].squeeze(0).to(torch.int32)
    # Fallback: concatenate.
    text = "\n".join(f"{m.role}: {_flatten_content(m.content)}" for m in messages) + "\nassistant:"
    return tokenizer.encode(text, return_tensors="pt").squeeze(0).to(torch.int32)


# ── Tool-call parsing (Gemma 4 mini-format → OpenAI tool_calls) ─────────────


# `<|"|>` is the template's quote-escape — strings inside argument blocks are
# wrapped as `<|"|>...<|"|>` and keys are bare. To emit valid JSON for OpenAI
# clients we (1) stash the escaped string spans, (2) quote bare keys, and (3)
# restore the strings as JSON-encoded literals.
_GEMMA_QUOTE = '<|"|>'
_GEMMA_QUOTED_RE = re.compile(re.escape(_GEMMA_QUOTE) + r"(.*?)" + re.escape(_GEMMA_QUOTE), re.DOTALL)
_GEMMA_BARE_KEY_RE = re.compile(r"(?<=[{,])\s*([A-Za-z_][\w\-\.]*)\s*:")
_GEMMA_TOOL_CALL_RE = re.compile(r"^\s*call:(?P<name>[A-Za-z_]\w*)\s*(?P<args>\{.*\})\s*$", re.DOTALL)


def _gemma_args_to_json(raw: str) -> str:
    """Convert a Gemma 4 tool-call argument block to a JSON string.

    Input format (from the chat template's `format_argument` macro):
      strings → ``<|"|>str<|"|>``    booleans → ``true``/``false``
      arrays  → ``[v1,v2,...]``      objects  → ``{key:value,...}``
      numbers → bare                 keys     → bare identifiers
    """
    placeholders: List[str] = []

    def stash(match: re.Match) -> str:
        placeholders.append(match.group(1))
        return f"\x00{len(placeholders) - 1}\x00"

    s = _GEMMA_QUOTED_RE.sub(stash, raw)
    s = _GEMMA_BARE_KEY_RE.sub(lambda m: '"' + m.group(1) + '":', s)

    def unstash(match: re.Match) -> str:
        return json.dumps(placeholders[int(match.group(1))])

    s = re.sub(r"\x00(\d+)\x00", unstash, s)
    return s


def _parse_gemma_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a single `call:NAME{ARGS}` payload into an OpenAI tool_call dict.

    Returns None when the payload doesn't match the documented shape — callers
    treat that as "the model hallucinated something we can't surface".
    """
    m = _GEMMA_TOOL_CALL_RE.match(raw)
    if not m:
        return None
    name = m.group("name")
    args_raw = m.group("args")
    args_json = "{}"
    try:
        candidate = _gemma_args_to_json(args_raw)
        json.loads(candidate)  # validate
        args_json = candidate
    except Exception:
        # Surface the raw block so clients can still see what the model
        # produced even if our reformatter choked on an unexpected shape.
        args_json = json.dumps(args_raw)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": args_json},
    }


def _format_prompt(tokenizer, prompt: str) -> torch.Tensor:
    return tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(torch.int32)


async def _drain_to_async(q: "asyncio.Queue[Optional[dict]]") -> AsyncIterator[dict]:
    """Yield events until the worker emits a None sentinel."""
    while True:
        item = await q.get()
        if item is None:
            return
        yield item


def build_app(engine: Engine, served_model_name: str = "gemma-4") -> FastAPI:
    app = FastAPI(title="Gemma4 OpenAI-compatible server")

    _load_env_file(_ENV_PATH)
    _admin_api_key = os.environ.get("PROXY_API_KEY")
    if not _admin_api_key:
        raise RuntimeError(f"PROXY_API_KEY is not set; add it to {_ENV_PATH}")
    _bearer = HTTPBearer(auto_error=False)

    def _require_admin_auth(
        creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    ) -> None:
        if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != _admin_api_key:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": served_model_name, "object": "model", "created": int(time.time()), "owned_by": "tenstorrent"}
            ],
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/admin/spec-stats")
    def admin_spec_stats():
        """Speculative-decode acceptance metrics.

        ``enabled`` is False unless the server was started with
        ``GEMMA4_SPECULATIVE_DECODE=1``. ``acceptance_rate`` is the fraction of
        drafter proposals that matched the target's actual token.
        """
        spec = getattr(engine, "_spec", None)
        if spec is None:
            return {"enabled": False}
        agg = spec.aggregate_stats()
        return {
            "enabled": True,
            "steps": getattr(engine, "_spec_steps_called", 0),
            "num_drafts": spec.num_drafts,
            **agg,
        }

    @app.post("/admin/patch", dependencies=[Depends(_require_admin_auth)])
    def admin_patch(body: _PatchBody):
        """exec() a Python snippet in a scope that includes the engine.

        Intended for live-rebinding ``engine._step_decode`` /
        ``engine._prefill_request`` with custom implementations:

            import types
            def new_step(self):
                ...                       # custom logic, has access to self
                for s in self.slots:
                    if s.request is not None:
                        s.request.debug.append({"cur_pos": s.cur_pos})
            engine._step_decode = types.MethodType(new_step, engine)

        Patched code can append anything to ``request.debug``; the value
        is surfaced on the API response (``debug`` field) so per-request
        diagnostics can flow back to the caller.

        Snippet scope includes: engine, Engine, _Slot, _Request, types,
        ttnn, torch, logger. Bind ``result`` to return a value.
        """
        import types

        scope: Dict[str, Any] = {
            "engine": engine,
            "Engine": Engine,
            "_Slot": _Slot,
            "_Request": _Request,
            "types": types,
            "ttnn": ttnn,
            "torch": torch,
            "logger": logger,
            "result": None,
        }
        try:
            exec(compile(body.code, "<admin/patch>", "exec"), scope)
        except Exception as e:
            logger.exception("Admin patch raised")
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"{type(e).__name__}: {e}"},
            )
        return {"ok": True, "result": repr(scope.get("result"))}

    def _new_request(
        input_ids: torch.Tensor,
        max_new_tokens: int,
        prompt_text: str,
        is_chat: bool,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
        enable_thinking: bool = False,
    ) -> _Request:
        return _Request(
            rid=uuid.uuid4().hex,
            prompt_text=prompt_text,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            is_chat=is_chat,
            created=time.time(),
            output_queue=asyncio.Queue(),
            output_loop=asyncio.get_running_loop(),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            enable_thinking=enable_thinking,
        )

    def _resolve_max_tokens(req_max: Optional[int]) -> int:
        if req_max is None or req_max <= 0:
            return 8192
        return int(req_max)

    async def _stream_completion(req: _Request, model_name: str, is_chat: bool) -> AsyncIterator[str]:
        cmpl_id = f"cmpl-{req.rid}"
        created = int(req.created)
        first = True
        finish_reason = "stop"
        any_tool_calls = False
        # finally: tell the worker to free the slot if the client disconnects
        # mid-stream (no-op on normal completion).
        try:
            async for event in _drain_to_async(req.output_queue):
                if "error" in event:
                    yield f"data: {json.dumps({'error': event['error']})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if "finish_reason" in event:
                    finish_reason = event["finish_reason"]
                    continue
                kind = event.get("kind", "content")
                if is_chat:
                    # Match the convention used by reasoning models on the
                    # OpenAI-compatible side (DeepSeek-R1, Qwen, etc.):
                    # `delta.reasoning_content` for thoughts,
                    # `delta.content` for the user-visible reply,
                    # `delta.tool_calls` for tool invocations. Role is only
                    # emitted on the first delta.
                    if kind == "tool_call":
                        any_tool_calls = True
                        tc = event["tool_call"]
                        delta = {
                            "tool_calls": [
                                {
                                    "index": event["index"],
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    },
                                }
                            ]
                        }
                    elif kind == "reasoning":
                        delta = {"reasoning_content": event["text"]}
                    else:
                        delta = {"content": event["text"]}
                    if first:
                        delta = {"role": "assistant", **delta}
                    chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                else:
                    # /v1/completions has no reasoning / tool_calls slot —
                    # drop everything but content so they don't leak into
                    # the prompt-completion text stream.
                    if kind != "content":
                        continue
                    chunk = {
                        "id": cmpl_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "text": event["text"], "finish_reason": None}],
                    }
                yield f"data: {json.dumps(chunk)}\n\n"
                first = False
            # OpenAI semantics: if the assistant produced any tool_calls and
            # we'd otherwise report "stop", upgrade to "tool_calls".
            if is_chat and any_tool_calls and finish_reason == "stop":
                finish_reason = "tool_calls"

            # Final chunk with finish_reason.
            if is_chat:
                final = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
            else:
                final = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
                }
            if req.debug:
                final["debug"] = req.debug
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            req.cancelled = True

    async def _collect_completion(
        req: _Request,
    ) -> tuple[str, str, List[Dict[str, Any]], str]:
        """Drain the output queue; return (content, reasoning_content, tool_calls, finish_reason)."""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        finish_reason = "stop"
        try:
            async for event in _drain_to_async(req.output_queue):
                if "error" in event:
                    raise HTTPException(status_code=500, detail=event["error"])
                if "finish_reason" in event:
                    finish_reason = event["finish_reason"]
                    continue
                kind = event.get("kind")
                if kind == "tool_call":
                    tool_calls.append(event["tool_call"])
                elif kind == "reasoning":
                    reasoning_parts.append(event["text"])
                else:
                    content_parts.append(event["text"])
        finally:
            req.cancelled = True
        return "".join(content_parts), "".join(reasoning_parts), tool_calls, finish_reason

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        input_ids = _format_prompt(engine.tokenizer, body.prompt)
        max_new = _resolve_max_tokens(body.max_tokens)
        req = _new_request(
            input_ids,
            max_new,
            body.prompt,
            is_chat=False,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            seed=body.seed,
        )
        engine.submit(req)
        model_name = body.model or served_model_name

        if body.stream:
            return StreamingResponse(_stream_completion(req, model_name, is_chat=False), media_type="text/event-stream")

        # Reasoning content and tool calls have no place in a /v1/completions
        # response — discard them. The chat endpoint surfaces both.
        text, _reasoning, _tool_calls, finish = await _collect_completion(req)
        response: Dict[str, Any] = {
            "id": f"cmpl-{req.rid}",
            "object": "text_completion",
            "created": int(req.created),
            "model": model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}],
            "usage": {
                "prompt_tokens": int(input_ids.shape[0]),
                "completion_tokens": len(engine.tokenizer.encode(text)) if text else 0,
                "total_tokens": int(input_ids.shape[0]) + (len(engine.tokenizer.encode(text)) if text else 0),
            },
        }
        if req.debug:
            response["debug"] = req.debug
        return JSONResponse(response)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        enable_thinking = _resolve_enable_thinking(body.verbosity)
        # tool_choice="none" suppresses tool definitions entirely; "auto" /
        # None / "required" / a forced-name object all forward the tools list
        # to the template. The Gemma 4 template doesn't gate on the kwarg, so
        # "required" / forced-name aren't strictly enforced here — the model
        # decides whether to emit a `<|tool_call>` block.
        choice = body.tool_choice
        if isinstance(choice, str) and choice.strip().lower() == "none":
            tools = None
        else:
            tools = body.tools or None
        input_ids = _format_chat_prompt(
            engine.tokenizer,
            body.messages,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        max_new = _resolve_max_tokens(body.max_tokens)
        prompt_text = "\n".join(f"{m.role}: {_flatten_content(m.content)}" for m in body.messages)
        req = _new_request(
            input_ids,
            max_new,
            prompt_text,
            is_chat=True,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            seed=body.seed,
            enable_thinking=enable_thinking,
        )
        engine.submit(req)
        model_name = body.model or served_model_name

        if body.stream:
            return StreamingResponse(_stream_completion(req, model_name, is_chat=True), media_type="text/event-stream")

        text, reasoning_text, tool_calls, finish = await _collect_completion(req)
        # OpenAI: when the assistant returns tool_calls, content is null and
        # finish_reason is "tool_calls" (unless we already hit "length" first).
        if tool_calls and finish == "stop":
            finish = "tool_calls"
        message: Dict[str, Any] = {"role": "assistant"}
        message["content"] = text if text else (None if tool_calls else "")
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
        response: Dict[str, Any] = {
            "id": f"chatcmpl-{req.rid}",
            "object": "chat.completion",
            "created": int(req.created),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": int(input_ids.shape[0]),
                "completion_tokens": len(engine.tokenizer.encode(text)) if text else 0,
                "total_tokens": int(input_ids.shape[0]) + (len(engine.tokenizer.encode(text)) if text else 0),
            },
        }
        if req.debug:
            response["debug"] = req.debug
        return JSONResponse(response)

    return app


# ── CLI / entry point ───────────────────────────────────────────────────────


# Standard MESH_DEVICE → (rows, cols) mapping used across the repo's demos.
# Accepts both the canonical `<NAME>x<N>` form (e.g. P150x8) and the inverted
# `<N>x<NAME>` form (e.g. 8xP150) for convenience.
_MESH_DEVICE_SHAPES = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P100": (1, 1),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def _resolve_mesh_shape(mesh_device_env: Optional[str]) -> tuple[int, int]:
    if not mesh_device_env:
        n = ttnn.get_num_devices()
        return (1, n) if n > 0 else (1, 1)
    if mesh_device_env in _MESH_DEVICE_SHAPES:
        return _MESH_DEVICE_SHAPES[mesh_device_env]
    # Tolerate inverted forms like `8xP150` → `P150x8`.
    if "x" in mesh_device_env:
        parts = mesh_device_env.split("x")
        if len(parts) == 2 and parts[0].isdigit() and not parts[1].isdigit():
            inverted = f"{parts[1]}x{parts[0]}"
            if inverted in _MESH_DEVICE_SHAPES:
                return _MESH_DEVICE_SHAPES[inverted]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Unrecognized MESH_DEVICE={mesh_device_env!r}; known: {sorted(_MESH_DEVICE_SHAPES)}")


# Trace region size in bytes — sized for three prefill traces (0.5x / 1x / 2x
# max_seq_len) plus the batch_32 decode trace, all captured back-to-back. Each
# bucket adds its own captured kernel graph; if you cut buckets you can drop
# this back toward ~90 MB.
_DEFAULT_TRACE_REGION_SIZE = 256_000_000


def _open_mesh_device(mesh_shape, trace_region_size):
    rows, cols = mesh_shape
    n = rows * cols
    available = ttnn.get_num_devices()
    if n > available:
        suggestion = f"--mesh-shape {rows}x{available}" if rows == 1 else "--mesh-shape <rows>x<cols>"
        raise RuntimeError(
            f"Requested {n} devices ({rows}x{cols}), only {available} available. "
            f"Try {suggestion} (e.g. for a P150x4 / N150x4 / 1x4-mock system, use --mesh-shape 1x4)."
        )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), trace_region_size=trace_region_size)


def run_server(args: argparse.Namespace) -> None:
    if args.mesh_shape:
        mesh_shape = _resolve_mesh_shape(args.mesh_shape)
    else:
        mesh_shape = _resolve_mesh_shape(os.environ.get("MESH_DEVICE"))
    multi_device = mesh_shape[0] * mesh_shape[1] > 1
    logger.info(f"Mesh shape: {mesh_shape[0]}x{mesh_shape[1]} (multi_device={multi_device})")

    if args.fabric and multi_device:
        # Match the demo: fabric_1d for multi-device meshes.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    mesh_device = _open_mesh_device(mesh_shape, args.trace_region_size)
    engine: Optional[Engine] = None
    try:
        engine = Engine(
            mesh_device=mesh_device,
            model_path=args.model_path,
            max_seq_len=args.max_seq_len,
            block_size=args.block_size,
            num_layers=args.num_layers,
            sliding_cache_len=args.sliding_cache_len,
            kv_cache_dtype=_KV_CACHE_DTYPES[args.kv_cache_dtype],
            max_prefill_bucket=args.max_prefill_bucket,
            batch=args.batch,
        )
        app = build_app(engine, served_model_name=args.served_model_name)
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    finally:
        if engine is not None:
            engine.stop()
        try:
            for submesh in mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
        except Exception:
            pass
        ttnn.close_mesh_device(mesh_device)
        if args.fabric and multi_device:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default=os.environ.get("HF_MODEL")
        or os.environ.get("GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"),
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help=(
            "Per-user average context budget. The shared KV pool is sized to "
            "batch*max_seq_len blocks worth of cumulative tokens; a single user "
            "may exceed this (up to the largest prefill bucket) as long as "
            "the cumulative usage across all active slots stays in budget. "
            "Must satisfy batch*max_seq_len >= max_user_seq_len."
        ),
    )
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument(
        "--batch",
        type=int,
        default=DECODE_BATCH,
        help="Number of concurrent decode slots (decode batch size). Default 32 (hardware decode width).",
    )
    p.add_argument(
        "--sliding-cache-len",
        type=int,
        default=DEFAULT_SLIDING_CACHE_LEN,
        help=(
            "Tokens of KV cache per user for sliding-window layers. Should be >= the "
            "model's sliding_window. Capped to 2*max_seq_len (the largest bucket)."
        ),
    )
    p.add_argument(
        "--mesh-shape",
        default=None,
        help="MESH_DEVICE-style name (P150x8, T3K, ...) or 'rowsxcols'. Defaults to $MESH_DEVICE.",
    )
    p.add_argument("--fabric", action="store_true", default=True, help="Enable FABRIC_1D for multi-device meshes")
    p.add_argument("--no-fabric", dest="fabric", action="store_false")
    p.add_argument(
        "--trace-region-size",
        type=int,
        default=int(os.environ.get("TT_METAL_TRACE_REGION_SIZE_BYTES", _DEFAULT_TRACE_REGION_SIZE)),
        help="Trace region size in bytes (default: 256MB). Sized for three prefill buckets + decode trace.",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=(int(os.environ.get("GEMMA4_NUM_LAYERS") or "0") or None),
        help="Override layer count for testing (default: $GEMMA4_NUM_LAYERS, or all layers if unset/0)",
    )
    p.add_argument("--served-model-name", default="gemma-4")
    p.add_argument(
        "--kv-cache-dtype",
        choices=tuple(_KV_CACHE_DTYPES.keys()),
        default="bfloat16",
        help="KV cache element dtype. bfloat8_b halves cache memory at some accuracy cost.",
    )
    p.add_argument(
        "--max-prefill-bucket",
        type=int,
        default=(
            int(os.environ.get("GEMMA4_MAX_PREFILL_BUCKET")) if os.environ.get("GEMMA4_MAX_PREFILL_BUCKET") else None
        ),
        help=(
            "Skip prefill bucket traces larger than this token count. The "
            "131072 bucket alone takes minutes to capture at startup; setting "
            "--max-prefill-bucket 65536 drops it (and any larger bucket). The "
            "largest remaining bucket also caps per-user context length. "
            "Default: no cap (all of "
            f"{PREFILL_BUCKET_LENS} are captured). "
            "Env: GEMMA4_MAX_PREFILL_BUCKET."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    run_server(_parse_args())
