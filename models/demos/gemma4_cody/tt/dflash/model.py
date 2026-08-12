# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter forward — TT implementation.

Mirrors the reference HF DFlash block (see ``/mnt/nas/qwen-coder-30b-a3b/dflash/
dflash.py`` and ``vllm-project/speculators/src/speculators/models/dflash/``).
The Gemma-4 variant uses Llama-style decoder layers (no q-norm / k-norm), 5
layers, hidden=5376, 32 Q heads × 16 KV heads at head_dim=256, intermediate
21504 with SwiGLU/silu, RMSNorm pre-norm, RoPE θ=10000.

Forward shape (Phase 2 / parity-first)
--------------------------------------

Single-user prefill-style shapes (B=1 in the outer dim) to match the HF
reference exactly. Multi-user (B>1) batched verify is a perf-concern follow-up.

  * ``aux_hiddens_concat`` — ``[1, 1, ctx_len, K*target_hidden]`` — caller
    has already concatenated the K aux target hiddens along the feature
    dim (after applying the trained ``-1`` offset from ``aux_hidden_state_layer_ids``).
  * ``noise_embeddings`` — ``[1, 1, block_size, hidden]``. Comes from
    ``target.embed_tokens(block_token_ids)``; the first slot is the most
    recently emitted "bonus" target token, the rest are ``mask_token_id``.
  * ``cos_full`` / ``sin_full`` — ``[1, 1, ctx_len + block_size, head_dim]``.
    RoPE covers BOTH the context (anchor) positions and the noise block;
    the kernel slices the trailing ``block_size`` rows for Q and uses the
    full range for K.

Returns ``(draft_logits, draft_hidden)`` — see :meth:`DFlashDrafter.forward`.

Attention semantics
-------------------

DFlash is **fully non-causal** within (context ∥ noise). The K/V cache is
``concat([k_proj(target_hidden), k_proj(noise)], dim=seq)`` — same
``k_proj`` / ``v_proj`` weights for both halves — and SDPA is called with
``is_causal=False``. The bonus-token position carries real conditioning;
positions 1..7 are the actual drafts and are predicted bidirectionally.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import ttnn
from models.demos.gemma4_cody.tt.attention.operations import (
    apply_per_head_norm,
    apply_rope,
    concat_heads,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)

# Per-layer hang-localizing checkpoints inside decode_step (GEMMA4_DFLASH_DEBUG=1).
_DFLASH_DEBUG = os.environ.get("GEMMA4_DFLASH_DEBUG") == "1"

# Op-section profiling for the traced propose (GEMMA4_DFLASH_OPPROF=1). Mirrors
# tt/attention/decode.py PV_OPPROF: the server flips DFLASH_OPPROF["active"] on
# around the (untraced) propose-trace WARMUP only, so the section syncs never
# run during capture/replay. Sums are wall-clock seconds across the 5 layers.
_DFLASH_OPPROF = os.environ.get("GEMMA4_DFLASH_OPPROF") == "1"
DFLASH_OPPROF = {
    "active": False,
    "qkv": 0.0,
    "kvwrite": 0.0,
    "rope_sdpa": 0.0,
    "ccl_oproj": 0.0,
    "mlp": 0.0,
    "head": 0.0,
}

from ..drafter_base import Drafter, DrafterOutput, RequiredTargetOutputs, TargetOutputs
from .config import DFlashConfig
from .weights import DFlashWeights, load_dflash_weights


@dataclass
class DFlashAnchorCache:
    """Per-user anchor KV cache for DFlash decode-time drafting.

    DFlash's parity :meth:`DFlashDrafter.forward` recomputes the context K/V
    (``k_proj``/``v_proj`` of the fused target hiddens) on every call — fine for
    a one-shot parity check, O(ctx_len) per step for decode. At decode time the
    context (the "anchors" = projected target-hidden states of committed tokens)
    grows by ``accepted + 1`` per step, so we cache the per-layer anchor K/V
    once at append time and reuse them across steps.

    One instance per server slot (B=1 logical user). Fields hold, per dflash
    layer, the already-(k_norm + RoPE)'d anchor K and the plain ``v_proj`` anchor
    V — exactly the ``k_ctx``/``v_ctx`` that :meth:`forward` would compute, but
    persisted. ``length`` is the number of valid anchors (== absolute position
    of the next anchor, since anchor ``a`` sits at RoPE position ``a``).

    v1 uses a **growing concat** (eager, dynamic shape) rather than a fixed
    ``[B, n_kv, max_anchors, hd]`` ring; that's the simplest correct form and
    keeps decode_step's op sequence identical to ``forward`` (a plain
    ``ttnn.concat`` of cached + noise K/V). Stage 5 swaps it for a fixed-shape
    ring + ``paged_update_cache`` once acceptance/perf are validated on hardware.
    """

    # Per dflash layer ℓ: K [1, n_kv_local, length, head_dim] (k_norm'd + RoPE'd),
    # V [1, n_kv_local, length, head_dim] (plain v_proj). ``None`` until first append.
    k: List[Optional["ttnn.Tensor"]] = field(default_factory=list)
    v: List[Optional["ttnn.Tensor"]] = field(default_factory=list)
    length: int = 0

    def reset(self) -> None:
        # Free cached tensors but PRESERVE the per-layer slot count — decode_step
        # / append_anchors index k[i]/v[i] for every layer, so the lists must
        # stay length num_hidden_layers (set entries back to None, don't empty).
        for t in self.k:
            if t is not None:
                ttnn.deallocate(t)
        for t in self.v:
            if t is not None:
                ttnn.deallocate(t)
        self.k = [None] * len(self.k)
        self.v = [None] * len(self.v)
        self.length = 0


class DFlashDrafter(Drafter):
    """TT DFlash drafter. 5 Llama-style layers, single-shot block diffusion."""

    def __init__(
        self,
        mesh_device,
        config: DFlashConfig,
        cache_dir: str | None = None,
        weight_dtype=ttnn.bfloat16,
        mesh_config=None,
        ccl_manager=None,
        *,
        safetensors_dir: str | None = None,
        load_embed_tokens: bool = True,
        write_cache_dir: str | None = None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.weights: DFlashWeights = load_dflash_weights(
            mesh_device=mesh_device,
            config=config,
            cache_dir=cache_dir,
            weight_dtype=weight_dtype,
            safetensors_dir=safetensors_dir,
            load_embed_tokens=load_embed_tokens,
            write_cache_dir=write_cache_dir,
        )
        self._ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # MLP/fc fidelity matches attention (HiFi2/bf16): bf8 + LoFi here was
        # measured to drop acceptance ~6 → ~2.4 tokens/step — drafter quality
        # is precision-bound, so we keep full mantissas everywhere.
        self._ck_mlp = self._ck
        # L1 block-sharded matmul configs at packed shapes (M = B*block):
        # mirrors the target model (~1.8x over DRAM-interleaved auto).
        self._l1_pc_cache = {}

        # ─── Traced-propose support (mirrors packed_decode_forward) ───────────
        # Fuse per-layer q/k/v into a single wqkv [1,1,hidden,(Q+2KV)_local] so
        # the proven split_qkv_heads_* + q_sharded_mem machinery applies for the
        # traced KV-cache writes (dflash's separate projections otherwise can't
        # drive nlp_create_qkv_heads_decode). Column-parallel like the sources.
        self._wqkv = [ttnn.concat([lw.q_proj, lw.k_proj, lw.v_proj], dim=3) for lw in self.weights.layers]
        # q_sharded_mem (height-sharded reshard spec for paged_update_cache),
        # learned once via a decode-split probe, cached by shape key.
        self._q_sharded_mem_cache: dict = {}

        # Tied-embedding variants (z-lab) ship NO own lm_head — draft logits use
        # the TARGET's lm_head over the full target vocab. The server sets this
        # to ``target.lm_head_weight`` (column-parallel on vocab) when
        # ``config.uses_target_head``; ``_head_weight`` picks it over the (None)
        # own head. ``None`` for the speculators variant, which has its own head.
        self._target_lm_head = None

    def _linear_l1(self, x, w, kcfg, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """ttnn.linear via the L1-block-sharded path at packed shapes
        (M % 256 == 0, M <= 2048); falls back to interleaved auto otherwise."""
        M, K, N = x.shape[-2], x.shape[-1], w.shape[-1]
        if M % 256 or M > 2048:
            return ttnn.linear(x, w, memory_config=memory_config, compute_kernel_config=kcfg)
        key = (M, K, N)
        pc_mem = self._l1_pc_cache.get(key)
        if pc_mem is None:
            from models.demos.gemma4_cody.tt.ccl import make_block_sharded_matmul_config

            pc_mem = make_block_sharded_matmul_config(M, K, N)
            self._l1_pc_cache[key] = pc_mem
        pc, in_mem = pc_mem
        x_sh = ttnn.to_memory_config(x, in_mem)
        out = ttnn.linear(x_sh, w, program_config=pc, memory_config=memory_config, compute_kernel_config=kcfg)
        x_sh.deallocate(True)
        return out

    def _head_weight(self):
        """The lm_head to apply after the final norm: the drafter's own narrow
        head when present (speculators), else the target's full-vocab head
        (z-lab tied variant, set via ``self._target_lm_head``)."""
        if self.weights.lm_head is not None:
            return self.weights.lm_head
        if self._target_lm_head is not None:
            return self._target_lm_head
        raise RuntimeError(
            "DFlash drafter has no lm_head: the checkpoint ships none (tied "
            "embeddings) and no target lm_head was provided. Set "
            "drafter._target_lm_head = target.lm_head_weight (see dflash_spec)."
        )

    # ─── Drafter abstract API ───────────────────────────────────────────────

    @property
    def required_target_outputs(self) -> RequiredTargetOutputs:
        return RequiredTargetOutputs(
            aux_hidden_layers=self.config.aux_hidden_layers,
            shared_kv_layer_types=(),
            needs_scaled_embedding=False,
            needs_next_token_embedding=False,
            block_size=self.config.block_size,
        )

    def propose(self, target_outputs: TargetOutputs) -> DrafterOutput:
        # Phase 5 will wire this up to the server. Phase 2/4 calls `forward`
        # directly for parity testing where we control input layout precisely.
        raise NotImplementedError(
            "DFlashDrafter.propose() is wired up in Phase 5 (server integration). "
            "For Phase 4 parity, call `forward(aux_hiddens_concat, noise_embeddings, ...)` directly."
        )

    # ─── Per-layer forward ──────────────────────────────────────────────────

    def _layer_forward(
        self,
        layer_idx: int,
        hidden,
        target_hidden_projected,
        cos_full,
        sin_full,
        cos_q=None,
        sin_q=None,
        attn_mask=None,
        intermediates=None,
    ):
        """Single Llama-style decoder layer over (target_hidden ∥ noise).

        Args:
            hidden: TT ``[1, 1, block_size, hidden]`` — noise stream.
            target_hidden_projected: TT ``[1, 1, ctx_len, hidden]`` — already
                run through ``fc + hidden_norm`` by the top-level forward.
            cos_full, sin_full: TT ``[1, 1, ctx_len + block_size, head_dim]``.
            cos_q, sin_q: optional pre-sliced ``[1, 1, block_size, head_dim]``
                for the noise-side RoPE. If ``None``, sliced lazily from
                ``cos_full``/``sin_full`` — keeping this kwarg lets the
                top-level forward slice ONCE and reuse across layers.

        Returns the layer's output stream — same shape as ``hidden``.
        """
        cfg = self.config
        layer_w = self.weights.layers[layer_idx]
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        num_heads_local = cfg.num_attention_heads // self.tp
        num_kv_local = cfg.num_key_value_heads // self.tp
        block_size = int(hidden.shape[2])
        ctx_len = int(target_hidden_projected.shape[2])

        # Optional per-sublayer capture for the parity test (gated: no-op unless
        # an ``intermediates`` list is threaded in from the top-level forward).
        def _cap(name, t):
            if intermediates is not None:
                intermediates.append((f"layer_{layer_idx}.{name}", ttnn.to_torch(ttnn.get_device_tensors(t)[0])))

        residual = hidden
        normed = ttnn.rms_norm(hidden, weight=layer_w.input_layernorm, epsilon=eps)
        _cap("normed", normed)

        # ─── Q/K/V projections ───────────────────────────────────────────────
        # Q is projected from the noise stream only — `q_proj @ noise`.
        # K/V are projected from BOTH the projected target context AND the noise
        # stream, then concatenated along the seq dim.
        q = ttnn.linear(normed, layer_w.q_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck)
        k_noise = ttnn.linear(
            normed, layer_w.k_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
        )
        v_noise = ttnn.linear(
            normed, layer_w.v_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
        )
        ttnn.deallocate(normed)

        k_ctx = ttnn.linear(
            target_hidden_projected,
            layer_w.k_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck,
        )
        v_ctx = ttnn.linear(
            target_hidden_projected,
            layer_w.v_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck,
        )

        # Concat along seq dim — [1, 1, ctx_len, KV_dim] ⊕ [1, 1, block_size, KV_dim]
        k = ttnn.concat([k_ctx, k_noise], dim=2)
        v = ttnn.concat([v_ctx, v_noise], dim=2)
        ttnn.deallocate(k_ctx)
        ttnn.deallocate(k_noise)
        ttnn.deallocate(v_ctx)
        ttnn.deallocate(v_noise)

        # ─── Reshape for per-head norm + SDPA ───────────────────────────────
        # Q: [1, 1, block_size, num_heads_local*head_dim] → [1, block_size, num_heads_local, head_dim]
        q = ttnn.reshape(q, (1, block_size, num_heads_local, head_dim))
        # K/V: [1, 1, ctx_len+block_size, num_kv_local*head_dim] → [1, ctx_len+block_size, num_kv_local, head_dim]
        k = ttnn.reshape(k, (1, ctx_len + block_size, num_kv_local, head_dim))
        v = ttnn.reshape(v, (1, ctx_len + block_size, num_kv_local, head_dim))

        # Per-head Q-norm and K-norm — applied before transposing heads forward
        # so the norm-weight (per-head-dim) broadcasts along the seq axis.
        # K-norm applies to the *concatenated* (context ∥ noise) K, matching
        # the HF reference's `self.k_norm(k)` after the concat.
        q = apply_per_head_norm(q, layer_w.q_norm, eps, with_scale=True)
        k = apply_per_head_norm(k, layer_w.k_norm, eps, with_scale=True)

        # Now transpose into [1, heads, S, head_dim] for SDPA + RoPE.
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)

        # ─── RoPE ────────────────────────────────────────────────────────────
        # K spans (ctx + noise) — RoPE over the full cos/sin cache.
        # Q spans noise only — RoPE over the trailing `block_size` rows.
        # ``ttnn.experimental.rotary_embedding`` in prefill mode multiplies along
        # the seq dim element-wise, so for Q we slice cos/sin first.
        k = ttnn.experimental.rotary_embedding(k, cos_full, sin_full, None)
        # cos_q/sin_q ideally pre-sliced ONCE at the top-level forward and
        # passed in — saves 4 redundant slice ops in the captured trace.
        if cos_q is None:
            cos_q = cos_full[:, :, ctx_len:, :]
        if sin_q is None:
            sin_q = sin_full[:, :, ctx_len:, :]
        q = ttnn.experimental.rotary_embedding(q, cos_q, sin_q, None)

        # ─── SDPA ────────────────────────────────────────────────────────────
        # Non-causal, no sliding window. Scale = 1/√head_dim (default for SDPA).
        # Pass scale explicitly to match HF — DFlash uses head_dim**-0.5.
        scale = head_dim**-0.5
        sdpa_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=scale,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # sdpa_out: [1, num_heads_local, block_size, head_dim]
        # Concat heads → [1, 1, block_size, num_heads_local*head_dim].
        sdpa_out = ttnn.experimental.nlp_concat_heads(sdpa_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # All-gather across TP so o_proj sees the full Q-dim.
        if self.tp > 1:
            from models.demos.gemma4_cody.tt.ccl import ccl_allgather

            sdpa_out = ccl_allgather(sdpa_out, self.mesh_config, self.ccl_manager, dim=3)
        _cap("attn_concat", sdpa_out)

        # ─── o_proj + residual ──────────────────────────────────────────────
        attn_out = ttnn.linear(
            sdpa_out, layer_w.o_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
        )
        ttnn.deallocate(sdpa_out)
        _cap("attn_out", attn_out)
        post_attn = ttnn.add(residual, attn_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_out)
        _cap("post_attn", post_attn)

        # ─── MLP block ──────────────────────────────────────────────────────
        residual2 = post_attn
        mlp_in = ttnn.rms_norm(residual2, weight=layer_w.post_attention_layernorm, epsilon=eps)
        _cap("mlp_in", mlp_in)

        gate = ttnn.linear(
            mlp_in, layer_w.mlp_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck_mlp
        )
        up = ttnn.linear(
            mlp_in, layer_w.mlp_up, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck_mlp
        )
        ttnn.deallocate(mlp_in)
        gate = ttnn.silu(gate)
        mlp_intermediate = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        mlp_out = ttnn.linear(
            mlp_intermediate,
            layer_w.mlp_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck_mlp,
        )
        ttnn.deallocate(mlp_intermediate)
        _cap("mlp_out", mlp_out)
        out = ttnn.add(residual2, mlp_out)
        ttnn.deallocate(residual2)
        ttnn.deallocate(mlp_out)
        return out

    # ─── Top-level forward ──────────────────────────────────────────────────

    # SDPA prefill's k_chunk_size is 64 by default, so the K sequence (ctx ∥
    # noise) needs to be a multiple of 64. We pad the noise stream up to the
    # next multiple of 64; combined with `ctx_len` being a multiple of 64 the
    # full key length lands on a kchunk boundary.
    _SEQ_PAD_MULTIPLE = 64

    @staticmethod
    def _pad_seq_to_tile(t, multiple=64):
        """Pad dim 2 (seq) up to the next ``multiple``.

        TT eltwise binary ops trip "Invalid subtile broadcast type" when one
        operand's seq dim isn't a tile multiple (TILE_SIZE=32), and SDPA
        prefill requires the K sequence to be a multiple of the kernel's
        k_chunk_size (64 by default). Padding to a multiple of 64 satisfies
        both. The padded positions are zeroed; the final-output slice trims
        them off so callers see the original block_size.
        """
        cur = int(t.shape[2])
        if cur % multiple == 0:
            return t, 0
        pad_amount = multiple - (cur % multiple)
        padded = ttnn.pad(
            t,
            [(0, 0), (0, 0), (0, pad_amount), (0, 0)],
            value=0.0,
        )
        return padded, pad_amount

    def forward(
        self,
        aux_hiddens_concat,
        noise_embeddings,
        cos_full,
        sin_full,
        return_intermediates: bool = False,
        head: bool = True,
    ):
        """Run the DFlash drafter end-to-end on one (single-user) input.

        Args
        ----
        aux_hiddens_concat: TT ``[1, 1, ctx_len, K*target_hidden]`` — caller
            concatenated the K aux target hiddens along the feature dim.
        noise_embeddings: TT ``[1, 1, block_size, hidden]`` — embedded
            (bonus ∥ mask × (block_size-1)) tokens.
        cos_full, sin_full: TT ``[1, 1, ctx_len + block_size, head_dim]``.
        return_intermediates: when True, returns a list of (name, torch_tensor)
            pairs for the per-layer hidden states and final norm output —
            used by the parity test.

        Returns
        -------
        Phase-2 contract: ``(draft_logits, draft_hidden)`` where

          * ``draft_logits``: TT ``[1, 1, block_size - 1, draft_vocab]`` —
            the HF reference slices ``[:, -block_size+1:, :]`` before applying
            ``lm_head``; we mirror that. The first noise position is the
            bonus and is not a draft prediction.
          * ``draft_hidden``: TT ``[1, 1, block_size, hidden]`` — final
            post-norm hidden, all 8 positions, exposed for downstream use
            (parity diagnostics, future cache-write paths).

        If ``return_intermediates``, returns ``(draft_logits, draft_hidden, intermediates)``.
        """
        cfg = self.config
        eps = cfg.rms_norm_eps

        # Pad noise stream's seq dim to a tile multiple so eltwise binaries
        # (RMSNorm gamma multiply, residual add) don't trip the subtile-
        # broadcast path. Original block_size is preserved for the final
        # output slice.
        original_block_size = int(noise_embeddings.shape[2])
        noise_embeddings, noise_pad = self._pad_seq_to_tile(noise_embeddings)
        cos_full_padded, _ = self._pad_seq_to_tile(cos_full)
        sin_full_padded, _ = self._pad_seq_to_tile(sin_full)

        # 1. fc + hidden_norm on the concatenated aux hiddens (bf8 fc → LoFi).
        target_hidden_projected = self._linear_l1(aux_hiddens_concat, self.weights.fc, self._ck_mlp)
        target_hidden_projected = ttnn.rms_norm(target_hidden_projected, weight=self.weights.hidden_norm, epsilon=eps)

        intermediates = []
        if return_intermediates:
            intermediates.append(
                ("target_hidden_projected", ttnn.to_torch(ttnn.get_device_tensors(target_hidden_projected)[0]))
            )

        # 2. Layer stack. Pre-slice cos_q/sin_q ONCE — reused across all 5 layers.
        ctx_len = int(target_hidden_projected.shape[2])
        cos_q = cos_full_padded[:, :, ctx_len:, :]
        sin_q = sin_full_padded[:, :, ctx_len:, :]

        # Mask the noise-pad KEY columns in SDPA. The noise stream is zero-padded
        # up to a 64-multiple; those pad rows are zero only in layer 0 — from
        # layer 1 on they carry the previous layer's (nonzero) output and act as
        # spurious keys, and even in layer 0 the zero pad-keys inflate the softmax
        # denominator. Mask K cols [ctx_len + block_size : ctx_len + block_padded]
        # (same scheme the decode path uses). ``None`` when the noise needs no pad.
        block_padded = int(noise_embeddings.shape[2])
        attn_mask = (
            self._build_noise_mask(ctx_len, original_block_size, block_padded, ctx_len + block_padded)
            if noise_pad > 0
            else None
        )

        hidden = noise_embeddings
        for i in range(cfg.num_hidden_layers):
            hidden = self._layer_forward(
                i,
                hidden,
                target_hidden_projected,
                cos_full_padded,
                sin_full_padded,
                cos_q=cos_q,
                sin_q=sin_q,
                attn_mask=attn_mask,
                intermediates=intermediates if return_intermediates else None,
            )
            if return_intermediates:
                intermediates.append((f"layer_{i}", ttnn.to_torch(ttnn.get_device_tensors(hidden)[0])))

        ttnn.deallocate(target_hidden_projected)
        if attn_mask is not None:
            ttnn.deallocate(attn_mask)

        # 3. Final norm.
        draft_hidden = ttnn.rms_norm(hidden, weight=self.weights.final_norm, epsilon=eps)

        # 4. Slice off the tile-pad on the noise stream so downstream callers
        # see the original block_size, not the padded one.
        if noise_pad > 0:
            draft_hidden = draft_hidden[:, :, :original_block_size, :]

        # 5. lm_head over the trailing (block_size - 1) positions.
        # HF reference: `target.lm_head(self(...)[:, -block_size+1:, :])`.
        # Slice last (block_size - 1) positions along dim 2 — drops the bonus.
        # ``head=False`` skips the lm_head and returns ``draft_logits=None`` —
        # used by the hidden-state parity test for the tied-embedding (z-lab)
        # variant, which ships no own head (the head would need the target's
        # full-vocab lm_head; the drafter-specific math is fully covered by the
        # hidden-state PCC, and the head is a shared linear validated elsewhere).
        if head:
            draft_hidden_for_head = draft_hidden[:, :, 1:, :]
            draft_logits = ttnn.linear(
                draft_hidden_for_head,
                self._head_weight(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self._ck,
            )
        else:
            draft_logits = None

        if return_intermediates:
            intermediates.append(("final_norm", ttnn.to_torch(ttnn.get_device_tensors(draft_hidden)[0])))
            return draft_logits, draft_hidden, intermediates
        return draft_logits, draft_hidden

    # ─── Decode-time drafting (own anchor KV cache) ─────────────────────────
    #
    # `forward` above is the parity one-shot. The methods below are the
    # decode-time path the server drives: an own anchor KV cache that grows by
    # `accepted + 1` per step (`append_anchors` on commit), and `decode_step`
    # which drafts `block_size` tokens reading that cache. The math is identical
    # to `forward` — same per-layer ops, same non-causal attention over
    # (anchors ∥ noise) — only the context K/V come from the cache instead of
    # being recomputed, and an explicit additive mask handles the cache/noise
    # tile-padding (`forward` zero-pads without a mask; here we mask the pad).

    def init_anchor_cache(self) -> DFlashAnchorCache:
        """Fresh empty anchor cache (one per server slot / logical user)."""
        cfg = self.config
        return DFlashAnchorCache(
            k=[None] * cfg.num_hidden_layers,
            v=[None] * cfg.num_hidden_layers,
            length=0,
        )

    def _project_anchor_hidden(self, aux_hiddens_concat):
        """fc + hidden_norm on concatenated aux taps → anchor hidden.

        aux_hiddens_concat: TT [1, 1, n_new, K*target_hidden]; the caller has
        concatenated the K aux target-hidden taps along the feature dim (the
        same layout `forward` consumes). Returns [1, 1, n_new, hidden].
        """
        cfg = self.config
        anchor_h = self._linear_l1(aux_hiddens_concat, self.weights.fc, self._ck_mlp)
        return ttnn.rms_norm(anchor_h, weight=self.weights.hidden_norm, epsilon=cfg.rms_norm_eps)

    def append_anchors(self, cache: DFlashAnchorCache, aux_hiddens_concat):
        """Append `n_new` anchors (the just-committed tokens) to the cache.

        Mirrors `_layer_forward`'s `k_ctx`/`v_ctx`, but persisted instead of
        recomputed: per layer the anchor K is `k_proj → per-head k_norm` and the
        anchor V is plain `v_proj`. K-norm is per-position, so applying it at
        append time and concatenating later is identical to `forward`'s
        "k_norm the full (ctx ∥ noise) K".

        RoPE is **NOT** applied here — anchors are stored un-rotated and the full
        concatenated K is RoPE'd once per layer in `decode_step` (where the seq
        is a clean tile multiple). RoPE-at-append would call
        ``rotary_embedding`` on a sub-tile seq (n_new < 32), which pads the seq
        to 32 and desyncs K's length from V's. RoPE is per-position, so rotating
        the full K at decode time is mathematically identical.

        Args:
            cache: the slot's anchor cache (mutated in place).
            aux_hiddens_concat: TT [1, 1, n_new, K*target_hidden] for the n_new
                committed positions.
        """
        cfg = self.config
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        num_kv_local = cfg.num_key_value_heads // self.tp
        n_new = int(aux_hiddens_concat.shape[2])

        anchor_h = self._project_anchor_hidden(aux_hiddens_concat)

        for i in range(cfg.num_hidden_layers):
            layer_w = self.weights.layers[i]
            k = ttnn.linear(
                anchor_h, layer_w.k_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
            )
            v = ttnn.linear(
                anchor_h, layer_w.v_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
            )
            # [1, n_new, num_kv_local, head_dim] → per-head k_norm → [1, kv, n_new, hd]
            k = ttnn.reshape(k, (1, n_new, num_kv_local, head_dim))
            v = ttnn.reshape(v, (1, n_new, num_kv_local, head_dim))
            k = apply_per_head_norm(k, layer_w.k_norm, eps, with_scale=True)
            k = ttnn.transpose(k, 1, 2)
            v = ttnn.transpose(v, 1, 2)
            if cache.k[i] is None:
                cache.k[i] = k
                cache.v[i] = v
            else:
                new_k = ttnn.concat([cache.k[i], k], dim=2)
                new_v = ttnn.concat([cache.v[i], v], dim=2)
                ttnn.deallocate(cache.k[i])
                ttnn.deallocate(cache.v[i])
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                cache.k[i] = new_k
                cache.v[i] = new_v

        ttnn.deallocate(anchor_h)
        cache.length += n_new

    def _build_noise_mask(self, anchor_len: int, real_block: int, block_padded: int, sk: int):
        """Additive SDPA mask [1, 1, block_padded, sk], bf16, replicated.

        Non-causal within (anchors ∥ noise): every real noise query row attends
        every real anchor [0:anchor_len] and every real noise position
        [anchor_len : anchor_len + real_block]; everything past that (the 56
        noise tile-pad rows + the K round-up to a 64 multiple) is masked. The
        mask is column-only (independent of the query row), so padded query rows
        stay finite (they attend ≥1 column) and are sliced off downstream.
        """
        valid = anchor_len + real_block
        mask = torch.zeros(1, 1, block_padded, sk, dtype=torch.float32)
        mask[:, :, :, valid:] = -1e9
        is_mesh = hasattr(self.mesh_device, "shape") and self.mesh_device.get_num_devices() > 1
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None
        return ttnn.from_torch(
            mask.to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def _decode_layer_forward(self, layer_idx, hidden, cache_k, cache_v, cos_full, sin_full, cos_q, sin_q, attn_mask):
        """One dflash layer at decode time: noise Q/K/V + cached anchor K/V.

        Identical to `_layer_forward` except the K/V context comes from the cache
        (`cache_k`/`cache_v` — k_norm'd, UN-roped) and SDPA takes the explicit
        `attn_mask`. The full concatenated K is RoPE'd here (over `cos_full`,
        seq == sk); Q is RoPE'd over the noise positions (`cos_q`). `hidden` is
        the noise stream [1, 1, block_padded, hidden].
        """
        cfg = self.config
        layer_w = self.weights.layers[layer_idx]
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        num_heads_local = cfg.num_attention_heads // self.tp
        num_kv_local = cfg.num_key_value_heads // self.tp
        block = int(hidden.shape[2])

        residual = hidden
        normed = ttnn.rms_norm(hidden, weight=layer_w.input_layernorm, epsilon=eps)

        q = ttnn.linear(normed, layer_w.q_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck)
        k_noise = ttnn.linear(
            normed, layer_w.k_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
        )
        v_noise = ttnn.linear(
            normed, layer_w.v_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
        )
        ttnn.deallocate(normed)

        q = ttnn.reshape(q, (1, block, num_heads_local, head_dim))
        k_noise = ttnn.reshape(k_noise, (1, block, num_kv_local, head_dim))
        v_noise = ttnn.reshape(v_noise, (1, block, num_kv_local, head_dim))
        q = apply_per_head_norm(q, layer_w.q_norm, eps, with_scale=True)
        k_noise = apply_per_head_norm(k_noise, layer_w.k_norm, eps, with_scale=True)
        q = ttnn.transpose(q, 1, 2)
        k_noise = ttnn.transpose(k_noise, 1, 2)
        v_noise = ttnn.transpose(v_noise, 1, 2)
        # Q RoPE over the noise positions [anchor_len : anchor_len+block_padded].
        q = ttnn.experimental.rotary_embedding(q, cos_q, sin_q, None)

        # K/V = cached anchors (k_norm'd, UN-roped) ∥ this step's noise (k_norm'd,
        # UN-roped). The FULL K is RoPE'd below — exactly forward()'s "k_norm then
        # RoPE the full (ctx ∥ noise) K". cache_k/cache_v share a seq length
        # (appended together), so K and V stay matched into SDPA.
        if cache_k is not None:
            k = ttnn.concat([cache_k, k_noise], dim=2)
            v = ttnn.concat([cache_v, v_noise], dim=2)
            ttnn.deallocate(k_noise)
            ttnn.deallocate(v_noise)
        else:
            k, v = k_noise, v_noise

        # Pad K/V seq to sk (== cos_full's seq, a multiple of 64); the pad is masked.
        sk = int(cos_full.shape[2])
        cur_kv = int(k.shape[2])
        if cur_kv < sk:
            pad = sk - cur_kv
            k = ttnn.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            v = ttnn.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        # RoPE the FULL K over [0:sk] (anchors at their positions, then noise, then pad).
        k = ttnn.experimental.rotary_embedding(k, cos_full, sin_full, None)

        scale = head_dim**-0.5
        sdpa_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=scale
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        sdpa_out = ttnn.experimental.nlp_concat_heads(sdpa_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.tp > 1:
            from models.demos.gemma4_cody.tt.ccl import ccl_allgather

            sdpa_out = ccl_allgather(sdpa_out, self.mesh_config, self.ccl_manager, dim=3)

        attn_out = ttnn.linear(
            sdpa_out, layer_w.o_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
        )
        ttnn.deallocate(sdpa_out)
        post_attn = ttnn.add(residual, attn_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_out)

        residual2 = post_attn
        mlp_in = ttnn.rms_norm(residual2, weight=layer_w.post_attention_layernorm, epsilon=eps)
        gate = ttnn.linear(
            mlp_in, layer_w.mlp_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck_mlp
        )
        up = ttnn.linear(
            mlp_in, layer_w.mlp_up, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck_mlp
        )
        ttnn.deallocate(mlp_in)
        gate = ttnn.silu(gate)
        mlp_intermediate = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        mlp_out = ttnn.linear(
            mlp_intermediate,
            layer_w.mlp_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck_mlp,
        )
        ttnn.deallocate(mlp_intermediate)
        out = ttnn.add(residual2, mlp_out)
        ttnn.deallocate(residual2)
        ttnn.deallocate(mlp_out)
        return out

    def decode_step(self, cache: DFlashAnchorCache, noise_embeddings, cos_full, sin_full, cos_q, sin_q):
        """Draft `block_size` tokens reading the slot's anchor cache.

        Args:
            cache: the slot's anchor cache (read-only here; grown via
                `append_anchors` on commit).
            noise_embeddings: TT [1, 1, block_size, hidden] — embed([bonus,
                mask × (block_size-1)]) from dflash's own (unscaled) embed_tokens.
            cos_full, sin_full: TT [1, 1, cache.length + block_padded, head_dim] —
                RoPE for the full (anchors ∥ noise) key span, positions
                [0 : cache.length + block_padded]. Padded to a 64-multiple here.
            cos_q, sin_q: TT [1, 1, block_padded, head_dim] — RoPE for the noise
                query positions [cache.length : cache.length + block_padded].

        Returns ``draft_logits`` TT [1, 1, block_size-1, draft_vocab] — the
        trailing block_size-1 positions (the bonus position is dropped), to be
        argmax'd + d2t-remapped by `draft_ids_from_logits`.
        """
        cfg = self.config
        eps = cfg.rms_norm_eps
        original_block_size = int(noise_embeddings.shape[2])
        noise, noise_pad = self._pad_seq_to_tile(noise_embeddings)
        # The server feeds noise from `embed_tokens` (ttnn.embedding → ROW_MAJOR);
        # rms_norm / the layer ops need TILE. Idempotent when already TILE (the
        # parity test passes TILE). Padded seq is a clean 64-multiple here.
        noise = ttnn.to_layout(noise, ttnn.TILE_LAYOUT)
        block_padded = int(noise.shape[2])

        # Pad the full-K RoPE cache to a 64 multiple (== the SDPA key length sk).
        # The mask zeroes everything past the valid (anchors ∥ real noise) span.
        cos_full_p, _ = self._pad_seq_to_tile(cos_full)
        sin_full_p, _ = self._pad_seq_to_tile(sin_full)
        sk = int(cos_full_p.shape[2])
        attn_mask = self._build_noise_mask(cache.length, original_block_size, block_padded, sk)

        hidden = noise
        for i in range(cfg.num_hidden_layers):
            if _DFLASH_DEBUG:
                ttnn.synchronize_device(self.mesh_device)
                print(
                    f"[dflash-ckpt]   decode_step entering layer {i} (sk={sk}, anchor_len={cache.length})", flush=True
                )
            hidden = self._decode_layer_forward(
                i, hidden, cache.k[i], cache.v[i], cos_full_p, sin_full_p, cos_q, sin_q, attn_mask
            )
        if _DFLASH_DEBUG:
            ttnn.synchronize_device(self.mesh_device)
            print("[dflash-ckpt]   decode_step layers done, lm_head next", flush=True)
        ttnn.deallocate(attn_mask)

        draft_hidden = ttnn.rms_norm(hidden, weight=self.weights.final_norm, epsilon=eps)
        if noise_pad > 0:
            draft_hidden = draft_hidden[:, :, :original_block_size, :]
        # Drop the bonus position; lm_head over the trailing block_size-1 drafts.
        draft_hidden_for_head = draft_hidden[:, :, 1:, :]
        draft_logits = ttnn.linear(
            draft_hidden_for_head,
            self._head_weight(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck,
        )
        ttnn.deallocate(draft_hidden)
        return draft_logits

    def _d2t_host(self) -> Optional["torch.Tensor"]:
        """d2t offset table on host (cached). int64 [draft_vocab] or None."""
        if self.weights.draft_id_to_target_id is None:
            return None
        if getattr(self, "_d2t_cache", None) is None:
            t = ttnn.to_torch(ttnn.get_device_tensors(self.weights.draft_id_to_target_id)[0])
            self._d2t_cache = t.reshape(-1).to(torch.int64)
        return self._d2t_cache

    def draft_ids_from_logits(self, draft_logits) -> "torch.Tensor":
        """argmax over the draft vocab + d2t offset remap → target-vocab ids.

        draft_logits: TT [1, 1, block_size-1, draft_vocab] (column-parallel on
        the vocab dim → concat shards). Returns torch int64 [block_size-1] in
        the **target** vocab: ``target_id = draft_id + d2t[draft_id]``.
        """
        is_mesh = hasattr(self.mesh_device, "shape") and self.mesh_device.get_num_devices() > 1
        if is_mesh and self.tp > 1:
            shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(draft_logits)]
            logits = torch.cat(shards, dim=-1)
        elif is_mesh:
            logits = ttnn.to_torch(ttnn.get_device_tensors(draft_logits)[0]).float()
        else:
            logits = ttnn.to_torch(draft_logits).float()
        draft_ids = logits[0, 0].argmax(dim=-1).to(torch.int64)  # [block_size-1]
        d2t = self._d2t_host()
        if d2t is None:
            return draft_ids  # Qwen variant: draft vocab == target vocab.
        return draft_ids + d2t[draft_ids]

    # ─── Traced packed propose (mirrors tt/attention/decode.py packed_decode_forward) ──
    #
    # The server drives this as a captured trace, B=32 batched, single forward.
    # The drafter's own KV is a FIXED-shape cache [B, n_kv_local, MAX_ANCHORS, hd]
    # (like the target's KV) so the propose can be traced. Anchors + this step's
    # noise are stored **un-RoPE'd** in the cache (writes never need RoPE → no
    # sub-tile-pad bug); the whole cache K is RoPE'd on read each layer (256 rows,
    # tile-aligned) and Q is RoPE'd at the noise positions. Noise rows are
    # slot-major/position-minor (row u*block+p), matching packed_decode_forward.

    def alloc_anchor_caches(self, batch: int, max_anchors: int):
        """Allocate the fixed-shape anchor KV cache: per layer [B, n_kv_local,
        max_anchors, head_dim], zero-init (trace-safe — see init_kv_cache)."""
        from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache

        caches = []
        for _ in range(self.config.num_hidden_layers):
            kv = init_kv_cache(
                self.mesh_device, self.config, max_batch_size=batch, max_seq_len=max_anchors, cache_dtype=ttnn.bfloat16
            )
            caches.append((kv[0], kv[1]))
        return caches

    def _q_sharded_mem(self, xqkv, n_users: int, qkv_dim: int):
        """Height-sharded ([B]-user) reshard spec for paged_update_cache, learned
        once via a decode-split probe and cached (mirrors packed_decode_forward §②).
        Also stashed as ``_q_sharded_mem_B`` for the eager append, which has no
        fused xqkv to probe."""
        cfg = self.config
        key = (n_users, qkv_dim, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, self.tp)
        spec = self._q_sharded_mem_cache.get(key)
        if spec is None:
            kv_replicated = cfg.num_key_value_heads < self.tp
            full = n_users == int(xqkv.shape[2])
            probe = xqkv if full else ttnn.slice(xqkv, [0, 0, 0, 0], [1, 1, n_users, qkv_dim])
            qp, kp, vp = split_qkv_heads_decode(probe, cfg, False, tp=self.tp, kv_replicated=kv_replicated)
            spec = qp.memory_config()
            self._q_sharded_mem_cache[key] = spec
            self._q_sharded_mem_B = spec
            ttnn.deallocate(qp)
            ttnn.deallocate(kp)
            ttnn.deallocate(vp)
            if not full:
                ttnn.deallocate(probe)
        return spec

    def _write_kv_per_p(self, k_cache, v_cache, tt_k, tt_v, write_idxs, B, n_pos, n_kv_local, head_dim, q_sharded_mem):
        """Write ``n_pos`` positions of K/V into the cache, one per call (mirrors
        packed_decode_forward §⑤ per-p loop). ``tt_k``/``tt_v`` are
        [1, n_kv_local, B*n_pos, hd]; row u*n_pos+p → slot u position p.
        ``write_idxs[p]`` is a [B] int32 cache index per slot (-1 ⇒ skip)."""
        tt_k_bp = ttnn.permute(tt_k, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,B*n_pos,nkv,hd]
        tt_v_bp = ttnn.permute(tt_v, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_k_view = ttnn.reshape(tt_k_bp, (1, B, n_pos, n_kv_local, head_dim))
        tt_v_view = ttnn.reshape(tt_v_bp, (1, B, n_pos, n_kv_local, head_dim))
        for p in range(n_pos):
            k_p = ttnn.reshape(
                ttnn.slice(tt_k_view, [0, 0, p, 0, 0], [1, B, p + 1, n_kv_local, head_dim]),
                (1, B, n_kv_local, head_dim),
            )
            v_p = ttnn.reshape(
                ttnn.slice(tt_v_view, [0, 0, p, 0, 0], [1, B, p + 1, n_kv_local, head_dim]),
                (1, B, n_kv_local, head_dim),
            )
            k_p = ttnn.to_memory_config(k_p, q_sharded_mem)
            v_p = ttnn.to_memory_config(v_p, q_sharded_mem)
            ttnn.experimental.paged_update_cache(k_cache, k_p, update_idxs_tensor=write_idxs[p])
            ttnn.experimental.paged_update_cache(v_cache, v_p, update_idxs_tensor=write_idxs[p])
            ttnn.deallocate(k_p)
            ttnn.deallocate(v_p)
        ttnn.deallocate(tt_k_bp)
        ttnn.deallocate(tt_v_bp)

    @staticmethod
    def _sdpa_head_splits(n_heads_local, n_kv_local, block, head_dim):
        """How many QUERY-HEAD-wise sub-ops to split the packed propose
        decode-SDPA into. The packed scheme folds ``n_heads_local*block`` query
        rows onto ``n_kv_local`` KV groups; the per-core SDPA-decode CBs (Q, QK
        scores, and the flash cross-core reduction buffer) scale with the packed
        query-head tile count ``PNHt = n_heads_local*block/32``. The heavy z-lab
        block-16 / head_dim-128 / 8-way-GQA drafter overflows the 1.5 MB L1 even
        at k_chunk=32, so we split the packed-head dim across ``n`` ops and
        concat — each op carries ``n_heads_local/n`` heads, i.e. ``PNHt/n``.

        Splitting on HEADS (not batch/slots) is what actually shrinks L1: it
        lowers PNHt directly. Each sub-op still runs on the FULL core grid with
        the SAME program config (we never shrink the grid or cap
        ``max_cores_per_head_batch``), so active-core count is preserved and the
        per-head K-reduction parallelism only goes up. Splitting batch instead
        would leave PNHt unchanged and *grow* the reduction buffer (fewer batch
        groups ⇒ more cores per KV head ⇒ more partials buffered per core).

        Head boundaries only: the ``block`` noise positions of a token attend to
        each other bidirectionally (K/V in the cache span
        [anchor_len:anchor_len+block], non-causal mask) and must stay in one op;
        a whole head's ``block`` rows always stay together, and each head
        sub-range maps onto its KV head(s) via GQA.

        ``GEMMA4_DFLASH_SDPA_HEAD_SPLITS`` overrides the count; it is reduced to
        the largest valid value ≤ the request (see ``_ok`` below)."""
        env = os.environ.get("GEMMA4_DFLASH_SDPA_HEAD_SPLITS")
        q_heads_per_kv = (n_heads_local * block) // max(1, n_kv_local)
        heavy = q_heads_per_kv * head_dim > 8192
        n = int(env) if env else (2 if heavy else 1)
        n = max(1, min(n, n_heads_local))
        g = max(1, n_heads_local // max(1, n_kv_local))  # GQA ratio (q heads / KV head)

        def _ok(n):
            if n_heads_local % n:  # whole heads per op
                return False
            hp = n_heads_local // n  # heads per op
            if (hp * block) % 32:  # query-row slice tile-aligned (32)
                return False
            return (hp % g == 0) or (g % hp == 0)  # clean GQA: whole or sub-KV groups

        while n > 1 and not _ok(n):
            n -= 1
        return n

    def _packed_propose_sdpa(self, q_packed, roped_k, v_cache, attn_mask, scale, sdpa_pc, sdpa_args, n_splits):
        """Run the packed propose decode-SDPA, optionally split along the packed
        QUERY-HEAD dim into ``n_splits`` full-grid ops + a concat (see
        ``_sdpa_head_splits``). ``q_packed`` is [1,B,n_heads_local*block,hd] with
        the head-major packing [h0b0..h0b15, h1b0..., ...]; ``roped_k``/
        ``v_cache`` are [B,n_kv_local,MA,hd]; ``attn_mask`` is
        [B,1,n_heads_local*block,MA]. Returns [1,B,n_heads_local*block,hd].
        Does NOT free its inputs.

        ``sdpa_args = (B, n_heads_local, n_kv_local, max_anchors, head_dim, block)``."""
        B, n_heads_local, n_kv_local, max_anchors, head_dim, block = sdpa_args
        if n_splits <= 1:
            return ttnn.transformer.scaled_dot_product_attention_decode(
                q_packed,
                roped_k,
                v_cache,
                is_causal=False,
                attn_mask=attn_mask,
                scale=scale,
                sliding_window_size=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=sdpa_pc,
            )
        heads_per = n_heads_local // n_splits  # whole heads per sub-op
        rows_per = heads_per * block  # packed query rows per sub-op
        g = max(1, n_heads_local // max(1, n_kv_local))  # GQA ratio (q heads per KV head)
        parts = []
        for s in range(n_splits):
            h_lo = s * heads_per
            r0, r1 = s * rows_per, (s + 1) * rows_per
            kv_lo = h_lo // g  # KV head(s) this head sub-range attends to
            kv_hi = ((h_lo + heads_per - 1) // g) + 1
            q_i = ttnn.slice(q_packed, [0, 0, r0, 0], [1, B, r1, head_dim])
            m_i = ttnn.slice(attn_mask, [0, 0, r0, 0], [B, 1, r1, max_anchors])
            # When this sub-op spans ALL local KV heads (e.g. n_kv_local==1 at
            # tp=8: every head group shares the lone KV head), the KV "slice"
            # would be a full-tensor no-op that ALIASES roped_k / v_cache rather
            # than copying — deallocating it would then free the shared input
            # (and v_cache is the persistent anchor cache). Share directly in
            # that case (read-only); only slice+free for a proper KV subset
            # (e.g. tp=4, n_kv_local==2 ⇒ one of the two heads per op).
            kv_full = kv_lo == 0 and kv_hi == n_kv_local
            if kv_full:
                k_i, v_i = roped_k, v_cache
            else:
                k_i = ttnn.slice(roped_k, [0, kv_lo, 0, 0], [B, kv_hi, max_anchors, head_dim])
                v_i = ttnn.slice(v_cache, [0, kv_lo, 0, 0], [B, kv_hi, max_anchors, head_dim])
            out_i = ttnn.transformer.scaled_dot_product_attention_decode(
                q_i,
                k_i,
                v_i,
                is_causal=False,
                attn_mask=m_i,
                scale=scale,
                sliding_window_size=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=sdpa_pc,
            )
            ttnn.deallocate(q_i)
            ttnn.deallocate(m_i)
            if not kv_full:
                ttnn.deallocate(k_i)
                ttnn.deallocate(v_i)
            parts.append(out_i)
        out = ttnn.concat(parts, dim=2)  # [1, B, n_heads_local*block, hd]
        for p in parts:
            ttnn.deallocate(p)
        return out

    def decode_forward_packed(
        self,
        noise_embeds,
        caches,
        noise_write_idxs,
        cos_bp,
        sin_bp,
        fixed_cos,
        fixed_sin,
        attn_mask,
        B,
        block,
        intermediates=None,
        head=True,
    ):
        """One traced, B-batched dflash propose forward.

        Args (all pre-allocated device buffers, refreshed by the server):
            noise_embeds: [1, 1, B*block, hidden] — embed([bonus, mask×(block-1)])/slot.
            caches: list of num_hidden_layers (k_cache, v_cache), each
                [B, n_kv_local, MAX_ANCHORS, head_dim] (un-RoPE'd).
            noise_write_idxs: list of ``block`` [B] int32 — cache write index
                anchor_len[u]+p per slot for noise position p.
            cos_bp/sin_bp: [1, 1, B*block, head_dim] — Q RoPE at noise positions.
            fixed_cos/sin: [1, 1, MAX_ANCHORS, head_dim] — anchor-relative RoPE
                for the whole cache K (position == cache index).
            attn_mask: [B, 1, n_heads_local*block, MAX_ANCHORS] additive (valid
                [0:anchor_len+block] per slot).

        Returns draft_logits [1, 1, B*(block-1), draft_vocab_local].
        """
        cfg = self.config
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        n_heads_local = cfg.num_attention_heads // self.tp
        n_kv_local = cfg.num_key_value_heads // self.tp
        kv_replicated = cfg.num_key_value_heads < self.tp
        qkv_dim = (n_heads_local + 2 * n_kv_local) * head_dim
        max_anchors = int(caches[0][0].shape[2])
        scale = head_dim**-0.5
        l1 = ttnn.L1_MEMORY_CONFIG
        # Packed SDPA-decode L1 budget: this scheme maps all `n_heads_local*block`
        # query rows of a slot onto its `n_kv_local` KV groups, so the per-core
        # circular buffers scale with the query-heads-per-KV-group
        # (`n_heads_local*block / n_kv_local`) × head_dim × k_chunk_size. The
        # block-8 / head_dim-256 speculator maps ≤16 q-heads/group and fits at
        # k_chunk=64; the z-lab block-16 / head_dim-128 / 8-way-GQA drafter maps
        # 128 q-heads/group (8 local heads × 16 block onto 1 KV group at tp=8)
        # and overflows the 1.5 MB L1. We pull TWO independent levers and keep
        # all cores busy (q_chunk is already a single 32-tile and the grid is
        # already maxed):
        #   1. halve the K chunk (64→32) for the heavy variant;
        #   2. split the decode-SDPA along the packed QUERY-HEAD dim into
        #      `n_sdpa_splits` full-grid ops + a concat (`_sdpa_head_splits` /
        #      `_packed_propose_sdpa`) — each op carries n_heads_local/n heads,
        #      cutting the packed query-head tile count (PNHt) the per-core CBs
        #      scale with. Every op keeps the full core grid and the same
        #      program config — we never shrink the grid or cap
        #      `max_cores_per_head_batch`, so active-core count and K-reduction
        #      parallelism are preserved. (Splitting batch would leave PNHt
        #      unchanged and grow the cross-core reduction buffer instead.)
        q_heads_per_kv = (n_heads_local * block) // max(1, n_kv_local)
        k_chunk = 32 if q_heads_per_kv * head_dim > 8192 else 64
        n_sdpa_splits = self._sdpa_head_splits(n_heads_local, n_kv_local, block, head_dim)
        sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4) if head_dim >= 512 else ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
            max_cores_per_head_batch=16,
        )
        sdpa_args = (B, n_heads_local, n_kv_local, max_anchors, head_dim, block)

        # Clone-then-free the input so layer 0's `ttnn.deallocate(residual)`
        # doesn't free a buffer the caller might still own (server's pre-trace
        # design previously held `noise_embeds` outside the trace; today it's
        # an in-trace embedding output, but the same convention keeps eager
        # callers safe). Freeing the original immediately means the in-trace
        # embedding+AG output doesn't sit dead for the entire layer loop.
        stream = ttnn.clone(noise_embeds, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,B*block,hidden]
        ttnn.deallocate(noise_embeds)
        _op = _DFLASH_OPPROF and DFLASH_OPPROF["active"]  # section timing (warmup only)

        # Optional per-sublayer capture for parity testing (gated: no-op unless an
        # ``intermediates`` list is passed — the traced server path passes None, so
        # this never D2Hs inside a captured trace). ``layer`` is the loop var below.
        def _cap(name, t):
            if intermediates is not None:
                intermediates.append((f"layer_{layer}.{name}", ttnn.to_torch(ttnn.get_device_tensors(t)[0])))

        for layer in range(cfg.num_hidden_layers):
            lw = self.weights.layers[layer]
            k_cache, v_cache = caches[layer]
            if _op:
                ttnn.synchronize_device(self.mesh_device)
                _t = time.perf_counter()
            residual = stream
            normed = ttnn.rms_norm(stream, weight=lw.input_layernorm, epsilon=eps)
            _cap("normed", normed)
            xqkv = self._linear_l1(normed, self._wqkv[layer], self._ck, memory_config=l1)
            ttnn.deallocate(normed)
            q_sharded_mem = self._q_sharded_mem(xqkv, B, qkv_dim)
            tt_q, tt_k, tt_v = split_qkv_heads_prefill(
                xqkv, cfg, False, tp=self.tp, kv_replicated=kv_replicated, memory_config=l1
            )
            ttnn.deallocate(xqkv)
            tt_q = apply_per_head_norm(tt_q, lw.q_norm, eps, with_scale=True, memory_config=l1)
            tt_k = apply_per_head_norm(tt_k, lw.k_norm, eps, with_scale=True, memory_config=l1)
            # Q RoPE at the noise positions; K stays un-RoPE'd (cache RoPE'd on read).
            tt_q = apply_rope(tt_q, cos_bp, sin_bp, token_index=None, memory_config=l1)
            # Park Q in DRAM across the KV-write loop (L1 pressure — see packed_decode §⑤).
            tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
            if _op:
                ttnn.synchronize_device(self.mesh_device)
                DFLASH_OPPROF["qkv"] += time.perf_counter() - _t
                _t = time.perf_counter()
            # Write this step's noise K/V (un-RoPE'd) into the cache at [anchor_len+p].
            self._write_kv_per_p(
                k_cache, v_cache, tt_k, tt_v, noise_write_idxs, B, block, n_kv_local, head_dim, q_sharded_mem
            )
            ttnn.deallocate(tt_k)
            ttnn.deallocate(tt_v)
            if _op:
                ttnn.synchronize_device(self.mesh_device)
                DFLASH_OPPROF["kvwrite"] += time.perf_counter() - _t
                _t = time.perf_counter()

            # RoPE the whole cache K on read (anchor-relative, fixed positions).
            roped_k = ttnn.reshape(k_cache, (1, B * n_kv_local, max_anchors, head_dim))
            roped_k = ttnn.experimental.rotary_embedding(roped_k, fixed_cos, fixed_sin, None)
            roped_k = ttnn.reshape(roped_k, (B, n_kv_local, max_anchors, head_dim))

            # Pack Q head-major [1, B, H_local*block, hd] (packed_decode §⑥).
            tt_q = ttnn.to_layout(tt_q, ttnn.ROW_MAJOR_LAYOUT)
            tt_q = ttnn.reshape(tt_q, (1, n_heads_local, B, block, head_dim))
            tt_q = ttnn.permute(tt_q, (0, 2, 1, 3, 4))
            tt_q = ttnn.reshape(tt_q, (1, B, n_heads_local * block, head_dim))
            q_packed = ttnn.to_layout(tt_q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(tt_q)
            # Masked decode-SDPA over the anchor cache (+ this step's noise K/V),
            # optionally split along the packed query-head dim to fit L1 — see
            # `_sdpa_head_splits`. Each sub-op runs on the full core grid.
            sdpa = self._packed_propose_sdpa(
                q_packed, roped_k, v_cache, attn_mask, scale, sdpa_pc, sdpa_args, n_sdpa_splits
            )
            ttnn.deallocate(q_packed)
            ttnn.deallocate(roped_k)
            # Unpack head-major → [1, B*block, H_local, hd] (packed_decode §⑦).
            sdpa = ttnn.to_layout(sdpa, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)
            sdpa = ttnn.reshape(sdpa, (1, B, n_heads_local, block, head_dim))
            sdpa = ttnn.permute(sdpa, (0, 1, 3, 2, 4))
            sdpa = ttnn.reshape(sdpa, (1, B * block, n_heads_local, head_dim))
            sdpa = ttnn.to_layout(sdpa, ttnn.TILE_LAYOUT, memory_config=l1)
            # concat_heads(is_decode_mode=True) transposes dim 1↔2 before
            # nlp_concat_heads so the op sees [1, H_local, B*block, hd] (heads on
            # dim 1) — calling nlp_concat_heads directly on [1, B*block, H, hd]
            # treats B*block as "num_heads" and blows past L1 (CBs sized for 256
            # heads). Mirrors packed_decode_forward §⑦.
            attn_out = concat_heads(sdpa, is_decode_mode=True, memory_config=l1)  # [1,1,B*block,H_local*hd]
            ttnn.deallocate(sdpa)
            if _op:
                ttnn.synchronize_device(self.mesh_device)
                DFLASH_OPPROF["rope_sdpa"] += time.perf_counter() - _t
                _t = time.perf_counter()
            if self.tp > 1:
                from models.demos.gemma4_cody.tt.ccl import ccl_allgather

                attn_out = ccl_allgather(attn_out, self.mesh_config, self.ccl_manager, dim=3)
            _cap("attn_concat", attn_out)
            attn_out = self._linear_l1(attn_out, lw.o_proj, self._ck)
            _cap("attn_out", attn_out)
            post_attn = ttnn.add(residual, attn_out)
            ttnn.deallocate(residual)
            ttnn.deallocate(attn_out)
            _cap("post_attn", post_attn)
            if _op:
                ttnn.synchronize_device(self.mesh_device)
                DFLASH_OPPROF["ccl_oproj"] += time.perf_counter() - _t
                _t = time.perf_counter()

            residual2 = post_attn
            mlp_in = ttnn.rms_norm(post_attn, weight=lw.post_attention_layernorm, epsilon=eps)
            _cap("mlp_in", mlp_in)
            gate = self._linear_l1(mlp_in, lw.mlp_gate, self._ck_mlp)
            up = self._linear_l1(mlp_in, lw.mlp_up, self._ck_mlp)
            ttnn.deallocate(mlp_in)
            gate = ttnn.silu(gate)
            inter = ttnn.mul(gate, up)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            mlp_out = self._linear_l1(inter, lw.mlp_down, self._ck_mlp)
            ttnn.deallocate(inter)
            _cap("mlp_out", mlp_out)
            stream = ttnn.add(residual2, mlp_out)
            ttnn.deallocate(residual2)
            ttnn.deallocate(mlp_out)
            if intermediates is not None:
                intermediates.append((f"layer_{layer}", ttnn.to_torch(ttnn.get_device_tensors(stream)[0])))
            if _op:
                ttnn.synchronize_device(self.mesh_device)
                DFLASH_OPPROF["mlp"] += time.perf_counter() - _t

        if _op:
            ttnn.synchronize_device(self.mesh_device)
            _t = time.perf_counter()
        draft_hidden = ttnn.rms_norm(stream, weight=self.weights.final_norm, epsilon=eps)  # [1,1,B*block,hidden]
        if intermediates is not None:
            intermediates.append(("final_norm", ttnn.to_torch(ttnn.get_device_tensors(draft_hidden)[0])))
        # ``head=False`` returns the post-norm hidden (all B*block positions) for the
        # head-less parity test (z-lab ships no own lm_head); the server passes head=True.
        if not head:
            return draft_hidden
        # Drop the bonus (position 0) per slot → lm_head over the block-1 drafts.
        dh = ttnn.reshape(draft_hidden, (1, B, block, cfg.hidden_size))
        dh = ttnn.slice(dh, [0, 0, 1, 0], [1, B, block, cfg.hidden_size])  # [1, B, block-1, hidden]
        dh = ttnn.reshape(dh, (1, 1, B * (block - 1), cfg.hidden_size))
        # Own narrow lm_head (speculators) or the target's full-vocab head
        # (z-lab tied variant). Greedy drafting argmaxes these logits, so the
        # z-lab `final_logit_softcapping` is intentionally NOT applied here
        # (tanh is monotonic ⇒ argmax-invariant), matching the target's verify.
        draft_logits = ttnn.linear(
            dh, self._head_weight(), memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self._ck
        )
        if _op:
            ttnn.synchronize_device(self.mesh_device)
            DFLASH_OPPROF["head"] += time.perf_counter() - _t
        return draft_logits

    def write_anchors_packed(self, caches, aux_concat, write_idxs, B, n_pos):
        """Append committed tokens as anchors (eager, off-trace; mirrors the
        propose KV write but un-RoPE'd and from the fused target hidden).

        aux_concat: [1, 1, B*n_pos, K*target_hidden] — slot u's committed
            positions at rows [u*n_pos : u*n_pos+n_pos] (padded for j>n_acc_u).
        write_idxs: list of ``n_pos`` [B] int32 — cache index anchor_len[u]+j
            per slot (-1 where slot u has no committed token at position j).
        """
        cfg = self.config
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        n_kv_local = cfg.num_key_value_heads // self.tp
        l1 = ttnn.L1_MEMORY_CONFIG
        q_sharded_mem = getattr(self, "_q_sharded_mem_B", None)
        if q_sharded_mem is None:
            raise RuntimeError("write_anchors_packed: q_sharded_mem not learned yet (run decode_forward_packed first)")
        anchor_h = self._project_anchor_hidden(aux_concat)  # [1,1,B*n_pos,hidden]
        for layer in range(cfg.num_hidden_layers):
            lw = self.weights.layers[layer]
            k_cache, v_cache = caches[layer]
            k = ttnn.linear(anchor_h, lw.k_proj, memory_config=l1, compute_kernel_config=self._ck)
            v = ttnn.linear(anchor_h, lw.v_proj, memory_config=l1, compute_kernel_config=self._ck)
            k = ttnn.reshape(k, (1, B * n_pos, n_kv_local, head_dim))
            v = ttnn.reshape(v, (1, B * n_pos, n_kv_local, head_dim))
            k = apply_per_head_norm(k, lw.k_norm, eps, with_scale=True, memory_config=l1)
            # To [1, n_kv_local, B*n_pos, hd] for the shared per-p writer.
            k = ttnn.transpose(k, 1, 2)
            v = ttnn.transpose(v, 1, 2)
            self._write_kv_per_p(k_cache, v_cache, k, v, write_idxs, B, n_pos, n_kv_local, head_dim, q_sharded_mem)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        ttnn.deallocate(anchor_h)
