# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy pure-PyTorch DFlash reference for parity testing.

Memory-budgeted reproduction of the DFlash forward — Qwen3-style decoder
layers (per-head q/k-norms despite the ``model_type: llama`` config),
``fc + hidden_norm`` context projection, narrow draft ``lm_head``. Adapted
from ``/mnt/nas/qwen-coder-30b-a3b/dflash/dflash.py`` and verified against
the actual ``RedHatAI/gemma-4-31B-it-speculator.dflash`` safetensors keys.

Why lazy
--------

Eager-loading the full ~4.3 B fp32 reference takes ~17 GB RAM, which OOMs
on hosts with single-digit-GB RAM (and ~9 GB even at bf16). Instead we
read each tensor on demand directly from the source safetensors via
:class:`models.demos.gemma4_cody.utils.lazy_state_dict.LazyStateDict`
— peak RSS is one layer's weights at fp32 (~1.8 GB) plus the small
top-level pieces (``fc``, ``norm``, ``hidden_norm``, ``lm_head``). The
math runs at fp32 throughout; weights are loaded bf16 from disk and cast
inside each matmul.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Llama/Qwen-style RMSNorm at fp32. ``weight`` may be bf16."""
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps)).to(x.dtype) * weight.to(x.dtype)


def _per_head_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm. ``x`` shape ``[..., num_heads, head_dim]``; ``weight`` shape ``[head_dim]``."""
    return _rms_norm(x, weight, eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _linear(x_fp32: torch.Tensor, w_bf16_or_fp32: torch.Tensor) -> torch.Tensor:
    """fp32 matmul where the weight may live in bf16 — cast in-line, free after."""
    return F.linear(x_fp32, w_bf16_or_fp32.float())


class LazyDFlashReference:
    """Streaming PyTorch DFlash reference.

    Resident state is bounded: small top-level tensors stay in RAM at bf16
    (~500 MB for the Gemma config), per-layer weights are loaded on demand,
    cast to fp32 inside each op, and freed before the next layer.

    Use as a callable: ``ref(aux, noise, cos, sin)`` returns
    ``(draft_logits[fp32], draft_hidden[fp32])``.
    """

    def __init__(self, cfg, lazy_state_dict, *, keep_top_level_fp32: bool = False):
        from ...utils.lazy_state_dict import LazyStateDict  # local import to avoid cycles

        assert isinstance(lazy_state_dict, LazyStateDict)
        self.cfg = cfg
        self.lsd = lazy_state_dict
        self._keep_fp32 = keep_top_level_fp32

        # Resident top-level pieces. Stored bf16 by default (LazyStateDict cast).
        cast = (lambda t: t.float()) if keep_top_level_fp32 else (lambda t: t)
        self.fc_w = cast(lazy_state_dict["fc.weight"])
        self.hidden_norm_w = cast(lazy_state_dict["hidden_norm.weight"])
        self.norm_w = cast(lazy_state_dict["norm.weight"])
        # ``lm_head`` is 344 MB bf16 / 689 MB fp32 — keep resident, it's used once per call.
        self.lm_head_w = cast(lazy_state_dict["lm_head.weight"])

    # ─── layer forward ──────────────────────────────────────────────────────

    def _layer_forward(
        self,
        hidden: torch.Tensor,
        target_hidden_projected: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        num_heads = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        num_kv_groups = num_heads // num_kv
        scale = head_dim**-0.5
        prefix = f"layers.{layer_idx}"
        lsd = self.lsd

        # ── Pre-attention norm + Q/K/V projections ───────────────────────────
        input_ln_w = lsd[f"{prefix}.input_layernorm.weight"]
        residual = hidden
        normed = _rms_norm(hidden, input_ln_w, eps).float()
        del input_ln_w

        # Q from noise (residual stream) only.
        q_proj = lsd[f"{prefix}.self_attn.q_proj.weight"]
        q = _linear(normed, q_proj)
        del q_proj
        # K/V from concat(target_hidden_projected, normed) along seq.
        k_proj = lsd[f"{prefix}.self_attn.k_proj.weight"]
        k_ctx = _linear(target_hidden_projected, k_proj)
        k_noise = _linear(normed, k_proj)
        del k_proj
        v_proj = lsd[f"{prefix}.self_attn.v_proj.weight"]
        v_ctx = _linear(target_hidden_projected, v_proj)
        v_noise = _linear(normed, v_proj)
        del v_proj, normed

        bsz, q_len, _ = hidden.shape
        ctx_len = target_hidden_projected.shape[1]

        q = q.view(bsz, q_len, num_heads, head_dim)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, num_kv, head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, num_kv, head_dim)
        del k_ctx, k_noise, v_ctx, v_noise

        # ── Per-head Q/K norms (Qwen3-style, present in the Gemma checkpoint) ──
        q_norm_w = lsd[f"{prefix}.self_attn.q_norm.weight"]
        k_norm_w = lsd[f"{prefix}.self_attn.k_norm.weight"]
        q = _per_head_rms_norm(q, q_norm_w, eps)
        k = _per_head_rms_norm(k, k_norm_w, eps)
        del q_norm_w, k_norm_w

        # Transpose into [B, heads, S, head_dim] for SDPA + RoPE.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── RoPE ───────────────────────────────────────────────────────────
        q, k = _apply_rope(q, k, cos, sin)

        # ── GQA + SDPA (non-causal) ────────────────────────────────────────
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
        del q, k, v

        # ── o_proj + residual ─────────────────────────────────────────────
        o_proj = lsd[f"{prefix}.self_attn.o_proj.weight"]
        attn_out = _linear(attn, o_proj)
        del o_proj
        hidden = residual.float() + attn_out
        del residual, attn_out

        # ── MLP block ─────────────────────────────────────────────────────
        post_ln_w = lsd[f"{prefix}.post_attention_layernorm.weight"]
        residual2 = hidden
        mlp_in = _rms_norm(hidden, post_ln_w, eps).float()
        del post_ln_w

        gate_w = lsd[f"{prefix}.mlp.gate_proj.weight"]
        gate = _linear(mlp_in, gate_w)
        del gate_w
        up_w = lsd[f"{prefix}.mlp.up_proj.weight"]
        up = _linear(mlp_in, up_w)
        del up_w, mlp_in
        intermediate = F.silu(gate) * up
        del gate, up
        down_w = lsd[f"{prefix}.mlp.down_proj.weight"]
        mlp_out = _linear(intermediate, down_w)
        del down_w, intermediate

        hidden = residual2 + mlp_out
        del residual2, mlp_out

        gc.collect()
        return hidden

    # ─── top-level forward ──────────────────────────────────────────────────

    def __call__(
        self,
        aux_hiddens_concat: torch.Tensor,
        noise: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_intermediates: bool = False,
    ):
        cfg = self.cfg

        # fc + hidden_norm on the concatenated aux hiddens.
        target_hidden_projected = _linear(aux_hiddens_concat.float(), self.fc_w)
        target_hidden_projected = _rms_norm(target_hidden_projected, self.hidden_norm_w, cfg.rms_norm_eps).float()

        intermediates = {}
        if return_intermediates:
            intermediates["target_hidden_projected"] = target_hidden_projected.detach().clone()

        hidden = noise.float()
        for i in range(cfg.num_hidden_layers):
            hidden = self._layer_forward(hidden, target_hidden_projected, cos, sin, i)
            if return_intermediates:
                intermediates[f"layer_{i}"] = hidden.detach().clone()

        del target_hidden_projected

        hidden = _rms_norm(hidden, self.norm_w, cfg.rms_norm_eps).float()
        if return_intermediates:
            intermediates["final_norm"] = hidden.detach().clone()

        # Drop the bonus position before the lm_head.
        draft_logits = _linear(hidden[:, 1:, :], self.lm_head_w)

        if return_intermediates:
            return draft_logits, hidden, intermediates
        return draft_logits, hidden


def build_rope_cache(
    positions: torch.Tensor, head_dim: int, theta: float, dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HF Llama-style RoPE: full rotary (no partial), default scaling.

    Args
    ----
    positions: int tensor ``[B, total_len]`` of absolute position indices.
    head_dim: per-head dim (256 for Gemma DFlash).
    theta: ``rope_theta`` (10000.0 for Gemma DFlash).

    Returns
    -------
    cos, sin: float tensors ``[B, total_len, head_dim]``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def load_reference_from_safetensors(cfg, model_path) -> LazyDFlashReference:
    """Build a streaming reference rooted at the DFlash safetensors directory.

    `model_path` must contain ``model.safetensors`` + ``config.json``.
    The reference holds open file handles to the safetensors shard; close
    by going out of scope or calling ``ref.lsd.close()``.
    """
    from ...utils.lazy_state_dict import LazyStateDict

    lsd = LazyStateDict(Path(model_path))
    return LazyDFlashReference(cfg, lsd)
