# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared attention operations for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (not the llama variant).
No Meta-format weight conversion needed. No transformation matrices needed.

Handles:
- Per-head RMSNorm (q_norm, k_norm, v_norm) via reshape trick
- Partial RoPE for global layers (split, rotate, concat)
- K=V tying (fused Q+K+K weight, standard nlp_create_qkv_heads split)
- No bias on any projection
- scaling=1.0 (no 1/sqrt(d_k))
"""

import ttnn
from models.demos.gemma4_cody.tt.ccl import (
    ccl_allreduce,
    ccl_matmul_reduce_scatter_allgather,
    make_block_sharded_matmul_config,
)

from .weights import AttentionWeights

# Packed-shape QKV/o_proj on L1-block-sharded activations (~1.8x over DRAM
# interleaved at M=512; mcast volume bound, not FLOPs). Per-shape config cache
# so trace replay reuses configs.
_L1_PC_CACHE = {}


def _l1_pc(m, k, n):
    key = (m, k, n)
    if key not in _L1_PC_CACHE:
        _L1_PC_CACHE[key] = make_block_sharded_matmul_config(m, k, n)
    return _L1_PC_CACHE[key]


def apply_qkv_projection(hidden_states, weights: AttentionWeights, memory_config=None):
    """Fused QKV matmul (no bias for Gemma4)."""
    M = hidden_states.shape[-2]
    # in0 shard + matmul CBs share a core's 1.46 MB L1; the CB budget in
    # make_block_sharded_matmul_config shrinks in0_block_w to fit up to 2048.
    if M % 256 == 0 and M <= 2048:
        pc, in_mem = _l1_pc(M, hidden_states.shape[-1], weights.wqkv.shape[-1])
        x_sh = ttnn.to_memory_config(hidden_states, in_mem)
        out = ttnn.linear(x_sh, weights.wqkv, program_config=pc, memory_config=memory_config)
        x_sh.deallocate(True)
        return out
    return ttnn.linear(hidden_states, weights.wqkv, memory_config=memory_config)


def split_qkv_heads_decode(xqkv_fused, config, is_global: bool, tp: int = 1, kv_replicated: bool = False):
    """
    Split fused QKV into separate head tensors for decode mode.
    When TP > 1, uses local head counts (global / tp).
    When kv_replicated (num_kv_heads < TP), each device has 1 KV head (GQA-assigned).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    # nlp_create_qkv_heads_decode miscomputes with DRAM input on Blackhole (#16667);
    # the op's own unit test skips that combination. Force L1 here so every caller is safe.
    if xqkv_fused.memory_config().buffer_type == ttnn.BufferType.DRAM:
        xqkv_fused = ttnn.to_memory_config(xqkv_fused, ttnn.L1_MEMORY_CONFIG)
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )


def split_qkv_heads_prefill(
    xqkv_fused, config, is_global: bool, tp: int = 1, kv_replicated: bool = False, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """
    Split fused QKV into separate head tensors for prefill mode.
    When TP > 1, uses local head counts (global / tp).
    When kv_replicated (num_kv_heads < TP), each device has 1 KV head (GQA-assigned).

    ``memory_config`` defaults to DRAM (true prefill: seq_len can be thousands
    of tokens and would not fit L1). The packed-decode caller overrides it to
    L1 so the split output — and the whole downstream activation stream — stays
    resident on L1; the input ``xqkv`` is already L1, so this is interleaved
    L1 → interleaved L1 (the op only emits sharded output for sharded input).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=memory_config,
    )


def apply_per_head_norm(tensor, weight, eps, with_scale=True, memory_config=None):
    """
    Apply RMSNorm per-head on the head_dim dimension.

    Input: [1, num_heads, S, head_dim]
    Process: reshape to [1, 1, num_heads*S, head_dim] -> rms_norm -> reshape back

    ``memory_config`` is forwarded to ``rms_norm``; pass L1 to keep the
    normed activation resident on L1 (packed-decode path). ``None`` keeps
    the op's default (follows the input's layout).
    """
    orig_shape = tensor.shape
    num_heads = orig_shape[1]
    seq_or_batch = orig_shape[2]
    head_dim = orig_shape[3]

    flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_or_batch, head_dim))
    if with_scale and weight is not None:
        normed = ttnn.rms_norm(flat, weight=weight, epsilon=eps, memory_config=memory_config)
    else:
        normed = ttnn.rms_norm(flat, epsilon=eps, memory_config=memory_config)

    return ttnn.reshape(normed, orig_shape)


def apply_rope(tensor, cos_cache, sin_cache, token_index=None, memory_config=None):
    """
    Apply HF-style rotary position embedding via ttnn.experimental.rotary_embedding.

    Prefill (token_index=None):
        tensor:  [1, heads, S, head_dim]
        cos/sin: [1, 1, max_seq_len, head_dim]
        Single call; the kernel multiplies element-wise along the seq dim.

    Decode — production path (cos pre-gathered, token_index=0):
        tensor:  [1, batch, heads, head_dim]
        cos/sin: [1, 1, batch, head_dim]   (one row per slot)

        The rotary_embedding kernel iterates cos/sin along the "Ht" axis
        (dim -2 of input). It does NOT index across input dim 1, so a
        naive call with batch on dim 1 would apply slot-0's angle to every
        slot. To get per-slot positions in a single op, we transpose
        dim 1 ↔ dim 2 so batch lives on the kernel's iteration axis, then
        run in prefill mode (element-wise tile multiply against the per-row
        cos/sin), then transpose back. 3 ops total (2 transposes + 1
        rotary) regardless of batch size.

    Decode — legacy single-user path (full cache, token_index=position):
        tensor:  [1, heads, 1, head_dim]
        cos/sin: [1, 1, max_seq_len, head_dim]
        Single call; kernel indexes the cache by token_index.
    """
    if token_index is None:
        # Prefill rotary kernel (also used by packed decode). Forward
        # memory_config so the rotated tensor can stay resident on L1.
        return ttnn.experimental.rotary_embedding(tensor, cos_cache, sin_cache, None, memory_config=memory_config)

    # Pre-gathered per-slot cos/sin → cos dim-2 matches tensor dim-1.
    if cos_cache.shape[2] == tensor.shape[1]:
        tensor_t = ttnn.transpose(tensor, 1, 2)
        rotated_t = ttnn.experimental.rotary_embedding(tensor_t, cos_cache, sin_cache)
        ttnn.deallocate(tensor_t)
        result = ttnn.transpose(rotated_t, 1, 2)
        ttnn.deallocate(rotated_t)
        return result

    # Legacy single-user: cos is the full cache, kernel indexes by token_index.
    orig_shape = tensor.shape
    result = ttnn.experimental.rotary_embedding(tensor, cos_cache, sin_cache, token_index)
    if result.shape[2] != orig_shape[2]:
        result = ttnn.reshape(
            result,
            (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]),
            (orig_shape[0], orig_shape[1], 32, orig_shape[3]),
        )
        result = result[:, :, : orig_shape[2]]
    return result


def concat_heads(tensor, is_decode_mode: bool, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Concatenate attention heads back to hidden dimension.

    ``memory_config`` defaults to DRAM; the packed-decode caller passes L1 to
    keep the concatenated activation resident on L1 ahead of ``o_proj``.
    """
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=memory_config)


def apply_output_projection(tensor, weights: AttentionWeights):
    """Apply output projection (no bias for Gemma4)."""
    M = tensor.shape[-2]
    if M % 256 == 0 and M <= 2048:
        pc, in_mem = _l1_pc(M, tensor.shape[-1], weights.o_proj.shape[-1])
        x_sh = ttnn.to_memory_config(tensor, in_mem)
        # DRAM interleaved output: feeds the all_reduce fabric op.
        out = ttnn.linear(x_sh, weights.o_proj, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_sh.deallocate(True)
        tensor.deallocate(True)
        return out
    out = ttnn.linear(tensor, weights.o_proj)
    tensor.deallocate(True)
    return out


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """Apply tensor-parallel allreduce if TP > 1."""
    return ccl_allreduce(tensor, mesh_config, ccl_manager)


def apply_fused_output_projection_and_allreduce(
    tensor,
    weights: AttentionWeights,
    mesh_config,
    ccl_manager,
    persistent_intermediate_buffer,
    persistent_output_buffer,
    program_config=None,
):
    """Row-parallel `o_proj` fused with reduce-scatter, then all-gather.

    When TP > 1 and persistent buffers are provided, replaces the
    `linear(o_proj) → all_reduce` pair with
    `matmul_reduce_scatter_async → all_gather_async`. Falls back to the
    unfused path otherwise (TP=1 or no persistent buffers — e.g. prefill
    with variable seq_len).
    """
    return ccl_matmul_reduce_scatter_allgather(
        tensor,
        weights.o_proj,
        persistent_intermediate_buffer,
        persistent_output_buffer,
        mesh_config,
        ccl_manager,
        program_config=program_config,
    )
