# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exact chunked prefill into Muse Glimmer's paged target KV cache."""

import math

import torch

import ttnn

from .kv_cache import PAGE_BLOCK_SIZE
from .operations import (
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)

PREFILL_Q_CHUNK_SIZE = 256
PREFILL_K_CHUNK_SIZE = 512
_SLIDING_MASKS = {}


def _query_chunk_size(chunk_start, logical_length):
    """Largest supported query tile aligned to both position and length."""
    for candidate in (PREFILL_Q_CHUNK_SIZE, 128, PAGE_BLOCK_SIZE):
        if chunk_start % candidate == 0 and logical_length % candidate == 0:
            return candidate
    raise ValueError("Chunked prefill requires a 64-token aligned query tile")


def _key_chunk_size(chunk_start, logical_length):
    """Use the largest L1-safe key tile compatible with the cache offset."""
    for candidate in (PREFILL_K_CHUNK_SIZE, 256, 128, PAGE_BLOCK_SIZE):
        if candidate <= logical_length and chunk_start % candidate == 0:
            return candidate
    raise ValueError("Chunked prefill requires a 64-token aligned key tile")


def _sliding_mask(device, query_length, tail_length, window):
    """Mask current queries against the retained tail and current K/V."""
    key = (id(device), query_length, tail_length, window)
    cached = _SLIDING_MASKS.get(key)
    if cached is not None:
        return cached
    query_positions = torch.arange(query_length).unsqueeze(1) + tail_length
    key_positions = torch.arange(tail_length + query_length).unsqueeze(0)
    allowed = (key_positions <= query_positions) & (key_positions >= query_positions - window + 1)
    mask = torch.where(allowed, 0.0, -1e9).to(torch.bfloat16).reshape(1, 1, query_length, tail_length + query_length)
    cached = ttnn.from_torch(
        mask,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _SLIDING_MASKS[key] = cached
    return cached


def _paged_fill(cache, values, page_table, chunk_start, logical_length):
    first_page = chunk_start // PAGE_BLOCK_SIZE
    page_count = (logical_length + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE
    chunk_page_table = ttnn.slice(page_table, [0, first_page], [1, first_page + page_count])
    ttnn.experimental.paged_fill_cache(cache, values, chunk_page_table, batch_idx=0)


def prefill_forward(
    hidden_states,
    cos,
    sin,
    weights,
    kv_cache,
    config,
    *,
    page_table,
    chunk_start,
    logical_length,
    previous_tail,
):
    if kv_cache is not None and (logical_length % ttnn.TILE_SIZE or chunk_start % PAGE_BLOCK_SIZE):
        raise ValueError("Chunked prefill starts must be page aligned and lengths must be tile aligned")
    if logical_length != int(hidden_states.shape[2]):
        raise ValueError("Chunked prefill input must not contain logical padding")

    qkv = apply_qkv_projection(hidden_states, weights)
    q, k, v = split_qkv_heads_prefill(qkv, config)
    qkv.deallocate(True)
    q = apply_per_head_norm(q, None, config.rms_norm_eps, with_scale=False)
    k = apply_per_head_norm(k, None, config.rms_norm_eps, with_scale=False)
    if config.has_rope:
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

    if kv_cache is None:
        output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=config.attention_scale,
            sliding_window_size=config.sliding_window if config.is_sliding else None,
        )
        for tensor in (q, k, v):
            ttnn.deallocate(tensor)
        output = concat_heads(output, is_decode_mode=False)
        return apply_output_projection(output, weights, hidden_states), None
    cache_k, cache_v = kv_cache
    if k.dtype != cache_k.dtype:
        old_k, old_v = k, v
        k = ttnn.typecast(k, cache_k.dtype)
        v = ttnn.typecast(v, cache_v.dtype)
        old_k.deallocate(True)
        old_v.deallocate(True)
    _paged_fill(cache_k, k, page_table, chunk_start, logical_length)
    _paged_fill(cache_v, v, page_table, chunk_start, logical_length)

    next_tail = None
    if config.is_sliding:
        tail_length = 0 if previous_tail is None else int(previous_tail[0].shape[2])
        if previous_tail is None:
            # Equal Q/K lengths let the kernel apply causal and window bounds
            # without reading an O(S^2) additive mask.
            output = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=config.attention_scale,
                sliding_window_size=config.sliding_window,
            )
        elif logical_length > config.sliding_window:
            # Only the first window of a new chunk can see the retained tail.
            # Run that prefix against tail+prefix with an explicit rectangular
            # mask, then use the native causal-window kernel for the rest. This
            # avoids streaming a chunk x (tail+chunk) dense mask on every SWA
            # layer while preserving exact attention at the chunk boundary.
            prefix_length = config.sliding_window
            prefix_q = ttnn.slice(q, [0, 0, 0, 0], [1, config.num_attention_heads, prefix_length, config.head_dim])
            prefix_k = ttnn.slice(k, [0, 0, 0, 0], [1, config.num_key_value_heads, prefix_length, config.head_dim])
            prefix_v = ttnn.slice(v, [0, 0, 0, 0], [1, config.num_key_value_heads, prefix_length, config.head_dim])
            context_k = ttnn.concat([previous_tail[0], prefix_k], dim=2)
            context_v = ttnn.concat([previous_tail[1], prefix_v], dim=2)
            mask = _sliding_mask(config.device, prefix_length, tail_length, config.sliding_window)
            prefix_output = ttnn.transformer.scaled_dot_product_attention(
                prefix_q,
                context_k,
                context_v,
                attn_mask=mask,
                is_causal=False,
                scale=config.attention_scale,
            )
            current_output = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=config.attention_scale,
                sliding_window_size=config.sliding_window,
            )
            suffix_output = ttnn.slice(
                current_output,
                [0, 0, prefix_length, 0],
                [1, config.num_attention_heads, logical_length, config.head_dim],
            )
            output = ttnn.concat([prefix_output, suffix_output], dim=2)
            for tensor in (
                prefix_q,
                prefix_k,
                prefix_v,
                context_k,
                context_v,
                prefix_output,
                suffix_output,
                current_output,
            ):
                ttnn.deallocate(tensor)
        else:
            context_k = ttnn.concat([previous_tail[0], k], dim=2)
            context_v = ttnn.concat([previous_tail[1], v], dim=2)
            mask = _sliding_mask(config.device, logical_length, tail_length, config.sliding_window)
            output = ttnn.transformer.scaled_dot_product_attention(
                q,
                context_k,
                context_v,
                attn_mask=mask,
                is_causal=False,
                scale=config.attention_scale,
            )

        keep = min(config.sliding_window, tail_length + logical_length)
        if logical_length >= keep:
            tail_source_k, tail_source_v = k, v
            begin = logical_length - keep
        else:
            tail_source_k, tail_source_v = context_k, context_v
            begin = tail_length + logical_length - keep
        # A full-range TT slice may alias its input.  The tail outlives k/v and
        # must therefore own its buffers across subsequent prefill/decode calls.
        tail_k = ttnn.slice(
            tail_source_k, [0, 0, begin, 0], [1, config.num_key_value_heads, begin + keep, config.head_dim]
        )
        tail_v = ttnn.slice(
            tail_source_v, [0, 0, begin, 0], [1, config.num_key_value_heads, begin + keep, config.head_dim]
        )
        next_tail = (
            ttnn.clone(tail_k, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.clone(tail_v, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )
        for tensor in previous_tail or ():
            ttnn.deallocate(tensor)
        if previous_tail is not None:
            if logical_length <= config.sliding_window:
                ttnn.deallocate(context_k)
                ttnn.deallocate(context_v)
    elif logical_length < PAGE_BLOCK_SIZE:
        output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=config.attention_scale,
        )
    else:
        scaled_q = ttnn.mul(q, config.attention_scale * math.sqrt(config.head_dim))
        ttnn.deallocate(q)
        q = scaled_q
        q_chunk = _query_chunk_size(chunk_start, logical_length)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=config.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=_key_chunk_size(chunk_start, logical_length),
            exp_approx_mode=False,
        )
        output = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            cache_k,
            cache_v,
            page_table,
            chunk_start_idx=chunk_start,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
        )

    for tensor in (q, k, v):
        ttnn.deallocate(tensor)
    output = concat_heads(output, is_decode_mode=False)
    return apply_output_projection(output, weights, hidden_states), next_tail
