# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged batch-1 packed verification attention for Muse Glimmer."""

import os

import ttnn

from .operations import (
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from .packed_cache import packed_kv_update
from .weights import AttentionWeights

PACKED_SDPA_SHORT_MAX_CORES = 16
PACKED_SDPA_LONG_MAX_CORES = 64
PACKED_SDPA_LONG_CONTEXT = 2048


def packed_decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_device,
    position_idx,
    cur_pos,
    page_table,
    page_index,
    page_offset,
    packed_p,
    real_p,
    rope_packed=None,
    retain_tail=True,
    decode_core_config=None,
):
    """Verify a physical packed block while every query shares one paged cache."""

    def sync_stage(stage):
        if decode_core_config is not None and os.getenv("MUSE_QKV_SYNC_DEBUG") == "1":
            ttnn.synchronize_device(
                mesh_device,
                sub_device_ids=[decode_core_config.worker_sub_device_id],
            )
            print(f"MUSE_QKV_ATTN_SYNC {stage}", flush=True)

    if kv_cache is None:
        raise ValueError("Packed verification requires a KV cache")
    if int(hidden_states.shape[2]) != packed_p:
        raise ValueError("Packed verification metadata does not match its input")

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    l1 = ttnn.L1_MEMORY_CONFIG

    xqkv = apply_qkv_projection(
        hidden_states,
        weights,
        memory_config=l1,
        decode_core_config=decode_core_config,
        has_previous_weight=config.layer_idx > 0,
    )
    sync_stage("qkv")

    if decode_core_config is not None:
        q, k, v = split_qkv_heads_decode(xqkv, config, decode_core_config)
        q_sharded, k_sharded, v_sharded = q, k, v
        q = ttnn.sharded_to_interleaved(q_sharded, l1)
        k = ttnn.sharded_to_interleaved(k_sharded, l1)
        v = ttnn.sharded_to_interleaved(v_sharded, l1)
        ttnn.deallocate(q_sharded)
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)
    else:
        q, k, v = split_qkv_heads_prefill(xqkv, config, memory_config=l1)
    ttnn.deallocate(xqkv)
    sync_stage("split_heads")
    q = apply_per_head_norm(
        q,
        None,
        config.rms_norm_eps,
        with_scale=False,
        memory_config=l1,
        decode_core_config=decode_core_config,
    )
    k = apply_per_head_norm(
        k,
        None,
        config.rms_norm_eps,
        with_scale=False,
        memory_config=l1,
        decode_core_config=decode_core_config,
    )
    sync_stage("head_norms")

    if rope_packed is None:
        cos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        owns_rope = True
    else:
        cos, sin = rope_packed
        owns_rope = False
    owns_transposed_rope = False
    q = apply_rope(q, cos, sin, memory_config=l1, decode_core_config=decode_core_config)
    k = apply_rope(k, cos, sin, memory_config=l1, decode_core_config=decode_core_config)
    sync_stage("rope")
    if owns_rope or owns_transposed_rope:
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

    k_cache, v_cache = kv_cache
    cache_slice_end = (
        [1, real_p, num_kv_heads, head_dim] if decode_core_config is not None else [1, num_kv_heads, real_p, head_dim]
    )
    write_k = ttnn.slice(
        k,
        [0, 0, 0, 0],
        cache_slice_end,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_core_grids=decode_core_config.target_compute_cores if decode_core_config is not None else None,
    )
    write_v = ttnn.slice(
        v,
        [0, 0, 0, 0],
        cache_slice_end,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_core_grids=decode_core_config.target_compute_cores if decode_core_config is not None else None,
    )
    if write_k.dtype != k_cache.dtype:
        old_write_k, old_write_v = write_k, write_v
        write_k = ttnn.typecast(write_k, k_cache.dtype)
        write_v = ttnn.typecast(write_v, v_cache.dtype)
        ttnn.deallocate(old_write_k)
        ttnn.deallocate(old_write_v)
    packed_kv_update(
        k_cache,
        v_cache,
        write_k,
        write_v,
        position_idx,
        count=real_p,
    )
    sync_stage("cache_update")
    pending_tail = (write_k, write_v) if config.is_sliding and retain_tail else None
    if pending_tail is None:
        ttnn.deallocate(write_k)
        ttnn.deallocate(write_v)

    # Treat packed positions as decode users which all map to the same physical
    # pages. Their individual cur_pos values supply exact causal/sliding masks,
    # eliminating the O(P * heads * max_seq_len) additive mask.
    if decode_core_config is not None:
        q_sharded = ttnn.interleaved_to_sharded(q, decode_core_config.sdpa_output_memory_config)
        ttnn.deallocate(q)
        q = q_sharded
    else:
        q_heads = q
        q = ttnn.permute(q_heads, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_heads)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(
            (decode_core_config.target_compute_grid.x, decode_core_config.target_compute_grid.y)
            if decode_core_config is not None
            else mesh_device.compute_with_storage_grid_size()
        ),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=(
            PACKED_SDPA_LONG_MAX_CORES if page_index * 64 >= PACKED_SDPA_LONG_CONTEXT else PACKED_SDPA_SHORT_MAX_CORES
        ),
        sub_core_grids=decode_core_config.target_compute_cores if decode_core_config is not None else None,
    )
    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        cur_pos_tensor=cur_pos,
        scale=config.attention_scale,
        sliding_window_size=config.sliding_window if config.is_sliding else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
    )
    sync_stage("sdpa")
    ttnn.deallocate(q)
    if decode_core_config is not None:
        output_sharded = ttnn.interleaved_to_sharded(output, decode_core_config.sdpa_output_memory_config)
        ttnn.deallocate(output)
        output = output_sharded
    output = concat_heads(
        output,
        is_decode_mode=True,
        memory_config=l1,
        decode_core_config=decode_core_config,
    )
    sync_stage("concat_heads")
    return (
        apply_output_projection(
            output,
            weights,
            hidden_states,
            decode_core_config=decode_core_config,
        ),
        pending_tail,
    )
