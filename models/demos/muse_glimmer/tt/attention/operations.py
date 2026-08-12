# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core single-device Muse Glimmer attention operations."""

import ttnn


def apply_qkv_projection(
    hidden_states, weights, memory_config=None, decode_core_config=None, has_previous_weight=False
):
    if decode_core_config is not None and weights.wqkv_prefetch is not None:
        return decode_core_config.qkv(hidden_states, weights.wqkv_prefetch, has_previous_weight=has_previous_weight)
    return ttnn.linear(
        hidden_states,
        weights.wqkv,
        memory_config=memory_config,
        core_grid=decode_core_config.target_compute_grid if decode_core_config is not None else None,
    )


def split_qkv_heads_decode(xqkv, config, decode_core_config=None):
    if xqkv.memory_config().buffer_type == ttnn.BufferType.DRAM:
        xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        memory_config=(
            decode_core_config.head_output_memory_config
            if decode_core_config is not None
            else ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        ),
    )


def split_qkv_heads_prefill(xqkv, config, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        transpose_k_heads=False,
        memory_config=memory_config,
    )


def apply_per_head_norm(tensor, weight, eps, with_scale=True, memory_config=None, decode_core_config=None):
    shape = tensor.shape
    flat = ttnn.reshape(
        tensor,
        (1, 1, shape[1] * shape[2], shape[3]),
        sub_core_grids=decode_core_config.target_compute_cores if decode_core_config is not None else None,
    )
    if decode_core_config is not None:
        norm_memory_config, program_config = decode_core_config.head_norm_config(
            int(flat.shape[-2]), int(flat.shape[-1])
        )
        sharded = ttnn.to_memory_config(flat, norm_memory_config)
        normed_sharded = ttnn.rms_norm(
            sharded,
            weight=weight if with_scale else None,
            epsilon=eps,
            memory_config=norm_memory_config,
            program_config=program_config,
        )
        ttnn.deallocate(sharded)
        normed = ttnn.to_memory_config(normed_sharded, memory_config or ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(normed_sharded)
        return ttnn.reshape(normed, shape, sub_core_grids=decode_core_config.target_compute_cores)
    normed = ttnn.rms_norm(
        flat,
        weight=weight if with_scale else None,
        epsilon=eps,
        memory_config=memory_config,
    )
    return ttnn.reshape(normed, shape)


def apply_rope(tensor, cos, sin, memory_config=None, decode_core_config=None):
    if decode_core_config is not None:
        width = int(tensor.shape[-1])
        half = width // 2
        starts = [0] * len(tensor.shape)
        first_end = list(int(dim) for dim in tensor.shape)
        first_end[-1] = half
        second_start = starts.copy()
        second_start[-1] = half
        second_end = list(int(dim) for dim in tensor.shape)
        first = ttnn.slice(tensor, starts, first_end, sub_core_grids=decode_core_config.target_compute_cores)
        second = ttnn.slice(tensor, second_start, second_end, sub_core_grids=decode_core_config.target_compute_cores)
        negative_second = ttnn.neg(second, sub_core_grids=decode_core_config.target_compute_cores)
        ttnn.deallocate(second)
        rotated = ttnn.concat(
            [negative_second, first],
            dim=-1,
            memory_config=memory_config,
            sub_core_grids=decode_core_config.target_compute_cores,
        )
        ttnn.deallocate(negative_second)
        ttnn.deallocate(first)
        direct = ttnn.multiply(tensor, cos, sub_core_grids=decode_core_config.target_compute_cores)
        crossed = ttnn.multiply(rotated, sin, sub_core_grids=decode_core_config.target_compute_cores)
        ttnn.deallocate(rotated)
        output = ttnn.add(
            direct,
            crossed,
            memory_config=memory_config,
            sub_core_grids=decode_core_config.target_compute_cores,
        )
        ttnn.deallocate(direct)
        ttnn.deallocate(crossed)
        return output
    return ttnn.experimental.rotary_embedding(tensor, cos, sin, None, memory_config=memory_config)


def concat_heads(tensor, is_decode_mode, memory_config=ttnn.DRAM_MEMORY_CONFIG, decode_core_config=None):
    if decode_core_config is not None:
        output = ttnn.experimental.nlp_concat_heads_decode(
            tensor,
            num_heads=int(tensor.shape[2]),
            sub_core_grids=decode_core_config.target_compute_cores,
        )
        if output.memory_config() != memory_config:
            converted = ttnn.to_memory_config(output, memory_config)
            ttnn.deallocate(output)
            output = converted
        return output
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=memory_config)


def apply_output_projection(tensor, weights, normalized_hidden_states, decode_core_config=None):
    core_grid = decode_core_config.target_compute_grid if decode_core_config is not None else None
    sub_core_grids = decode_core_config.target_compute_cores if decode_core_config is not None else None
    gate = ttnn.linear(normalized_hidden_states, weights.gate_proj, core_grid=core_grid)
    gated = ttnn.multiply(
        tensor,
        gate,
        input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        sub_core_grids=sub_core_grids,
    )
    tensor.deallocate(True)
    gate.deallocate(True)
    output = ttnn.linear(gated, weights.o_proj, core_grid=core_grid)
    gated.deallocate(True)
    return output
