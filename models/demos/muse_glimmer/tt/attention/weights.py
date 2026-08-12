# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer attention weight loading."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder, get_cache_file_name


@dataclass(frozen=True)
class AttentionWeights:
    wqkv: ttnn.Tensor
    gate_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    wqkv_prefetch: tuple[ttnn.Tensor, ...] | None = None


def load_attention_weights(
    device,
    state_dict,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
    load_qkv_prefetch=False,
):
    qkv_cache = get_cache_file_name(tensor_cache_path, "wqkv")
    gate_cache = get_cache_file_name(tensor_cache_path, "gate_proj")
    output_cache = get_cache_file_name(tensor_cache_path, "o_proj")
    qkv = cached_tensor_placeholder(qkv_cache, weight_dtype, ttnn.TILE_LAYOUT)
    qkv_source = None
    gate = cached_tensor_placeholder(gate_cache, weight_dtype, ttnn.TILE_LAYOUT)
    output = cached_tensor_placeholder(output_cache, weight_dtype, ttnn.TILE_LAYOUT)
    if qkv is None:
        qkv_source = (
            torch.cat(
                [
                    state_dict["q_proj.weight"].transpose(-2, -1),
                    state_dict["k_proj.weight"].transpose(-2, -1),
                    state_dict["v_proj.weight"].transpose(-2, -1),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        qkv = qkv_source
    if gate is None:
        gate = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
    if output is None:
        output = state_dict["o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)

    def as_weight(tensor, cache_name):
        return ttnn.as_tensor(
            tensor,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    qkv_prefetch = None
    if load_qkv_prefetch:
        if qkv_source is None:
            # A cache-hit placeholder intentionally carries no shape. Muse
            # Glimmer has one target projection shape across all layers.
            k, n = 6656, 4608
        else:
            k, n = int(qkv_source.shape[-2]), int(qkv_source.shape[-1])
        dram_banks = device.dram_grid_size().x
        if n % dram_banks:
            raise ValueError(f"QKV width {n} must divide over {dram_banks} DRAM banks")
        dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_banks - 1, 0))])
        num_parts = 1
        part_width = n // num_parts
        padded_k = ((k + 72 * ttnn.TILE_SIZE - 1) // (72 * ttnn.TILE_SIZE)) * (72 * ttnn.TILE_SIZE)
        prefetch_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_grid,
                (padded_k, part_width // dram_banks),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        parts = []
        for part in range(num_parts):
            prefetch_cache = get_cache_file_name(
                tensor_cache_path, f"wqkv_prefetch_k{padded_k}_1way_part{part}_dram_sharded"
            )
            prefetch_source = cached_tensor_placeholder(prefetch_cache, weight_dtype, ttnn.TILE_LAYOUT)
            if prefetch_source is None:
                if qkv_source is None:
                    qkv_source = (
                        torch.cat(
                            [
                                state_dict["q_proj.weight"].transpose(-2, -1),
                                state_dict["k_proj.weight"].transpose(-2, -1),
                                state_dict["v_proj.weight"].transpose(-2, -1),
                            ],
                            dim=-1,
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                start = part * part_width
                prefetch_source = qkv_source[..., start : start + part_width]
                if padded_k != k:
                    prefetch_source = torch.nn.functional.pad(prefetch_source, (0, 0, 0, padded_k - k))
            parts.append(
                ttnn.as_tensor(
                    prefetch_source,
                    device=device,
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=prefetch_cache,
                    memory_config=prefetch_memory_config,
                )
            )
        qkv_prefetch = tuple(parts)

    return AttentionWeights(
        wqkv=as_weight(qkv, qkv_cache),
        gate_proj=as_weight(gate, gate_cache),
        o_proj=as_weight(output, output_cache),
        wqkv_prefetch=qkv_prefetch,
    )
