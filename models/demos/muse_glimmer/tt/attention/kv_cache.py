# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged KV cache for the batch-1 Muse Glimmer target."""

import ttnn

PAGE_BLOCK_SIZE = 64


def init_kv_cache(device, config, max_seq_len, cache_dtype=ttnn.bfloat16, block_size=PAGE_BLOCK_SIZE):
    if device.get_num_devices() != 1:
        raise ValueError("Muse Glimmer supports one device only")
    if max_seq_len % block_size:
        raise ValueError(f"max_seq_len must be divisible by the KV page size ({block_size})")
    shape = [max_seq_len // block_size, config.num_key_value_heads, block_size, config.head_dim]

    def allocate():
        tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            cache_dtype,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.fill(tensor, 0.0, output_tensor=tensor)
        return tensor

    return [allocate(), allocate()]
