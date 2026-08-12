# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer RMSNorm weight loading and execution."""

import ttnn
from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder


class RMSNorm:
    def __init__(self, device, hidden_size, state_dict, eps, tensor_cache_path=None, centered=False):
        self.eps = eps
        self._decode_sharded = None
        self._decode_compute_kernel = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        cache_name = f"{tensor_cache_path}.weight" if tensor_cache_path else None
        weight = cached_tensor_placeholder(cache_name, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        if weight is None:
            weight = state_dict["weight"]
            if centered:
                weight = weight + 1.0
            weight = weight.reshape(1, 1, -1, ttnn.TILE_SIZE).contiguous()
        self.weight = ttnn.as_tensor(
            weight,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_tiles = hidden_size // ttnn.TILE_SIZE
        # Muse's 6,656-wide state divides cleanly over 16 workers. Restrict
        # this to one-tile AR activations; prefill keeps the original path.
        if hidden_size % ttnn.TILE_SIZE == 0 and hidden_tiles % 16 == 0:
            grid = (8, 2)
            memory_config = ttnn.create_sharded_memory_config(
                shape=(1, 1, ttnn.TILE_SIZE, hidden_size),
                core_grid=ttnn.CoreGrid(y=grid[1], x=grid[0]),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            self._decode_sharded = (
                memory_config,
                ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=grid,
                    subblock_w=1,
                    block_h=1,
                    block_w=hidden_tiles // 16,
                    inplace=False,
                ),
            )

    def forward(self, hidden_states, *, decode_sharded=False):
        if decode_sharded and self._decode_sharded is not None and int(hidden_states.shape[2]) == ttnn.TILE_SIZE:
            memory_config, program_config = self._decode_sharded
            original_memory_config = hidden_states.memory_config()
            sharded = ttnn.to_memory_config(hidden_states, memory_config)
            normalized = ttnn.rms_norm(
                sharded,
                weight=self.weight,
                epsilon=self.eps,
                memory_config=memory_config,
                program_config=program_config,
                compute_kernel_config=self._decode_compute_kernel,
            )
            ttnn.deallocate(sharded)
            interleaved = ttnn.to_memory_config(normalized, original_memory_config)
            ttnn.deallocate(normalized)
            return interleaved
        return ttnn.rms_norm(hidden_states, weight=self.weight, epsilon=self.eps)
