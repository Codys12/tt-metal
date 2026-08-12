# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer's dense SwiGLU feed-forward block."""

import math
import os

import torch

import ttnn
from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder, get_cache_file_name


class SharedMLP:
    def __init__(self, device, state_dict, dtype=ttnn.bfloat4_b, tensor_cache_path=None, load_prefetch=False):
        def load(name):
            cache_name = get_cache_file_name(tensor_cache_path, f"{name}.weight")
            tensor = cached_tensor_placeholder(cache_name, dtype, ttnn.TILE_LAYOUT)
            if tensor is None:
                tensor = state_dict[f"{name}.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                tensor,
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.gate_proj = load("gate_proj")
        self.up_proj = load("up_proj")
        self.down_proj = load("down_proj")
        self.gate_proj_prefetch = None
        self.up_proj_prefetch = None
        self.prefetch_output_width = int(self.gate_proj.shape[-1])
        if load_prefetch:
            dram_banks = device.dram_grid_size().x
            dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_banks - 1, 0))])

            def load_prefetch_parts(name, num_parts, k, n):
                receiver_ring = 8 * int(os.getenv("MUSE_PREFETCH_RECEIVERS_PER_BANK", "3"))
                ring_tile_width = receiver_ring * ttnn.TILE_SIZE
                padded_k = math.ceil(k / ring_tile_width) * ring_tile_width
                padded_n = math.ceil(n / ring_tile_width) * ring_tile_width
                # Ring-aligned parts bound receiver L1 and give every streamed
                # tensor a fixed-page geometry.
                width_groups = padded_n // ring_tile_width
                groups_per_part = [
                    width_groups // num_parts + (part < width_groups % num_parts) for part in range(num_parts)
                ]
                part_widths = [groups * ring_tile_width for groups in groups_per_part]
                parts = []
                start = 0
                source = None
                for part, part_width in enumerate(part_widths):
                    memory_config = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(
                            dram_grid,
                            (padded_k, part_width // dram_banks),
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    )
                    cache_name = get_cache_file_name(
                        tensor_cache_path,
                        f"{name}_prefetch_k{padded_k}_n{padded_n}_{num_parts}way_{part_width}_part{part}_dram_sharded",
                    )
                    cached = cached_tensor_placeholder(cache_name, dtype, ttnn.TILE_LAYOUT)
                    if cached is None and source is None:
                        source = state_dict[f"{name}.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
                        if padded_k != k or padded_n != n:
                            source = torch.nn.functional.pad(source, (0, padded_n - n, 0, padded_k - k))
                    parts.append(
                        ttnn.as_tensor(
                            source[..., start : start + part_width] if cached is None else cached,
                            device=device,
                            dtype=dtype,
                            layout=ttnn.TILE_LAYOUT,
                            cache_file_name=cache_name,
                            memory_config=memory_config,
                        )
                    )
                    start += part_width
                return tuple(parts)

            self.gate_proj_prefetch = load_prefetch_parts(
                "gate_proj", 3, int(self.gate_proj.shape[-2]), int(self.gate_proj.shape[-1])
            )
            self.up_proj_prefetch = load_prefetch_parts(
                "up_proj", 3, int(self.up_proj.shape[-2]), int(self.up_proj.shape[-1])
            )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # Packed verification always occupies one 32-token tile.  The
        # automatic 1-D matmul config uses a two-tile K block for the down
        # projection, which leaves the Blackhole DRAM readers under-filled.
        # Four K tiles nearly halves this projection's device time.
        self.decode_down_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def __call__(self, hidden_states, decode_core_config=None):
        core_grid = decode_core_config.target_compute_grid if decode_core_config is not None else None
        sub_core_grids = decode_core_config.target_compute_cores if decode_core_config is not None else None
        if decode_core_config is not None and self.gate_proj_prefetch is not None and self.up_proj_prefetch is not None:
            gate = decode_core_config.streamed_projection(
                hidden_states,
                self.gate_proj_prefetch,
                next_weight=self.up_proj_prefetch[0],
            )
            up = decode_core_config.streamed_projection(
                hidden_states,
                self.up_proj_prefetch,
            )
            gate = decode_core_config.trim_projection(gate, self.prefetch_output_width)
            up = decode_core_config.trim_projection(up, self.prefetch_output_width)
        else:
            gate = ttnn.linear(
                hidden_states,
                self.gate_proj,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=core_grid,
            )
            up = ttnn.linear(
                hidden_states,
                self.up_proj,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=core_grid,
            )
        intermediate = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            sub_core_grids=sub_core_grids,
        )
        gate.deallocate(True)
        up.deallocate(True)
        down_program_config = (
            self.decode_down_program_config
            if decode_core_config is None and int(intermediate.shape[2]) == ttnn.TILE_SIZE
            else None
        )
        output = ttnn.linear(
            intermediate,
            self.down_proj,
            program_config=down_program_config,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=core_grid if down_program_config is None else None,
        )
        intermediate.deallocate(True)
        return output
