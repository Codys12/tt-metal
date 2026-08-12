# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Worker-core and prefetched-matmul policy for Muse Glimmer AR decode."""

from __future__ import annotations

import math
import os

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class MuseDecodeCoreConfig:
    """Keep target decode programs off the prefetch sender cores.

    The sender cores occupy columns 9 and 10 on their DRAM-bank-affine rows.
    Normal decode programs use the contiguous 9x10 rectangle at the origin, while
    prefetched matmuls consume their weights on the prefetcher's receiver ring.
    """

    def __init__(self, prefetcher):
        self.prefetcher = prefetcher
        self.worker_sub_device_id = prefetcher.worker_sub_device_id
        self.receiver_sub_device_id = prefetcher.receiver_sub_device_id
        self.compute_grid = ttnn.CoreGrid(x=8, y=8)
        self.compute_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
        self.target_compute_grid = ttnn.CoreGrid(x=9, y=10)
        self.target_compute_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8, 9))])
        # Keep reductions off the GCB receiver rows.  Argmax is not a
        # streaming consumer, and its multicore implementation reuses CB
        # indices that are live on the receiver ring while the prefetch
        # sub-device is active.
        self.argmax_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5))])
        self.receiver_cores = prefetcher.to_core_range_set(
            prefetcher.core_config.receiver_cores(sender_active=True, receiver_active=True)
        )
        ring_size = prefetcher.ring_size
        self.ring_size = ring_size
        self._input_memory_configs = {}
        self._projection_configs = {}
        self.next_projection_weight = None
        self.qkv_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )
        self.head_output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.target_compute_cores,
                [ttnn.TILE_SIZE, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        sdpa_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(0, 0),
            ttnn.TILE_SIZE,
            self.target_compute_cores,
            row_wise=True,
        )
        self.sdpa_output_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, 128),
            core_grid=sdpa_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    @staticmethod
    def isolated_sender_mapping(receivers_per_bank: int = 3):
        if receivers_per_bank not in (2, 3):
            raise ValueError("Muse's isolated decode topology uses two or three receivers per DRAM bank")
        bank_ordered_rows = (9, 1, 7, 3, 0, 2, 6, 4)
        if receivers_per_bank == 2:
            return {(10, row): [(8, row), (9, row)] for row in bank_ordered_rows}
        overflow_receivers = [(x, 8) for x in range(8)]
        return {
            (10, row): [(8, row), (9, row), overflow_receivers[index]] for index, row in enumerate(bank_ordered_rows)
        }

    @staticmethod
    def full_bandwidth_sender_mapping():
        """Use two readers per bank, feeding five and four receivers."""
        bank_ordered_rows = (9, 1, 7, 3, 0, 2, 6, 4)
        mapping = {}
        for row in bank_ordered_rows:
            mapping[(9, row)] = [(x, row) for x in range(5)]
            mapping[(10, row)] = [(x, row) for x in range(5, 9)]
        return mapping

    def copy(self, tensor, *, memory_config=None, output_tensor=None):
        if tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
            if output_tensor is None:
                return ttnn.clone(tensor, memory_config=memory_config)
            return ttnn.assign(tensor, output_tensor)
        return ttnn.typecast(
            tensor,
            tensor.dtype,
            memory_config=memory_config,
            output_tensor=output_tensor,
            sub_core_grids=self.compute_cores,
        )

    def target_copy(self, tensor, *, memory_config=None, output_tensor=None):
        if tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
            if output_tensor is None:
                return ttnn.clone(tensor, memory_config=memory_config)
            return ttnn.assign(tensor, output_tensor)
        return ttnn.typecast(
            tensor,
            tensor.dtype,
            memory_config=memory_config,
            output_tensor=output_tensor,
            sub_core_grids=self.target_compute_cores,
        )

    def head_norm_config(self, rows: int, width: int):
        width_tiles = width // ttnn.TILE_SIZE
        grid = ttnn.CoreGrid(x=width_tiles, y=1)
        memory_config = ttnn.create_sharded_memory_config(
            shape=(rows, width),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(width_tiles, 1),
            subblock_w=1,
            block_h=math.ceil(rows / ttnn.TILE_SIZE),
            block_w=1,
            inplace=False,
        )
        return memory_config, program_config

    def qkv(self, hidden_states, weights, *, has_previous_weight):
        # The global-CB matmul now waits and releases bounded K-block chunks.
        # It is the synchronization point; intervening AR ops may run while
        # the producer fills the next layer's QKV pages.
        return self.streamed_projection(hidden_states, weights)

    def _input_memory_config(self, width):
        width = int(width)
        if width not in self._input_memory_configs:
            width_per_core = math.ceil(math.ceil(width / self.ring_size) / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            self._input_memory_configs[width] = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, width_per_core),
                core_grid=self.receiver_cores,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        return self._input_memory_configs[width]

    def _projection_config(self, k, width):
        """Build the sharding and matmul descriptor for one streamed slice."""
        k = int(k)
        width = int(width)
        key = (k, width)
        if key in self._projection_configs:
            return self._projection_configs[key]
        if width % (self.ring_size * ttnn.TILE_SIZE):
            raise ValueError(
                f"Streamed projection width {width} must be divisible by "
                f"ring_size * tile_size ({self.ring_size * ttnn.TILE_SIZE})"
            )
        per_core_n = width // self.ring_size // ttnn.TILE_SIZE
        in0_block_w = k // self.ring_size // ttnn.TILE_SIZE
        while in0_block_w > 0 and (k // ttnn.TILE_SIZE) % in0_block_w:
            in0_block_w -= 1
        in0_block_w = max(1, in0_block_w)
        out_subblock_w = min(8, per_core_n)
        while per_core_n % out_subblock_w:
            out_subblock_w -= 1
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width // self.ring_size),
            core_grid=self.receiver_cores,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(
                len(self.prefetcher.dram_banks()),
                self.ring_size // len(self.prefetcher.dram_banks()),
            ),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=ttnn.CoreRangeSet(set()),
            num_global_cb_receivers=self.prefetcher.num_receiver_cores,
            untilize_out=False,
        )
        result = (output_memory_config, program_config, per_core_n * in0_block_w, (k // ttnn.TILE_SIZE) // in0_block_w)
        self._projection_configs[key] = result
        return result

    def streamed_projection(self, hidden_states, weights, *, fence_before=False, next_weight=None):
        """Run ring-aligned streamed projection slices and concatenate them."""
        # Receiver and ordinary AR cores share one consumer sub-device. Their
        # program order therefore carries activation dependencies while the
        # DRAM senders remain independently dispatchable.
        if fence_before:
            self.prefetch_fence(hidden_states, weights[0])
            self._sync_qkv("projection_fence")
        k = int(weights[0].shape[-2])
        input_states = hidden_states
        if int(hidden_states.shape[-1]) < k:
            input_states = ttnn.pad(
                hidden_states,
                [(0, 0), (0, 0), (0, 0), (0, k - int(hidden_states.shape[-1]))],
                value=0.0,
                use_multicore=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        elif int(hidden_states.shape[-1]) != k:
            raise ValueError(f"Activation width {hidden_states.shape[-1]} exceeds streamed weight K {k}")
        sharded_input = ttnn.interleaved_to_sharded(input_states, self._input_memory_config(k))
        if input_states is not hidden_states:
            ttnn.deallocate(input_states)
        self._sync_qkv("input_shard")
        outputs = []
        for part, weight in enumerate(weights):
            output_memory_config, program_config, _, _ = self._projection_config(weight.shape[-2], weight.shape[-1])
            sharded_output = ttnn.linear(
                sharded_input,
                weight,
                program_config=program_config,
                memory_config=output_memory_config,
                compute_kernel_config=self.qkv_compute_config,
                dtype=ttnn.bfloat16,
                global_cb=self.prefetcher.global_cb,
                sub_device_id=self.receiver_sub_device_id,
            )
            self._sync_qkv(f"matmul_{part}")
            output = ttnn.sharded_to_interleaved(sharded_output, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(sharded_output)
            outputs.append(output)
        ttnn.deallocate(sharded_input)
        if len(outputs) == 1:
            return outputs[0]
        combined = ttnn.concat(outputs, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG, sub_core_grids=self.compute_cores)
        for output in outputs:
            ttnn.deallocate(output)
        return combined

    def trim_projection(self, output, width):
        """Drop ring-alignment padding from a streamed projection."""
        width = int(width)
        if int(output.shape[-1]) == width:
            return output
        trimmed = ttnn.slice(
            output,
            [0, 0, 0, 0],
            [int(output.shape[0]), int(output.shape[1]), int(output.shape[2]), width],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            sub_core_grids=self.target_compute_cores,
        )
        output.deallocate(True)
        return trimmed

    def _sync_qkv(self, stage):
        if os.getenv("MUSE_QKV_SYNC_DEBUG") == "1":
            ttnn.synchronize_device(
                self.prefetcher.mesh_device,
                sub_device_ids=[self.worker_sub_device_id],
            )
            print(f"MUSE_QKV_CORE_SYNC {stage}", flush=True)

    def prefetch_fence(self, dependency, weight):
        """Wait until the next streamed weight is fully resident in the GCB."""
        remote_cb_index = 31
        weight_dtype = weight.dtype
        tile_bytes = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[weight_dtype]
        _, _, weight_block_tiles, weight_pages = self._projection_config(weight.shape[-2], weight.shape[-1])
        weight_page_bytes = weight_block_tiles * tile_bytes
        cb = ttnn.CBDescriptor(
            total_size=self.prefetcher.global_cb.size(),
            core_ranges=self.receiver_cores,
            format_descriptors=[],
        )
        cb.remote_format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=remote_cb_index,
                data_format=weight_dtype,
                page_size=16,
            )
        ]
        cb.set_global_circular_buffer(self.prefetcher.global_cb)
        kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/muse_glimmer/tt/attention/kernels/prefetch_fence.cpp",
            core_ranges=self.receiver_cores,
            ncrisc_named_compile_time_args=[
                ("cb_remote", remote_cb_index),
                ("num_aligned_pages", weight_page_bytes * weight_pages // 16),
            ],
        )
        program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=[cb])
        return ttnn.generic_op([dependency, dependency], program)
