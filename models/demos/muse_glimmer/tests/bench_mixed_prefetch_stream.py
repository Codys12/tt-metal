# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replay the mixed Muse QKV/MLP remote-CB stream without loading the model."""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.demos.muse_glimmer.tt.decode_core_config import MuseDecodeCoreConfig
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.prefetcher import Prefetcher


def run(layers: int, repetitions: int, trace: bool, receivers_per_bank: int) -> None:
    mesh = None
    prefetcher = None
    trace_id = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=256_000_000)
        prefetcher = Prefetcher(
            mesh,
            num_tensors=7,
            num_layers=layers,
            num_receiver_cores=receivers_per_bank,
            receiver_mapping_override=(
                MuseDecodeCoreConfig.full_bandwidth_sender_mapping()
                if receivers_per_bank == 9
                else MuseDecodeCoreConfig.isolated_sender_mapping(receivers_per_bank)
            ),
            model_name="Muse-Glimmer-30B",
        )
        prefetcher.init(Mode.DECODE)

        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})

        def weight(width: int, dtype):
            memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (6912, width // 8),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            return ttnn.from_torch(
                torch.randn(1, 1, 6912, width, dtype=torch.bfloat16),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        qkv = (weight(4608, ttnn.bfloat8_b),)
        mlp = tuple(weight(6912, ttnn.bfloat4_b) for _ in range(3))
        stream = qkv + mlp + mlp
        for _ in range(layers):
            for tensor in stream:
                prefetcher.insert_tensor(tensor)

        core_config = MuseDecodeCoreConfig(prefetcher)
        activation = ttnn.from_torch(
            torch.randn(1, 1, 32, 6656, dtype=torch.bfloat16),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def forward():
            output = None
            for layer in range(layers):
                output = core_config.streamed_projection(activation, qkv, next_weight=mlp[0])
                ttnn.deallocate(output)
                output = core_config.streamed_projection(activation, mlp, next_weight=mlp[0])
                ttnn.deallocate(output)
                output = core_config.streamed_projection(activation, mlp)
                ttnn.deallocate(output)
            return output

        prefetcher.run()
        forward()
        mesh.reset_sub_device_stall_group()
        ttnn.synchronize_device(mesh)
        print("MIXED_PREFETCH direct=pass", flush=True)
        if not trace:
            return

        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        prefetcher.run()
        forward()
        mesh.reset_sub_device_stall_group()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        start = time.perf_counter()
        for _ in range(repetitions):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        elapsed_us = (time.perf_counter() - start) * 1e6 / repetitions
        print(
            f"MIXED_PREFETCH replay=pass layers={layers} trace_us={elapsed_us:.2f} "
            f"per_projection_us={elapsed_us / layers / 3:.2f}",
            flush=True,
        )
    finally:
        if mesh is not None:
            try:
                mesh.reset_sub_device_stall_group()
            except Exception:
                pass
        if prefetcher is not None and prefetcher.garbage is not None:
            try:
                prefetcher.stop()
            except Exception:
                pass
        if mesh is not None:
            if trace_id is not None:
                try:
                    ttnn.release_trace(mesh, trace_id)
                except Exception:
                    pass
            try:
                mesh.clear_loaded_sub_device_manager()
            except Exception:
                pass
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--receivers-per-bank", type=int, choices=(2, 3, 9), default=9)
    args = parser.parse_args()
    run(args.layers, args.repetitions, args.trace, args.receivers_per_bank)
