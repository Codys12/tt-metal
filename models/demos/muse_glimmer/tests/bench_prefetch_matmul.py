# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure a single-token Glimmer matmul fed by the Blackhole DRAM prefetcher."""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.prefetcher import VERIFIED_MODEL_CONFIGS, Prefetcher


def _round_up(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def run(
    k: int,
    n: int,
    receivers_per_bank: int,
    dtype: str,
    repetitions: int,
    performance_mode: bool,
    trace: bool,
    layers: int,
    include_input_reshard: bool,
    edge_senders: bool,
    distinct_weights: bool,
    receiver_start_x: int,
    math_fidelity: str,
    dst_full_sync: bool,
    separate_cq: bool,
    check_pcc: bool,
) -> None:
    os.environ["HF_MODEL"] = "MuseGlimmerPrefetchBench"
    VERIFIED_MODEL_CONFIGS["MuseGlimmerPrefetchBench"] = {
        "dim": k,
        # Admission uses a conservative BFP8 estimate. The benchmark's wider
        # BFP4 tensors are checked from their actual tile sizes at insertion.
        "hidden_dim": min(n, 2304),
        "n_heads": 32,
        "n_kv_heads": 2,
    }

    mesh = None
    prefetcher = None
    prefetch_started = False
    trace_id = None
    try:
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            trace_region_size=64_000_000,
            num_command_queues=2 if separate_cq else 1,
        )
        receiver_mapping = None
        if edge_senders:
            if receivers_per_bank > 9:
                raise ValueError("The edge-sender topology supports at most nine receivers per bank")
            bank_ordered_rows = (9, 1, 7, 3, 0, 2, 6, 4)
            if receivers_per_bank == 2:
                receiver_mapping = {(10, row): [(8, row), (9, row)] for row in bank_ordered_rows}
            elif receivers_per_bank == 3:
                overflow_receivers = [(x, 8) for x in range(8)]
                receiver_mapping = {
                    (10, row): [(8, row), (9, row), overflow_receivers[bank_idx]]
                    for bank_idx, row in enumerate(bank_ordered_rows)
                }
            else:
                receiver_mapping = {}
                for row in bank_ordered_rows:
                    receiver_mapping[(9, row)] = [(x, row) for x in range(5)]
                    receiver_mapping[(10, row)] = [(x, row) for x in range(5, 9)]
        prefetcher = Prefetcher(
            mesh,
            num_tensors=1,
            num_layers=layers,
            num_receiver_cores=receivers_per_bank,
            receiver_mapping_override=receiver_mapping,
        )
        prefetcher.enable_performance_mode = performance_mode
        ring_size = prefetcher.ring_size
        dram_banks = len(prefetcher.dram_banks())
        n_padded = _round_up(math.ceil(n / ring_size), ttnn.TILE_SIZE) * ring_size

        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_banks - 1, 0))})
        weight_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_grid,
                (k, n_padded // dram_banks),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[dtype]
        if os.getenv("MUSE_PREFETCH_IDENTITY_WEIGHTS", "0") == "1":
            if k != n_padded:
                raise ValueError("identity weights require k == n_padded")
            torch_weights = [
                torch.eye(k, dtype=torch.bfloat16).reshape(1, 1, k, k) for _ in range(layers if distinct_weights else 1)
            ]
        else:
            torch_weights = [
                torch.randn(1, 1, k, n_padded, dtype=torch.bfloat16) for _ in range(layers if distinct_weights else 1)
            ]
        weights = [
            ttnn.from_torch(
                torch_weight,
                device=mesh,
                dtype=tt_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for torch_weight in torch_weights
        ]

        receiver_grid = prefetcher.to_core_range_set(
            prefetcher.core_config.receiver_cores(sender_active=True, receiver_active=True)
        )
        input_width_per_core = _round_up(math.ceil(k / ring_size), ttnn.TILE_SIZE)
        input_memory_config = ttnn.create_sharded_memory_config(
            shape=(32, input_width_per_core),
            core_grid=receiver_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(32, n_padded // ring_size),
            core_grid=receiver_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        torch_activation = torch.randn(1, 1, 32, k, dtype=torch.bfloat16)
        activation = ttnn.from_torch(
            torch_activation,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        interleaved_activation = None
        if include_input_reshard:
            interleaved_activation = ttnn.from_torch(
                torch_activation,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        in0_block_w = input_width_per_core // ttnn.TILE_SIZE
        while in0_block_w > 0 and (k // ttnn.TILE_SIZE) % in0_block_w:
            in0_block_w -= 1
        per_core_n = n_padded // ring_size // ttnn.TILE_SIZE
        out_subblock_w = min(8, per_core_n)
        while per_core_n % out_subblock_w:
            out_subblock_w -= 1
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(dram_banks, receivers_per_bank),
            in0_block_w=max(1, in0_block_w),
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=ttnn.CoreRangeSet(set()),
            num_global_cb_receivers=receivers_per_bank,
            untilize_out=False,
        )
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity={"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}[math_fidelity],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=dst_full_sync,
        )

        prefetcher.init(Mode.DECODE)
        for layer_idx in range(layers):
            prefetcher.insert_tensor(weights[layer_idx] if distinct_weights else weights[0])

        def matmul():
            output = None
            for layer_idx in range(layers):
                if include_input_reshard:
                    ttnn.interleaved_to_sharded(
                        interleaved_activation,
                        input_memory_config,
                        preallocated_output=activation,
                    )
                output = ttnn.linear(
                    activation,
                    weights[layer_idx] if distinct_weights else weights[0],
                    program_config=program_config,
                    memory_config=output_memory_config,
                    compute_kernel_config=compute_config,
                    dtype=ttnn.bfloat16,
                    global_cb=prefetcher.global_cb,
                    sub_device_id=prefetcher.worker_sub_device_id,
                )
            return output

        # Compile both programs once before capture.
        prefetcher.run()
        prefetch_started = True
        output = matmul()
        mesh.reset_sub_device_stall_group()
        ttnn.synchronize_device(mesh)
        if check_pcc:
            actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).float()
            expected = torch.matmul(
                torch_activation.float(),
                torch_weights[-1 if distinct_weights else 0].float(),
            )
            quantized_weight = ttnn.to_torch(ttnn.get_device_tensors(weights[-1 if distinct_weights else 0])[0]).float()
            quantized_expected = torch.matmul(torch_activation.float(), quantized_weight)
            _, pcc = comp_pcc(expected, actual, 0.0)
            _, quantized_pcc = comp_pcc(quantized_expected, actual, 0.0)
            print(f"PREFETCH_PCC {float(pcc):.8f}", flush=True)
            print(f"PREFETCH_QUANTIZED_PCC {float(quantized_pcc):.8f}", flush=True)
            shard_width = n_padded // ring_size
            shard_pccs = []
            for shard_idx in range(ring_size):
                shard_start = shard_idx * shard_width
                shard_end = shard_start + shard_width
                _, shard_pcc = comp_pcc(
                    expected[..., shard_start:shard_end],
                    actual[..., shard_start:shard_end],
                    0.0,
                )
                shard_pccs.append(f"{float(shard_pcc):.4f}")
            print(f"PREFETCH_SHARD_PCC {' '.join(shard_pccs)}", flush=True)
            if os.getenv("MUSE_PREFETCH_IDENTITY_WEIGHTS", "0") == "1":
                shard_map = []
                for actual_shard_idx in range(ring_size):
                    actual_start = actual_shard_idx * shard_width
                    actual_end = actual_start + shard_width
                    candidates = []
                    for expected_shard_idx in range(ring_size):
                        expected_start = expected_shard_idx * shard_width
                        expected_end = expected_start + shard_width
                        _, candidate_pcc = comp_pcc(
                            expected[..., expected_start:expected_end],
                            actual[..., actual_start:actual_end],
                            0.0,
                        )
                        candidates.append(float(candidate_pcc))
                    shard_map.append(str(max(range(ring_size), key=lambda idx: candidates[idx])))
                print(f"PREFETCH_SHARD_MAP {' '.join(shard_map)}", flush=True)
        if not trace:
            print(
                "PREFETCH_RESULT "
                f"ring={ring_size} k={k} n={n} n_padded={n_padded} dtype={dtype} "
                f"performance_mode={performance_mode} global_cb_bytes={prefetcher.global_cb_size} "
                f"layers={layers} input_reshard={include_input_reshard} edge_senders={edge_senders} "
                f"distinct_weights={distinct_weights} "
                f"receiver_start_x={receiver_start_x} "
                "compile_run=pass",
                flush=True,
            )
            return

        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        if not separate_cq:
            prefetcher.run()
        output = matmul()
        if not separate_cq:
            mesh.reset_sub_device_stall_group()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)

        def replay():
            if separate_cq:
                mesh.set_sub_device_stall_group([prefetcher.worker_sub_device_id])
                with ttnn.command_queue(1):
                    prefetcher.run()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            if separate_cq:
                ttnn.synchronize_device(mesh, sub_device_ids=[prefetcher.worker_sub_device_id])
                mesh.reset_sub_device_stall_group()
                ttnn.synchronize_device(mesh)

        replay()

        start = time.perf_counter()
        for _ in range(repetitions):
            replay()
        if not separate_cq:
            ttnn.synchronize_device(mesh)
        latency_us = (time.perf_counter() - start) * 1e6 / repetitions
        print(
            "PREFETCH_RESULT "
            f"ring={ring_size} k={k} n={n} n_padded={n_padded} dtype={dtype} "
            f"performance_mode={performance_mode} global_cb_bytes={prefetcher.global_cb_size} "
            f"layers={layers} input_reshard={include_input_reshard} "
            f"edge_senders={edge_senders} "
            f"distinct_weights={distinct_weights} "
            f"receiver_start_x={receiver_start_x} "
            f"math_fidelity={math_fidelity} dst_full_sync={dst_full_sync} "
            f"trace_us={latency_us:.2f} per_matmul_us={latency_us / layers:.2f}",
            flush=True,
        )
        del output
    finally:
        if mesh is not None:
            try:
                mesh.reset_sub_device_stall_group()
            except Exception:
                pass
        if prefetcher is not None and prefetch_started:
            try:
                prefetcher.stop()
            except Exception:
                pass
        if mesh is not None:
            try:
                mesh.clear_loaded_sub_device_manager()
            except Exception:
                pass
            if trace_id is not None:
                try:
                    ttnn.release_trace(mesh, trace_id)
                except Exception:
                    pass
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=6656)
    parser.add_argument("--n", type=int, default=4608)
    parser.add_argument("--receivers-per-bank", type=int, choices=(1, 2, 3, 8, 9, 10), default=8)
    parser.add_argument("--dtype", choices=("bfp8", "bfp4"), default="bfp8")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--include-input-reshard", action="store_true")
    parser.add_argument("--edge-senders", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--performance-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distinct-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--receiver-start-x", type=int, default=0)
    parser.add_argument("--math-fidelity", choices=("lofi", "hifi2"), default="hifi2")
    parser.add_argument("--dst-full-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate-cq", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-pcc", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    run(
        args.k,
        args.n,
        args.receivers_per_bank,
        args.dtype,
        args.repetitions,
        args.performance_mode,
        args.trace,
        args.layers,
        args.include_input_reshard,
        args.edge_senders,
        args.distinct_weights,
        args.receiver_start_x,
        args.math_fidelity,
        args.dst_full_sync,
        args.separate_cq,
        args.check_pcc,
    )
