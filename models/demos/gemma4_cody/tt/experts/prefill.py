# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device expert prefill forward using sparse_matmul.

Following gpt_oss pattern: sparse_matmul computes ALL experts (all-ones sparsity),
then routing weights are applied after down projection to select active experts.

For prefill (seq_len > 1), the sequence is reshaped into tile-sized groups:
  [1, 1, seq_len, H] -> [1, seq_len/32, 32, H]
"""

import os
import time as _time

import torch

import ttnn
from models.demos.gemma4_cody.tt.ccl import ccl_allreduce

from .decode import _build_sparse_matmul_config
from .operations import apply_geglu, expert_kernel_config
from .weights import ExpertWeights

TILE_SIZE = 32

# Env-gated MoE sub-op profiling (DIFF_MOE_PROF=1). No-op for the AR gemma4 path.
_MOE_PROF = os.getenv("DIFF_MOE_PROF") == "1"
MOE_PROF_TIMES = {}


def _moe_mark(dev, label, t0):
    if not _MOE_PROF:
        return 0.0
    ttnn.synchronize_device(dev)
    now = _time.perf_counter()
    MOE_PROF_TIMES[label] = MOE_PROF_TIMES.get(label, 0.0) + (now - t0)
    return now


def create_prefill_sparsity(mesh_device, num_experts):
    """Create all-ones sparsity mask for prefill (all experts active).

    Following gpt_oss: prefill computes all experts, routing weights
    zero out inactive experts afterward.

    Returns:
        [1, 1, 1, num_experts] ROW_MAJOR bfloat16 tensor on device
    """
    is_mesh = hasattr(mesh_device, "shape")
    sparsity = torch.ones(1, 1, 1, num_experts, dtype=torch.bfloat16)
    return ttnn.from_torch(
        sparsity,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _process_prefill_chunk(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    prefill_sparsity,
):
    """Process a single chunk of the sequence in prefill mode.

    Args:
        hidden_states: [1, 1, chunk_len, hidden_size] on device
        routing_weights: [1, 1, chunk_len, num_experts] on device
    Returns:
        [1, 1, chunk_len, hidden_size] expert output
    """
    chunk_len = hidden_states.shape[2]
    num_experts = config.num_experts
    hidden_size = config.hidden_size

    group_size = chunk_len // TILE_SIZE

    # Reshape sequence into tile groups: [1, 1, S, H] -> [1, S/32, 32, H]
    hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))

    # All-ones sparsity repeated across groups
    sparsity = ttnn.repeat(prefill_sparsity, (1, 1, group_size, 1))

    # nnz = all experts × groups (compute every expert for every group)
    nnz = num_experts * group_size

    output_tile = ttnn.Tile([32, 32])
    intermediate_size = weights.intermediate_size_per_device
    # gate/up contract K=hidden (88 tiles); in0_block_w=1 ⇒ 88 serial K-steps and
    # they dominate the MoE. Larger in0_block_w contracts more K per step.
    # DIFF_MOE_IN0BW (default 1 = unchanged for the AR gemma4 path).
    _in0bw = int(os.getenv("DIFF_MOE_IN0BW", "1"))
    gate_up_config = _build_sparse_matmul_config(TILE_SIZE, intermediate_size, in0_block_w=_in0bw)
    down_config = _build_sparse_matmul_config(TILE_SIZE, hidden_size)
    kcfg = expert_kernel_config(hidden_states.device())
    _dev = hidden_states.device()
    _t = _time.perf_counter() if _MOE_PROF else 0.0

    # Gate projection
    gate = ttnn.sparse_matmul(
        hidden_grouped,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=kcfg,
        dtype=ttnn.bfloat16,
    )
    _t = _moe_mark(_dev, "gate_mm", _t)
    sm_intermediate = gate.shape[-1]
    gate = ttnn.transpose(gate, 1, 3)
    gate = ttnn.reshape(gate, (1, num_experts, chunk_len, sm_intermediate))

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_grouped,
        weights.up_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=kcfg,
        dtype=ttnn.bfloat16,
    )
    hidden_grouped.deallocate(True)
    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, (1, num_experts, chunk_len, sm_intermediate))
    _t = _moe_mark(_dev, "up_mm", _t)

    # GeGLU activation
    down_input = apply_geglu(gate, up)
    down_input = ttnn.reshape(down_input, (1, num_experts, chunk_len, sm_intermediate))
    _t = _moe_mark(_dev, "geglu+transposes", _t)

    # Down projection — all experts, using base sparsity (not repeated)
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=prefill_sparsity,
        nnz=num_experts,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        compute_kernel_config=kcfg,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    _t = _moe_mark(_dev, "down_mm", _t)

    # Reshape to [1, E, S, H]
    next_states = ttnn.reshape(down, (1, num_experts, chunk_len, hidden_size))

    # Apply routing weights to select active experts (gpt_oss pattern)
    # routing_weights: [1, 1, S, E] -> [1, E, S, 1] for broadcast mul
    routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))  # [1, E, S, 1]
    next_states = ttnn.mul(next_states, routing_permuted)

    # Reduce across experts dimension using fast_reduce_nc (gpt_oss pattern)
    next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
    next_states = ttnn.reshape(next_states, (1, 1, chunk_len, hidden_size))
    _t = _moe_mark(_dev, "route+reduce", _t)

    return next_states


# Maximum tokens per chunk for sparse_matmul prefill.
# sparse_matmul treats the group dimension (seq_len/32) as part of the M dimension,
# which increases num_blocks_y and can overflow the core grid.
# Using chunk_size=32 means group_size=1, so num_blocks_y=1 and the
# blocks come entirely from N-dimension — guaranteed to fit.
PREFILL_CHUNK_SIZE = 32


def prefill_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    prefill_sparsity,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
):
    """
    On-device expert prefill forward using sparse_matmul.

    Following gpt_oss pattern:
    1. Chunk long sequences so group_size fits in the core grid
    2. sparse_matmul gate/up/down with all-ones sparsity (compute ALL experts)
    3. Apply routing weights after down projection to select active experts
    4. Reduce across expert dimension

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] on device (seq_len must be multiple of 32)
        routing_weights: [1, 1, seq_len, num_experts] on device (dense routing from router)
        weights: ExpertWeights
        config: Gemma4ExpertConfig
        prefill_sparsity: [1, 1, 1, num_experts] all-ones sparsity mask (cached, do not deallocate)

    Returns:
        output: [1, 1, seq_len, hidden_size] on device
    """
    seq_len = hidden_states.shape[2]
    hidden_size = config.hidden_size

    assert seq_len % TILE_SIZE == 0, f"Prefill seq_len must be multiple of {TILE_SIZE}, got {seq_len}"

    # Chunk long sequences to avoid sparse_matmul core overflow
    if seq_len > PREFILL_CHUNK_SIZE:
        hidden_chunks = ttnn.split(hidden_states, PREFILL_CHUNK_SIZE, dim=2)
        routing_chunks = ttnn.split(routing_weights, PREFILL_CHUNK_SIZE, dim=2)
    else:
        hidden_chunks = [hidden_states]
        routing_chunks = [routing_weights]

    result_acc = None
    for h_chunk, r_chunk in zip(hidden_chunks, routing_chunks):
        chunk_result = _process_prefill_chunk(h_chunk, r_chunk, weights, config, prefill_sparsity)

        if result_acc is None:
            result_acc = chunk_result
        else:
            result_concat = ttnn.concat([result_acc, chunk_result], dim=2)
            result_acc.deallocate(True)
            chunk_result.deallocate(True)
            result_acc = result_concat

    # All-reduce after row-parallel down_proj
    if mesh_config is not None and mesh_config.tp > 1:
        result_acc = ccl_allreduce(result_acc, mesh_config, ccl_manager)

    return result_acc
