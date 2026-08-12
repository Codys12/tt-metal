# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has BOTH a shared MLP and routed MoE experts.
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x))
intermediate_size = 2112, no bias.

HF weight shapes:
  gate_proj.weight: [intermediate_size, hidden_size] = [2112, 2816]
  up_proj.weight:   [intermediate_size, hidden_size] = [2112, 2816]
  down_proj.weight: [hidden_size, intermediate_size] = [2816, 2112]

TP fusion (down_proj):
  The row-parallel `down_proj` is followed by an all-reduce. With TP > 1
  and persistent buffers allocated, this is replaced by
  `matmul_reduce_scatter_async` followed by `all_gather_async`, fusing
  the matmul output streaming directly into the reduce-scatter.
"""

import ttnn
from models.demos.gemma4_cody.tt.ccl import (
    ccl_allreduce,
    ccl_matmul_reduce_scatter_allgather,
    make_block_sharded_matmul_config,
    make_fused_matmul_program_config,
    make_reduce_scatter_persistent_buffers,
)
from models.demos.gemma4_cody.utils.general_utils import cached_tensor_placeholder, get_cache_file_name

# L1-block-sharded activations for packed-shape gate/up (~1.8x over DRAM
# interleaved). Per-shape config cache so trace replay reuses configs.
_L1_PC_CACHE = {}

# LoFi math for the MLP matmuls (bf4/bf8 weights carry 4-8 mantissa bits;
# 4-bit activation truncation is below quantization noise; ~2x ceiling).
# Attention's L1-pc matmuls also land LoFi (ttnn auto-fidelity drops to LoFi
# whenever a program_config is passed without a kernel config) — measured
# harmless: HiFi2-vs-LoFi A/B on attention moved acceptance by 0.
_LOFI_KCFG = None


def _mlp_kernel_config(mesh_device):
    global _LOFI_KCFG
    if _LOFI_KCFG is None:
        _LOFI_KCFG = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    return _LOFI_KCFG


def _l1_pc(m, k, n, gelu=False, out_l1=True):
    key = (m, k, n, gelu, out_l1)
    if key not in _L1_PC_CACHE:
        res = make_block_sharded_matmul_config(m, k, n, with_out_mem=out_l1)
        pc, mem = res[0], res[1]
        out_mem = res[2] if out_l1 else None
        if gelu:
            pc.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)
        _L1_PC_CACHE[key] = (pc, mem, out_mem)
    return _L1_PC_CACHE[key]


class SharedMLP:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        max_local_batch_size=1,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        if tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        gate_cache_name = get_cache_file_name(tensor_cache_path, f"gate_proj.weight{tp_suffix}")
        up_cache_name = get_cache_file_name(tensor_cache_path, f"up_proj.weight{tp_suffix}")
        down_cache_name = get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}")

        if state_dict:
            gate_proj_weight = cached_tensor_placeholder(gate_cache_name, dtype, ttnn.TILE_LAYOUT)
            up_proj_weight = cached_tensor_placeholder(up_cache_name, dtype, ttnn.TILE_LAYOUT)
            down_proj_weight = cached_tensor_placeholder(down_cache_name, dtype, ttnn.TILE_LAYOUT)
            if gate_proj_weight is None:
                gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            if up_proj_weight is None:
                up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            if down_proj_weight is None:
                down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        # gate/up: column-parallel (shard output dim across TP devices)
        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=gate_cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=up_cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # down: row-parallel (shard input dim, allreduce after)
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=down_cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Persistent buffers for fused matmul_reduce_scatter on down_proj.
        # Decode shape: [1, 1, TILE-padded batch, hidden_size]. We pre-allocate
        # for the max_local_batch_size and reuse the same buffers per call.
        # Prefill (variable seq_len) falls back to the unfused all-reduce path.
        self._fused_intermediate = None
        self._fused_output = None
        self._fused_program_config = None
        if tp > 1 and ccl_manager is not None:
            tile_pad = max(32, ((max_local_batch_size + 31) // 32) * 32)
            self._fused_decode_seq = tile_pad
            self._fused_intermediate, self._fused_output = make_reduce_scatter_persistent_buffers(
                mesh_device=mesh_device,
                matmul_output_shape=(1, 1, tile_pad, self.hidden_size),
                tp=tp,
                dtype=ttnn.bfloat16,
            )
            # The fused matmul_reduce_scatter_async op requires an explicit
            # program_config (no auto-derivation, unlike ttnn.linear).
            # in_dim_per_device is down_proj's contraction dim per device —
            # derive it from the (row-sharded) weight to stay tile-exact.
            self._fused_program_config = make_fused_matmul_program_config(
                matmul_output_shape=(1, 1, tile_pad, self.hidden_size),
                in_dim_per_device=self.down_proj.shape[-2],
            )

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel (no comm needed).
        down is row-parallel: when persistent buffers exist and the seq dim
        matches, use matmul_reduce_scatter_async + all_gather_async; otherwise
        fall back to linear + ccl_allreduce.
        """
        M = hidden_states.shape[-2]
        kcfg = _mlp_kernel_config(self.mesh_device)
        # M <= 1024: x + gate + up stay L1-resident end-to-end (saves both
        # DRAM round-trips). M = 2048 (prefill bucket): the three shards no
        # longer fit next to the CBs - shard the input only, emit DRAM.
        if M % 256 == 0 and M <= 2048:
            chain_l1 = M <= 1024
            K, N = self.gate_proj.shape[-2], self.gate_proj.shape[-1]
            gate_pc, in_mem, gate_out_mem = _l1_pc(M, K, N, gelu=True, out_l1=chain_l1)
            up_pc, _, _ = _l1_pc(M, K, N, out_l1=chain_l1)
            out_mem = gate_out_mem if chain_l1 else ttnn.DRAM_MEMORY_CONFIG
            x_sh = ttnn.to_memory_config(hidden_states, in_mem)
            gate = ttnn.linear(
                x_sh, self.gate_proj, program_config=gate_pc, memory_config=out_mem, compute_kernel_config=kcfg
            )
            up = ttnn.linear(
                x_sh, self.up_proj, program_config=up_pc, memory_config=out_mem, compute_kernel_config=kcfg
            )
            x_sh.deallocate(True)
        else:
            gate = ttnn.linear(hidden_states, self.gate_proj, compute_kernel_config=kcfg)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            up = ttnn.linear(hidden_states, self.up_proj, compute_kernel_config=kcfg)

        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        tp = self.mesh_config.tp if self.mesh_config else 1
        seq_dim = hidden.shape[2]
        can_fuse = tp > 1 and self._fused_intermediate is not None and seq_dim == self._fused_decode_seq

        if can_fuse:
            return ccl_matmul_reduce_scatter_allgather(
                hidden,
                self.down_proj,
                self._fused_intermediate,
                self._fused_output,
                self.mesh_config,
                self.ccl_manager,
                program_config=self._fused_program_config,
                compute_kernel_config=kcfg,
            )

        if seq_dim % 256 == 0 and seq_dim <= 2048:
            # hidden is block-sharded for M <= 1024 (sharded mul output);
            # at 2048 it's DRAM interleaved — reshard. Emit DRAM for AR.
            chain_l1 = seq_dim <= 1024
            down_pc, down_in_mem, _ = _l1_pc(
                seq_dim, self.down_proj.shape[-2], self.down_proj.shape[-1], out_l1=chain_l1
            )
            if not chain_l1:
                hidden = ttnn.to_memory_config(hidden, down_in_mem)
            output = ttnn.linear(
                hidden,
                self.down_proj,
                program_config=down_pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=kcfg,
            )
        else:
            output = ttnn.linear(hidden, self.down_proj, compute_kernel_config=kcfg)
        hidden.deallocate(True)

        if tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)
        return output
