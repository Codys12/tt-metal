# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Decoder Layer.

Each layer has 7 RMSNorms + layer_scalar:
  - input_layernorm: before attention
  - post_attention_layernorm: after attention, before residual add
  - pre_feedforward_layernorm: before shared MLP
  - post_feedforward_layernorm: after combined MLP+MoE, before final residual add
  - post_feedforward_layernorm_1: after shared MLP output (MoE path only)
  - pre_feedforward_layernorm_2: before expert input (MoE path only)
  - post_feedforward_layernorm_2: after expert output (MoE path only)
  - layer_scalar: learned per-layer scalar

Forward flow (matching HF exactly):
  residual = x
  x = input_layernorm(x)
  x = self_attn(x)
  x = post_attention_layernorm(x)
  x = residual + x

  residual = x
  x = pre_feedforward_layernorm(x)
  x = mlp(x)

  if enable_moe_block:
    x_1 = post_feedforward_layernorm_1(x)
    x_flat = residual.reshape(-1, H)     # router input = pre-norm residual
    _, top_k_w, top_k_idx = router(x_flat)
    x_2 = pre_feedforward_layernorm_2(x_flat)
    x_2 = experts(x_2, top_k_idx, top_k_w)
    x_2 = post_feedforward_layernorm_2(x_2)
    x = x_1 + x_2

  x = post_feedforward_layernorm(x)
  x = residual + x
  x *= layer_scalar
"""

import time

import torch

import ttnn
from models.demos.gemma4_cody.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.attention.decode import _PV_OPPROF, PV_OPPROF
from models.demos.gemma4_cody.tt.gemma4_attention_config import get_attention_program_config
from models.demos.gemma4_cody.tt.moe import MoEBlock
from models.demos.gemma4_cody.tt.rms_norm import RMSNorm
from models.demos.gemma4_cody.tt.shared_mlp import SharedMLP
from models.demos.gemma4_cody.utils.general_utils import cached_tensor_placeholder, get_cache_file_name
from models.demos.gemma4_cody.utils.substate import substate


class Gemma4DecoderLayer:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype,
        tensor_cache_path,
        mesh_config,
        max_seq_len,
        max_local_batch_size,
        transformation_mats=None,  # Legacy — ignored (HF-style RoPE needs no transformation mats)
    ):
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.hidden_size = hf_config.hidden_size
        self.layer_type = hf_config.layer_types[layer_idx]
        self.enable_moe_block = hf_config.enable_moe_block
        self.hidden_size_per_layer_input = getattr(hf_config, "hidden_size_per_layer_input", 0) or 0

        # Try both key formats (HF uses "model.language_model.layers", tests use "model.layers")
        layer_state = {}
        if state_dict:
            for prefix in [f"model.language_model.layers.{layer_idx}", f"model.layers.{layer_idx}"]:
                layer_state = substate(state_dict, prefix)
                if layer_state:
                    break

        def _norm(name, with_scale=True, enable_sharded_decode=False):
            return RMSNorm(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=substate(layer_state, name) if layer_state else {},
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/{name}" if tensor_cache_path else None,
                mesh_config=mesh_config,
                with_scale=with_scale,
                enable_sharded_decode=enable_sharded_decode,
            )

        # 4 norms present on every layer — opt into the width-sharded multi-core
        # decode path. The adapter still handles input/output reshard internally,
        # so call sites below are unchanged. Standalone the reshard cost roughly
        # cancels the kernel gain at hidden=2816; the win compounds once the
        # surrounding ops in this layer keep their sharded layout (follow-up).
        self.input_layernorm = _norm("input_layernorm", enable_sharded_decode=True)
        self.post_attention_layernorm = _norm("post_attention_layernorm", enable_sharded_decode=True)
        self.pre_feedforward_layernorm = _norm("pre_feedforward_layernorm", enable_sharded_decode=True)
        self.post_feedforward_layernorm = _norm("post_feedforward_layernorm", enable_sharded_decode=True)

        # 3 additional norms for MoE layers
        if self.enable_moe_block:
            self.post_feedforward_layernorm_1 = _norm("post_feedforward_layernorm_1")
            self.pre_feedforward_layernorm_2 = _norm("pre_feedforward_layernorm_2")
            self.post_feedforward_layernorm_2 = _norm("post_feedforward_layernorm_2")

        # Layer scalar
        if layer_state and "layer_scalar" in layer_state:
            self.layer_scalar = layer_state["layer_scalar"].item()
        else:
            self.layer_scalar = 1.0

        # Attention
        attn_config = Gemma4AttentionConfig(hf_config, layer_idx)
        attn_program_config = get_attention_program_config(attn_config, mesh_config, is_decode=True)
        self.self_attn = Gemma4Attention(
            mesh_device=mesh_device,
            config=attn_config,
            state_dict=substate(layer_state, "self_attn") if layer_state else {},
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attn_program_config,
            layer_idx=layer_idx,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/self_attn" if tensor_cache_path else None,
            max_batch_size=max_local_batch_size,
        )

        # Shared/dense MLP (HF key: "mlp") — bfloat4_b for MLP weights
        self.shared_mlp = SharedMLP(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(layer_state, "mlp") if layer_state else {},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=ttnn.bfloat4_b,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/mlp" if tensor_cache_path else None,
            max_local_batch_size=max_local_batch_size,
        )

        # MoE block (router + routed experts) — bfloat4_b for expert weights (router stays bf16)
        if self.enable_moe_block:
            self.moe = MoEBlock(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=layer_state,  # MoE expects "router.*" and "experts.*" keys
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                dtype=ttnn.bfloat4_b,
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/moe" if tensor_cache_path else None,
            )

        # Per-layer input embeddings (E2B/E4B feature)
        if self.hidden_size_per_layer_input:
            pli_prefix = f"{tensor_cache_path}/layer_{layer_idx}" if tensor_cache_path else None
            gate_cache_name = get_cache_file_name(pli_prefix, "per_layer_input_gate")
            proj_cache_name = get_cache_file_name(pli_prefix, "per_layer_projection")

            if layer_state and "per_layer_input_gate.weight" in layer_state:
                gate_w = cached_tensor_placeholder(gate_cache_name, dtype, ttnn.TILE_LAYOUT)
                proj_w = cached_tensor_placeholder(proj_cache_name, dtype, ttnn.TILE_LAYOUT)
                if gate_w is None:
                    gate_w = layer_state["per_layer_input_gate.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
                if proj_w is None:
                    proj_w = layer_state["per_layer_projection.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            else:
                gate_w = None
                proj_w = None

            self.per_layer_input_gate = ttnn.as_tensor(
                gate_w,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=gate_cache_name,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.per_layer_projection = ttnn.as_tensor(
                proj_w,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=proj_cache_name,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.post_per_layer_input_norm = _norm("post_per_layer_input_norm")

    def __call__(
        self,
        hidden_states,
        rope_mats,
        position_idx,
        page_table,
        kv_cache,
        is_decode,
        token_index=None,
        per_layer_input=None,
        shared_kv=None,
        keep_kv=False,
        is_kv_shared=False,
        position_idx_cache=None,
        page_table_sliding=None,
        position_idx_cache_sliding_write=None,
        position_idx_cache_sliding_sdpa=None,
        packed=None,
    ):
        """
        Decoder layer forward pass.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            rope_mats: precomputed RoPE matrices
            position_idx: current position index
            page_table: paged attention page table
            kv_cache: KV cache for this layer
            is_decode: True for decode mode
            shared_kv: optional (tt_k, tt_v) from source layer for KV sharing (prefill only)
            keep_kv: if True, keep K/V alive for sharing with later layers (prefill only)
            is_kv_shared: if True, this layer shares KV from source (skip K/V proj + cache update)

        Returns:
            hidden_states: [1, 1, seq_len, hidden_size] on device
        """
        # 1. Attention block: norm -> attn -> post_attn_norm -> residual add
        residual = hidden_states
        normed = self.input_layernorm.forward(hidden_states)
        attn_output = self.self_attn(
            normed,
            rope_mats=rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            token_index=token_index,
            shared_kv=shared_kv,
            keep_kv=keep_kv,
            is_kv_shared=is_kv_shared,
            position_idx_cache=position_idx_cache,
            page_table_sliding=page_table_sliding,
            position_idx_cache_sliding_write=position_idx_cache_sliding_write,
            position_idx_cache_sliding_sdpa=position_idx_cache_sliding_sdpa,
            packed=packed,
        )

        if isinstance(attn_output, torch.Tensor):
            hidden_states = residual
        else:
            attn_output = self.post_attention_layernorm.forward(attn_output)
            hidden_states = ttnn.add(residual, attn_output)
            residual.deallocate(True)
            attn_output.deallocate(True)

        # 2. MLP + MoE block
        _op_mlp = _PV_OPPROF and PV_OPPROF["active"] and packed is not None
        if _op_mlp:
            ttnn.synchronize_device(self.mesh_device)
            _op_mlp_t = time.perf_counter()
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(hidden_states)
        mlp_output = self.shared_mlp(normed)
        normed.deallocate(True)
        if _op_mlp:
            ttnn.synchronize_device(self.mesh_device)
            PV_OPPROF["mlp"] += time.perf_counter() - _op_mlp_t

        if self.enable_moe_block:
            # post_feedforward_layernorm_1 on MLP output
            mlp_normed = self.post_feedforward_layernorm_1.forward(mlp_output)
            mlp_output.deallocate(True)

            # Router input = pre-MLP residual, expert input = normed residual
            # All on device — no CPU round-trip
            residual_for_router = residual
            expert_input = self.pre_feedforward_layernorm_2.forward(residual_for_router)

            # MoE: router(residual) → dense_routing → experts(normed_input, routing)
            expert_output = self.moe(residual_for_router, expert_input)
            expert_input.deallocate(True)

            # post_feedforward_layernorm_2 on expert output
            expert_normed = self.post_feedforward_layernorm_2.forward(expert_output)
            expert_output.deallocate(True)

            # Combine: mlp_normed + expert_normed
            hidden_states = ttnn.add(mlp_normed, expert_normed)
            mlp_normed.deallocate(True)
            expert_normed.deallocate(True)
        else:
            hidden_states = mlp_output

        # post_feedforward_layernorm -> residual add
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)

        hidden_states = combined

        # Per-layer input embeddings (E2B/E4B) — BEFORE layer_scalar (matching HF order)
        if self.hidden_size_per_layer_input and per_layer_input is not None and hasattr(self, "per_layer_input_gate"):
            residual_pli = hidden_states
            gated = ttnn.linear(hidden_states, self.per_layer_input_gate)
            gated = ttnn.gelu(gated, fast_and_approximate_mode=True)
            gated = ttnn.mul(gated, per_layer_input)
            projected = ttnn.linear(gated, self.per_layer_projection)
            normed_pli = self.post_per_layer_input_norm.forward(projected)
            hidden_states = ttnn.add(residual_pli, normed_pli)
            if len(hidden_states.shape) > 4:
                hidden_states = ttnn.reshape(hidden_states, (1, 1, hidden_states.shape[-2], self.hidden_size))

        # Layer scalar — AFTER PLI (matching HF order)
        if self.layer_scalar != 1.0:
            hidden_states = ttnn.mul(hidden_states, self.layer_scalar)

        return hidden_states
