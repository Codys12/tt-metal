# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Router: RMSNorm -> scale -> linear -> softmax -> topk -> normalize -> per_expert_scale -> scatter

Fully on-device, trace-compatible. Returns dense routing weights [1,1,S,E] on device for sparse_matmul.
Gemma4 uses softmax-THEN-topk (opposite of GPT-OSS).

Following gpt_oss topk.py pattern: normalize via softmax on top-k subset,
scatter into dense tensor on device, per_expert_scale via broadcast mul.
"""


import ttnn
from models.demos.gemma4_cody.tt.rms_norm import RMSNorm
from models.demos.gemma4_cody.utils.general_utils import cached_tensor_placeholder, get_cache_file_name
from models.demos.gemma4_cody.utils.substate import substate


class Gemma4Router:
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.mesh_device = mesh_device
        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.top_k_experts
        self.hidden_size = hf_config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5

        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            tensor_cache_path=f"{tensor_cache_path}/norm" if tensor_cache_path else None,
            with_scale=False,
        )

        scale_cache_name = get_cache_file_name(tensor_cache_path, "scale")
        proj_cache_name = get_cache_file_name(tensor_cache_path, "proj.weight")
        per_expert_scale_cache_name = get_cache_file_name(tensor_cache_path, "per_expert_scale")

        if state_dict:
            scale_weight = cached_tensor_placeholder(scale_cache_name, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            proj_weight = cached_tensor_placeholder(proj_cache_name, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            per_expert_scale_raw = cached_tensor_placeholder(
                per_expert_scale_cache_name, ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            if scale_weight is None:
                scale_weight = state_dict["scale"].reshape(1, 1, 1, -1)
            if proj_weight is None:
                proj_weight = substate(state_dict, "proj")["weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            if per_expert_scale_raw is None:
                per_expert_scale_raw = state_dict["per_expert_scale"]
        else:
            scale_weight = None
            proj_weight = None
            per_expert_scale_raw = None

        is_mesh = hasattr(mesh_device, "shape")
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        self.scale = ttnn.as_tensor(
            scale_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=scale_cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.proj_weight = ttnn.as_tensor(
            proj_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=proj_cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Per-expert scale on device [1, 1, 1, num_experts] for broadcast mul with dense routing
        if per_expert_scale_raw is not None:
            per_expert_scale = per_expert_scale_raw
            if per_expert_scale.dim() > 0:
                per_expert_scale = per_expert_scale.reshape(1, 1, 1, -1)
            self.per_expert_scale = ttnn.as_tensor(
                per_expert_scale,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_mapper,
                cache_file_name=per_expert_scale_cache_name,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.per_expert_scale = None

    def __call__(self, hidden_states):
        """
        Route tokens to experts. Fully on-device, trace-compatible.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device

        Returns:
            dense_routing: [1, 1, seq_len, num_experts] on device — weights at selected experts, zeros elsewhere
        """
        # 1. RMSNorm — on device
        normed = self.norm.forward(hidden_states)

        # 2. Scale — on device
        scaled = ttnn.mul(normed, self.scale)
        normed.deallocate(True)
        scaled = ttnn.mul(scaled, self.scalar_root_size)

        # 3. Linear projection → [1, 1, seq_len, num_experts] — on device
        expert_scores = ttnn.linear(scaled, self.proj_weight)
        scaled.deallocate(True)

        # 4. Softmax over all experts — on device
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)

        # 5. TopK — on device → values [1,1,S,k], indices [1,1,S,k]
        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)

        # 6. Normalize top-k weights via softmax on the subset (gpt_oss pattern)
        top_k_values = ttnn.softmax(top_k_values, dim=-1)

        # 7. Scatter into dense [1,1,S,E] — fully on device
        dense_routing = ttnn.scatter(
            ttnn.zeros_like(router_probs),
            dim=-1,
            index=top_k_indices,
            src=top_k_values,
        )
        router_probs.deallocate(True)
        top_k_values.deallocate(True)
        top_k_indices.deallocate(True)

        # 8. Per-expert scale — broadcast mul on device
        if self.per_expert_scale is not None:
            dense_routing = ttnn.mul(dense_routing, self.per_expert_scale)

        return dense_routing
