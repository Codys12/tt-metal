# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native Muse Glimmer DFlash weight loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn
from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder, get_cache_file_name
from models.demos.muse_glimmer.utils.lazy_state_dict import LazyStateDict


@dataclass(frozen=True)
class DFlashLayerWeights:
    input_layernorm: ttnn.Tensor
    post_attention_layernorm: ttnn.Tensor
    qkv_proj: ttnn.Tensor
    kv_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor
    k_norm: ttnn.Tensor
    mlp_gate: ttnn.Tensor
    mlp_up: ttnn.Tensor
    mlp_down: ttnn.Tensor


@dataclass(frozen=True)
class DFlashWeights:
    fc: ttnn.Tensor
    hidden_norm: ttnn.Tensor
    final_norm: ttnn.Tensor
    layers: list[DFlashLayerWeights]


def load_dflash_weights(
    mesh_device,
    config,
    cache_dir: str | Path | None = None,
    attention_dtype=ttnn.bfloat8_b,
    mlp_dtype=ttnn.bfloat4_b,
    fc_dtype=ttnn.bfloat8_b,
    *,
    safetensors_dir: str | Path | None = None,
) -> DFlashWeights:
    if mesh_device.get_num_devices() != 1:
        raise ValueError("Muse Glimmer DFlash supports one device only")
    if safetensors_dir is None:
        raise ValueError("safetensors_dir must point at meta-models/Muse-Glimmer-30B-assistant")

    source = LazyStateDict(Path(safetensors_dir))
    cache_root = Path(cache_dir or Path(safetensors_dir) / "tensor_cache_tt_bf16")
    cache_root.mkdir(parents=True, exist_ok=True)

    def matmul(key: str, dtype, cache_key: str | None = None):
        stem = get_cache_file_name(cache_root, cache_key or key)
        host = cached_tensor_placeholder(stem, dtype, ttnn.TILE_LAYOUT)
        if host is None:
            host = source[key].transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
        return ttnn.as_tensor(
            host,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=stem,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def norm(key: str, cache_key: str | None = None):
        stem = get_cache_file_name(cache_root, cache_key or key)
        host = cached_tensor_placeholder(stem, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        if host is None:
            host = source[key].reshape(1, 1, -1, ttnn.TILE_SIZE).contiguous()
        return ttnn.as_tensor(
            host,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=stem,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    layers = []
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        q_proj = matmul(f"{prefix}.self_attn.q_proj.weight", attention_dtype)
        k_proj = matmul(f"{prefix}.self_attn.k_proj.weight", attention_dtype)
        v_proj = matmul(f"{prefix}.self_attn.v_proj.weight", attention_dtype)
        qkv_proj = ttnn.concat([q_proj, k_proj, v_proj], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kv_proj = ttnn.concat([k_proj, v_proj], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_proj)
        ttnn.deallocate(k_proj)
        ttnn.deallocate(v_proj)
        layers.append(
            DFlashLayerWeights(
                input_layernorm=norm(f"{prefix}.input_layernorm.weight"),
                post_attention_layernorm=norm(f"{prefix}.post_attention_layernorm.weight"),
                qkv_proj=qkv_proj,
                kv_proj=kv_proj,
                o_proj=matmul(f"{prefix}.self_attn.o_proj.weight", attention_dtype),
                q_norm=norm(f"{prefix}.self_attn.q_norm.weight"),
                k_norm=norm(f"{prefix}.self_attn.k_norm.weight"),
                mlp_gate=matmul(f"{prefix}.mlp.gate_proj.weight", mlp_dtype),
                mlp_up=matmul(f"{prefix}.mlp.up_proj.weight", mlp_dtype),
                mlp_down=matmul(f"{prefix}.mlp.down_proj.weight", mlp_dtype),
            )
        )

    weights = DFlashWeights(
        fc=matmul("encoder.fc.weight", fc_dtype, "encoder.fc.weight"),
        hidden_norm=norm("encoder.output_norm_enc.weight", "encoder.output_norm_enc.weight"),
        final_norm=norm("norm.weight"),
        layers=layers,
    )
    source.close()
    return weights
