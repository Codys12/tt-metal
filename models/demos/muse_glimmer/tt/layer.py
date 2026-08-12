# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer dense decoder layer."""

import os

import ttnn
from models.demos.muse_glimmer.tt.attention import MuseGlimmerAttention, MuseGlimmerAttentionConfig
from models.demos.muse_glimmer.tt.rms_norm import RMSNorm
from models.demos.muse_glimmer.tt.shared_mlp import SharedMLP
from models.demos.muse_glimmer.utils.substate import substate


class MuseGlimmerDecoderLayer:
    def __init__(
        self,
        device,
        hf_config,
        state_dict,
        layer_idx,
        attention_dtype,
        mlp_dtype,
        tensor_cache_path,
        mlp_cache_path,
        max_seq_len,
        create_kv_cache=False,
        kv_cache_dtype=ttnn.bfloat16,
        load_qkv_prefetch=False,
    ):
        layer_state = substate(state_dict, f"model.language_model.layers.{layer_idx}")

        def norm(name, eps):
            cache = f"{tensor_cache_path}/layer_{layer_idx}/{name}" if tensor_cache_path else None
            return RMSNorm(
                device,
                hf_config.hidden_size,
                substate(layer_state, name),
                eps,
                cache,
                centered=True,
            )

        self.input_layernorm = norm("input_layernorm", hf_config.rms_norm_eps)
        self.post_attention_layernorm = norm("post_attention_layernorm", hf_config.post_norm_eps)
        self.pre_feedforward_layernorm = norm("pre_feedforward_layernorm", hf_config.rms_norm_eps)
        self.post_feedforward_layernorm = norm("post_feedforward_layernorm", hf_config.post_norm_eps)

        attention_cache = f"{tensor_cache_path}/layer_{layer_idx}/self_attn" if tensor_cache_path else None
        self.self_attn = MuseGlimmerAttention(
            device,
            MuseGlimmerAttentionConfig(hf_config, layer_idx),
            substate(layer_state, "self_attn"),
            attention_cache,
            max_seq_len,
            create_kv_cache,
            kv_cache_dtype,
            attention_dtype,
            load_qkv_prefetch,
        )
        mlp_cache = f"{mlp_cache_path}/layer_{layer_idx}/mlp" if mlp_cache_path else None
        self.mlp = SharedMLP(
            device,
            substate(layer_state, "mlp"),
            mlp_dtype,
            mlp_cache,
            load_prefetch=load_qkv_prefetch and os.getenv("MUSE_PREFETCH_MLP", "0") == "1",
        )
        self.decode_core_config = None
        self.layer_idx = layer_idx
        self.num_layers = hf_config.num_hidden_layers

    def __call__(self, hidden_states, rope_mats, *, kv_cache=None, is_decode=False, packed=None, ar_decode=False):
        def sync_stage(stage):
            if ar_decode and os.getenv("MUSE_QKV_SYNC_DEBUG") == "1":
                ttnn.synchronize_device(
                    self.self_attn.config.device,
                    sub_device_ids=[self.decode_core_config.worker_sub_device_id],
                )
                print(f"MUSE_QKV_SYNC {stage}", flush=True)

        residual = hidden_states
        normalized = self.input_layernorm.forward(hidden_states, decode_sharded=ar_decode)
        sync_stage("input_norm")
        if ar_decode and self.decode_core_config is not None and self.mlp.gate_proj_prefetch is not None:
            self.decode_core_config.next_projection_weight = self.mlp.gate_proj_prefetch[0]
        elif ar_decode and self.decode_core_config is not None:
            self.decode_core_config.next_projection_weight = (
                self.self_attn.weights.wqkv_prefetch[0] if self.layer_idx + 1 < self.num_layers else None
            )
        attention = self.self_attn(
            normalized,
            rope_mats,
            is_decode=is_decode,
            kv_cache=kv_cache,
            packed=packed,
            decode_core_config=self.decode_core_config if ar_decode else None,
        )
        sync_stage("attention")
        normalized.deallocate(True)
        attention = self.post_attention_layernorm.forward(attention, decode_sharded=ar_decode)
        sync_stage("post_attention_norm")
        hidden_states = ttnn.add(
            residual,
            attention,
            sub_core_grids=self.decode_core_config.target_compute_cores
            if ar_decode and self.decode_core_config
            else None,
        )
        residual.deallocate(True)
        attention.deallocate(True)
        sync_stage("attention_residual")

        residual = hidden_states
        normalized = self.pre_feedforward_layernorm.forward(hidden_states, decode_sharded=ar_decode)
        sync_stage("pre_feedforward_norm")
        feed_forward = self.mlp(normalized, decode_core_config=self.decode_core_config if ar_decode else None)
        sync_stage("mlp")
        normalized.deallocate(True)
        feed_forward = self.post_feedforward_layernorm.forward(feed_forward, decode_sharded=ar_decode)
        sync_stage("post_feedforward_norm")
        hidden_states = ttnn.add(
            residual,
            feed_forward,
            sub_core_grids=self.decode_core_config.target_compute_cores
            if ar_decode and self.decode_core_config
            else None,
        )
        residual.deallocate(True)
        feed_forward.deallocate(True)
        sync_stage("feedforward_residual")
        return hidden_states
