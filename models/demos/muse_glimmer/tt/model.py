# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1, single-device Muse Glimmer target model."""

import os

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.muse_glimmer.tt.attention.kv_cache import PAGE_BLOCK_SIZE
from models.demos.muse_glimmer.tt.layer import MuseGlimmerDecoderLayer
from models.demos.muse_glimmer.tt.rms_norm import RMSNorm
from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder, get_cache_file_name
from models.demos.muse_glimmer.utils.substate import substate


def _default_rope_inv_freq(head_dim, theta):
    exponent = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    return (1.0 / (np.float32(theta) ** exponent)).astype(np.float32, copy=False)


def _create_rope_cache_tensors(config, max_seq_len, layer_type):
    """Create local RoPE tables and identity tables for full-attention NoPE."""
    layer_idx = list(config.layer_types).index(layer_type)
    theta = float(config.layer_rope_theta[layer_idx])
    if theta == 0.0:
        shape = (1, max_seq_len, config.head_dim)
        return torch.ones(shape), torch.zeros(shape)
    positions = np.arange(max_seq_len, dtype=np.float32)
    frequencies = np.outer(positions, _default_rope_inv_freq(config.head_dim, theta))
    angles = np.concatenate((frequencies, frequencies), axis=-1).astype(np.float32, copy=False)
    return torch.from_numpy(np.cos(angles)[None].copy()), torch.from_numpy(np.sin(angles)[None].copy())


def create_rope_caches(device, config, max_seq_len):
    prefill = {}
    decode = {}
    cpu = {}
    for layer_type in dict.fromkeys(config.layer_types):
        cos, sin = _create_rope_cache_tensors(config, max_seq_len, layer_type)
        prefill[layer_type] = (
            ttnn.from_torch(cos.unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            ttnn.from_torch(sin.unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        )
        decode[layer_type] = (
            ttnn.from_torch(cos.squeeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            ttnn.from_torch(sin.squeeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        )
        cpu[layer_type] = (cos.squeeze(0).contiguous(), sin.squeeze(0).contiguous())
    return prefill, decode, cpu


class MuseGlimmerModel:
    def __init__(
        self,
        device,
        hf_config,
        state_dict,
        attention_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat4_b,
        lm_head_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        mlp_cache_path=None,
        max_seq_len=2048,
        create_kv_cache=True,
        kv_cache_dtype=ttnn.bfloat16,
        load_qkv_prefetch=False,
    ):
        if device.get_num_devices() != 1:
            raise ValueError("Muse Glimmer supports one device only")
        self.device = device
        self.hf_config = hf_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.output_multiplier = hf_config.output_multiplier
        self.final_logit_softcapping = hf_config.final_logit_softcapping
        self.embed_norm_eps = hf_config.rms_norm_eps
        self.max_seq_len = max_seq_len
        self.page_block_size = PAGE_BLOCK_SIZE
        self._state_dict = state_dict
        self._embed_key = "model.language_model.embed_tokens.weight"
        self._embed_weight_cpu = None

        self.rope_caches, self.rope_caches_2d, self.rope_caches_cpu = create_rope_caches(device, hf_config, max_seq_len)
        page_ids = torch.arange(max_seq_len // self.page_block_size, dtype=torch.int32).reshape(1, -1)
        self.page_table = ttnn.from_torch(
            page_ids,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if not hasattr(state_dict, "get_tensor_rows"):
            self._embed_weight_cpu = state_dict[self._embed_key]

        lm_head_cache = get_cache_file_name(tensor_cache_path, "lm_head.weight")
        lm_head = cached_tensor_placeholder(lm_head_cache, lm_head_dtype, ttnn.TILE_LAYOUT)
        if lm_head is None:
            lm_head = state_dict["lm_head.weight"].transpose(0, 1).unsqueeze(0).unsqueeze(0)
        self.lm_head_weight = ttnn.as_tensor(
            lm_head,
            device=device,
            dtype=lm_head_dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=lm_head_cache,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        layer_count = hf_config.num_hidden_layers
        self.layers = []
        for layer_idx in range(layer_count):
            logger.info(f"Loading Muse Glimmer decoder layer {layer_idx + 1}/{layer_count}")
            self.layers.append(
                MuseGlimmerDecoderLayer(
                    device,
                    hf_config,
                    state_dict,
                    layer_idx,
                    attention_dtype,
                    mlp_dtype,
                    tensor_cache_path,
                    mlp_cache_path or tensor_cache_path,
                    max_seq_len,
                    create_kv_cache,
                    kv_cache_dtype,
                    load_qkv_prefetch,
                )
            )
        self.tt_kv_cache = [layer.self_attn.kv_cache for layer in self.layers]

        norm_key = "model.language_model.norm"
        norm_cache = f"{tensor_cache_path}/final_norm" if tensor_cache_path else None
        self.norm = RMSNorm(
            device,
            hf_config.hidden_size,
            substate(state_dict, norm_key),
            hf_config.rms_norm_eps,
            norm_cache,
        )
        self._aux_tap_layers = ()
        self.decode_core_config = None

    def configure_decode_cores(self, decode_core_config):
        """Enable the worker-only policy for AR verification, leaving prefill unchanged."""
        self.decode_core_config = decode_core_config
        for layer in self.layers:
            layer.decode_core_config = decode_core_config

    def configure_aux_taps(self, layer_indices):
        taps = tuple(int(layer_idx) for layer_idx in layer_indices)
        if any(layer_idx < 0 or layer_idx >= len(self.layers) for layer_idx in taps):
            raise ValueError(f"Auxiliary layer taps must be in [0, {len(self.layers)})")
        self._aux_tap_layers = taps

    def reset_prefill_state(self):
        for layer in self.layers:
            layer.self_attn.reset_prefill_state()

    def clone_prefill_state(self):
        """Take an owning snapshot of the sliding tails at a request boundary."""
        snapshot = []
        for layer in self.layers:
            tail = layer.self_attn.prefill_tail
            snapshot.append(
                None
                if tail is None
                else tuple(ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG) for tensor in tail)
            )
        return snapshot

    def restore_prefill_state(self, snapshot):
        if len(snapshot) != len(self.layers):
            raise ValueError("Invalid Muse Glimmer prefill-state snapshot")
        self.reset_prefill_state()
        for layer, tail in zip(self.layers, snapshot):
            layer.self_attn.prefill_tail = tail

    def commit_decode_tails(self, count):
        for layer in self.layers:
            layer.self_attn.commit_pending_tail(count)

    def _rope_for_layer(self, layer_idx, seq_len, is_decode, start_position=0):
        layer_type = self.hf_config.layer_types[layer_idx]
        if is_decode:
            return self.rope_caches_2d[layer_type]
        cos, sin = self.rope_caches[layer_type]
        return (
            cos[:, :, start_position : start_position + seq_len, :],
            sin[:, :, start_position : start_position + seq_len, :],
        )

    def __call__(
        self,
        hidden_states,
        *,
        is_decode=False,
        packed=None,
        return_aux_hidden=False,
        last_token_only=False,
        compute_logits=True,
    ):
        logical_seq_len = int(hidden_states.shape[2])
        if not is_decode and packed is None:
            packed = {
                "page_table": self.page_table,
                "chunk_start": 0,
                "logical_length": logical_seq_len,
            }
        if not is_decode and logical_seq_len % ttnn.TILE_SIZE:
            padded = ((logical_seq_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            hidden_states = ttnn.pad(
                hidden_states,
                [(0, 0), (0, 0), (0, padded - logical_seq_len), (0, 0)],
                value=0.0,
            )
        seq_len = int(hidden_states.shape[2])
        tap_set = set(self._aux_tap_layers) if return_aux_hidden else set()
        taps = {}
        ar_decode = is_decode and packed is not None and packed.get("retain_tail", True) is False
        for layer_idx, layer in enumerate(self.layers):
            if ar_decode and os.getenv("MUSE_QKV_DEBUG") == "1":
                print(f"MUSE_QKV_DEBUG layer={layer_idx}", flush=True)
            start_position = 0 if is_decode else int(packed["chunk_start"])
            hidden_states = layer(
                hidden_states,
                self._rope_for_layer(layer_idx, seq_len, is_decode, start_position),
                kv_cache=self.tt_kv_cache[layer_idx],
                is_decode=is_decode,
                packed=packed,
                ar_decode=ar_decode,
            )
            if ar_decode and os.getenv("MUSE_QKV_SYNC_DEBUG") == "1":
                ttnn.synchronize_device(
                    self.device,
                    sub_device_ids=[self.decode_core_config.worker_sub_device_id],
                )
            if ar_decode and os.getenv("MUSE_QKV_DEBUG") == "1":
                print(f"MUSE_QKV_DEBUG layer_done={layer_idx}", flush=True)
            if layer_idx in tap_set:
                if ar_decode and os.getenv("MUSE_QKV_DEBUG") == "1":
                    print(f"MUSE_QKV_DEBUG tap={layer_idx}", flush=True)
                taps[layer_idx] = (
                    self.decode_core_config.target_copy(hidden_states, memory_config=hidden_states.memory_config())
                    if ar_decode and self.decode_core_config is not None
                    else ttnn.clone(hidden_states)
                )

        if not compute_logits:
            hidden_states.deallocate(True)
            if return_aux_hidden:
                return None, None, [taps[layer_idx] for layer_idx in self._aux_tap_layers]
            return None
        if last_token_only:
            last_index = int(packed["real_p"]) - 1 if is_decode else logical_seq_len - 1
            last = ttnn.slice(
                hidden_states,
                [0, 0, last_index, 0],
                [1, 1, last_index + 1, self.hidden_size],
            )
            hidden_states.deallocate(True)
            hidden_states = last
        if ar_decode and os.getenv("MUSE_QKV_DEBUG") == "1":
            print("MUSE_QKV_DEBUG final_norm", flush=True)
        hidden_states = self.norm.forward(hidden_states, decode_sharded=ar_decode)
        if ar_decode and os.getenv("MUSE_QKV_DEBUG") == "1":
            print("MUSE_QKV_DEBUG lm_head", flush=True)
        logits = ttnn.linear(
            hidden_states,
            self.lm_head_weight,
            core_grid=self.decode_core_config.target_compute_grid
            if ar_decode and self.decode_core_config is not None
            else None,
        )
        hidden_states.deallocate(True)
        decode_cores = (
            self.decode_core_config.target_compute_cores if ar_decode and self.decode_core_config is not None else None
        )
        logits = ttnn.mul(logits, self.output_multiplier, sub_core_grids=decode_cores)
        if self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap, sub_core_grids=decode_cores)
            logits = ttnn.tanh(logits, sub_core_grids=decode_cores)
            logits = ttnn.mul(logits, cap, sub_core_grids=decode_cores)
        if return_aux_hidden:
            return logits, None, [taps[layer_idx] for layer_idx in self._aux_tap_layers]
        return logits

    def _embedding_rows(self, input_ids):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        flat = input_ids.reshape(-1)
        if self._embed_weight_cpu is not None:
            rows = self._embed_weight_cpu.index_select(0, flat)
        else:
            rows = self._state_dict.get_tensor_rows(self._embed_key, flat)
        return rows.reshape(*input_ids.shape, self.hidden_size)

    def embed_input_ids(self, input_ids):
        embeddings = self._embedding_rows(input_ids).float()
        embeddings *= torch.rsqrt(embeddings.square().mean(dim=-1, keepdim=True) + self.embed_norm_eps)
        return ttnn.from_torch(
            embeddings.to(torch.bfloat16).unsqueeze(0),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    def raw_token_embeddings(self, input_ids):
        return self._embedding_rows(input_ids).to(torch.bfloat16)
