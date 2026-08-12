# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer target attention: causal prefill and packed verification only."""

import math

import torch
import ttnn

from .decode import packed_decode_forward
from .kv_cache import PAGE_BLOCK_SIZE, init_kv_cache
from .prefill import prefill_forward
from .weights import load_attention_weights


class MuseGlimmerAttentionConfig:
    def __init__(self, model_config, layer_idx):
        self.layer_idx = layer_idx
        self.num_hidden_layers = model_config.num_hidden_layers
        self.layer_type = model_config.layer_types[layer_idx]
        self.hidden_size = model_config.hidden_size
        self.num_attention_heads = model_config.num_attention_heads
        self.num_key_value_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim
        self.rms_norm_eps = model_config.rms_norm_eps
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = model_config.sliding_window if self.is_sliding else None
        self.has_rope = bool(model_config.layer_rope_theta[layer_idx])
        self.attention_scale = model_config.qk_scale_factor / math.sqrt(self.head_dim)


class MuseGlimmerAttention:
    def __init__(
        self,
        device,
        config,
        state_dict,
        tensor_cache_path=None,
        max_seq_len=2048,
        create_kv_cache=False,
        kv_cache_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        load_qkv_prefetch=False,
    ):
        if device.get_num_devices() != 1:
            raise ValueError("Muse Glimmer attention supports one device only")
        self.device = device
        self.config = config
        self.config.device = device
        self.weights = load_attention_weights(
            device,
            state_dict,
            weight_dtype,
            tensor_cache_path,
            load_qkv_prefetch=load_qkv_prefetch,
        )
        self.kv_cache = init_kv_cache(device, config, max_seq_len, kv_cache_dtype) if create_kv_cache else None
        self.page_table = ttnn.from_torch(
            torch.arange(max_seq_len // PAGE_BLOCK_SIZE, dtype=torch.int32).reshape(1, -1),
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.prefill_tail = None
        self.pending_tail = None

    def reset_prefill_state(self):
        for pair in (self.prefill_tail, self.pending_tail):
            if pair is not None:
                for tensor in pair:
                    ttnn.deallocate(tensor)
        self.prefill_tail = None
        self.pending_tail = None

    def commit_pending_tail(self, count):
        if self.pending_tail is None:
            return
        if count < 0 or count > int(self.pending_tail[0].shape[2]):
            raise ValueError("Invalid packed-tail commit length")
        if count:
            pieces = []
            for old, pending in zip(self.prefill_tail or (None, None), self.pending_tail):
                new_view = ttnn.slice(
                    pending, [0, 0, 0, 0], [1, self.config.num_key_value_heads, count, self.config.head_dim]
                )
                # Full-range slices can be views.  Keep an owning DRAM copy
                # before pending_tail is released below.
                new = ttnn.clone(new_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if old is not None:
                    combined = ttnn.concat([old, new], dim=2)
                    ttnn.deallocate(old)
                    ttnn.deallocate(new)
                else:
                    combined = new
                keep = min(self.config.sliding_window, int(combined.shape[2]))
                if keep < int(combined.shape[2]):
                    trimmed_view = ttnn.slice(
                        combined,
                        [0, 0, int(combined.shape[2]) - keep, 0],
                        [1, self.config.num_key_value_heads, int(combined.shape[2]), self.config.head_dim],
                    )
                    trimmed = ttnn.clone(trimmed_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(combined)
                    combined = trimmed
                pieces.append(combined)
            self.prefill_tail = tuple(pieces)
        for tensor in self.pending_tail:
            ttnn.deallocate(tensor)
        self.pending_tail = None

    def __call__(self, hidden_states, rope_mats, *, is_decode, kv_cache=None, packed=None, decode_core_config=None):
        cache = kv_cache or self.kv_cache
        cos, sin = rope_mats
        if not is_decode:
            if packed is None:
                packed = {
                    "page_table": self.page_table,
                    "chunk_start": 0,
                    "logical_length": int(hidden_states.shape[2]),
                }
            output, self.prefill_tail = prefill_forward(
                hidden_states,
                cos,
                sin,
                self.weights,
                cache,
                self.config,
                page_table=packed["page_table"],
                chunk_start=packed["chunk_start"],
                logical_length=packed["logical_length"],
                previous_tail=self.prefill_tail,
            )
            return output
        if packed is None:
            raise ValueError("Muse Glimmer decode requires a packed DFlash verification block")
        rope_packed = packed.get("rope_packed", {}).get(self.config.layer_type)
        output, self.pending_tail = packed_decode_forward(
            hidden_states,
            cos,
            sin,
            self.weights,
            cache,
            self.config,
            self.device,
            packed["position_idx"],
            packed["cur_pos"],
            packed["page_table"],
            packed["page_index"],
            packed["page_offset"],
            packed["p"],
            packed["real_p"],
            rope_packed,
            packed.get("retain_tail", True),
            decode_core_config,
        )
        return output


__all__ = ["MuseGlimmerAttention", "MuseGlimmerAttentionConfig", "PAGE_BLOCK_SIZE"]
