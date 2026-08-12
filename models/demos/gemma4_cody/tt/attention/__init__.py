# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Attention module.

Uses HF-style ttnn.experimental.rotary_embedding — no Meta-format weight conversion,
no transformation matrices. Cos/sin caches are passed directly.

Supports two layer types:
- sliding_attention: head_dim=256, 8 KV heads, separate K/V, full RoPE, window=1024
- full_attention: head_dim=512, 2 KV heads, K=V tying, partial RoPE (0.25), full context
"""

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, Mode
from models.demos.gemma4_cody.tt.ccl import (
    make_fused_matmul_program_config,
    make_reduce_scatter_persistent_buffers,
)

from .weights import AttentionWeights, load_attention_weights
from .kv_cache import init_kv_cache
from .decode import decode_forward, packed_decode_forward
from .prefill import prefill_forward


class Gemma4AttentionConfig:
    """Configuration for a single attention layer, derived from HF config + layer type."""

    def __init__(self, hf_config, layer_idx):
        self.layer_type = hf_config.layer_types[layer_idx]
        self.hidden_size = hf_config.hidden_size
        self.num_attention_heads = hf_config.num_attention_heads
        self.rms_norm_eps = hf_config.rms_norm_eps

        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_kv_tying = getattr(hf_config, "attention_k_eq_v", False) and not self.is_sliding

        if self.is_sliding:
            self.num_key_value_heads = hf_config.num_key_value_heads
            self.head_dim = hf_config.head_dim
            self.sliding_window = hf_config.sliding_window
            self.rope_theta = hf_config.rope_theta
            self.partial_rotary_factor = 1.0
        else:
            # Global KV heads: use num_global_key_value_heads if set, else fall back to sliding
            global_kv = getattr(hf_config, "num_global_key_value_heads", None)
            self.num_key_value_heads = global_kv if global_kv else hf_config.num_key_value_heads
            self.head_dim = getattr(hf_config, "global_head_dim", hf_config.head_dim)
            self.sliding_window = None
            self.rope_theta = hf_config.global_rope_theta
            self.partial_rotary_factor = hf_config.partial_rotary_factor

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads


class Gemma4Attention:
    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        layer_idx,
        tensor_cache_path=None,
        create_kv_cache=False,
        max_batch_size=1,
        max_seq_len=131072,
        kv_cache_dtype=ttnn.bfloat16,
        # Legacy parameter — ignored (no longer needed with HF-style RoPE)
        transformation_mats=None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.layer_idx = layer_idx

        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=ttnn.bfloat8_b,
            tensor_cache_path=tensor_cache_path,
        )

        if create_kv_cache:
            self.kv_cache = init_kv_cache(
                mesh_device=mesh_device,
                config=config,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                cache_dtype=kv_cache_dtype,
            )
        else:
            self.kv_cache = None
        # Per-layer staging for the loop-free packed KV write — assigned by the
        # model after all layers are built (shared layers alias the source's).
        self.kv_staging = None

        # Persistent buffers for fused o_proj matmul_reduce_scatter (decode).
        # Decode shape: [1, 1, TILE-padded batch, padded_hidden]. Prefill
        # (variable seq_len) falls through to the unfused linear+all_reduce path.
        self._fused_intermediate = None
        self._fused_output = None
        self._fused_program_config = None
        tp = mesh_config.tp if mesh_config else 1
        if tp > 1 and ccl_manager is not None:
            tile_pad = max(32, ((max_batch_size + 31) // 32) * 32)
            local_hidden = config.hidden_size // tp
            padded_local_hidden = ((local_hidden + 31) // 32) * 32
            padded_hidden = padded_local_hidden * tp
            self._fused_decode_seq = tile_pad
            self._fused_intermediate, self._fused_output = make_reduce_scatter_persistent_buffers(
                mesh_device=mesh_device,
                matmul_output_shape=(1, 1, tile_pad, padded_hidden),
                tp=tp,
                dtype=ttnn.bfloat16,
            )
            # The fused matmul_reduce_scatter_async op requires an explicit
            # program_config (unlike ttnn.linear, no auto-derivation).
            # in_dim_per_device must be the o_proj matmul's *contraction* dim
            # per device (num_heads*head_dim/tp), which for Gemma4 differs
            # from hidden_size/tp — derive it from the weight itself.
            self._fused_program_config = make_fused_matmul_program_config(
                matmul_output_shape=(1, 1, tile_pad, padded_hidden),
                in_dim_per_device=self.weights.o_proj.shape[-2],
            )

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        token_index=None,
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
        Attention forward pass — dispatches to on-device decode or prefill.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            rope_mats: (cos_cache, sin_cache) TT tensors, shape [1, 1, max_seq_len, head_dim]
            position_idx: position tensor for KV cache update (decode only)
            page_table: paged attention page table
            kv_cache: [k_cache, v_cache] or None
            is_decode: True for decode mode
            token_index: int position for decode RoPE slicing (decode only)
            shared_kv: optional (tt_k, tt_v) from source layer for KV sharing (prefill only)
            keep_kv: if True, keep K/V alive for sharing with later layers (prefill only)
            is_kv_shared: if True, this layer shares KV from source (skip K/V proj + cache update)
        """
        cache = kv_cache or self.kv_cache
        cos_cache, sin_cache = rope_mats

        if is_decode and packed is not None:
            # Speculative-decode packed verify: P query positions/slot in one
            # forward. `packed` carries the per-row positions, the P cache-write
            # index tensors, and the per-layer-type head-major attn mask.
            #
            # ``rope_packed`` (optional): per-layer-type pre-gathered
            # ``(cos_bp, sin_bp)`` for the hoisted decode path. The gather
            # depends only on position_idx and the layer-type's 2D cache —
            # identical across all layers of a given type — so the server
            # computes it once per type per step and shares it. Falls back
            # to per-layer gather when absent (tests, legacy callers).
            rope_packed = None
            rope_dict = packed.get("rope_packed")
            if rope_dict is not None:
                rope_packed = rope_dict.get(self.config.layer_type)
            return packed_decode_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                position_idx=packed["position_idx"],
                kv_write_idxs=packed["kv_write_idxs"],
                kv_write_idxs_sliding=packed.get("kv_write_idxs_sliding"),
                attn_mask=packed["attn_mask"][self.config.layer_type],
                packed_p=packed["p"],
                page_table=page_table,
                page_table_sliding=page_table_sliding,
                ccl_manager=self.ccl_manager,
                rope_packed=rope_packed,
                up_kv_write=packed.get("up_kv_write", False),
                kv_write_up_masked_full=packed.get("kv_write_up_masked_full"),
                kv_write_up_masked_sliding=packed.get("kv_write_up_masked_sliding"),
                page_table_up_full=packed.get("page_table_up_full"),
                page_table_up_sliding=packed.get("page_table_up_sliding"),
                # Loop-free KV write via persistent staging. ``merge_idx`` is
                # layer-independent (staging positions, not physical pages), so
                # it is shared; only the fill destination ``hot_pt`` differs
                # full vs sliding. ``kv_staging`` is this layer's resident copy.
                kv_staging=self.kv_staging,
                merge_idx=packed.get("merge_idx"),
                hot_pt=packed.get("hot_pt"),
                hot_pt_sliding=packed.get("hot_pt_sliding"),
                # ``embedding`` merge (transpose-free row-gather); embed_idx is
                # per-head-flattened (nkv differs full vs sliding).
                kv_merge=packed.get("kv_merge", "embedding"),
                embed_idx=packed.get("embed_idx_full"),
                embed_idx_sliding=packed.get("embed_idx_sliding"),
            )

        if is_decode:
            return decode_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                position_idx=position_idx,
                token_index=token_index,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                is_kv_shared=is_kv_shared,
                position_idx_cache=position_idx_cache,
                fused_intermediate_buffer=self._fused_intermediate,
                fused_output_buffer=self._fused_output,
                fused_program_config=self._fused_program_config,
                page_table_sliding=page_table_sliding,
                position_idx_cache_sliding_write=position_idx_cache_sliding_write,
                position_idx_cache_sliding_sdpa=position_idx_cache_sliding_sdpa,
            )
        else:
            tt_out, kept_kv = prefill_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                shared_kv=shared_kv,
                keep_kv=keep_kv,
                page_table_sliding=page_table_sliding,
            )
            self._last_kv = kept_kv
            return tt_out
