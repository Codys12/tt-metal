# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 native DFlash decode for Muse Glimmer.

The assistant consumes five target hidden-state taps, conditions a 16-token
non-causal diffusion block on cached committed anchors, and reuses the target
LM head. Position 0 is the target bonus token; positions 1..15 are drafts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.muse_glimmer.tt.attention.operations import (
    apply_per_head_norm,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)

from .config import DFlashConfig
from .fixed_cache import fixed_kv_update
from .weights import DFlashWeights, load_dflash_weights


@dataclass
class DFlashAnchorCache:
    """Per-layer K/V for committed target-token anchors."""

    # Per dflash layer ℓ: K [1, n_kv_local, length, head_dim] (k_norm'd + RoPE'd),
    # V [1, n_kv_local, length, head_dim] (plain v_proj). ``None`` until first append.
    k: list[ttnn.Tensor | None] = field(default_factory=list)
    v: list[ttnn.Tensor | None] = field(default_factory=list)
    length: int = 0
    physical_length: int = 0
    # (logical rows, tile-padded rows) for each append.  TT matmuls round a
    # short anchor chunk's sequence dimension up to a tile, so these gaps must
    # be masked and must not consume RoPE positions.
    chunks: list[tuple[int, int]] = field(default_factory=list)

    @property
    def cached_length(self):
        return sum(logical for logical, _ in self.chunks)

    @property
    def start_position(self):
        return self.length - self.cached_length


class DFlashDrafter:
    """Muse Glimmer's five-layer, single-device block-diffusion drafter."""

    def __init__(
        self,
        mesh_device,
        config: DFlashConfig,
        cache_dir: str | None = None,
        attention_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat4_b,
        fc_dtype=ttnn.bfloat8_b,
        *,
        safetensors_dir: str | None = None,
        target_lm_head,
    ):
        if mesh_device.get_num_devices() != 1:
            raise ValueError("Muse Glimmer DFlash supports one device only")
        self.mesh_device = mesh_device
        self.config = config
        self.weights: DFlashWeights = load_dflash_weights(
            mesh_device=mesh_device,
            config=config,
            cache_dir=cache_dir,
            attention_dtype=attention_dtype,
            mlp_dtype=mlp_dtype,
            fc_dtype=fc_dtype,
            safetensors_dir=safetensors_dir,
        )
        self._ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._ck_mlp = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._decode_down_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        self.target_lm_head = target_lm_head

    @staticmethod
    def _linear(x, w, kcfg, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.linear(
            x,
            w,
            memory_config=memory_config,
            compute_kernel_config=kcfg,
            core_grid=ttnn.CoreGrid(x=8, y=8),
        )

    def _head_weight(self):
        """Muse Glimmer's assistant intentionally reuses the target LM head."""
        if self.target_lm_head is None:
            raise RuntimeError("target_lm_head is required by the native Muse Glimmer assistant")
        return self.target_lm_head

    # SDPA prefill's k_chunk_size is 64 by default, so the K sequence (ctx ∥
    # noise) needs to be a multiple of 64. We pad the noise stream up to the
    # next multiple of 64; combined with `ctx_len` being a multiple of 64 the
    # full key length lands on a kchunk boundary.
    _SEQ_PAD_MULTIPLE = 64

    @staticmethod
    def _pad_seq_to_tile(t, multiple=64):
        """Pad dim 2 (seq) up to the next ``multiple``.

        TT eltwise binary ops trip "Invalid subtile broadcast type" when one
        operand's seq dim isn't a tile multiple (TILE_SIZE=32), and SDPA
        prefill requires the K sequence to be a multiple of the kernel's
        k_chunk_size (64 by default). Padding to a multiple of 64 satisfies
        both. The padded positions are zeroed; the final-output slice trims
        them off so callers see the original block_size.
        """
        cur = int(t.shape[2])
        if cur % multiple == 0:
            return t, 0
        pad_amount = multiple - (cur % multiple)
        padded = ttnn.pad(
            t,
            [(0, 0), (0, 0), (0, pad_amount), (0, 0)],
            value=0.0,
        )
        return padded, pad_amount

    @staticmethod
    def _pad_seq_to_length(t, length):
        cur = int(t.shape[2])
        if cur == length:
            return t
        if cur > length:
            raise ValueError(f"Cannot pad sequence from {cur} down to {length}")
        return ttnn.pad(t, [(0, 0), (0, 0), (0, length - cur), (0, 0)], value=0.0)

    def _build_physical_rope(self, cos, sin, chunks, real_noise, physical_noise, sk):
        """Expand logical RoPE rows into the TT cache's padded chunk layout."""

        def expand(table):
            pieces = []
            logical_offset = 0
            for logical_rows, physical_rows in chunks:
                piece = table[:, :, logical_offset : logical_offset + logical_rows, :]
                pieces.append(self._pad_seq_to_length(piece, physical_rows))
                logical_offset += logical_rows
            noise = table[:, :, logical_offset : logical_offset + real_noise, :]
            pieces.append(self._pad_seq_to_length(noise, physical_noise))
            physical = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2)
            return self._pad_seq_to_length(physical, sk)

        return expand(cos), expand(sin)

    def init_anchor_cache(self) -> DFlashAnchorCache:
        """Fresh empty anchor cache (one per server slot / logical user)."""
        cfg = self.config
        return DFlashAnchorCache(
            k=[None] * cfg.num_hidden_layers,
            v=[None] * cfg.num_hidden_layers,
            length=0,
        )

    @staticmethod
    def clone_anchor_cache(cache: DFlashAnchorCache) -> DFlashAnchorCache:
        """Take an owning snapshot for restoring a reusable prompt boundary."""
        return DFlashAnchorCache(
            k=[
                None if tensor is None else ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for tensor in cache.k
            ],
            v=[
                None if tensor is None else ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for tensor in cache.v
            ],
            length=cache.length,
            physical_length=cache.physical_length,
            chunks=list(cache.chunks),
        )

    @staticmethod
    def clear_anchor_cache(cache: DFlashAnchorCache | None):
        if cache is None:
            return
        for tensor in cache.k + cache.v:
            if tensor is not None:
                ttnn.deallocate(tensor)
        cache.k = [None] * len(cache.k)
        cache.v = [None] * len(cache.v)
        cache.length = 0
        cache.physical_length = 0
        cache.chunks.clear()

    def _project_anchor_hidden(self, aux_hiddens_concat):
        """fc + hidden_norm on concatenated aux taps → anchor hidden.

        aux_hiddens_concat: TT [1, 1, n_new, K*target_hidden]; the caller has
        concatenated the K aux target-hidden taps along the feature dim.
        Returns [1, 1, n_new, hidden].
        """
        cfg = self.config
        anchor_h = self._linear(aux_hiddens_concat, self.weights.fc, self._ck)
        if os.getenv("MUSE_APPEND_SYNC_ENABLED") == "1":
            print("MUSE_APPEND anchor_fc_start", flush=True)
            ttnn.synchronize_device(self.mesh_device)
            print("MUSE_APPEND anchor_fc_end", flush=True)
        normalized = ttnn.rms_norm(anchor_h, weight=self.weights.hidden_norm, epsilon=cfg.rms_norm_eps)
        if os.getenv("MUSE_APPEND_SYNC_ENABLED") == "1":
            print("MUSE_APPEND anchor_norm_start", flush=True)
            ttnn.synchronize_device(self.mesh_device)
            print("MUSE_APPEND anchor_norm_end", flush=True)
        return normalized

    def append_anchors(self, cache: DFlashAnchorCache, aux_hiddens_concat):
        """Append `n_new` anchors (the just-committed tokens) to the cache.

        Per layer, anchor K is `k_proj → per-head k_norm` and anchor V is plain
        `v_proj`. K-norm is per-position, so it is safe to apply on append.

        RoPE is **NOT** applied here — anchors are stored un-rotated and the full
        concatenated K is RoPE'd once per layer in `decode_step` (where the seq
        is a clean tile multiple). RoPE-at-append would call
        ``rotary_embedding`` on a sub-tile seq (n_new < 32), which pads the seq
        to 32 and desyncs K's length from V's. RoPE is per-position, so rotating
        the full K at decode time is mathematically identical.

        Args:
            cache: the slot's anchor cache (mutated in place).
            aux_hiddens_concat: TT [1, 1, n_new, K*target_hidden] for the n_new
                committed positions.
        """
        cfg = self.config
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        num_kv_local = cfg.num_key_value_heads
        n_new = int(aux_hiddens_concat.shape[2])

        anchor_h = self._project_anchor_hidden(aux_hiddens_concat)
        n_physical = int(anchor_h.shape[2])

        for i in range(cfg.num_hidden_layers):
            layer_w = self.weights.layers[i]
            kv = ttnn.linear(
                anchor_h,
                layer_w.kv_proj,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self._ck,
                core_grid=ttnn.CoreGrid(x=8, y=8),
            )
            kv_width = num_kv_local * head_dim
            k = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, n_physical, kv_width])
            v = ttnn.slice(kv, [0, 0, 0, kv_width], [1, 1, n_physical, 2 * kv_width])
            # [1, n_new, num_kv_local, head_dim] → per-head k_norm → [1, kv, n_new, hd]
            k = ttnn.reshape(k, (1, n_physical, num_kv_local, head_dim))
            v = ttnn.reshape(v, (1, n_physical, num_kv_local, head_dim))
            k = apply_per_head_norm(k, layer_w.k_norm, eps, with_scale=True)
            k = ttnn.transpose(k, 1, 2)
            v = ttnn.transpose(v, 1, 2)
            ttnn.deallocate(kv)
            if cache.k[i] is None:
                cache.k[i] = k
                cache.v[i] = v
            else:
                new_k = ttnn.concat([cache.k[i], k], dim=2)
                new_v = ttnn.concat([cache.v[i], v], dim=2)
                ttnn.deallocate(cache.k[i])
                ttnn.deallocate(cache.v[i])
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                cache.k[i] = new_k
                cache.v[i] = new_v

        ttnn.deallocate(anchor_h)
        cache.length += n_new
        cache.physical_length += n_physical
        cache.chunks.append((n_new, n_physical))
        self._trim_anchor_cache(cache)

    def _trim_anchor_cache(self, cache: DFlashAnchorCache):
        """Discard whole old append chunks outside the assistant's sliding window.

        Whole-chunk eviction preserves TT tile padding. The retained prefix can
        be at most one append chunk larger than the exact window; the attention
        mask below still excludes every out-of-window position.
        """
        keep = self.config.sliding_window - 1
        remove_physical = 0
        while len(cache.chunks) > 1 and cache.cached_length - cache.chunks[0][0] >= keep:
            _, physical = cache.chunks.pop(0)
            remove_physical += physical
        if not remove_physical:
            return
        old_physical = cache.physical_length
        for i in range(self.config.num_hidden_layers):
            new_k_view = ttnn.slice(
                cache.k[i],
                [0, 0, remove_physical, 0],
                [1, self.config.num_key_value_heads, old_physical, self.config.head_dim],
            )
            new_v_view = ttnn.slice(
                cache.v[i],
                [0, 0, remove_physical, 0],
                [1, self.config.num_key_value_heads, old_physical, self.config.head_dim],
            )
            new_k = ttnn.clone(new_k_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            new_v = ttnn.clone(new_v_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(cache.k[i])
            ttnn.deallocate(cache.v[i])
            cache.k[i], cache.v[i] = new_k, new_v
        cache.physical_length -= remove_physical

    def _build_layout_mask(self, cache, real_block: int, block_padded: int, sk: int):
        """Additive SDPA mask [1, 1, block_padded, sk], bf16, replicated.

        Non-causal within (anchors ∥ noise): every real noise query row attends
        every real anchor [0:anchor_len] and every real noise position
        [anchor_len : anchor_len + real_block]; everything past that (the 56
        noise tile-pad rows + the K round-up to a 64 multiple) is masked. The
        mask is column-only (independent of the query row), so padded query rows
        stay finite (they attend ≥1 column) and are sliced off downstream.
        """
        mask = torch.full((1, 1, block_padded, sk), -1e9, dtype=torch.float32)
        physical_offset = 0
        absolute_offset = cache.start_position
        window = self.config.sliding_window
        for logical_rows, physical_rows in cache.chunks:
            key_positions = torch.arange(absolute_offset, absolute_offset + logical_rows)
            for query in range(real_block):
                query_position = cache.length + query
                allowed = (query_position - key_positions).abs() <= window
                mask[:, :, query, physical_offset : physical_offset + logical_rows][..., allowed] = 0
            physical_offset += physical_rows
            absolute_offset += logical_rows
        mask[:, :, :, physical_offset : physical_offset + real_block] = 0
        return ttnn.from_torch(
            mask.to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    def _decode_layer_forward(self, layer_idx, hidden, cache_k, cache_v, cos_full, sin_full, cos_q, sin_q, attn_mask):
        """One dflash layer at decode time: noise Q/K/V + cached anchor K/V.

        Identical to `_layer_forward` except the K/V context comes from the cache
        (`cache_k`/`cache_v` — k_norm'd, UN-roped) and SDPA takes the explicit
        `attn_mask`. The full concatenated K is RoPE'd here (over `cos_full`,
        seq == sk); Q is RoPE'd over the noise positions (`cos_q`). `hidden` is
        the noise stream [1, 1, block_padded, hidden].
        """
        cfg = self.config
        layer_w = self.weights.layers[layer_idx]
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim
        num_heads_local = cfg.num_attention_heads
        num_kv_local = cfg.num_key_value_heads
        block = int(hidden.shape[2])

        residual = hidden
        normed = ttnn.rms_norm(hidden, weight=layer_w.input_layernorm, epsilon=eps)

        qkv = ttnn.linear(
            normed,
            layer_w.qkv_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck,
            core_grid=ttnn.CoreGrid(x=8, y=8),
        )
        ttnn.deallocate(normed)

        q_width = num_heads_local * head_dim
        kv_width = num_kv_local * head_dim
        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, block, q_width])
        k_noise = ttnn.slice(qkv, [0, 0, 0, q_width], [1, 1, block, q_width + kv_width])
        v_noise = ttnn.slice(qkv, [0, 0, 0, q_width + kv_width], [1, 1, block, q_width + 2 * kv_width])

        q = ttnn.reshape(q, (1, block, num_heads_local, head_dim))
        k_noise = ttnn.reshape(k_noise, (1, block, num_kv_local, head_dim))
        v_noise = ttnn.reshape(v_noise, (1, block, num_kv_local, head_dim))
        q = apply_per_head_norm(q, layer_w.q_norm, eps, with_scale=True)
        k_noise = apply_per_head_norm(k_noise, layer_w.k_norm, eps, with_scale=True)
        q = ttnn.transpose(q, 1, 2)
        k_noise = ttnn.transpose(k_noise, 1, 2)
        v_noise = ttnn.transpose(v_noise, 1, 2)
        ttnn.deallocate(qkv)
        # Q RoPE over the noise positions [anchor_len : anchor_len+block_padded].
        q = ttnn.experimental.rotary_embedding(q, cos_q, sin_q, None)

        # K/V = cached anchors (k_norm'd, UN-roped) ∥ this step's noise (k_norm'd,
        # UN-roped). cache_k/cache_v share a sequence length, so K and V stay
        # matched into SDPA.
        if cache_k is not None:
            k = ttnn.concat([cache_k, k_noise], dim=2)
            v = ttnn.concat([cache_v, v_noise], dim=2)
            ttnn.deallocate(k_noise)
            ttnn.deallocate(v_noise)
        else:
            k, v = k_noise, v_noise

        # Pad K/V seq to sk (== cos_full's seq, a multiple of 64); the pad is masked.
        sk = int(cos_full.shape[2])
        cur_kv = int(k.shape[2])
        if cur_kv < sk:
            pad = sk - cur_kv
            k = ttnn.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            v = ttnn.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        # RoPE the FULL K over [0:sk] (anchors at their positions, then noise, then pad).
        k = ttnn.experimental.rotary_embedding(k, cos_full, sin_full, None)

        scale = head_dim**-0.5
        sdpa_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=scale
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        sdpa_out = ttnn.experimental.nlp_concat_heads(sdpa_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out = ttnn.linear(
            sdpa_out,
            layer_w.o_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck,
            core_grid=ttnn.CoreGrid(x=8, y=8),
        )
        ttnn.deallocate(sdpa_out)
        post_attn = ttnn.add(residual, attn_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_out)

        residual2 = post_attn
        mlp_in = ttnn.rms_norm(residual2, weight=layer_w.post_attention_layernorm, epsilon=eps)
        gate = ttnn.linear(
            mlp_in,
            layer_w.mlp_gate,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck_mlp,
            core_grid=ttnn.CoreGrid(x=8, y=8),
        )
        up = ttnn.linear(
            mlp_in,
            layer_w.mlp_up,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck_mlp,
            core_grid=ttnn.CoreGrid(x=8, y=8),
        )
        ttnn.deallocate(mlp_in)
        mlp_intermediate = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        mlp_out = ttnn.linear(
            mlp_intermediate,
            layer_w.mlp_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck_mlp,
            core_grid=ttnn.CoreGrid(x=8, y=8),
        )
        ttnn.deallocate(mlp_intermediate)
        out = ttnn.add(residual2, mlp_out)
        ttnn.deallocate(residual2)
        ttnn.deallocate(mlp_out)
        return out

    def decode_step(self, cache: DFlashAnchorCache, noise_embeddings, cos_full, sin_full):
        """Draft `block_size` tokens reading the slot's anchor cache.

        Args:
            cache: the slot's anchor cache (read-only here; grown via
                `append_anchors` on commit).
            noise_embeddings: TT [1, 1, block_size, hidden] — embed([bonus,
                mask × (block_size-1)]) from dflash's own (unscaled) embed_tokens.
            cos_full, sin_full: TT [1, 1, cache.length + block_padded, head_dim] —
                RoPE for the full (anchors ∥ noise) key span, positions
                [0 : cache.length + block_padded]. Padded to a 64-multiple here.
        Returns ``draft_logits`` TT [1, 1, block_size-1, draft_vocab] for PCC
        comparison. The traced server uses ``fixed_propose_forward`` instead.
        """
        cfg = self.config
        eps = cfg.rms_norm_eps
        original_block_size = int(noise_embeddings.shape[2])
        noise, noise_pad = self._pad_seq_to_tile(noise_embeddings)
        # RMSNorm and the layer ops require TILE layout.
        noise = ttnn.to_layout(noise, ttnn.TILE_LAYOUT)
        block_padded = int(noise.shape[2])

        # K/V contains the logical anchors followed by a physically padded
        # noise block.  Round that physical length to the SDPA K chunk and pad
        # the logical ``anchors || real noise`` RoPE cache to the same extent.
        # Padding the combined logical cache directly is wrong for batch=1:
        # e.g. 17 anchors + 16 noise rounds to 64, while K/V is 17 + 64 = 81.
        key_len = cache.physical_length + block_padded
        sk = ((key_len + self._SEQ_PAD_MULTIPLE - 1) // self._SEQ_PAD_MULTIPLE) * self._SEQ_PAD_MULTIPLE
        cos_full_p, sin_full_p = self._build_physical_rope(
            cos_full,
            sin_full,
            cache.chunks,
            original_block_size,
            block_padded,
            sk,
        )
        cos_q = cos_full_p[:, :, cache.physical_length : cache.physical_length + block_padded, :]
        sin_q = sin_full_p[:, :, cache.physical_length : cache.physical_length + block_padded, :]
        attn_mask = self._build_layout_mask(cache, original_block_size, block_padded, sk)

        hidden = noise
        for i in range(cfg.num_hidden_layers):
            hidden = self._decode_layer_forward(
                i, hidden, cache.k[i], cache.v[i], cos_full_p, sin_full_p, cos_q, sin_q, attn_mask
            )
        ttnn.deallocate(attn_mask)

        draft_hidden = ttnn.rms_norm(hidden, weight=self.weights.final_norm, epsilon=eps)
        if noise_pad > 0:
            draft_hidden = draft_hidden[:, :, :original_block_size, :]
        # Drop the bonus position; lm_head over the trailing block_size-1 drafts.
        draft_hidden_for_head = draft_hidden[:, :, 1:, :]
        draft_logits = ttnn.linear(
            draft_hidden_for_head,
            self._head_weight(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck,
            core_grid=ttnn.CoreGrid(x=8, y=8),
        )
        ttnn.deallocate(draft_hidden)
        return draft_logits

    # Fixed-cache path used by the traced batch-1 server. The eager growing
    # cache above remains the prefill/PCC oracle; decode converts its retained
    # logical anchors once at the prompt boundary and thereafter uses only the
    # fixed buffers below.

    def alloc_fixed_anchor_caches(self, capacity: int):
        if capacity % 64:
            raise ValueError("DFlash fixed cache capacity must be a multiple of 64")
        shape = ttnn.Shape([1, self.config.num_key_value_heads, capacity, self.config.head_dim])

        def allocate():
            tensor = ttnn.allocate_tensor_on_device(
                shape,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill(tensor, 0.0, output_tensor=tensor)
            return tensor

        return [(allocate(), allocate()) for _ in range(self.config.num_hidden_layers)]

    @staticmethod
    def _compact_dynamic_cache(cache: DFlashAnchorCache, tensor):
        pieces = []
        physical_offset = 0
        for logical_rows, physical_rows in cache.chunks:
            pieces.append(
                ttnn.slice(
                    tensor,
                    [0, 0, physical_offset, 0],
                    [1, int(tensor.shape[1]), physical_offset + logical_rows, int(tensor.shape[3])],
                )
            )
            physical_offset += physical_rows
        if not pieces:
            return None
        # A full-range slice may alias its parent. The fixed-cache conversion
        # deallocates the compact tensor after fill_cache, so make the
        # single-chunk case owning as well; otherwise it silently frees the
        # dynamic prompt cache used by parity/debug snapshots.
        compact = (
            ttnn.clone(pieces[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if len(pieces) == 1
            else ttnn.concat(pieces, dim=2)
        )
        return compact

    def prepare_fixed_anchor_caches(self, dynamic_cache, fixed_caches, *, max_new_tokens: int):
        """Copy a prompt's retained anchors into static trace-owned caches.

        Leave room for the requested completion so no cache compaction or
        allocation can occur in the AR interval. Dropping older assistant
        anchors can only affect speculation acceptance; target verification
        remains exact.
        """
        capacity = int(fixed_caches[0][0].shape[2])
        room = min(max_new_tokens, capacity - self.config.block_size)
        retain = max(0, min(dynamic_cache.cached_length, capacity - self.config.block_size - room))
        logical_start = dynamic_cache.cached_length - retain
        for layer, (fixed_k, fixed_v) in enumerate(fixed_caches):
            compact_k = self._compact_dynamic_cache(dynamic_cache, dynamic_cache.k[layer])
            compact_v = self._compact_dynamic_cache(dynamic_cache, dynamic_cache.v[layer])
            if retain:
                if logical_start:
                    k_view = ttnn.slice(
                        compact_k,
                        [0, 0, logical_start, 0],
                        [1, self.config.num_key_value_heads, logical_start + retain, self.config.head_dim],
                    )
                    v_view = ttnn.slice(
                        compact_v,
                        [0, 0, logical_start, 0],
                        [1, self.config.num_key_value_heads, logical_start + retain, self.config.head_dim],
                    )
                else:
                    k_view, v_view = compact_k, compact_v
                ttnn.fill_cache(fixed_k, k_view, batch_idx=0)
                ttnn.fill_cache(fixed_v, v_view, batch_idx=0)
            if compact_k is not None:
                ttnn.deallocate(compact_k)
                ttnn.deallocate(compact_v)
        absolute_start = dynamic_cache.length - retain
        return retain, absolute_start

    @staticmethod
    def rotate_fixed_anchor_caches(fixed_caches, cos, sin):
        """Convert fixed prompt K buffers to their absolute-position RoPE form once."""
        for fixed_k, _ in fixed_caches:
            roped_k = apply_rope(fixed_k, cos, sin)
            ttnn.assign(roped_k, fixed_k)
            ttnn.deallocate(roped_k)

    @staticmethod
    def _write_fixed_kv(k_cache, v_cache, k, v, write_control):
        """Trace-safe contiguous K/V write controlled by ``[base, count]``."""
        fixed_kv_update(k_cache, v_cache, k, v, write_control)

    def fixed_propose_forward(
        self,
        noise,
        caches,
        write_control,
        cos_q,
        sin_q,
        attn_mask,
        decode_core_config=None,
    ):
        """Static-shape DFlash propose suitable for capture and replay."""
        cfg = self.config
        block = cfg.block_size
        heads, kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        capacity = int(caches[0][0].shape[2])
        sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
            max_cores_per_head_batch=16,
        )
        stream = (
            decode_core_config.copy(noise, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if decode_core_config is not None
            else ttnn.clone(noise, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        sub_core_grids = decode_core_config.compute_cores if decode_core_config is not None else None
        for layer, (k_cache, v_cache) in enumerate(caches):
            weights = self.weights.layers[layer]
            residual = stream
            normed = ttnn.rms_norm(stream, weight=weights.input_layernorm, epsilon=cfg.rms_norm_eps)
            qkv = ttnn.linear(
                normed,
                weights.qkv_proj,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self._ck,
                core_grid=ttnn.CoreGrid(x=8, y=8),
            )
            ttnn.deallocate(normed)
            q, k, v = split_qkv_heads_prefill(qkv, cfg, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(qkv)
            q = ttnn.slice(q, [0, 0, 0, 0], [1, heads, block, head_dim])
            k = ttnn.slice(k, [0, 0, 0, 0], [1, kv_heads, block, head_dim])
            v = ttnn.slice(v, [0, 0, 0, 0], [1, kv_heads, block, head_dim])
            q = apply_per_head_norm(
                q,
                weights.q_norm,
                cfg.rms_norm_eps,
                with_scale=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                decode_core_config=decode_core_config,
            )
            k = apply_per_head_norm(
                k,
                weights.k_norm,
                cfg.rms_norm_eps,
                with_scale=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                decode_core_config=decode_core_config,
            )
            q = apply_rope(q, cos_q, sin_q, memory_config=ttnn.L1_MEMORY_CONFIG, decode_core_config=decode_core_config)
            k = apply_rope(k, cos_q, sin_q, memory_config=ttnn.L1_MEMORY_CONFIG, decode_core_config=decode_core_config)
            # rotary_embedding exposes its tile-padded sequence (32). Trim the
            # logical 16 rows in row-major before flattening head-major Q.
            q = ttnn.to_layout(
                q,
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                sub_core_grids=sub_core_grids,
            )
            q = ttnn.slice(q, [0, 0, 0, 0], [1, heads, block, head_dim], sub_core_grids=sub_core_grids)
            self._write_fixed_kv(k_cache, v_cache, k, v, write_control)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            q_packed = ttnn.reshape(q, (1, 1, heads * block, head_dim))
            q_packed = ttnn.to_layout(
                q_packed,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                sub_core_grids=sub_core_grids,
            )
            ttnn.deallocate(q)
            # A 32-head × 16-position packed query overflows the decode
            # kernel's per-core CBs as one op. Keep whole GQA groups together.
            # Production needs four slices to stay below Blackhole's L1 CB
            # limit; the compact parity configuration uses two.
            num_parts = 4 if kv_heads >= 8 else (2 if kv_heads >= 2 else 1)
            parts = []
            heads_per_part = heads // num_parts
            kv_per_part = kv_heads // num_parts
            rows_per_part = heads_per_part * block
            for part in range(num_parts):
                row_start = part * rows_per_part
                kv_start = part * kv_per_part
                q_part = ttnn.slice(
                    q_packed,
                    [0, 0, row_start, 0],
                    [1, 1, row_start + rows_per_part, head_dim],
                    sub_core_grids=sub_core_grids,
                )
                k_part = ttnn.slice(
                    k_cache,
                    [0, kv_start, 0, 0],
                    [1, kv_start + kv_per_part, capacity, head_dim],
                    sub_core_grids=sub_core_grids,
                )
                v_part = ttnn.slice(
                    v_cache,
                    [0, kv_start, 0, 0],
                    [1, kv_start + kv_per_part, capacity, head_dim],
                    sub_core_grids=sub_core_grids,
                )
                mask_part = ttnn.slice(
                    attn_mask,
                    [0, 0, row_start, 0],
                    [1, 1, row_start + rows_per_part, capacity],
                    sub_core_grids=sub_core_grids,
                )
                parts.append(
                    ttnn.transformer.scaled_dot_product_attention_decode(
                        q_part,
                        k_part,
                        v_part,
                        is_causal=False,
                        attn_mask=mask_part,
                        scale=head_dim**-0.5,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=sdpa_pc,
                    )
                )
                for tensor in (q_part, k_part, v_part, mask_part):
                    ttnn.deallocate(tensor)
            sdpa = ttnn.concat(parts, dim=2, sub_core_grids=sub_core_grids)
            for tensor in parts:
                ttnn.deallocate(tensor)
            ttnn.deallocate(q_packed)
            sdpa = ttnn.reshape(sdpa, (1, heads, block, head_dim), sub_core_grids=sub_core_grids)
            attn_out = concat_heads(sdpa, is_decode_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(sdpa)
            attn_out = ttnn.linear(
                attn_out,
                weights.o_proj,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self._ck,
                core_grid=ttnn.CoreGrid(x=8, y=8),
            )
            stream = ttnn.add(residual, attn_out, sub_core_grids=sub_core_grids)
            ttnn.deallocate(residual)
            ttnn.deallocate(attn_out)

            residual = stream
            mlp_in = ttnn.rms_norm(stream, weight=weights.post_attention_layernorm, epsilon=cfg.rms_norm_eps)
            gate = ttnn.linear(
                mlp_in, weights.mlp_gate, compute_kernel_config=self._ck_mlp, core_grid=ttnn.CoreGrid(x=8, y=8)
            )
            up = ttnn.linear(
                mlp_in, weights.mlp_up, compute_kernel_config=self._ck_mlp, core_grid=ttnn.CoreGrid(x=8, y=8)
            )
            ttnn.deallocate(mlp_in)
            intermediate = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                sub_core_grids=sub_core_grids,
            )
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            down = ttnn.linear(
                intermediate,
                weights.mlp_down,
                compute_kernel_config=self._ck_mlp,
                program_config=self._decode_down_program_config,
            )
            ttnn.deallocate(intermediate)
            stream = ttnn.add(residual, down, sub_core_grids=sub_core_grids)
            ttnn.deallocate(residual)
            ttnn.deallocate(down)

        hidden = ttnn.rms_norm(stream, weight=self.weights.final_norm, epsilon=cfg.rms_norm_eps)
        ttnn.deallocate(stream)
        # An unaligned tiled slice lowers to the wide-block multicore
        # untilize path, whose selected rectangle can escape a sparse
        # sub-device. Use one worker for the tiny 16-row conversion, then
        # retile the 15 draft rows on the worker set.
        single_worker = (
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
            if decode_core_config is not None
            else None
        )
        hidden_rm = ttnn.untilize(
            hidden,
            use_multicore=decode_core_config is not None,
            sub_core_grids=single_worker,
        )
        drafts = ttnn.slice(
            hidden_rm,
            [0, 0, 1, 0],
            [1, 1, block, cfg.hidden_size],
            sub_core_grids=sub_core_grids,
        )
        ttnn.deallocate(hidden_rm)
        drafts = ttnn.to_layout(drafts, ttnn.TILE_LAYOUT, sub_core_grids=sub_core_grids)
        ttnn.deallocate(hidden)
        logits = ttnn.linear(
            drafts, self._head_weight(), compute_kernel_config=self._ck, core_grid=ttnn.CoreGrid(x=8, y=8)
        )
        ttnn.deallocate(drafts)
        return logits

    def fixed_append_forward(self, caches, aux_concat, write_control, cos, sin, decode_core_config=None):
        """Project and append accepted target taps into fixed caches."""

        def debug_sync(stage):
            if os.getenv("MUSE_APPEND_SYNC_ENABLED") == "1":
                print(f"MUSE_APPEND {stage}_start", flush=True)
                ttnn.synchronize_device(self.mesh_device)
                print(f"MUSE_APPEND {stage}_end", flush=True)

        cfg = self.config
        block = cfg.block_size
        anchor_h = self._project_anchor_hidden(aux_concat)
        debug_sync("anchor_projection")
        for layer, (k_cache, v_cache) in enumerate(caches):
            weights = self.weights.layers[layer]
            kv = ttnn.linear(
                anchor_h, weights.kv_proj, compute_kernel_config=self._ck, core_grid=ttnn.CoreGrid(x=8, y=8)
            )
            debug_sync(f"layer_{layer}_kv_projection")
            kv_width = cfg.num_key_value_heads * cfg.head_dim
            worker_cores = decode_core_config.compute_cores if decode_core_config else None
            k = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, block, kv_width], sub_core_grids=worker_cores)
            v = ttnn.slice(kv, [0, 0, 0, kv_width], [1, 1, block, 2 * kv_width], sub_core_grids=worker_cores)
            k = ttnn.reshape(
                k,
                (1, block, cfg.num_key_value_heads, cfg.head_dim),
                sub_core_grids=worker_cores,
            )
            v = ttnn.reshape(
                v,
                (1, block, cfg.num_key_value_heads, cfg.head_dim),
                sub_core_grids=worker_cores,
            )
            k = apply_per_head_norm(
                k,
                weights.k_norm,
                cfg.rms_norm_eps,
                with_scale=True,
                decode_core_config=decode_core_config,
            )
            debug_sync(f"layer_{layer}_head_norm")
            k = ttnn.transpose(k, 1, 2, sub_core_grids=worker_cores)
            v = ttnn.transpose(v, 1, 2, sub_core_grids=worker_cores)
            k = apply_rope(k, cos, sin, decode_core_config=decode_core_config)
            debug_sync(f"layer_{layer}_rope")
            ttnn.deallocate(kv)
            self._write_fixed_kv(k_cache, v_cache, k, v, write_control)
            debug_sync(f"layer_{layer}_cache_write")
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        ttnn.deallocate(anchor_h)
