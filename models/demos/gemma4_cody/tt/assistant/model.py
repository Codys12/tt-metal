# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4Assistant (drafter) TT model — forward implementation.

Per-layer pipeline mirrors cody's target ``Gemma4DecoderLayer`` so each
device's work matches a target layer 1:1:

  * Norms route through the shared ``RMSNorm`` adapter with
    ``enable_sharded_decode=True`` (width-sharded multi-core kernel,
    ≈32 cores at hidden=1024 vs ≈4 in the interleaved fallback).
  * ``q_proj`` is column-parallel; SDPA runs against the target's per-device
    KV-cache shard.
  * ``o_proj`` is row-parallel: the local SDPA output is already heads-sharded,
    so we drop the redundant all-gather and feed o_proj its native sharded
    input; an ``all_reduce`` sums partial outputs back to the replicated
    residual stream.
  * ``mlp_gate`` / ``mlp_up`` are column-parallel (each device does
    ``intermediate/tp`` of the GeGLU compute); ``mlp_down`` is row-parallel
    + all_reduce.

Implementation notes
--------------------

  1. **Q-only attention** — no fused QKV. We do ``ttnn.linear`` against
     ``q_proj`` directly and use the externally-supplied K, V.
  2. **Per-layer-type head_dim** — sliding layers (0..2) use head_dim=256;
     full-attention layer (3) uses head_dim=512. We thread that through
     the reshape + RoPE + SDPA calls.
  3. **No K/V cache writes** — we never call ``paged_update_cache``. The
     target's KV is read-only from the drafter's perspective.

Activation: ``gelu_pytorch_tanh`` per config ⇒ ``ttnn.gelu(..., fast_and_approximate_mode=True)``.

When ``shared_kv`` is provided as raw (non-paged) K/V tensors (test path),
SDPA falls back to ``scaled_dot_product_attention_decode``. The production
server passes the target's paged caches and page tables, in which case
SDPA uses ``paged_scaled_dot_product_attention_decode``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import ttnn
from models.demos.gemma4_cody.tt.attention.operations import apply_per_head_norm, apply_rope
from models.demos.gemma4_cody.tt.ccl import ccl_allgather, ccl_allreduce

from .config import Gemma4AssistantConfig
from .weights import DrafterWeights, load_drafter_weights


class Gemma4AssistantModel:
    """TT drafter model. 4 layers (3 sliding + 1 full), Q-only attention."""

    def __init__(
        self,
        mesh_device,
        config: Gemma4AssistantConfig,
        cache_dir: str | None = None,
        weight_dtype=ttnn.bfloat16,
        mesh_config=None,
        ccl_manager=None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        # TP factor — the drafter's attention is column-parallel on the query
        # heads so each device's SDPA matches the target's per-device KV shard.
        self.tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.weights: DrafterWeights = load_drafter_weights(
            mesh_device=mesh_device,
            config=config,
            cache_dir=cache_dir,
            weight_dtype=weight_dtype,
            mesh_config=mesh_config,
        )
        # HiFi2 + bf16 dest accumulation for the matmuls / SDPA, matching the
        # ttnn SDPA default profile.
        self._ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Row-parallel MLP/o_proj path is only usable when we have full CCL
        # plumbing (mesh_config + ccl_manager). Without it, weights.py loaded
        # those matrices as replicated and we run the legacy ccl_allgather
        # path before o_proj instead. ``weights.py`` makes the same decision
        # — keep these in lockstep.
        self._row_parallel = self.tp > 1 and mesh_config is not None and ccl_manager is not None

    def _all_reduce(self, tensor):
        """All-reduce across TP devices, summing partial outputs.

        Used after row-parallel ``o_proj`` and ``mlp_down``; on TP=1 this
        is a no-op so the same call site works in single-device tests.
        """
        if not self._row_parallel:
            return tensor
        return ccl_allreduce(tensor, self.mesh_config, self.ccl_manager)

    def _layer_forward(
        self,
        layer_idx: int,
        residual,
        shared_K,
        shared_V,
        cos_pos,
        sin_pos,
        cur_pos_tensor,
        page_table=None,
    ):
        """Single-layer forward.

        Args:
            residual: TT [1, 1, B, hidden_size] — replicated across TP.
            shared_K, shared_V: TT tensors. Two supported formats:
                - Raw (non-paged): shape [B, nkv_local, kv_len, head_dim_for_layer]
                  (already RoPE'd and per-head-normed by the target).
                - Paged: same paged-KV-cache tensors that cody's target uses
                  for this layer-type's deepest layer. Must be paired with
                  ``page_table``.
            cos_pos, sin_pos: TT [1, 1, B, head_dim] (per-slot RoPE).
            cur_pos_tensor: TT int32 [B] for SDPA cur_pos.
            page_table: TT page-table tensor. When provided, SDPA uses the
                paged variant (production cody path).
        """
        cfg = self.config
        eps = cfg.rms_norm_eps
        head_dim = cfg.layer_head_dim(layer_idx)
        num_heads = cfg.num_attention_heads
        # q_proj is column-parallel: this device holds num_heads/tp query heads.
        num_heads_local = num_heads // self.tp
        layer_w = self.weights.layers[layer_idx]
        is_sliding = cfg.layer_types[layer_idx] == "sliding_attention"
        _dbg = getattr(self, "_dbg_capture", None)
        _cap = _dbg is not None and layer_idx == getattr(self, "_dbg_layer", 0)

        # ─── Attention block ────────────────────────────────────────────────
        normed = layer_w.input_layernorm.forward(residual)
        if _cap:
            # Capture every device shard as a list — the parity test reassembles
            # column-parallel tensors. Device 0 alone is one TP shard, not the
            # full tensor (see test_assistant_parity._dbg_full).
            _dbg["ln1"] = [ttnn.to_torch(_t) for _t in ttnn.get_device_tensors(normed)]

        # Q projection (column-parallel): [1,1,B,hidden] → [1,1,B,num_heads_local*head_dim].
        # Output forced to DRAM — the per-head norm and (especially) the
        # experimental rotary_embedding kernel below do not handle
        # interleaved-L1 input cleanly and the failure mode is a SIGILL,
        # not a catchable validation error. Mirrors the target's
        # ``tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)``
        # in ``tt/attention/decode.py`` between the split and the per-head
        # norm. The tensor is small (B*q_dim_local bf16 ≈ 64 KB / device on
        # the drafter) so the DRAM hop is cheap.
        q = ttnn.linear(
            normed,
            layer_w.q_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._ck,
        )
        ttnn.deallocate(normed)
        if _cap:
            _dbg["q_proj"] = [ttnn.to_torch(_t) for _t in ttnn.get_device_tensors(q)]

        # Reshape to [1, B, num_heads_local, head_dim] for per-head norm.
        B = residual.shape[2]
        q = ttnn.reshape(q, (1, B, num_heads_local, head_dim))

        # Per-head Q norm.
        q = apply_per_head_norm(q, layer_w.q_norm, eps, with_scale=True)
        if _cap:
            _dbg["q_norm"] = [ttnn.to_torch(_t) for _t in ttnn.get_device_tensors(q)]

        # RoPE.
        q = apply_rope(q, cos_pos, sin_pos, token_index=0)
        if _cap:
            _dbg["q_rope"] = [ttnn.to_torch(_t) for _t in ttnn.get_device_tensors(q)]

        # SDPA: drafter Q is [1, B, num_heads_local, head_dim]; K/V from caller.
        sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4) if head_dim >= 512 else ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
            max_cores_per_head_batch=16,
        )
        sliding_window = cfg.sliding_window if is_sliding else None
        if page_table is not None:
            # Production path: K/V live in the target's paged cache.
            sdpa_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                shared_K,
                shared_V,
                cur_pos_tensor=cur_pos_tensor,
                page_table_tensor=page_table,
                scale=1.0,
                sliding_window_size=sliding_window,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=sdpa_pc,
                compute_kernel_config=self._ck,
            )
        else:
            # Non-paged path: raw K, V tensors. Used by parity / smoke tests.
            sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                shared_K,
                shared_V,
                cur_pos_tensor=cur_pos_tensor,
                scale=1.0,
                sliding_window_size=sliding_window,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=sdpa_pc,
                compute_kernel_config=self._ck,
            )
        ttnn.deallocate(q)
        if _cap:
            _dbg["sdpa_raw"] = [ttnn.to_torch(_t) for _t in ttnn.get_device_tensors(sdpa_out)]

        # Merge the head axis into the hidden axis. On the row-parallel path
        # the SDPA output [1, B, num_heads_local, head_dim] reshapes directly
        # into o_proj's per-device input [1, 1, B, num_heads_local*head_dim]
        # — no all-gather. On the legacy path (no CCL) o_proj is replicated
        # and needs the full ``num_heads*head_dim`` input, so we all-gather
        # the sharded heads back to full width first.
        sdpa_out_shape = sdpa_out.shape  # [1, B, num_heads_local, head_dim]
        B_dim = int(sdpa_out_shape[1])
        merged_width = int(sdpa_out_shape[2]) * int(sdpa_out_shape[3])
        sdpa_out = ttnn.reshape(sdpa_out, (1, 1, B_dim, merged_width))
        if not self._row_parallel and self.tp > 1:
            sdpa_out = ccl_allgather(sdpa_out, self.mesh_config, self.ccl_manager, dim=3)

        # o_proj: row-parallel + all-reduce when CCL is wired, replicated
        # matmul otherwise. Same pattern as the target's attention tail
        # (apply_fused_output_projection_and_allreduce → unfused fallback
        # in ccl.py:ccl_matmul_reduce_scatter_allgather when persistent
        # buffers are absent).
        attn_out = ttnn.linear(sdpa_out, layer_w.o_proj, compute_kernel_config=self._ck)
        ttnn.deallocate(sdpa_out)
        attn_out = self._all_reduce(attn_out)
        if _cap:
            _dbg["o_proj"] = [ttnn.to_torch(_t) for _t in ttnn.get_device_tensors(attn_out)]

        # post_attention_layernorm + residual
        attn_out = layer_w.post_attention_layernorm.forward(attn_out)
        post_attn = ttnn.add(residual, attn_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_out)

        # ─── MLP block ──────────────────────────────────────────────────────
        residual2 = post_attn
        mlp_in = layer_w.pre_feedforward_layernorm.forward(residual2)

        # gate / up are column-parallel: per-device output is
        # [1, 1, B, intermediate/tp]. No comm needed; the elementwise
        # gelu * up stays sharded.
        gate = ttnn.linear(mlp_in, layer_w.mlp_gate, compute_kernel_config=self._ck)
        up = ttnn.linear(mlp_in, layer_w.mlp_up, compute_kernel_config=self._ck)
        ttnn.deallocate(mlp_in)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
        mlp_intermediate = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # Row-parallel down: weight is [intermediate/tp, hidden], partial-sum
        # output, all-reduce restores the replicated residual stream.
        mlp_out = ttnn.linear(mlp_intermediate, layer_w.mlp_down, compute_kernel_config=self._ck)
        ttnn.deallocate(mlp_intermediate)
        mlp_out = self._all_reduce(mlp_out)

        mlp_out = layer_w.post_feedforward_layernorm.forward(mlp_out)
        out = ttnn.add(residual2, mlp_out)
        ttnn.deallocate(residual2)
        ttnn.deallocate(mlp_out)

        # layer_scalar is a single scalar weight; multiply the output by it.
        out = ttnn.mul(out, layer_w.layer_scalar)
        return out

    def forward(
        self,
        target_last_hidden,
        shared_kv: Dict[str, Tuple],
        cos_pos_full,
        sin_pos_full,
        cos_pos_sliding,
        sin_pos_sliding,
        cur_pos_tensor,
        page_table_full=None,
        page_table_sliding=None,
        return_intermediates=False,
        cur_pos_tensor_sliding=None,
    ):
        """Run the drafter end-to-end against the target's hidden + KV state.

        Args:
            target_last_hidden: TT tensor [1, 1, B, 2*backbone_hidden=10752].
                The target's last-layer hidden state concatenated with its
                next-token embedding (the HF reference does this — see
                modeling_gemma4_assistant.py for the exact pairing).
            shared_kv: dict mapping layer_type → (K, V) TT tensors.
                K, V shape: [1, B, nkv_local, head_dim_for_layer].
            cos_pos_full, sin_pos_full: RoPE cos/sin for full-attention layer
                (head_dim=512). [1, 1, B, head_dim_full].
            cos_pos_sliding, sin_pos_sliding: RoPE cos/sin for sliding layers
                (head_dim=256). [1, 1, B, head_dim_sliding].
            cur_pos_tensor: TT int32 [B] — per-slot SDPA position for the
                full-attention layer (raw cur_pos = target KV extent).
            cur_pos_tensor_sliding: TT int32 [B] — per-slot SDPA position for
                the sliding layers. The target's sliding KV is a ring buffer of
                W=1024; its SDPA reads use ``min(cur_pos, W-1)`` so the kernel
                walks the whole ring. The drafter reads that same ring and must
                use the same clamped position — passing raw cur_pos here reads
                wrong ring slots once cur_pos >= W. Defaults to ``cur_pos_tensor``
                when None (correct only while cur_pos < W, e.g. parity tests).

        Returns:
            (out_hidden, logits) where
                out_hidden: TT [1, 1, B, backbone_hidden=5376]
                logits: TT [1, 1, B, vocab_size] — TP-sharded on the vocab dim
                    (the lm-head / embed_tokens weight is column-parallel, like
                    the target's). Each device holds [1,1,B,vocab/tp]; the
                    consumer reconstructs the global argmax across shards.
        """
        # pre_projection: [1, 1, B, 2*backbone] → [1, 1, B, hidden]. The
        # weight is replicated, so each device gets the full replicated
        # residual stream — same starting state as a target decoder layer.
        h = ttnn.linear(
            target_last_hidden,
            self.weights.pre_projection,
            compute_kernel_config=self._ck,
        )

        intermediates = []
        if return_intermediates:
            intermediates.append(("pre_projection", ttnn.to_torch(ttnn.get_device_tensors(h)[0])))

        if cur_pos_tensor_sliding is None:
            cur_pos_tensor_sliding = cur_pos_tensor

        for i in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[i]
            K, V = shared_kv[layer_type]
            if layer_type == "full_attention":
                cos_pos, sin_pos = cos_pos_full, sin_pos_full
                page_table = page_table_full
                layer_cur_pos = cur_pos_tensor
            else:
                cos_pos, sin_pos = cos_pos_sliding, sin_pos_sliding
                page_table = page_table_sliding
                layer_cur_pos = cur_pos_tensor_sliding
            h = self._layer_forward(i, h, K, V, cos_pos, sin_pos, layer_cur_pos, page_table=page_table)
            if return_intermediates:
                intermediates.append((f"layer_{i}", ttnn.to_torch(ttnn.get_device_tensors(h)[0])))

        # Final norm (shared sharded-decode adapter).
        h = self.weights.final_norm.forward(h)

        # post_projection: [1, 1, B, hidden] → [1, 1, B, backbone]
        out_hidden = ttnn.linear(h, self.weights.post_projection, compute_kernel_config=self._ck)

        # Tied lm_head: logits = h @ embed_tokens.T
        # embed_tokens is column-parallel ([1,1,hidden,vocab/tp] per device);
        # output is [1,1,B,vocab/tp] sharded — consumer reconstructs global
        # argmax via the cross-shard topk path (server._build_drafter_fwd).
        logits = ttnn.linear(h, self.weights.embed_tokens, compute_kernel_config=self._ck)

        if return_intermediates:
            intermediates.append(("final_norm", ttnn.to_torch(ttnn.get_device_tensors(h)[0])))
            return out_hidden, logits, intermediates
        return out_hidden, logits
