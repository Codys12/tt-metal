# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DiffusionGemma on TT hardware — block-diffusion wrapper around Gemma4Model.

Architecture facts (vendored from transformers 5.11):
- Encoder = AR Gemma4 MoE stack; weights tied with decoder except per-layer
  `layer_scalar`. Encoder builds prompt KV.
- Decoder = same layers, bidirectional self-attention over a canvas, reading
  encoder KV (sliding layers see the last (window-1) prefix tokens).
- Decoder inputs = embed(canvas) + self-conditioning MLP(softmax(prev logits)
  @ embed * scale), then RMSNorm (no scale).
- attention scale 1.0; final logit softcap 30.0; lm_head tied to embeddings.

Implementation: encoder pass = standard causal prefill capturing dense
per-layer K/V (keep_kv); decoder pass = 256-token non-causal SDPA over
[prefix-slice + canvas K/V]. No paged caches; whole prefix re-encoded each
block (simple bring-up; canvas is 256 so blocks are few).
"""

import json
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.demos.gemma4_cody.tt.attention.operations import (
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from models.demos.gemma4_cody.tt.ccl import CCLManager, ccl_allgather, ccl_allreduce, make_block_sharded_matmul_config
from models.demos.gemma4_cody.tt.model import Gemma4Model, _create_rope_cache_tensors
from models.demos.gemma4_cody.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_cody.tt.shared_mlp import _mlp_kernel_config
from models.demos.gemma4_cody.utils.lazy_state_dict import LazyStateDict

NEG_INF = -1e9


# ── Config / state-dict adaptation ────────────────────────────────────────


def load_diffusion_config(model_path):
    """Parse gemma-diff config.json without AutoConfig (model_type unknown to venv HF).

    Returns (model_args, hf_text_ns, canvas_length).
    """
    with open(f"{model_path}/config.json") as f:
        cfg = json.load(f)
    tc = dict(cfg["text_config"])

    ns = SimpleNamespace(**tc)
    args = Gemma4ModelArgs.from_hf_config(ns)
    # v_proj is absent on full-attention layers => K==V tying, like gemma4 26B.
    args.attention_k_eq_v = True
    args.enable_moe_block = True
    args.num_kv_shared_layers = 0
    args._hf_text_config = ns  # rope cache creation reads layer_types/rope_parameters
    return args, ns, cfg.get("canvas_length", 256)


class DiffusionStateDictView:
    """Lazy key-translating view: gemma4_cody names -> diffusion checkpoint names."""

    _MAP = (
        ("model.layers.", "model.decoder.layers."),
        ("model.embed_tokens.", "model.decoder.embed_tokens."),
        ("model.norm.", "model.decoder.norm."),
    )

    def __init__(self, sd, prefix=""):
        self._sd = sd
        self._prefix = prefix

    def _real(self, key):
        full = self._prefix + key
        for new, old in self._MAP:
            if full.startswith(new):
                return old + full[len(new) :]
        return full

    def _virtual(self, real_key):
        for new, old in self._MAP:
            if real_key.startswith(old):
                return new + real_key[len(old) :]
        return real_key

    def substate(self, key):
        return DiffusionStateDictView(self._sd, f"{self._prefix}{key}.")

    def __getitem__(self, key):
        return self._sd[self._real(key)]

    def __contains__(self, key):
        return self._real(key) in self._sd

    def __iter__(self):
        plen = len(self._prefix)
        for rk in self._sd:
            vk = self._virtual(rk)
            if vk.startswith(self._prefix):
                yield vk[plen:]

    def __len__(self):
        return sum(1 for _ in self)

    def keys(self):
        return list(self)

    def get_tensor_rows(self, key, row_indices):
        return self._sd.get_tensor_rows(self._real(key), row_indices)


# ── TT model ──────────────────────────────────────────────────────────────


class TTDiffusionGemma:
    def __init__(
        self,
        mesh_device,
        model_path,
        num_layers=None,
        max_seq_len=4096,
        dtype=ttnn.bfloat16,
        tensor_cache_tag="diff",
        cache_gen_only=False,
    ):
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        args, hf_text_ns, canvas_length = load_diffusion_config(model_path)
        if num_layers is not None:
            args.num_hidden_layers = num_layers
        self.args = args
        self.canvas_length = canvas_length
        self.sliding_window = args.sliding_window

        is_mesh = hasattr(mesh_device, "shape")
        num_devices = mesh_device.get_num_devices() if is_mesh else 1
        if is_mesh and num_devices > 1:
            self.mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
            self.ccl_manager = CCLManager(mesh_device, num_links=2) if not cache_gen_only else None
        else:
            self.mesh_config = MeshConfig((1, 1), decode=ModeConfig(tp=1))
            self.ccl_manager = None
        self.replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        sd = DiffusionStateDictView(LazyStateDict(model_path))
        self.state_dict = sd

        # Separate tensor cache dir per weight set; mixing with AR gemma4 caches
        # would silently load the wrong tensors (cache filename = layer/op only).
        import os

        cache_root = os.getenv("TT_CACHE_PATH") or model_path
        tensor_cache_path = f"{cache_root}/tensor_cache_{tensor_cache_tag}_bf16"
        os.makedirs(tensor_cache_path, exist_ok=True)

        # Soft-embedding matmul weight: [vocab, hidden/tp] TILE, cached (device
        # to_layout on 262k rows stalls; built once on the cache-gen host).
        from models.demos.gemma4_cody.utils.general_utils import cached_tensor_placeholder, get_cache_file_name

        tp = self.mesh_config.tp
        embed_tile_cache = get_cache_file_name(tensor_cache_path, f"embed_tokens_tile_tp{tp}")
        embed_tile_t = cached_tensor_placeholder(embed_tile_cache, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        if embed_tile_t is None:
            embed_tile_t = sd["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0)
        self.embed_tile = ttnn.as_tensor(
            embed_tile_t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self.mesh_config.column_parallel(mesh_device) if tp > 1 else self.replicate,
            cache_file_name=embed_tile_cache,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # V-sharded embedding [vocab/tp, hidden], row-parallel on vocab — matches
        # the lm_head column-parallel vocab split (both shard V into the same tp
        # chunks). Lets the self-conditioning consume the pre-allgather logit
        # shards directly: embmm 11.84->3.52ms and no 134MB logits allgather.
        embed_shard_cache = get_cache_file_name(tensor_cache_path, f"embed_shard_tp{tp}")
        embed_shard_t = cached_tensor_placeholder(embed_shard_cache, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        if embed_shard_t is None:
            embed_shard_t = sd["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0)
        self.embed_shard = ttnn.as_tensor(
            embed_shard_t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self.mesh_config.row_parallel(mesh_device) if tp > 1 else self.replicate,
            cache_file_name=embed_shard_cache,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        logger.info("Building Gemma4Model backbone (no paged KV cache)")
        self.model = Gemma4Model(
            mesh_device=mesh_device,
            hf_config=args,
            state_dict=sd,
            ccl_manager=self.ccl_manager,
            dtype=dtype,
            tensor_cache_path=tensor_cache_path,
            mesh_config=self.mesh_config,
            max_seq_len=max_seq_len,
            max_local_batch_size=1,
            num_layers=args.num_hidden_layers,
            create_kv_cache=False,
            create_rope_cache=not cache_gen_only,
        )
        if cache_gen_only:
            # Mock devices can't run from_torch uploads; all as_tensor cache
            # files are written at this point.
            return
        # Demo path: the fused matmul_reduce_scatter buffers are batch=32 decode
        # buffers; this model never runs the fused decode path — free them.
        for layer in self.model.layers:
            for mod in (layer.self_attn, layer.shared_mlp):
                for attr in ("_fused_intermediate", "_fused_output"):
                    buf = getattr(mod, attr, None)
                    if buf is not None:
                        buf.deallocate(True)
                        setattr(mod, attr, None)

        # Per-layer (decoder, encoder) scalars; layer.layer_scalar is swapped per pass.
        raw_sd = sd._sd
        self.decoder_scalars = []
        self.encoder_scalars = []
        for i in range(args.num_hidden_layers):
            self.decoder_scalars.append(raw_sd[f"model.decoder.layers.{i}.layer_scalar"].float().item())
            self.encoder_scalars.append(raw_sd[f"model.encoder.language_model.layers.{i}.layer_scalar"].float().item())

        # Self-conditioning weights (decoder only): pre_norm (scaled), gate/up/down.
        def _w(name, transpose=True):
            t = raw_sd[f"model.decoder.self_conditioning.{name}.weight"]
            if transpose:
                t = t.transpose(0, 1)
            # rms_norm gamma must be [1, 1, H/32, 32] row-major
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0) if transpose else t.reshape(1, 1, -1, 32),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT if transpose else ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.replicate,
            )

        self.sc_gate = _w("gate_proj")
        self.sc_up = _w("up_proj")
        self.sc_down = _w("down_proj")
        self.sc_pre_norm = _w("pre_norm", transpose=False)
        self.embed_scale = args.hidden_size**0.5
        self.eps = args.rms_norm_eps

        # Host-side rope tables (fp32) for exact arbitrary-offset slicing.
        self.rope_host = {}
        for lt in dict.fromkeys(args.layer_types):
            cos, sin = _create_rope_cache_tensors(hf_text_ns, max_seq_len, lt)
            self.rope_host[lt] = (cos.squeeze(0), sin.squeeze(0))  # [max_seq, head_dim]

        # State: dense per-layer prefix K/V [1, n_local_kv, P_padded, head_dim]
        self.prefix_kv = [None] * args.num_hidden_layers
        self.prefix_len = 0  # valid (unpadded) prefix length
        self.prefix_padded = 0

        # Traced-decode state (persistent input buffers reused across canvases;
        # rope/masks/sc_buf/trace rebuilt per canvas in decode_prepare/_finish).
        self._toks_dev = None
        self._inv_temp_dev = None  # 1/prev_temp for self-conditioning
        self._inv_samp_dev = None  # 1/temp for the sampler entropy
        self._toks_host = None
        self._inv_temp_host = None
        self._inv_samp_host = None
        self._sc_buf = None
        self._decode_rope = None
        self._decode_masks = None
        self._trace_id = None
        self._trace_logits = None
        self._trace_argmax = None
        self._trace_ent = None
        self._trace_partials = None
        # entropy reduction precision (bf16 unless DIFF_ENTROPY_FP32=1)
        import os as _os

        self._entropy_fp32 = _os.getenv("DIFF_ENTROPY_FP32") == "1"
        # one-time on-device-reduction self-check vs host (DIFF_REDUCE_CHECK=1)
        self._reduce_check = _os.getenv("DIFF_REDUCE_CHECK") == "1"
        self._reduce_checked = False
        # sync-bracketed per-component timing on eager steps (DIFF_PROFILE=1).
        # DIFF_NO_TRACE forces every step through the eager path so a WARM
        # (post-compile) step can be profiled — step 2 eager is the compile run
        # and its per-op times are dominated by one-time kernel compilation.
        self._profile = _os.getenv("DIFF_PROFILE") == "1"
        self._no_trace = _os.getenv("DIFF_NO_TRACE") == "1"
        # MoE gate/up contract K=hidden (88 tiles); in0_block_w=1 starves on 88
        # serial K-steps (~4× slower). 8 K-tiles/step ≈ 4× faster gate/up.
        _os.environ.setdefault("DIFF_MOE_IN0BW", "8")
        # Experimental: all-experts MoE as batched ttnn.matmul at M=256 instead of
        # 128 serial per-expert sparse_matmuls (DIFF_MOE_BATCHED=1).
        self._moe_batched = _os.getenv("DIFF_MOE_BATCHED") == "1"
        # CONCAT MoE: gate/up over ALL experts as ONE big-N matmul (N=E*I → full
        # core grid) + batched down. The fix for the MoE under-utilization
        # (f_moe 6746→529 ms). Default ON.
        self._moe_concat = _os.getenv("DIFF_MOE_CONCAT", "1") == "1"
        # Reduce-sharding: compute argmax+entropy partials on the PRE-allgather
        # sharded logits ([S, V/tp], 4× less width) + cross-shard host combine,
        # instead of the full [S, V] reduce. Default ON (warm step 2.38→0.91s).
        self._reduce_shard = _os.getenv("DIFF_REDUCE_SHARD", "1") == "1"
        # Batched FFN all-reduce: concat the MoE-down + shared-MLP-down partials into
        # ONE [S,2H] collective. The collective itself is ~0.38ms/layer cheaper, but
        # measured NEUTRAL in the full trace (the per-layer all-reduces already
        # overlap matmul compute, so they're off the critical path; concat+slice
        # overhead cancels the saving). Kept gated, default OFF.
        self._batch_ffn_ar = _os.getenv("DIFF_BATCH_FFN_AR", "0") == "1"
        # Diagnostic: split the warm replay into write / device-compute / host-readback
        # (forces a sync after execute_trace) to size the clock-invariant host floor.
        self._time_readback = _os.getenv("DIFF_TIME_READBACK", "0") == "1"
        # Packed-partials readback (BROKEN — default OFF): concat of four width-1
        # TILE-padded [S,1] partials is a sub-tile concat that corrupts a_val, which
        # flips the cross-shard argmax winner at near-ties (238/256, garbage output)
        # while entropy stays in tolerance. The readback DID drop 9→4ms, so the win
        # is real but needs a concat-free path (composer-gather read). Kept OFF.
        self._pack_partials = _os.getenv("DIFF_PACK_PARTIALS", "0") == "1"
        # Composer-gather readback: read each partial's tp shards in ONE to_torch
        # (mesh composer) instead of tp individual reads → 16 reads → 4, no on-device
        # concat (so no sub-tile corruption). Clock-invariant ~5ms. Default ON.
        self._gather_read = _os.getenv("DIFF_GATHER_READ", "1") == "1"
        # Trace-ablation: DIFF_SKIP="attn,moe,shared,sc,lmhead,reduce" replaces a
        # component with a cheap shape-preserving op (zeros_like). Its TRACE
        # contribution then shows up as the warm-step delta vs the unablated trace
        # (the serial DIFF_PROFILE sum can't isolate it — it inflates dispatch
        # uniformly). Empty by default (no effect on the real path).
        self._skip = set(filter(None, _os.getenv("DIFF_SKIP", "").split(",")))
        self._vocab_per_dev = None
        if self._moe_concat:
            logger.info("Building concatenated MoE gate/up weights")
            self._build_moe_concat()
        self._prof_on = False
        self._prof_times = {}
        self._eager_real_count = 0
        self._pt = None

    # ── helpers ───────────────────────────────────────────────────────────

    def _pstart(self):
        if self._prof_on:
            ttnn.synchronize_device(self.mesh_device)
            import time as _t

            self._pt = _t.perf_counter()

    def _pmark(self, label):
        """Sync the device and accumulate elapsed since the last mark into label."""
        if self._prof_on:
            import time as _t

            ttnn.synchronize_device(self.mesh_device)
            now = _t.perf_counter()
            self._prof_times[label] = self._prof_times.get(label, 0.0) + (now - self._pt)
            self._pt = now

    def _from_torch(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=self.mesh_device, dtype=dtype, layout=layout, mesh_mapper=self.replicate)

    def _to_torch(self, t):
        if hasattr(self.mesh_device, "shape"):
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        return ttnn.to_torch(t)

    def _rope_slice(self, layer_type, start, length):
        cos, sin = self.rope_host[layer_type]
        sl = lambda t: self._from_torch(t[start : start + length].reshape(1, 1, length, -1))
        return sl(cos), sl(sin)

    def _embed(self, token_ids):
        """token_ids: torch [S] -> [1,1,S,H] tile tensor on device (scaled, gathered)."""
        toks = self._from_torch(token_ids.view(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        emb = self.model.embed_tokens(toks)
        emb = ttnn.reshape(emb, (1, 1, token_ids.numel(), self.args.hidden_size))
        return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

    def reset(self):
        for kv in self.prefix_kv:
            if kv is not None:
                kv[0].deallocate(True)
                kv[1].deallocate(True)
        self.prefix_kv = [None] * self.args.num_hidden_layers
        self.prefix_len = 0
        self.prefix_padded = 0

    # ── encoder ───────────────────────────────────────────────────────────

    def encode_prefix(self, input_ids):
        """Causal-encode the full prefix (replaces existing KV). input_ids: torch [P]."""
        self.reset()
        p = input_ids.numel()
        p_pad = max(128, ((p + 31) // 32) * 32)
        ids = torch.nn.functional.pad(input_ids, (0, p_pad - p), value=0)
        hidden = self._embed(ids)

        for i, layer in enumerate(self.model.layers):
            layer.layer_scalar = self.encoder_scalars[i]
            cos, sin = self._rope_slice(layer.layer_type, 0, p_pad)
            hidden = layer(
                hidden,
                rope_mats=(cos, sin),
                position_idx=None,
                page_table=None,
                kv_cache=None,
                is_decode=False,
                keep_kv=True,
            )
            cos.deallocate(True)
            sin.deallocate(True)
            self.prefix_kv[i] = layer.self_attn._last_kv
            layer.self_attn._last_kv = None
            layer.layer_scalar = self.decoder_scalars[i]
        hidden.deallocate(True)
        self.prefix_len = p
        self.prefix_padded = p_pad

    # ── decoder ───────────────────────────────────────────────────────────

    def _canvas_masks(self):
        """Additive bf16 masks per layer type: [1,1,S,window+S], built once per prefix."""
        s = self.canvas_length
        p, p_pad = self.prefix_len, self.prefix_padded
        masks = {}
        full = torch.zeros(1, 1, s, p_pad + s)
        full[:, :, :, p:p_pad] = NEG_INF
        masks["full_attention"] = self._from_torch(full)

        lo = max(0, p - self.sliding_window + 1)
        lo32 = (lo // 32) * 32
        win = torch.zeros(1, 1, s, (p_pad - lo32) + s)
        win[:, :, :, : lo - lo32] = NEG_INF
        win[:, :, :, p - lo32 : p_pad - lo32] = NEG_INF
        masks["sliding_attention"] = self._from_torch(win)
        self._sliding_lo32 = lo32
        return masks

    def _canvas_attention(self, layer, normed, rope, masks):
        cfg = layer.self_attn.config
        weights = layer.self_attn.weights
        tp = self.mesh_config.tp
        cos, sin = rope

        xqkv = apply_qkv_projection(normed, weights)
        q, k, v = split_qkv_heads_prefill(xqkv, cfg, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated)
        xqkv.deallocate(True)
        q = apply_per_head_norm(q, weights.q_norm_weight, cfg.rms_norm_eps, with_scale=True)
        k = apply_per_head_norm(k, weights.k_norm_weight, cfg.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, cfg.rms_norm_eps, with_scale=False)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        pk, pv = self.prefix_kv[layer.layer_idx]
        if cfg.is_sliding and self._sliding_lo32 > 0:
            lo = self._sliding_lo32
            pk = ttnn.slice(pk, (0, 0, lo, 0), (pk.shape[0], pk.shape[1], pk.shape[2], pk.shape[3]))
            pv = ttnn.slice(pv, (0, 0, lo, 0), (pv.shape[0], pv.shape[1], pv.shape[2], pv.shape[3]))
            owns_prefix = True
        else:
            owns_prefix = False
        k_all = ttnn.concat([pk, k], dim=2)
        v_all = ttnn.concat([pv, v], dim=2)
        k.deallocate(True)
        v.deallocate(True)
        if owns_prefix:
            pk.deallocate(True)
            pv.deallocate(True)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=masks[cfg.layer_type], is_causal=False, scale=1.0
        )
        q.deallocate(True)
        k_all.deallocate(True)
        v_all.deallocate(True)

        out = concat_heads(attn, is_decode_mode=False)
        attn.deallocate(True)
        out = ttnn.linear(out, weights.o_proj)
        if self.mesh_config.tp > 1:
            out = ccl_allreduce(out, self.mesh_config, self.ccl_manager)
        return out

    def _self_conditioning(self, embeds, sc_input, inv_temp, sc_mode):
        """embeds: [1,1,S,H]. sc_input: [1,1,S,V] device logits (real mode) or None.

        inv_temp is 1/temperature, either a python float (eager / parity path) or
        a device [1,1,1,1] tensor (traced path — temperature varies per step but the
        op graph must stay constant, so it is fed as a runtime scalar).
        sc_mode: "zeros" (step 1, no self-conditioning) or "real".
        """
        if sc_mode == "real" and "sc" in self._skip:
            return ttnn.rms_norm(embeds, epsilon=self.eps)  # ablate softmax+embmm+SC-MLP
        if sc_mode == "zeros":
            soft = ttnn.mul(embeds, 0.0)  # zeros_like
        elif self._reduce_shard:
            # V-sharded SC: sc_input is the pre-allgather logit shard [1,1,S,V/tp].
            # softmax(x)@E folded as (exp(x-m)@E)/Z. m is the GLOBAL per-token max
            # (all-gather the per-shard maxes, then max) — a constant upper bound
            # would underflow exp(x-c) into bf16 garbage when the real max << cap.
            # Z and the partial embedding then sum across shards via all-reduce.
            if isinstance(inv_temp, (int, float)):
                scaled = ttnn.mul(sc_input, float(inv_temp))
            else:
                scaled = ttnn.multiply(sc_input, inv_temp)  # broadcast [1,1,1,1]
            m_local = ttnn.max(scaled, dim=-1, keepdim=True)  # [1,1,S,1] per shard
            if self.mesh_config.tp > 1:
                m_gathered = ccl_allgather(m_local, self.mesh_config, self.ccl_manager, dim=3)  # [1,1,S,tp]
                m_local.deallocate(True)
                m = ttnn.max(m_gathered, dim=-1, keepdim=True)  # [1,1,S,1] global
                m_gathered.deallocate(True)
            else:
                m = m_local
            sub = ttnn.subtract(scaled, m)  # exp(x-m) in (0,1] — bf16-safe
            scaled.deallocate(True)
            m.deallocate(True)
            ex = ttnn.exp(sub)
            sub.deallocate(True)
            z_local = ttnn.sum(ex, dim=-1, keepdim=True)  # [1,1,S,1]
            z = ccl_allreduce(z_local, self.mesh_config, self.ccl_manager) if self.mesh_config.tp > 1 else z_local
            if self.mesh_config.tp > 1:
                z_local.deallocate(True)
            ex = ttnn.typecast(ex, ttnn.bfloat16)
            self._pmark("sc_softmax")
            # embed_shard [1,1,V/tp,H] row-parallel — partial expected-embedding
            # per shard, summed across shards; 4× less K than the full-vocab embmm.
            soft = ttnn.matmul(ex, self.embed_shard)  # [1,1,S,H]
            ex.deallocate(True)
            if self.mesh_config.tp > 1:
                soft = ccl_allreduce(soft, self.mesh_config, self.ccl_manager)
            soft = ttnn.divide(soft, z)  # normalize on the small [S, H]
            z.deallocate(True)
            soft = ttnn.mul(soft, self.embed_scale)
            self._pmark("sc_embmm")
        else:
            if isinstance(inv_temp, (int, float)):
                scaled = ttnn.mul(sc_input, float(inv_temp))
            else:
                scaled = ttnn.multiply(sc_input, inv_temp)  # broadcast [1,1,1,1]
            # Manual softmax: ttnn.softmax over the 262144-wide row is ~9× slower
            # than the same max/exp/sum primitives (cf. the entropy reduce). Fold
            # the normalization past the matmul — softmax(x)@E = (exp(x-m)@E)/Z —
            # so the /Z runs on the small [S, H/tp] result, not the [S, V] probs.
            m = ttnn.max(scaled, dim=-1, keepdim=True)
            sub = ttnn.subtract(scaled, m)  # broadcast over the vocab dim
            scaled.deallocate(True)
            m.deallocate(True)
            ex = ttnn.exp(sub)
            sub.deallocate(True)
            z = ttnn.sum(ex, dim=-1, keepdim=True)  # [1,1,S,1] normalizer
            ex = ttnn.typecast(ex, ttnn.bfloat16)
            self._pmark("sc_softmax")
            # tied embedding weight is [1,1,V,H/tp] column-parallel — matmul + gather
            soft = ttnn.matmul(ex, self.embed_tile)
            ex.deallocate(True)
            soft = ttnn.divide(soft, z)  # normalize on the sharded [S, H/tp]
            z.deallocate(True)
            if self.mesh_config.tp > 1:
                soft = ccl_allgather(soft, self.mesh_config, self.ccl_manager)
            soft = ttnn.mul(soft, self.embed_scale)
            self._pmark("sc_embmm")
        normed = ttnn.rms_norm(soft, weight=self.sc_pre_norm, epsilon=self.eps)
        soft.deallocate(True)
        gate = ttnn.gelu(ttnn.linear(normed, self.sc_gate), fast_and_approximate_mode=False)
        up = ttnn.linear(normed, self.sc_up)
        normed.deallocate(True)
        h = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)
        sc = ttnn.linear(h, self.sc_down)
        h.deallocate(True)
        combined = ttnn.add(embeds, sc)
        sc.deallocate(True)
        out = ttnn.rms_norm(combined, epsilon=self.eps)
        combined.deallocate(True)
        self._pmark("sc_mlp")
        return out

    def _cat_expert_w(self, t):
        """[1, E, H, I] -> [1, 1, H, E*I] (expert-major), so gate/up over all
        experts is ONE big-N matmul. Built once at model build. permute/reshape
        reject bf4, so dequant→relayout→requant (concat weight stays bf4 ~ the
        original size; a bf16 concat would be ~8 GB)."""
        t16 = ttnn.typecast(t, ttnn.bfloat16)  # bf4 -> bf16 for permute/reshape
        t16 = ttnn.permute(t16, (0, 2, 1, 3))  # [1, H, E, I]
        ei = t16.shape[2] * t16.shape[3]
        t16 = ttnn.reshape(t16, (1, 1, t16.shape[1], ei))  # [1, 1, H, E*I]
        cat = ttnn.typecast(t16, ttnn.bfloat4_b)  # back to bf4
        t16.deallocate(True)
        return cat

    def _cat_down_w(self, t):
        """[1, E, I, H] -> [1, 1, E*I, H] (expert-major along the contraction
        dim). Lets the routing-folded geglu output run ONE big down matmul:
        out = sum_e W_down_e @ (routing_e * g_e) = (routing⊙g) @ down_cat.
        reshape rejects bf4, so dequant→reshape→requant (stays bf4-sized)."""
        t16 = ttnn.typecast(t, ttnn.bfloat16)
        ei = t16.shape[1] * t16.shape[2]
        t16 = ttnn.reshape(t16, (1, 1, ei, t16.shape[3]))  # [1, 1, E*I, H]
        cat = ttnn.typecast(t16, ttnn.bfloat4_b)
        t16.deallocate(True)
        return cat

    def _build_moe_concat(self):
        """Pre-build the concatenated gate/up/down weights for every MoE layer."""
        import torch as _torch

        e = i = None
        for layer in self.model.layers:
            w = layer.moe.experts.weights
            layer._gate_cat = self._cat_expert_w(w.gate_proj)
            layer._up_cat = self._cat_expert_w(w.up_proj)
            layer._down_cat = self._cat_down_w(w.down_proj)
            layer._n_experts = e = w.gate_proj.shape[1]
            layer._inter = i = w.intermediate_size_per_device
        # Static [E, E*I] expand matrix: row e is 1 across its own I columns, 0
        # elsewhere. routing[1,1,S,E] @ expand -> [1,1,S,E*I] broadcasts each
        # expert's weight across its intermediate block via a cheap matmul,
        # replacing the wide-g tile-repacking reshapes (the real m_route cost).
        exp = _torch.repeat_interleave(_torch.eye(e), i, dim=1).unsqueeze(0).unsqueeze(0)  # [1,1,E,E*I]
        self._route_expand = ttnn.from_torch(
            exp,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=self.replicate,
        )
        # Block-sharded program config for the concat DOWN matmul [S, E*I] @ [E*I, H].
        # K=E*I is huge (768 tiles); the auto config serializes K (1.36ms). Block-
        # sharding the in0 cuts mcast volume -> 0.53ms incl reshard (measured).
        _down_res = make_block_sharded_matmul_config(self.canvas_length, e * i, self.args.hidden_size)
        self._down_pc, self._down_in_mem = _down_res[0], _down_res[1]

    def _moe_concat_forward(self, layer, router_in, expert_in, defer_reduce=False):
        """All-experts MoE: gate/up as ONE big-N matmul each (full core grid),
        routing folded into the geglu output, then ONE concat down matmul (no
        per-expert batched down or 128× intermediate). Same dense gpt_oss math."""
        from models.demos.gemma4_cody.tt.experts.operations import apply_geglu

        dram = ttnn.DRAM_MEMORY_CONFIG
        routing = layer.moe.router(router_in)  # [1,1,S,E], top-k masked
        self._pmark("m_router")
        gate = ttnn.linear(expert_in, layer._gate_cat, memory_config=dram)  # [1,1,S,E*I]
        up = ttnn.linear(expert_in, layer._up_cat, memory_config=dram)
        g = apply_geglu(gate, up)  # [1,1,S,E*I]
        gate.deallocate(True)
        up.deallocate(True)
        self._pmark("m_gateup")
        # Fold the top-k routing weights into the geglu output per expert, then
        # do ONE concat down matmul instead of a batched-128 down + mask + reduce.
        # down is linear: sum_e W_down_e @ (routing_e * g_e) = (routing⊙g) @ down_cat,
        # which also avoids materializing the [1,E,S,H] (128×) intermediate.
        # Broadcast each expert's weight across its I-block via a cheap expand
        # matmul (routing[S,E] @ [E,E*I]) — no tile-repacking reshape of wide g.
        rexp = ttnn.matmul(routing, self._route_expand, memory_config=dram)  # [1,1,S,E*I]
        routing.deallocate(True)
        g = ttnn.mul(g, rexp)  # scale each expert's intermediate by its weight (0 if unselected)
        rexp.deallocate(True)
        self._pmark("m_route")
        g_sh = ttnn.to_memory_config(g, self._down_in_mem)  # block-shard in0 for the huge-K down
        g.deallocate(True)
        out = ttnn.matmul(g_sh, layer._down_cat, program_config=self._down_pc, memory_config=dram)  # [1,1,S,H]
        g_sh.deallocate(True)
        if self.mesh_config.tp > 1 and not defer_reduce:
            out = ccl_allreduce(out, self.mesh_config, self.ccl_manager)  # row-parallel down partials
        self._pmark("m_down")
        return out

    def _moe_batched_forward(self, layer, router_in, expert_in):
        """All-experts MoE as BATCHED ttnn.matmul at M=256 (the whole canvas) —
        replaces the prefill path's 128 serial per-expert sparse_matmuls at M=32.
        expert_in [1,1,S,H] broadcasts across the 128-expert batch of the weights.
        Same dense gpt_oss math (compute all experts, then routing-mask + sum)."""
        from models.demos.gemma4_cody.tt.experts.operations import apply_geglu

        w = layer.moe.experts.weights
        dram = ttnn.DRAM_MEMORY_CONFIG
        routing = layer.moe.router(router_in)  # [1,1,S,E], top-k masked
        s = expert_in.shape[2]
        h = expert_in.shape[-1]
        e = w.gate_proj.shape[1]
        # gate/up: [1,E,S,H] @ [1,E,H,I] -> [1,E,S,I]. The default matmul won't
        # batch-broadcast [1,1]→[1,E], so replicate the hidden across experts.
        xin = ttnn.repeat(expert_in, [1, e, 1, 1], memory_config=dram)  # [1,E,S,H]
        gate = ttnn.matmul(xin, w.gate_proj, memory_config=dram)
        up = ttnn.matmul(xin, w.up_proj, memory_config=dram)
        xin.deallocate(True)
        g = apply_geglu(gate, up)  # [1,E,S,I]
        gate.deallocate(True)
        up.deallocate(True)
        down = ttnn.matmul(g, w.down_proj, memory_config=dram)  # [1,E,S,H]
        g.deallocate(True)
        # routing-weighted sum over experts (gpt_oss pattern)
        rperm = ttnn.permute(routing, (0, 3, 2, 1))  # [1,E,S,1]
        routing.deallocate(True)
        down = ttnn.mul(down, rperm)
        rperm.deallocate(True)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        down.deallocate(True)
        out = ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, s, h))
        if self.mesh_config.tp > 1:
            out = ccl_allreduce(out, self.mesh_config, self.ccl_manager)
        return out

    def _shared_mlp_m256(self, layer, x, defer_reduce=False):
        """Shared MLP (GeGLU) at full M=256 in ONE pass. SharedMLP.__call__'s L1
        block path overflows the grid at M=256 (num_blocks_x 9>8), but the auto
        matmul config runs gate/up/down fine — 2.7× faster than 2× M=128 halves
        (1.83->0.68ms/layer, measured). Mirrors SharedMLP's plain GeGLU path."""
        smlp = layer.shared_mlp
        kcfg = _mlp_kernel_config(self.mesh_device)
        gate = ttnn.linear(x, smlp.gate_proj, compute_kernel_config=kcfg)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
        up = ttnn.linear(x, smlp.up_proj, compute_kernel_config=kcfg)
        h = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(h, smlp.down_proj, compute_kernel_config=kcfg)
        h.deallocate(True)
        if self.mesh_config.tp > 1 and not defer_reduce:
            out = ccl_allreduce(out, self.mesh_config, self.ccl_manager)
        return out

    def _decode_body(self, hidden, rope, masks):
        """Layer stack + final norm + chunked lm_head + softcap (+ allgather).

        Pure device ops. `hidden` is consumed; `rope`/`masks` are read-only and
        owned by the caller (so the traced path can keep them persistent across
        replays). Returns device logits [1,1,S,V].
        """
        for layer in self.model.layers:
            residual = hidden
            normed = layer.input_layernorm.forward(hidden)
            if "attn" in self._skip:
                attn = ttnn.mul(normed, 0.0)  # zeros_like [S,H] — isolates attn trace cost
            else:
                attn = self._canvas_attention(layer, normed, rope[layer.layer_type], masks)
            normed.deallocate(True)
            attn = layer.post_attention_layernorm.forward(attn)
            hidden = ttnn.add(residual, attn)
            residual.deallocate(True)
            attn.deallocate(True)
            self._pmark("attn")

            # FFN is row-independent and runs entirely at full M=256: the concat
            # MoE and the shared MLP each fire once per layer (full core grid).
            residual = hidden
            # Batch the two FFN-down all-reduces into one [S,2H] collective: defer
            # both internal reduces, concat the partials, all-reduce once, split.
            # Only the concat-MoE path returns a deferrable [S,H] partial.
            batch_ar = (
                self._batch_ffn_ar
                and self.mesh_config.tp > 1
                and self._moe_concat
                and "moe" not in self._skip
                and "shared" not in self._skip
            )
            moe_in = layer.pre_feedforward_layernorm_2.forward(residual)
            if "moe" in self._skip:
                moe_out = ttnn.mul(moe_in, 0.0)  # zeros_like — isolates MoE trace cost
            elif self._moe_concat:
                moe_out = self._moe_concat_forward(layer, residual, moe_in, defer_reduce=batch_ar)
            elif self._moe_batched:
                moe_out = self._moe_batched_forward(layer, residual, moe_in)
            else:
                moe_out = layer.moe(residual, moe_in)
            moe_in.deallocate(True)
            self._pmark("f_moe")
            # Shared MLP at full M=256 in ONE pass (2.7× vs the 2× M=128 halves);
            # _shared_mlp_m256 calls the plain GeGLU matmuls directly so it skips
            # SharedMLP.__call__'s L1 block path that overflows the grid at M=256.
            normed = layer.pre_feedforward_layernorm.forward(residual)
            if "shared" in self._skip:
                mlp_out = ttnn.mul(normed, 0.0)  # zeros_like — isolates shared-MLP trace cost
            else:
                mlp_out = self._shared_mlp_m256(layer, normed, defer_reduce=batch_ar)
            normed.deallocate(True)
            if batch_ar:
                # [S,H]+[S,H] -> [S,2H] -> one all-reduce -> split back. Each device
                # holds its row-parallel partial; the all-reduce sums across TP.
                s = moe_out.shape[2]
                hw = moe_out.shape[-1]
                cat = ttnn.concat([moe_out, mlp_out], dim=3)  # [1,1,S,2H]
                moe_out.deallocate(True)
                mlp_out.deallocate(True)
                cat = ccl_allreduce(cat, self.mesh_config, self.ccl_manager)
                moe_out = ttnn.slice(cat, [0, 0, 0, 0], [1, 1, s, hw])
                mlp_out = ttnn.slice(cat, [0, 0, 0, hw], [1, 1, s, 2 * hw])
                cat.deallocate(True)
            moe_n = layer.post_feedforward_layernorm_2.forward(moe_out)
            moe_out.deallocate(True)
            mlp_n = layer.post_feedforward_layernorm_1.forward(mlp_out)
            mlp_out.deallocate(True)
            self._pmark("f_shared")
            h = ttnn.add(mlp_n, moe_n)
            mlp_n.deallocate(True)
            moe_n.deallocate(True)
            h = layer.post_feedforward_layernorm.forward(h)
            hidden = ttnn.add(residual, h)
            residual.deallocate(True)
            h.deallocate(True)
            if self.decoder_scalars[layer.layer_idx] != 1.0:
                hidden = ttnn.mul(hidden, self.decoder_scalars[layer.layer_idx])
            self._pmark("f_combine")

        hidden = self.model.norm.forward(hidden)
        self._pmark("final_norm")
        # lm_head as ONE M=256 matmul (auto config handles it at 1.94ms/dev,
        # measured) — the prior 8×32-row chunking + concat was unnecessary.
        logits = ttnn.linear(hidden, self.model.lm_head_weight)
        hidden.deallocate(True)
        self._pmark("lm_head")
        cap = self.args.final_logit_softcapping
        logits = ttnn.mul(logits, 1.0 / cap)
        logits = ttnn.tanh(logits)
        logits = ttnn.mul(logits, cap)
        # Reduce partials on the SHARDED logits (pre-allgather, 4× less width).
        partials = self._reduce_partials(logits) if self._reduce_shard else None
        # With sharded reductions + V-sharded SC, nothing needs the full-vocab
        # logits — keep them sharded (skip the 134MB all-gather, feed the shards
        # straight back into the next step's SC). Only the non-sharded fallback
        # all-gathers.
        if self.mesh_config.tp > 1 and not self._reduce_shard:
            logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)
        self._pmark("softcap_allgather")
        return logits, partials

    def decode_canvas(self, canvas_ids, sc_logits_tt=None, temperature=1.0, masks=None):
        """One eager denoising forward (parity / non-traced path).

        canvas_ids: torch [S]; sc_logits_tt: device [1,1,S,V] raw softcapped logits
        from the previous step (temperature applied here on device), or None (step 1).
        Returns (host bf16 logits [S, V], device logits handle for next step).
        """
        s = canvas_ids.numel()
        masks = masks or self._canvas_masks()
        embeds = self._embed(canvas_ids)
        sc_mode = "zeros" if sc_logits_tt is None else "real"
        hidden = self._self_conditioning(embeds, sc_logits_tt, 1.0 / temperature, sc_mode)
        embeds.deallocate(True)
        rope = {lt: self._rope_slice(lt, self.prefix_len, s) for lt in dict.fromkeys(self.args.layer_types)}
        logits, _ = self._decode_body(hidden, rope, masks)
        for cos, sin in rope.values():
            cos.deallocate(True)
            sin.deallocate(True)
        logits_host = self._to_torch(logits).reshape(s, -1)  # bf16; cast chunked downstream
        return logits_host, logits

    # ── traced decode (device-side trace replay) ───────────────────────────
    #
    # The denoising inner loop runs the same 30-layer forward ~16× per canvas
    # with only the canvas token ids and the self-conditioning temperature
    # changing. Capturing it as a ttnn trace removes per-step host dispatch
    # (hundreds of ops → one execute_trace). Within a canvas the prefix (KV,
    # rope, masks) is constant, so one trace serves every step; across canvases
    # the prefix grows, so the trace is released and re-captured per canvas.
    #
    # Per-step dynamic inputs, fed without breaking the captured graph:
    #   * canvas ids  → persistent uint32 buffer, copy_host_to_device each step
    #   * temperature → persistent [1,1,1,1] scalar (broadcast-mul in SC)
    #   * prev logits → persistent sc_buf, read at the start of self-conditioning.
    #     The new logits are copied back into sc_buf *after* each replay (an eager
    #     device→device copy, NOT inside the trace — tt-metal forbids data writes
    #     during capture). The copy touches no new allocation, so it is safe
    #     post-capture, the same way copy_host_to_device feeds traced decoders.
    #
    # Step 1 (zeros SC) and step 2 (real SC) run eagerly: step 1 has a different
    # op graph (no SC matmul) and step 2 is the compile run that warms the traced
    # path's kernels before capture.

    def _embed_dev(self, toks_dev):
        """Embed a persistent [1,S] uint32 token buffer → [1,1,S,H] tile tensor."""
        emb = self.model.embed_tokens(toks_dev)
        emb = ttnn.reshape(emb, (1, 1, self.canvas_length, self.args.hidden_size))
        return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

    def _decode_core(self, toks_dev, sc_input, inv_temp, rope, masks, sc_mode):
        """Forward from a device token buffer to device logits. Captured verbatim
        when tracing (no host writes, no feedback copy inside)."""
        self._pstart()
        embeds = self._embed_dev(toks_dev)
        self._pmark("embed")
        hidden = self._self_conditioning(embeds, sc_input, inv_temp, sc_mode)
        embeds.deallocate(True)
        return self._decode_body(hidden, rope, masks)

    def _ensure_decode_buffers(self):
        s = self.canvas_length
        if self._toks_dev is None:
            self._toks_dev = ttnn.from_torch(
                torch.zeros(1, s, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.replicate,
            )
            samp_dtype = ttnn.float32 if self._entropy_fp32 else ttnn.bfloat16
            self._inv_temp_dev = ttnn.from_torch(
                torch.ones(1, 1, 1, 1),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self.replicate,
            )
            self._inv_samp_dev = ttnn.from_torch(
                torch.ones(1, 1, 1, 1),
                device=self.mesh_device,
                dtype=samp_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self.replicate,
            )
        self._toks_host = torch.zeros(1, s, dtype=torch.int32)
        self._inv_temp_host = torch.ones(1, 1, 1, 1)
        self._inv_samp_host = torch.ones(1, 1, 1, 1)

    def _write_inputs(self, canvas_ids, inv_temp, inv_samp):
        """Stage per-step dynamic inputs into the persistent device buffers:
        canvas ids, the self-conditioning 1/temp, and the sampler 1/temp."""
        samp_dtype = ttnn.float32 if self._entropy_fp32 else ttnn.bfloat16
        self._toks_host[0, :] = canvas_ids.to(torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                self._toks_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.replicate
            ),
            self._toks_dev,
        )
        self._inv_temp_host[0, 0, 0, 0] = inv_temp
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                self._inv_temp_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.replicate
            ),
            self._inv_temp_dev,
        )
        self._inv_samp_host[0, 0, 0, 0] = inv_samp
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self._inv_samp_host, dtype=samp_dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.replicate),
            self._inv_samp_dev,
        )

    def decode_prepare(self):
        """After encode_prefix: build persistent rope + masks for the new prefix
        and ensure the reusable input buffers exist. Call once per canvas."""
        self._ensure_decode_buffers()
        self._decode_masks = self._canvas_masks()
        self._decode_rope = {
            lt: self._rope_slice(lt, self.prefix_len, self.canvas_length) for lt in dict.fromkeys(self.args.layer_types)
        }
        self._sc_buf = None
        self._trace_id = None
        self._trace_logits = None
        self._trace_partials = None

    def _reduce(self, logits, inv_samp):
        """On-device per-token reductions so only [S]-sized tensors are read back
        instead of the 134 MB [S, V] logits: the argmax token id (temperature-
        independent) and the temperature-scaled entropy the sampler thresholds on.

        Entropy is the fp32-stable form H = log(Z) - sum(p*lf)/Z, lf = x - max(x),
        x = logits/temp — matching sampler.token_entropy. Returns ([1,1,S,1] argmax,
        [1,1,S,1] entropy)."""
        amax = ttnn.argmax(logits, dim=-1, keepdim=True)
        x = ttnn.typecast(logits, ttnn.float32) if self._entropy_fp32 else logits
        xs = ttnn.mul(x, inv_samp)  # logits / temp
        if self._entropy_fp32:
            x.deallocate(True)
        m = ttnn.max(xs, dim=-1, keepdim=True)
        lf = ttnn.subtract(xs, m)  # broadcast over the vocab dim
        xs.deallocate(True)
        m.deallocate(True)
        ex = ttnn.exp(lf)
        z = ttnn.sum(ex, dim=-1, keepdim=True)
        exlf = ttnn.mul(ex, lf)
        lf.deallocate(True)
        ex.deallocate(True)
        s_exlf = ttnn.sum(exlf, dim=-1, keepdim=True)
        exlf.deallocate(True)
        ratio = ttnn.divide(s_exlf, z)
        s_exlf.deallocate(True)
        logz = ttnn.log(z)
        z.deallocate(True)
        ent = ttnn.subtract(logz, ratio)
        logz.deallocate(True)
        ratio.deallocate(True)
        self._pmark("reduce")
        return amax, ent

    def _read_col(self, t):
        """Read a [1,1,S,1] device column to a [S] host tensor."""
        return self._to_torch(t).reshape(-1)[: self.canvas_length]

    def _read_shards(self, t):
        """Read every TP shard of a [1,1,S,1] tensor → [tp, S] host (fp32)."""
        if self._gather_read and hasattr(self.mesh_device, "shape") and self.mesh_config.tp > 1:
            # ONE to_torch gathers all tp shards (host-side concat of per-device reads
            # along dim=3) → [1,1,S,tp]; transpose to [tp,S]. No on-device concat, so
            # values are bit-exact per shard — just 1 call instead of tp.
            tp = self.mesh_config.tp
            g = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3)).float()
            g = g.reshape(self.canvas_length, -1)[:, :tp]  # [S, tp]
            return g.transpose(0, 1).contiguous()  # [tp, S]
        devs = ttnn.get_device_tensors(t) if hasattr(self.mesh_device, "shape") else [t]
        return torch.stack([ttnn.to_torch(d).float().reshape(-1)[: self.canvas_length] for d in devs])

    def _read_shards_packed(self, t):
        """Read every TP shard of a packed [1,1,S,4] tensor → [tp, S, 4] host (fp32)."""
        devs = ttnn.get_device_tensors(t) if hasattr(self.mesh_device, "shape") else [t]
        return torch.stack([ttnn.to_torch(d).float().reshape(self.canvas_length, -1)[:, :4] for d in devs])

    def _reduce_partials(self, logits):
        """Per-shard reductions on the SHARDED [1,1,S,V/tp] logits (4× less width
        than the all-gathered). Returns (a_idx, a_val, Z, SX) of [1,1,S,1] tensors;
        combined cross-shard on host in _combine_partials."""
        self._vocab_per_dev = logits.shape[-1]
        # top-1 gives BOTH the local argmax index and its value in one op, and is
        # BIT-EXACT vs torch.argmax (256/256). It runs at ~19ms over the 65536-wide
        # shard; a topk-free max+eq+arg trick is ~12ms cheaper but was not bit-exact
        # on ties / large indices, so the exact topk is kept (sampling must not drift).
        if "argmax" in self._skip or "reduce" in self._skip:
            # replace topk with a plain max-reduce → the delta isolates topk's
            # marginal cost over a bare reduction (index is a dummy for timing).
            a_val = ttnn.max(logits, dim=-1, keepdim=True)
            a_idx = a_val
        else:
            # ttnn.topk returns a GARBAGE index *and* value at V/tp == 32768 (tp=8):
            # 32/256 index match + inf values (measured). It is correct at >= 49152,
            # so pad the shard up to a valid width with -inf (never the max, so the
            # index is unaffected) just for the topk INDEX. The VALUE comes from a
            # plain max-reduce, which is reliable at any width.
            W = self._vocab_per_dev
            if W < 49152:
                lp = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 49152 - W)], value=-1e30)
                _tv, a_idx = ttnn.topk(lp, 1, dim=-1, largest=True, sorted=False)
                lp.deallocate(True)
            else:
                _tv, a_idx = ttnn.topk(logits, 1, dim=-1, largest=True, sorted=False)
            _tv.deallocate(True)
            a_val = ttnn.max(logits, dim=-1, keepdim=True)
        self._pmark("r_argmax")
        if "entropy" in self._skip or "reduce" in self._skip:
            return (a_idx, a_val, a_val, a_val)  # dummy (z, sx) — isolates entropy trace cost
        x = ttnn.typecast(logits, ttnn.float32) if self._entropy_fp32 else logits
        xs = ttnn.mul(x, self._inv_samp_dev)  # logits / temp
        if self._entropy_fp32:
            x.deallocate(True)
        # max(xs) == max(logits)*inv_samp == a_val*inv_samp (inv_samp>0): reuse the
        # max we just computed for a_val instead of a 2nd wide max-reduce (bit-exact),
        # so the reliable-max + topk-index pair costs the same as the old single topk.
        if self._entropy_fp32:
            av = ttnn.typecast(a_val, ttnn.float32)
            m = ttnn.mul(av, self._inv_samp_dev)
            av.deallocate(True)
        else:
            m = ttnn.mul(a_val, self._inv_samp_dev)
        lf = ttnn.subtract(xs, m)
        xs.deallocate(True)
        m.deallocate(True)
        ex = ttnn.exp(lf)
        z = ttnn.sum(ex, dim=-1, keepdim=True)
        exlf = ttnn.mul(ex, lf)
        lf.deallocate(True)
        ex.deallocate(True)
        sx = ttnn.sum(exlf, dim=-1, keepdim=True)
        exlf.deallocate(True)
        self._pmark("r_entropy")
        if self._pack_partials:
            return self._pack4(a_idx, a_val, z, sx)
        return (a_idx, a_val, z, sx)

    def _pack4(self, *parts):
        """Concat the 4 [1,1,S,1] reduction partials into ONE [1,1,S,4] fp32 tensor.
        fp32 because a_idx is an exact vocab index (>256, unrepresentable in bf16);
        z/sx/a_val upcast exactly. The host then reads 4 device tensors (one/shard)
        instead of 16, at the same per-read tile-row cost."""
        cols = [t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32) for t in parts]
        packed = ttnn.concat(cols, dim=3)  # [1,1,S,4]
        for t, c in zip(parts, cols):
            c.deallocate(True)
            if c is not t:
                t.deallocate(True)
        return packed

    def _combine_partials(self, partials, inv_samp):
        """Cross-shard combine of per-device reduction partials → (argmax [S],
        entropy [S]) on host. argmax: pick the shard with the max raw logit. entropy:
        re-base each shard's (Z, SX) to the global max and sum (exact)."""
        if isinstance(partials, (tuple, list)):
            a_idx, a_val, z, sx = (self._read_shards(t) for t in partials)  # each [tp, S]
        else:
            packed = self._read_shards_packed(partials)  # [tp, S, 4]
            a_idx, a_val, z, sx = packed[..., 0], packed[..., 1], packed[..., 2], packed[..., 3]
        tp = a_val.shape[0]
        vpd = self._vocab_per_dev
        # global argmax (raw logits, temperature-independent)
        winner = a_val.argmax(0)  # [S]
        cols = torch.arange(a_idx.shape[1])
        argmax = (winner * vpd + a_idx[winner, cols]).long()  # [S]
        # global entropy from per-shard (m, Z, SX) with m_d = max(logits_d)/temp
        m = a_val * inv_samp  # [tp, S]
        M = m.max(0).values  # [S]
        d = m - M  # [tp, S] ≤ 0
        ed = torch.exp(d)
        gz = (z * ed).sum(0)  # [S]
        gsx = (ed * (sx + d * z)).sum(0)  # [S]
        ent = (torch.log(gz) - gsx / gz).float()  # [S]
        return argmax, ent

    def decode_eager(self, canvas_ids, inv_temp, inv_samp, sc_mode):
        """One eager forward over the persistent buffers (steps 1 and 2). Updates
        the sc_buf feedback for the next step. Returns (argmax [S], entropy [S])."""
        self._write_inputs(canvas_ids, inv_temp if sc_mode == "real" else 1.0, inv_samp)
        if sc_mode == "real":
            self._eager_real_count += 1
        # Profile real-SC eager steps. Step 2 (the 1st real eager) is the compile
        # run — its per-op times are one-time kernel compilation, NOT warm execution.
        # With DIFF_NO_TRACE the later real eager steps are warm (compiled); read those.
        self._prof_on = self._profile and sc_mode == "real"
        if self._prof_on:
            self._prof_times = {}
            from models.demos.gemma4_cody.tt.experts.prefill import MOE_PROF_TIMES

            MOE_PROF_TIMES.clear()
        logits, partials = self._decode_core(
            self._toks_dev, self._sc_buf, self._inv_temp_dev, self._decode_rope, self._decode_masks, sc_mode
        )
        if partials is None:
            amax, ent = self._reduce(logits, self._inv_samp_dev)
        if self._prof_on:
            self._prof_on = False
            total = sum(self._prof_times.values())
            order = [
                "embed",
                "sc_softmax",
                "sc_embmm",
                "sc_mlp",
                "attn",
                "f_shared",
                "m_router",
                "m_gateup",
                "m_down",
                "m_route",
                "f_moe",
                "f_combine",
                "final_norm",
                "lm_head",
                "softcap_allgather",
                "r_argmax",
                "r_entropy",
                "reduce",
            ]
            parts = " ".join(f"{k}={self._prof_times.get(k,0)*1e3:.0f}ms" for k in order if k in self._prof_times)
            kind = "COMPILE" if self._eager_real_count == 1 else f"warm#{self._eager_real_count}"
            logger.info(f"[profile {kind}] device-stage breakdown (sum={total*1e3:.0f}ms): {parts}")
            from models.demos.gemma4_cody.tt.experts.prefill import MOE_PROF_TIMES

            if MOE_PROF_TIMES:
                moe_parts = " ".join(f"{k}={v*1e3:.0f}ms" for k, v in MOE_PROF_TIMES.items())
                logger.info(f"[profile {kind}] MoE sub-ops (summed over chunks×layers): {moe_parts}")
        if self._sc_buf is None:
            # first logits define the sc/feedback buffer shape (vocab is padded)
            self._sc_buf = ttnn.clone(logits)
        else:
            ttnn.copy(logits, self._sc_buf)  # self-condition next step on these logits
        if partials is None:
            out = (self._read_col(amax).long(), self._read_col(ent).float())
            amax.deallocate(True)
            ent.deallocate(True)
        else:
            out = self._combine_partials(partials, inv_samp)
            for p in partials if isinstance(partials, (tuple, list)) else [partials]:
                p.deallocate(True)
        if self._reduce_check and not self._reduce_checked:
            self._reduce_check_vs_host(logits, inv_samp, out)
        logits.deallocate(True)
        return out

    def _reduce_check_vs_host(self, logits, inv_samp, out):
        """One-time correctness check of the on-device reductions at the real vocab
        size: read the full logits this once, recompute argmax+entropy on host, log
        the discrepancy. Host-heavy, so gated by DIFF_REDUCE_CHECK=1."""
        from models.demos.gemma4_cody.diffusion.sampler import token_entropy

        self._reduce_checked = True
        if self._reduce_shard and hasattr(self.mesh_device, "shape") and self.mesh_config.tp > 1:
            # logits are V-sharded (no all-gather); concat the per-device vocab
            # shards in device order (== the column-parallel split) for the ref.
            devs = ttnn.get_device_tensors(logits)
            lh = torch.cat([ttnn.to_torch(d).reshape(self.canvas_length, -1).float() for d in devs], dim=-1)
        else:
            lh = self._to_torch(logits).reshape(self.canvas_length, -1).float()
        a_ref = lh.argmax(-1)
        e_ref = token_entropy(lh * inv_samp)  # processed = logits / temp
        a_dev, e_dev = out
        amatch = int((a_dev == a_ref).sum())
        eerr = (e_dev - e_ref).abs()
        logger.info(
            f"[reduce-check] argmax {amatch}/{self.canvas_length} match; "
            f"entropy maxerr={eerr.max():.4f} meanerr={eerr.mean():.4f} "
            f"(dtype={'fp32' if self._entropy_fp32 else 'bf16'})"
        )

    def decode_capture(self):
        """Capture the real-SC forward + reductions into a trace. sc_buf already
        holds the previous step's logits; capture records the ops without executing
        them. The feedback copy is intentionally left OUT of the trace (writes are
        not allowed during capture) — decode_replay does it eagerly afterward."""
        ttnn.synchronize_device(self.mesh_device)
        self._trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._trace_logits, self._trace_partials = self._decode_core(
            self._toks_dev, self._sc_buf, self._inv_temp_dev, self._decode_rope, self._decode_masks, "real"
        )
        if self._trace_partials is None:
            self._trace_argmax, self._trace_ent = self._reduce(self._trace_logits, self._inv_samp_dev)
        ttnn.end_trace_capture(self.mesh_device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

    def decode_replay(self, canvas_ids, inv_temp, inv_samp):
        """Replay the captured trace for one step, feed the logits back into sc_buf
        for the next step, and read back only the [S] reductions. The full [S, V]
        logits never leave the device. Returns (argmax [S], entropy [S])."""
        if getattr(self, "_time_readback", False):
            return self._decode_replay_timed(canvas_ids, inv_temp, inv_samp)
        self._write_inputs(canvas_ids, inv_temp, inv_samp)
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        ttnn.copy(self._trace_logits, self._sc_buf)  # eager feedback (post-capture): allocates nothing
        if self._trace_partials is None:
            return self._read_col(self._trace_argmax).long(), self._read_col(self._trace_ent).float()
        return self._combine_partials(self._trace_partials, inv_samp)

    def _decode_replay_timed(self, canvas_ids, inv_temp, inv_samp):
        """decode_replay with a forced sync between device compute and host readback,
        so write / compute / readback are separable. Diagnostic only."""
        import time as _t

        t0 = _t.perf_counter()
        self._write_inputs(canvas_ids, inv_temp, inv_samp)
        t1 = _t.perf_counter()
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        ttnn.copy(self._trace_logits, self._sc_buf)
        ttnn.synchronize_device(self.mesh_device)  # wait out the device compute
        t2 = _t.perf_counter()
        if self._trace_partials is None:
            out = self._read_col(self._trace_argmax).long(), self._read_col(self._trace_ent).float()
        else:
            out = self._combine_partials(self._trace_partials, inv_samp)
        t3 = _t.perf_counter()
        logger.info(
            f"[replay-time] write={1e3*(t1-t0):.1f} compute+copy={1e3*(t2-t1):.1f} "
            f"readback+combine={1e3*(t3-t2):.1f} ms"
        )
        return out

    def decode_step(self, canvas_ids, inv_temp, inv_samp, step_index):
        """Drive one denoising step (1-based within the canvas). Step 1 runs the
        zeros-SC graph eagerly, step 2 is the real-SC compile run, then the trace
        is captured lazily and every later step is a replay. Returns the device
        reductions (argmax [S], entropy [S]) for the host accept/stop logic."""
        if step_index == 1:
            return self.decode_eager(canvas_ids, inv_temp, inv_samp, "zeros")
        if self._no_trace:  # warm-profiling mode: never capture, stay eager
            return self.decode_eager(canvas_ids, inv_temp, inv_samp, "real")
        if step_index == 2:
            return self.decode_eager(canvas_ids, inv_temp, inv_samp, "real")
        if self._trace_id is None:
            self.decode_capture()
        return self.decode_replay(canvas_ids, inv_temp, inv_samp)

    def decode_finish(self):
        """Release the trace and per-canvas persistent tensors (rope/masks/sc_buf)."""
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._trace_id)
            self._trace_id = None
        self._trace_logits = None
        self._trace_argmax = None
        self._trace_ent = None
        self._trace_partials = None
        if self._sc_buf is not None:
            self._sc_buf.deallocate(True)
            self._sc_buf = None
        if self._decode_rope is not None:
            for cos, sin in self._decode_rope.values():
                cos.deallocate(True)
                sin.deallocate(True)
            self._decode_rope = None
        if self._decode_masks is not None:
            for m in self._decode_masks.values():
                m.deallocate(True)
            self._decode_masks = None
