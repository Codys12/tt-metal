# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative-decode orchestrator (server side) — TRACED.

Mirrors the MTP drafter setup (`speculative.py` + the server's drafter trace):
the dflash propose is a SINGLE batched forward captured as its own trace, with
all device buffers pre-allocated in `__init__` (before the server's
`_capture_traces`) and refreshed per step via `copy_host_to_device_tensor`. The
drafter owns a FIXED-shape anchor KV cache (`DFlashDrafter.alloc_anchor_caches`)
so the forward has static shapes; `decode_forward_packed` writes this step's
noise into the cache and runs a masked decode-SDPA (mirrors
`packed_decode_forward`). The drafter gets its own `CCLManager` (independent
semaphores) and the server `synchronize_device`s between trace executions, so
the propose's CCL never contends with the captured target traces.

Per step the server: `refresh_propose` → execute the dflash trace →
`read_drafts` → (reuse) packed verify → `_commit_packed_verify` →
`append_committed_ondevice` (commit accepted tokens as anchors).
"""

from __future__ import annotations

import os
from typing import List

import torch

import ttnn
from models.demos.gemma4_cody.tt.dflash import DFlashConfig, DFlashDrafter
from models.demos.gemma4_cody.tt.dflash._reference import build_rope_cache

from .speculative import _SlotState

_DFLASH_DEBUG = os.environ.get("GEMMA4_DFLASH_DEBUG") == "1"


def dflash_ckpt(mesh_device, msg, sync=True):
    """Hang-localizing checkpoint (GEMMA4_DFLASH_DEBUG=1): sync then print."""
    if not _DFLASH_DEBUG:
        return
    if sync and mesh_device is not None:
        ttnn.synchronize_device(mesh_device)
    print(f"[dflash-ckpt] {msg}", flush=True)


def _round_up(n, m):
    return ((n + m - 1) // m) * m


class DflashSpeculativeDecoder:
    def __init__(
        self,
        mesh_device,
        model,
        dflash_path: str,
        dflash_cache_dir: str | None,
        num_slots: int,
        mesh_config=None,
        ccl_manager=None,
    ):
        dflash_ckpt(None, "DflashSpeculativeDecoder.__init__: enter", sync=False)
        self.mesh_device = mesh_device
        self.model = model
        self.batch = num_slots
        self._is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
        self._replicate = ttnn.ReplicateTensorToMesh(mesh_device) if self._is_mesh else None
        self.tp = mesh_device.shape[1] if self._is_mesh else 1

        cfg = DFlashConfig.from_hf_path(dflash_path)
        self.config = cfg
        self.block = cfg.block_size
        self.block_size = cfg.block_size  # alias used by server logging
        self.num_drafts = cfg.block_size - 1  # ⇒ packed-verify P = block_size
        self.mask_token_id = cfg.mask_token_id
        print(
            f"[dflash] variant={cfg.variant} block_size={cfg.block_size} "
            f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} head_dim={cfg.head_dim} "
            f"K_aux={cfg.num_aux_layers} aux_taps={list(cfg.aux_hidden_layers)} "
            f"draft_vocab={cfg.draft_vocab_size} uses_target_head={cfg.uses_target_head} "
            f"rope_theta={cfg.rope_theta} sliding_window={cfg.sliding_window}",
            flush=True,
        )
        if cfg.sliding_window:
            print(
                f"[dflash] NOTE: checkpoint declares sliding_window={cfg.sliding_window} "
                f"(layer_types={list(cfg.layer_types)}) but the TT propose SDPA attends ALL anchors "
                "(full non-causal). Harmless while anchor_len+block_size <= sliding_window; for longer "
                "contexts acceptance will diverge from the reference until per-layer windowing lands.",
                flush=True,
            )
        # Fixed cache depth (anchor-relative positions), a multiple of 64 for the
        # SDPA k_chunk; eligibility keeps anchor_len + block <= max_anchors.
        # GEMMA4_DFLASH_MAX_ANCHORS overrides the config-derived depth (the z-lab
        # config carries no max_anchors → DFlashConfig defaults it to 3072).
        max_anchors_cfg = int(os.environ.get("GEMMA4_DFLASH_MAX_ANCHORS", cfg.max_anchors))
        self.max_anchors = _round_up(max_anchors_cfg, 64)
        self.aux_hidden_layers = cfg.aux_hidden_layers
        self.hidden = cfg.hidden_size
        self.head_dim = cfg.head_dim
        self.n_heads_local = cfg.num_attention_heads // self.tp
        self.n_kv_local = cfg.num_key_value_heads // self.tp

        # Own CCLManager — independent semaphores so the (traced) propose CCL
        # never contends with the captured target traces. Built here, before
        # the server's _capture_traces.
        if self._is_mesh and self.tp > 1 and ccl_manager is not None:
            from models.demos.gemma4_cody.tt.ccl import CCLManager

            self._ccl_manager = CCLManager(mesh_device, num_links=ccl_manager.num_links, topology=ccl_manager.topology)
        else:
            self._ccl_manager = None

        dflash_ckpt(self.mesh_device, "DflashSpeculativeDecoder: loading drafter weights")
        # Source weights from the converted tensorbin cache when it exists; else
        # stream straight from the checkpoint's model.safetensors (the lazy
        # loader). The fallback lets a new checkpoint (e.g. the z-lab block-16)
        # be brought up without a separate convert_weights pass, and avoids
        # silently reading a stale cache built for a different checkpoint.
        use_cache = bool(dflash_cache_dir) and os.path.isdir(dflash_cache_dir)
        drafter_kwargs = dict(
            mesh_device=mesh_device,
            config=cfg,
            mesh_config=mesh_config,
            ccl_manager=self._ccl_manager,
            load_embed_tokens=False,  # reuse the target embedding (CCL-free) for noise.
        )
        if use_cache:
            drafter_kwargs["cache_dir"] = dflash_cache_dir
            print(f"[dflash] loading drafter weights from tensorbin cache {dflash_cache_dir}", flush=True)
        else:
            drafter_kwargs["safetensors_dir"] = dflash_path
            # Persist a tensorbin cache as we stream so the NEXT server run loads
            # from the fast eager path (atomic-rename on full success — a crash
            # mid-load leaves no partial cache). Skipped when no cache dir is
            # known (TT_CACHE_PATH unset).
            drafter_kwargs["write_cache_dir"] = dflash_cache_dir
            print(
                f"[dflash] cache dir {dflash_cache_dir!r} absent — streaming drafter weights from "
                f"{dflash_path}/model.safetensors (lazy)"
                + (f"; will write cache to {dflash_cache_dir} for next run" if dflash_cache_dir else ""),
                flush=True,
            )
        self.drafter = DFlashDrafter(**drafter_kwargs)
        # Tied-embedding variants (z-lab) ship no own lm_head: draft logits use
        # the TARGET's full-vocab lm_head. read_drafts reconstructs the global id
        # from the per-shard argmax via draft_vocab_size (== target vocab) and
        # applies no d2t remap (the checkpoint ships none).
        if cfg.uses_target_head:
            if getattr(self.model, "lm_head_weight", None) is None:
                raise RuntimeError("DFlash zlab variant needs the target lm_head, but model.lm_head_weight is None.")
            self.drafter._target_lm_head = self.model.lm_head_weight
            print("[dflash] using target lm_head for draft logits (tied-embedding variant)", flush=True)
        dflash_ckpt(self.mesh_device, "DflashSpeculativeDecoder: drafter loaded")

        # Host RoPE cache, anchor-relative (position i == cache index i).
        pos = torch.arange(self.max_anchors + self.block + 1).unsqueeze(0)
        cos, sin = build_rope_cache(pos, head_dim=self.head_dim, theta=cfg.rope_theta)
        self._cos_host = cos[0].to(torch.float32)  # [max_pos, head_dim]
        self._sin_host = sin[0].to(torch.float32)

        # Mesh config kept for the in-trace embed all-gather.
        self._mesh_config = mesh_config

        self._anchor_len = [0] * num_slots
        self.slots: List[_SlotState] = [_SlotState() for _ in range(num_slots)]

        self._preallocate()

    # ── device buffer helpers (mirror server._alloc_device_tensor/_host_tensor) ──

    def _alloc(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, device=self.mesh_device, layout=layout, dtype=dtype, mesh_mapper=self._replicate)

    def _host(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, layout=layout, dtype=dtype, mesh_mapper=self._replicate)

    def _preallocate(self):
        """Allocate the anchor cache + all propose-trace input buffers BEFORE
        the server captures traces. Per-step refresh is
        `copy_host_to_device_tensor` only — never allocates in the hot loop.

        The noise *embedding* is computed INSIDE the propose trace
        (`ttnn.embedding` + AG, mirrors `target.embed_tokens` but unscaled), so
        its output is allocated at capture time and reused on replay. The only
        per-step input is `_ids_dev` (the B*block noise token IDs)."""
        B, block, MA, hd, hidden = self.batch, self.block, self.max_anchors, self.head_dim, self.hidden
        dflash_ckpt(self.mesh_device, f"preallocate: caches start (B={B}, MA={MA})", sync=False)
        self.caches = self.drafter.alloc_anchor_caches(B, MA)
        dflash_ckpt(self.mesh_device, "preallocate: caches done")
        # Noise token IDs [1, B*block] uint32. ttnn.embedding consumes this
        # inside the trace; the embedded output never crosses the trace edge.
        self._ids_dev = self._alloc(torch.zeros(1, B * block, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dflash_ckpt(self.mesh_device, "preallocate: _ids_dev done")
        # On-device propose-input build: the only per-step H2D is anchor_len
        # [1, B] (-1 = idle) and the bonus token [1, B]. ids / q_pos / valid /
        # noise write idxs all derive from these in-trace; baked iota table.
        self._lf_anchor = self._alloc(torch.full((1, B), -1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._lf_bonus = self._alloc(torch.zeros(1, B, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._iota_block = self._alloc(
            (torch.arange(B * block, dtype=torch.int32) % block).reshape(1, B * block),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        # Position-major iota (arange // B): for the per-p noise write idxs;
        # row p's slots are contiguous so the trace can slice them per p.
        self._iota_pos_major = self._alloc(
            (torch.arange(B * block, dtype=torch.int32) // B).reshape(1, B * block),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        # Per-noise-position cache write index [B] int32 (one tensor per p).
        self._noise_write_idxs = [
            self._alloc(torch.full((B,), -1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(block)
        ]
        # The noise-position Q RoPE and the additive SDPA mask are built ON
        # DEVICE inside the propose trace (build_propose_fwd) from pre-baked
        # tables, indexed by these two tiny per-step int tensors — instead of a
        # host build + ~3 large H2D copies/step (the bulk of the old refresh,
        # ~31 ms). Mirrors the packed-verify trace's on-device mask gather.
        # Q-RoPE position per noise row [1, B*block] (anchor_len[u]+p; idle ⇒ 0).
        self._q_pos_idx = self._alloc(torch.zeros(1, B * block, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # Per-slot valid key length [1, B] (anchor_len[u]+block; idle ⇒ 0 ⇒ all-masked).
        self._valid_len = self._alloc(torch.zeros(1, B, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dflash_ckpt(self.mesh_device, "preallocate: propose-side buffers done")
        # Fixed whole-cache RoPE (positions 0..MA-1), constant across steps.
        self._fixed_cos = self._alloc(
            self._cos_host[:MA].reshape(1, 1, MA, hd).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
        )
        self._fixed_sin = self._alloc(
            self._sin_host[:MA].reshape(1, 1, MA, hd).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
        )
        dflash_ckpt(self.mesh_device, "preallocate: fixed-cos/sin done")
        # In-trace gather tables (constant; built once). cos/sin 2D embedding
        # weights [MA+block, hd] (ROW_MAJOR) for the noise-position Q RoPE.
        P_max = MA + block
        self._cos_table_2d = self._alloc(
            self._cos_host[:P_max].to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        self._sin_table_2d = self._alloc(
            self._sin_host[:P_max].to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        # Additive-mask embedding weight [MA+1, MA] (TILE, like the packed-verify
        # mask table): row L is the mask for a query whose valid key span is
        # [0:L] (0 for col<L, -1e9 otherwise). Gathered by per-slot valid_len.
        _cols = torch.arange(MA).unsqueeze(0)
        _Ls = torch.arange(MA + 1).unsqueeze(1)
        _mask_table = torch.where(_cols < _Ls, 0.0, -1e9).to(torch.bfloat16)  # [MA+1, MA]
        self._mask_table = self._alloc(_mask_table, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        dflash_ckpt(self.mesh_device, "preallocate: gather tables done")
        # Append-trace input buffers. Refreshed by `append_committed_ondevice`
        # then consumed by the captured `_dflash_append_trace` — never
        # reallocated. K_hidden is `fc_in_features` = the K aux-tap hiddens
        # concatenated along the feature dim (what `_project_anchor_hidden`
        # consumes). Each widx[p] is a [B] int32 cache index (-1 ⇒ skip slot).
        #
        # NOTE: `_aux_dev` is the largest single allocation here (~14 MB at
        # 1×1×256×26880 bf16). Going through `ttnn.from_torch + mesh_mapper`
        # at this size has tripped a SIGILL (host packer / replication path);
        # `init_kv_cache` runs into the same class of issue and uses
        # ``allocate_tensor_on_device + ttnn.fill`` instead. Mirror that.
        K_hidden = self.config.fc_in_features
        self._K_hidden = K_hidden
        dflash_ckpt(self.mesh_device, f"preallocate: aux_dev start ({B*block}x{K_hidden} bf16 TILE)", sync=False)
        self._aux_dev = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, B * block, K_hidden]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        dflash_ckpt(self.mesh_device, "preallocate: aux_dev allocated; filling")
        ttnn.fill(self._aux_dev, 0.0, output_tensor=self._aux_dev)
        dflash_ckpt(self.mesh_device, "preallocate: aux_dev filled")
        self._anchor_widx_devs = [
            self._alloc(torch.full((B,), -1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(block)
        ]
        # On-device aux-tap row-gather index [1, B*block] u32 (Phase 2a). Gathers
        # the committed positions' aux taps — kept ON DEVICE — straight into
        # `_aux_dev`, replacing the per-step D2H(K taps)+host-cat+H2D round-trip
        # (the largest per-step transfer). See `append_committed_ondevice`.
        self._aux_gidx_dev = self._alloc(
            torch.zeros(1, B * block, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        dflash_ckpt(self.mesh_device, "preallocate: anchor_widx_devs done")
        ttnn.synchronize_device(self.mesh_device)
        dflash_ckpt(self.mesh_device, "preallocate: complete")

    # ── trace closure ───────────────────────────────────────────────────────

    def build_propose_fwd(self):
        """Closure captured as the dflash propose trace. Reads `_ids_dev` (the
        noise token IDs) and all other pre-allocated buffers; embeds the IDs +
        all-gathers the column-parallel hidden dim INSIDE the trace (mirrors
        `target.embed_tokens` minus the `*sqrt(H)` scale — dflash uses unscaled
        embeddings for noise); runs the 5-layer block forward; then argmaxes the
        draft logits ON DEVICE (`ttnn.topk` k=1). Returns ``(topk_val,
        topk_idx)`` — each [1,1,B*(block-1),1] per vocab shard. The full
        [B*(block-1),draft_vocab] logits never cross the trace edge, so the hot
        loop only D2Hs the per-row (value,index) pair (mirrors the packed-verify
        trace)."""
        from models.demos.gemma4_cody.tt.ccl import ccl_allgather

        B, block, hidden = self.batch, self.block, self.hidden
        # ttnn.topk requires its row count (∏ of the leading dims) to be a
        # multiple of 32. The draft logits have B*(block-1) rows — 224 at the
        # standard B=32 (a clean 7·32, so pad_h==0 and this is a no-op), but the
        # server supports batch<32 where B*(block-1) can be unaligned (e.g.
        # B=16 ⇒ 112). Pad the row dim up to the next 32 then slice back in
        # `read_drafts`. The amount is constant (B/block fixed), baked into the trace.
        n_rows = B * (block - 1)
        pad_rows = (-n_rows) % 32

        def fwd():
            # ── On-device propose-input build ────────────────────────────────
            # Push: anchor_len [1,B] + bonus token [1,B] (last bytes). Derive:
            #   ids[u, p]   = idle ? 0 : (p == 0 ? bonus[u] : mask_token_id)
            #   q_pos[u, p] = L[u] + p (idle → 0)
            #   valid[u]    = L[u] + block (idle 0)
            #   widx[p][u]  = L[u] + p (-1 idle ⇒ skipped)
            # Was 3 + block H2Ds + B·block host loop; now 2 H2Ds.
            B_, block_ = self.batch, self.block

            # ttnn.repeat_interleave on int32 quantizes through bfloat16
            # (bonus 45518→45568 — bisected 2026-06-05; broke acceptance 6→3:
            # every step's bonus token embedded as a wrong nearby id).
            # `_ri()` = bit-exact interleave via ttnn.repeat.
            def _ri(x, k):
                n = x.shape[-1]
                return ttnn.reshape(ttnn.repeat(ttnn.reshape(x, [n, 1]), [1, k]), [1, n * k])

            active = ttnn.gez(self._lf_anchor)  # [1,B] 1/0
            L_rep = _ri(self._lf_anchor, block_)
            act_rep = _ri(active, block_)
            bonus_rep = _ri(self._lf_bonus, block_)
            is0 = ttnn.eqz(self._iota_block)
            not0 = ttnn.rsub(is0, 1)  # 1 - is0
            ids = ttnn.add(ttnn.multiply(is0, bonus_rep), ttnn.multiply(not0, self.mask_token_id))
            ids = ttnn.multiply(ids, act_rep)
            ttnn.assign(ttnn.typecast(ids, ttnn.uint32), self._ids_dev)
            q_pos = ttnn.multiply(ttnn.add(L_rep, self._iota_block), act_rep)
            ttnn.assign(ttnn.typecast(q_pos, ttnn.uint32), self._q_pos_idx)
            valid = ttnn.multiply(ttnn.add(self._lf_anchor, block_), active)
            ttnn.assign(ttnn.typecast(valid, ttnn.uint32), self._valid_len)
            # Position-major widx: row p occupies [p*B, (p+1)*B) — contiguous
            # per-p slices for the paged_update_cache index tensors.
            L_pm = ttnn.repeat(self._lf_anchor, [1, block_])  # [1, block*B]
            act_pm = ttnn.repeat(active, [1, block_])
            widx_all = ttnn.add(L_pm, self._iota_pos_major)  # idle rows: -1 + p
            idle_pen = ttnn.multiply(ttnn.rsub(act_pm, 1), -(self.max_anchors + self.block))
            widx_all = ttnn.add(widx_all, idle_pen)  # idle → always < 0 ⇒ skip
            for p in range(block_):
                w = ttnn.slice(widx_all, [0, p * B_], [1, (p + 1) * B_])
                ttnn.assign(ttnn.reshape(w, [B_]), self._noise_write_idxs[p])

            # ttnn.embedding output: column-parallel [1, B*block, hidden_local]
            # ROW_MAJOR. The allocation happens at capture-time (recorded into
            # the trace) and is reused on replay — zero host work in the hot
            # loop. The dflash drafter ingests the verifier's UNSCALED embed
            # (no `mul self.embed_scale`).
            sharded = ttnn.embedding(self._ids_dev, self.model.embedding_weight, dtype=ttnn.bfloat16)
            if self._mesh_config is not None and self.tp > 1:
                sharded = ttnn.unsqueeze_to_4D(sharded)  # [1,1,B*block,hidden_local]
                sharded = ccl_allgather(sharded, self._mesh_config, self._ccl_manager, dim=3)
            else:
                sharded = ttnn.reshape(sharded, (1, 1, B * block, hidden))
            # rms_norm in decode_forward_packed needs TILE.
            noise = ttnn.to_layout(sharded, ttnn.TILE_LAYOUT)
            # In-trace gather of the noise-position Q RoPE and the additive SDPA
            # mask from pre-baked tables (mirrors _build_packed_verify_fwd). The
            # host pushes only `_q_pos_idx` / `_valid_len` now — not the built
            # [1,1,B*block,hd] cos/sin or the [B,1,H*block,MA] mask.
            cos_bp = ttnn.unsqueeze_to_4D(
                ttnn.embedding(self._q_pos_idx, self._cos_table_2d, layout=ttnn.TILE_LAYOUT)
            )  # [1,1,B*block,hd]
            sin_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(self._q_pos_idx, self._sin_table_2d, layout=ttnn.TILE_LAYOUT))
            m = ttnn.embedding(self._valid_len, self._mask_table, layout=ttnn.TILE_LAYOUT)  # [1,B,MA]
            m = ttnn.reshape(m, (B, 1, 1, self.max_anchors))
            attn_mask = ttnn.repeat(m, [1, 1, self.n_heads_local * block, 1])  # [B,1,H_local*block,MA]
            ttnn.deallocate(m)
            draft_logits = self.drafter.decode_forward_packed(
                noise,
                self.caches,
                self._noise_write_idxs,
                cos_bp,
                sin_bp,
                self._fixed_cos,
                self._fixed_sin,
                attn_mask,
                B,
                block,
            )
            ttnn.deallocate(cos_bp)
            ttnn.deallocate(sin_bp)
            ttnn.deallocate(attn_mask)
            # On-device argmax (k=1). Keeps the [1,1,B*(block-1),draft_vocab_local]
            # logits resident on device — the hot loop reads back only the per-row
            # (value, index) pair instead of D2H-ing the full draft-vocab logits
            # (~100 MB/step at draft_vocab=32000, B=32). Column-parallel lm_head ⇒
            # topk runs per shard; `read_drafts` reconstructs the global id.
            if pad_rows:
                draft_logits = ttnn.pad(draft_logits, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
            draft_val, draft_idx = ttnn.topk(draft_logits, k=1, dim=-1)
            ttnn.deallocate(draft_logits)
            return draft_val, draft_idx

        return fwd

    def build_append_fwd(self):
        """Closure captured as the dflash append trace. Reads only pre-allocated
        buffers (`_aux_dev`, `_anchor_widx_devs`) and writes the just-committed
        anchors into `self.caches`. ``write_idxs[p] == -1`` for every slot ⇒
        no-op writes, so capture-time warmup leaves the cache untouched.

        Depends on `_q_sharded_mem_B`, which `decode_forward_packed` learns on
        its first call — must warmup propose before append (the capture order
        in `server._capture_traces` enforces this)."""

        def fwd():
            self.drafter.write_anchors_packed(
                self.caches,
                self._aux_dev,
                self._anchor_widx_devs,
                self.batch,
                self.block,
            )
            return None

        return fwd

    # ── per-step refresh ──────────────────────────────────────────────────────

    def refresh_propose(self, verify_slots, server_slots):
        """Host→device refresh of the propose inputs: TWO [1, B] int copies —
        ids / q_pos / valid / per-p noise write idxs all derive on device
        inside the propose trace from ``anchor_len`` and the bonus token (see
        build_propose_fwd's preamble). -1 anchor ⇒ idle slot ⇒ ids/positions 0
        and write idxs negative ⇒ paged updates skipped."""
        B = self.batch
        sl = torch.tensor(verify_slots, dtype=torch.int64)
        anchor = torch.full((1, B), -1, dtype=torch.int32)
        anchor[0, sl] = torch.tensor([self._anchor_len[u] for u in verify_slots], dtype=torch.int32)
        bonus = torch.zeros(1, B, dtype=torch.int32)
        bonus[0, sl] = torch.tensor([int(server_slots[u].next_token) for u in verify_slots], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(self._host(anchor, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self._lf_anchor)
        ttnn.copy_host_to_device_tensor(self._host(bonus, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self._lf_bonus)

    def read_drafts(self, topk_val, topk_idx, verify_slots) -> List[List[int]]:
        """On-device-argmax draft read. ``topk_val`` / ``topk_idx`` are the
        propose trace's ``ttnn.topk(k=1)`` outputs over the draft logits
        ([1,1,B*(block-1),1] each, row u*(block-1)+j ⇒ slot u draft j) → per-slot
        block-1 target-vocab ids (winning-shard argmax + d2t offset remap),
        slot-indexed.

        The draft-vocab lm_head is column-parallel, so topk ran per shard: the
        global draft id is ``local_idx + shard * (draft_vocab // tp)`` for the
        shard with the largest value (mirrors ``server._read_packed_verify``).
        Reading only the (val, idx) pair avoids D2H-ing the full
        [B*(block-1), draft_vocab] logits every step."""
        block = self.block
        B = self.batch
        if self._is_mesh and self.tp > 1:
            shard_vocab = self.config.draft_vocab_size // self.tp
            vals = torch.stack(
                [ttnn.to_torch(s).float().reshape(-1) for s in ttnn.get_device_tensors(topk_val)], dim=0
            )  # [tp, B*(block-1)]
            idxs = torch.stack(
                [ttnn.to_torch(s).reshape(-1).to(torch.int64) for s in ttnn.get_device_tensors(topk_idx)], dim=0
            )  # [tp, B*(block-1)]
            win = vals.argmax(dim=0)  # [B*(block-1)] — winning shard per row
            rows = torch.arange(win.shape[0])
            ids = idxs[win, rows] + win.to(torch.int64) * shard_vocab
        elif self._is_mesh:
            ids = ttnn.to_torch(ttnn.get_device_tensors(topk_idx)[0]).reshape(-1).to(torch.int64)
        else:
            ids = ttnn.to_torch(topk_idx).reshape(-1).to(torch.int64)
        # Drop the topk row-padding (build_propose_fwd pads B*(block-1) up to a
        # 32-multiple for the topk kernel); only the first B*(block-1) are real.
        ids = ids[: B * (block - 1)].reshape(B, block - 1)  # [B, block-1] global draft ids
        d2t = self.drafter._d2t_host()
        if d2t is not None:
            ids = ids + d2t[ids]
        drafts: List[List[int]] = [[] for _ in range(B)]
        for u in verify_slots:
            drafts[u] = [int(x) for x in ids[u].tolist()]
        return drafts

    # ── commit-side anchor append ─────────────────────────────────────────────

    def append_committed_ondevice(self, commits, B_v, aux_taps) -> bool:
        """On-device anchor append (Phase 2a). Gathers the committed positions'
        aux taps — which stay ON DEVICE (verify-trace outputs) — directly into
        the pre-allocated `_aux_dev` via one `ttnn.embedding` row-gather, then
        refreshes `_anchor_widx_devs`. Replaces the old host path's D2H(K taps)
        + host torch-cat + H2D(`_aux_dev`) round-trip with ~3 device
        ops + a tiny index H2D (mirrors `_packed_fill_kv_loopfree_embed`).

        `commits`: list of (slot_idx, verify_row_r, n_acc).
        `aux_taps`: list of K device tensors, each [1, 1, B_v*P, hidden]
            (residual-stream taps, replicated across TP). The append TRACE
            (`write_anchors_packed`) still consumes `_aux_dev` unchanged.
        """
        B, block, P = self.batch, self.block, self.block  # packed P == block_size
        # gidx (anchor_len-INDEPENDENT) is built by the shared staticmethod the
        # parity test also calls — so test and production use IDENTICAL logic.
        gidx, any_write = self.aux_gather_index(commits, B, block, P)
        if not any_write:
            return False
        # widx (anchor_len-dependent) names each committed position's cache index.
        widx = [torch.full((B,), -1, dtype=torch.int32) for _ in range(block)]
        for slot_idx, r, n_acc in commits:
            L = self._anchor_len[slot_idx]
            for j in range(n_acc + 1):
                widx[j][slot_idx] = L + j
            self._anchor_len[slot_idx] = L + (n_acc + 1)
        ttnn.copy_host_to_device_tensor(
            self._host(gidx.reshape(1, B * block), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self._aux_gidx_dev
        )
        self.gather_committed_aux_ondevice(aux_taps, self._aux_gidx_dev, self._aux_dev, B_v, P)
        for j in range(block):
            ttnn.copy_host_to_device_tensor(
                self._host(widx[j], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self._anchor_widx_devs[j]
            )
        return True

    @staticmethod
    def aux_gather_index(commits, batch, block, P):
        """Build the [batch*block] row-gather index for the on-device aux gather
        (Phase 2a): `_aux_dev` row (slot*block + j) ← verify-aux row (r*P + j) for
        each committed (slot, verify_row r, n_acc), j in 0..n_acc. Uncommitted
        rows default to src row 0 (their cache write is skipped by widx=-1).
        anchor_len-INDEPENDENT, so it is shared VERBATIM by `append_committed_ondevice`
        and the parity test — guaranteeing test == production index logic.
        """
        gidx = torch.zeros(batch * block, dtype=torch.int32)
        any_write = False
        for slot_idx, r, n_acc in commits:
            for j in range(n_acc + 1):
                gidx[slot_idx * block + j] = r * P + j
            any_write = True
        return gidx, any_write

    @staticmethod
    def gather_committed_aux_ondevice(aux_taps, gidx_dev, aux_dev, B_v, P):
        """On-device gather of the committed positions' aux taps into the
        pre-allocated `aux_dev` (Phase 2a). Concats the K taps along the feature
        dim → [1,1,B_v*P,K*hidden], then a **dim-2 `ttnn.gather`** by `gidx_dev`
        writes the result into `aux_dev` IN PLACE via `ttnn.assign`.

        Mirrors `tt/attention/decode.py::_packed_fill_kv_loopfree` (the dim-2
        gather variant). The `*_embed` variant flattens the feature dim into the
        embedding-table rows, which SCRAMBLES a K*hidden=32256-wide row (it is
        only exercised at head_dim=256 in the verify path) — caught by
        `test_dflash_fast_path.py::test_ondevice_aux_gather_parity`. Keeping the
        feature dim as a real axis (dim-2 gather) avoids that.

        aux_taps: K device tensors, each [1,1,B_v*P,hidden] (replicated across TP).
        gidx_dev: [1, n_out] uint32 row map — out row i ← src row gidx[i].
        aux_dev:  [1,1,n_out,K*hidden] destination (n_out == B*block).
        """
        K = len(aux_taps)
        hidden = int(aux_taps[0].shape[-1])
        Kh = K * hidden
        n_out = int(gidx_dev.shape[-1])
        dram = ttnn.DRAM_MEMORY_CONFIG
        src = ttnn.concat(
            [ttnn.reshape(a, (1, 1, B_v * P, hidden)) for a in aux_taps], dim=-1, memory_config=dram
        )  # [1,1,B_v*P,Kh]
        # Full-shape dim-2 (row) gather index [1,1,n_out,Kh] — idx[..,o,f] = gidx[o]
        # broadcast over the feature dim (mirrors `_packed_fill_kv_loopfree`).
        idx = ttnn.reshape(gidx_dev, (1, 1, n_out, 1))
        idx = ttnn.repeat(idx, [1, 1, 1, Kh])  # ROW_MAJOR broadcast
        idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
        merged = ttnn.gather(src, dim=2, index=idx, memory_config=dram)  # [1,1,n_out,Kh]
        ttnn.deallocate(idx)
        ttnn.deallocate(src)
        ttnn.assign(merged, aux_dev)
        ttnn.deallocate(merged)

    @staticmethod
    def ondevice_n_accepted(draft_ids, target_ids, T):
        """On-device greedy speculative accept count (Phase 2b core), pure ttnn
        so it can run in the verify-trace tail. GLOBAL token ids (float):
          draft_ids:  [B_v, T]   the drafter's proposed tokens
          target_ids: [B_v, P]   the target's argmax per position (P = T+1)
        Returns n_acc [B_v, 1] (float) = length of the longest prefix with
        draft==target. Because accepted drafts EQUAL the target there, the emitted
        tokens are exactly ``target_ids[:, :n_acc+1]`` — so only ``target_ids`` +
        ``n_acc`` need leave the device (not the drafts). Validated against the
        host greedy match in `test_dflash_fast_path.py`.
        """
        B_v = int(draft_ids.shape[0])
        tgt_pref = ttnn.slice(target_ids, [0, 0], [B_v, T])  # [B_v, T]
        match = ttnn.eq(draft_ids, tgt_pref)  # 1.0 where draft==target, else 0.0
        # n_acc = Σ_k Π_{j<=k} match[j]  (cumulative AND ⇒ count of leading 1s).
        run = ttnn.slice(match, [0, 0], [B_v, 1])  # running prefix-product [B_v,1]
        n_acc = run
        for k in range(1, T):
            col = ttnn.slice(match, [0, k], [B_v, k + 1])
            run = ttnn.mul(run, col)
            n_acc = ttnn.add(n_acc, run)
        return n_acc  # [B_v,1] float; host rounds to int

    # ── lifecycle + stats ─────────────────────────────────────────────────────

    def anchor_len_at(self, i: int) -> int:
        return self._anchor_len[i]

    def reset_slot(self, i: int) -> None:
        # Stale cache rows are harmless: anchor_len=0 ⇒ the mask attends no anchors.
        self._anchor_len[i] = 0
        self.slots[i].reset()

    def move_slot(self, dst: int, src: int) -> None:
        # The per-slot anchor cache lives at batch index `src` in a device tensor;
        # moving it would be a device copy. v1 instead drops dst's anchors
        # (anchor_len=0 ⇒ re-bootstraps from the next commits — a brief acceptance
        # dip after a compaction, never incorrect since the bonus always advances).
        if dst == src:
            return
        self.slots[dst] = self.slots[src]
        self.slots[src] = _SlotState()
        self._anchor_len[dst] = 0
        self._anchor_len[src] = 0

    def verify(self, slot_idx, draft_tokens, target_top1_per_position) -> int:
        n = 0
        for i, d in enumerate(draft_tokens):
            if i >= len(target_top1_per_position):
                break
            if target_top1_per_position[i] == d:
                n += 1
            else:
                break
        s = self.slots[slot_idx]
        s.n_accepted += n
        s.record_outcomes(n, len(draft_tokens))
        return n

    def commit(self, slot_idx, n_accepted, bonus=None) -> None:
        s = self.slots[slot_idx]
        s.history.extend(s.pending_drafts[:n_accepted])
        if bonus is not None:
            s.history.append(int(bonus))
            s.cur_pos += n_accepted + 1
        else:
            s.cur_pos += n_accepted
        s.pending_drafts = []

    def aggregate_stats(self) -> dict:
        tp_ = sum(s.n_proposed for s in self.slots)
        ta = sum(s.n_accepted for s in self.slots)
        ts = sum(s.n_steps_with_drafts for s in self.slots)
        recent = [o for s in self.slots for o in s.recent_outcomes]
        return {
            "total_proposed": tp_,
            "total_accepted": ta,
            "total_steps": ts,
            "acceptance_rate": (ta / tp_) if tp_ > 0 else 0.0,
            "mean_accepted_per_step": (ta / ts) if ts > 0 else 0.0,
            "mean_tokens_per_step": ((ta + ts) / ts) if ts > 0 else 0.0,
            "windowed_acceptance": (sum(recent) / len(recent)) if recent else 0.0,
            "windowed_n": len(recent),
            "active_slots": sum(1 for s in self.slots if s.n_proposed > 0),
        }
