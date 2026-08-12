# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import os
import time

import ttnn

from .operations import (
    apply_allreduce,
    apply_fused_output_projection_and_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

# Op-profiling for the packed-verify forward (GEMMA4_PV_OPPROF). The server
# sets PV_OPPROF["active"] around the packed-verify trace *warmup* only, so the
# section timers (and their device syncs) never run during trace capture or
# replay or any hot-path step. When the env flag is unset it is a no-op.
_PV_OPPROF = os.environ.get("GEMMA4_PV_OPPROF") == "1"
PV_OPPROF = {
    "active": False,
    "attn_prep": 0.0,
    "sdpa": 0.0,
    "mlp": 0.0,
    # fine-grained attn_prep sub-sections
    "qkv_split_norm": 0.0,
    "rope": 0.0,
    "kv_write": 0.0,
}


# SDPA-decode compute grid. 12x10 = max usable on P150 (13x10 trips the
# 341-runtime-arg writer cap at B=32); per-batch core count stays bounded by
# max_cores_per_head_batch=16, so the wide grid means more parallel batch rows.
# Flat for sliding (S_k=192); helps global S_k=4096 SDPA.
def _sdpa_grid(config):
    return ttnn.CoreCoord(12, 10)


# Cache for the L1 height-sharded ``MemoryConfig`` that
# ``nlp_create_qkv_heads_decode`` produces. The spec only depends on shape
# constants (B, num_heads_local, head_dim, qkv_dim, kv_replicated/global flags)
# — all fixed for a given model+batch. By caching across calls we avoid the
# per-layer probe decode-split that exists ONLY to discover this spec.
#
# Populated on first call (during the un-traced compile pass) and read from
# every subsequent call. Inside trace capture the probe is skipped entirely.
_Q_SHARDED_MEM_CACHE: dict = {}


def _q_sharded_mem_key(B, qkv_dim, config, weights, tp):
    """Hashable key for the q_sharded_mem cache. Captures every input that
    affects ``nlp_create_qkv_heads_decode``'s output shard spec."""
    return (
        int(B),
        int(qkv_dim),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        bool(weights.is_global),
        bool(weights.kv_replicated),
        int(tp),
    )


def decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    token_index,
    page_table=None,
    ccl_manager=None,
    is_kv_shared=False,
    position_idx_cache=None,
    fused_intermediate_buffer=None,
    fused_output_buffer=None,
    fused_program_config=None,
    page_table_sliding=None,
    position_idx_cache_sliding_write=None,
    position_idx_cache_sliding_sdpa=None,
):
    """
    Single-token decode attention, fully on device.

    Args:
        hidden_states: [1, 1, batch, hidden_size] on device
        cos_cache: [max_seq_len, head_dim] 2D cache for embedding lookup, or [1,1,max_seq_len,head_dim] 4D
        sin_cache: same format as cos_cache
        weights: AttentionWeights container
        kv_cache: [k_cache, v_cache] TT tensors (for shared layers, this is the source layer's cache)
        config: Gemma4AttentionConfig
        mesh_config: MeshConfig
        mesh_device: TT device
        position_idx: [batch] tensor of current positions for KV cache update + RoPE embedding lookup
        token_index: int position for legacy RoPE slicing (unused when cos_cache is 2D)
        page_table: optional paged attention table (full layers; also fallback for sliding)
        ccl_manager: optional CCL manager for TP > 1
        is_kv_shared: if True, skip K/V projection and cache update (use source layer's KV cache)
        page_table_sliding: optional page_table for sliding-window cache (smaller per-user range)
        position_idx_cache_sliding_write: [batch] int32 of cur_pos % W for ring-buffer writes
        position_idx_cache_sliding_sdpa: [batch] int32 of min(cur_pos, W-1) for SDPA reads
    """
    tp = mesh_config.tp if mesh_config else 1

    # 1. Fused QKV projection
    # L1 output required — nlp_create_qkv_heads_decode miscomputes with DRAM input on Blackhole (#16667).
    xqkv = apply_qkv_projection(hidden_states, weights, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 2. Split into Q, K, V heads
    tt_q, tt_k, tt_v = split_qkv_heads_decode(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

    # 3. Per-head norms (move to DRAM for rms_norm, restore sharded for RoPE)
    q_sharded_mem = tt_q.memory_config()
    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if is_kv_shared:
        # KV-shared layer: discard own K/V, use source layer's KV cache directly
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    else:
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # 4. RoPE — use on-device embedding lookup for trace compatibility
    use_embedding_rope = len(cos_cache.shape) == 2  # 2D cache = embedding lookup mode
    if use_embedding_rope:
        # Gather position-specific cos/sin via ttnn.embedding (fully on-device, trace-safe)
        # position_idx: [1, 32] uint32 padded tensor for embedding lookup
        cos_pos = ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)  # [1, batch_pad, head_dim]
        sin_pos = ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
        cos_pos = ttnn.unsqueeze_to_4D(cos_pos)  # [1, 1, batch_pad, head_dim]
        sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
        batch = tt_q.shape[1]
        if cos_pos.shape[2] != batch:
            cos_pos = cos_pos[:, :, :batch, :]
            sin_pos = sin_pos[:, :, :batch, :]
        # rotary_embedding expects cos/sin as [1, 1, *, head_dim] — token_index=0 indexes position 0
        # which holds the data for the actual current position (gathered by embedding above)
        tt_q = apply_rope(tt_q, cos_pos, sin_pos, token_index=0)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_pos, sin_pos, token_index=0)
    else:
        # Slice the current-position row first, matching the production
        # embedding-lookup path. Direct token_index slicing in rotary_embedding
        # gives incorrect decode results on TP mesh paths.
        cos_pos = cos_cache[:, :, token_index : token_index + 1, :]
        sin_pos = sin_cache[:, :, token_index : token_index + 1, :]
        tt_q = apply_rope(tt_q, cos_pos, sin_pos, token_index=0)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_pos, sin_pos, token_index=0)

    # 5. KV cache update — skip for KV-shared layers (source layer already updated the cache)
    # Use a rank-1 int32 position tensor for cache ops / SDPA. The RoPE
    # embedding path uses a different padded [1, batch_pad] uint32 tensor.
    # For sliding layers: cache writes use cur_pos % W (ring buffer);
    # SDPA reads use min(cur_pos, W-1) so the kernel iterates the whole ring.
    base_pos = position_idx_cache if position_idx_cache is not None else position_idx
    if len(base_pos.shape) == 2 and base_pos.shape[0] == 1 and base_pos.shape[1] == 1:
        base_pos = ttnn.reshape(base_pos, (1,))

    if config.is_sliding and position_idx_cache_sliding_write is not None:
        cache_write_pos = position_idx_cache_sliding_write
        if len(cache_write_pos.shape) == 2 and cache_write_pos.shape[0] == 1 and cache_write_pos.shape[1] == 1:
            cache_write_pos = ttnn.reshape(cache_write_pos, (1,))
    else:
        cache_write_pos = base_pos

    if config.is_sliding and position_idx_cache_sliding_sdpa is not None:
        sdpa_pos = position_idx_cache_sliding_sdpa
        if len(sdpa_pos.shape) == 2 and sdpa_pos.shape[0] == 1 and sdpa_pos.shape[1] == 1:
            sdpa_pos = ttnn.reshape(sdpa_pos, (1,))
    else:
        sdpa_pos = base_pos

    layer_page_table = page_table_sliding if (config.is_sliding and page_table_sliding is not None) else page_table

    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        if not is_kv_shared:
            # After HF-style RoPE, tensors may be in DRAM. Move to HEIGHT_SHARDED for cache update.
            tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
            tt_v = ttnn.to_memory_config(tt_v, q_sharded_mem)

            if layer_page_table is not None:
                ttnn.experimental.paged_update_cache(
                    k_cache, tt_k, update_idxs_tensor=cache_write_pos, page_table=layer_page_table
                )
                ttnn.experimental.paged_update_cache(
                    v_cache, tt_v, update_idxs_tensor=cache_write_pos, page_table=layer_page_table
                )
            else:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_write_pos)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_write_pos)
    else:
        k_cache = tt_k
        v_cache = tt_v

    # 6. SDPA (scale=1.0)
    sliding_window = config.sliding_window if config.is_sliding else None

    # Always pass an explicit decode config. Without one, SDPA decode uses the
    # full available grid as max_cores_per_head_batch, which can exceed the
    # tree-reduction kernel limit of 64 cores/head on Blackhole meshes.
    # Large head_dim global layers use a smaller grid.
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=_sdpa_grid(config),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )

    if layer_page_table is not None:
        tt_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=sdpa_pos,
            page_table_tensor=layer_page_table,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=sdpa_pos,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    tt_q.deallocate(True)

    # 7. Concat heads + fused output projection + allreduce
    tt_out = concat_heads(tt_sdpa, is_decode_mode=True)
    tt_out = apply_fused_output_projection_and_allreduce(
        tt_out,
        weights,
        mesh_config,
        ccl_manager,
        fused_intermediate_buffer,
        fused_output_buffer,
        program_config=fused_program_config,
    )

    return tt_out


def _verify_head_splits(B, H_local, nkv_local, P, head_dim, grid=None):
    """How many QUERY-HEAD-wise sub-ops to split the packed-verify decode-SDPA
    into (mirrors the dflash propose's ``DFlashDrafter._sdpa_head_splits``).

    The packed verify folds ``H_local*P`` query rows onto ``nkv_local`` KV
    groups; the per-core SDPA-decode CBs (Q, QK scores, and the flash cross-core
    reduction buffer) scale with the packed query-head tile count
    ``PNHt = H_local*P/32``. Gemma4's GLOBAL-attention layers (``global_head_dim
    == 512`` ⇒ DHt 16, and a single local KV head at tp≥4) overflow the 1.5 MB
    L1 even at the default k_chunk; we split the packed-head dim across ``n``
    full-grid ops + a concat, carrying ``H_local/n`` heads (PNHt/n) each. Sliding
    layers (head_dim 256, ``H_local*P/nkv·head_dim == 8192`` — not over the
    threshold) fit and are left as one op.

    Enabled ONLY when ``nkv_local == 1``: then every query head maps to the lone
    KV head, so each sub-op can SHARE the full (paged) K/V cache + page_table
    with NO cache slicing — any head subset is GQA-correct. (With >1 local KV
    head a head split would need to slice the paged cache per op, which is not
    worth it; such configs are left unsplit.) Each op keeps the full core grid
    and program config, so active-core count is preserved.

    The count is **B_v-adaptive**: the cross-core reduction CB scales with
    ``PNHt·(cores_per_head − 1)``, and ``cores_per_head`` grows as B_v shrinks
    (fewer batch groups over the same grid — e.g. 8 at B_v=4 but 16 at B_v≤2 on
    the 32-core global grid), so small buckets need more splitting. We estimate
    the per-core CB footprint (matching ``sdpa_decode_program_factory`` for the
    single-KV-head case) and pick the smallest valid split that fits a
    conservative L1 budget — never capping ``cores_per_head``.

    ``GEMMA4_PV_SDPA_HEAD_SPLITS`` overrides the count (clamped to the largest
    valid split ≤ the request)."""
    if nkv_local != 1:
        return 1
    # Valid splits: whole heads per op AND 32-tile-aligned query-row slices.
    valid = [d for d in range(1, H_local + 1) if H_local % d == 0 and ((H_local // d) * P) % 32 == 0]
    env = os.environ.get("GEMMA4_PV_SDPA_HEAD_SPLITS")
    if env:
        want = max(1, min(int(env), H_local))
        return max(d for d in valid if d <= want)
    if (H_local * P) * head_dim <= 8192:  # not heavy (sliding) — one op fits
        return 1
    # Mirror sdpa_decode_program_factory's core allocation (num_kv_heads == 1)
    # and CB tile counts; keep grid / max_cores_per_head_batch / k_chunk in sync
    # with `sdpa_program_config` below. Budget in 2 KB tiles, with margin under
    # the 768-tile (1.5 MB) L1 cap (the estimate runs ~4% high — conservative).
    if grid is None:
        grid = 32 if head_dim >= 512 else 64
    max_cores_per_head, k_chunk = 16, 64
    cores_per_head = max(1, min(grid, max_cores_per_head * B) // max(1, B))
    DHt, Sk = head_dim // 32, max(1, k_chunk // 32)
    BUDGET_TILES = 720

    def per_core_tiles(n):
        pnht = (H_local * P) // (32 * n)
        return (
            2 * pnht * DHt  # Q-in + tilized-Q
            + 5 * pnht * DHt  # 3 out-im + out-stats + out-final (vDHt == DHt)
            + 2 * pnht * Sk  # attn-mask + QK-im
            + 11 * pnht  # softmax stats CBs
            + 4 * Sk * DHt  # K + V (double-buffered)
            + 3  # scale / identity
            + (pnht * DHt + 2 * pnht) * (cores_per_head - 1)  # cross-core flash reduction
        )

    for d in valid:  # ascending ⇒ smallest split that fits the budget
        if per_core_tiles(d) <= BUDGET_TILES:
            return d
    return valid[-1]  # most aggressive split available


def _packed_verify_sdpa(q_packed, k, v, layer_page_table, attn_mask, scale, pc, n_splits, B, H_local, P, head_dim):
    """Run the packed-verify decode-SDPA, optionally head-split into ``n_splits``
    full-grid ops + a concat (see ``_verify_head_splits``). Splitting SHARES the
    (paged) K/V cache + page_table across sub-ops unsliced — valid only for a
    single local KV head — and slices only Q and the additive mask on the packed
    query-head dim. Handles both the paged and non-paged SDPA-decode entry
    points. Does NOT free its inputs. Returns [1, B, H_local*P, head_dim]."""

    def _call(q, m):
        if layer_page_table is not None:
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k,
                v,
                page_table_tensor=layer_page_table,
                is_causal=False,
                attn_mask=m,
                scale=scale,
                sliding_window_size=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
            )
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=m,
            scale=scale,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
        )

    if n_splits <= 1:
        return _call(q_packed, attn_mask)
    s_k = attn_mask.shape[3] if attn_mask is not None else None
    rows_per = (H_local // n_splits) * P  # packed query rows per sub-op (32-aligned)
    parts = []
    for s in range(n_splits):
        r0, r1 = s * rows_per, (s + 1) * rows_per
        q_i = ttnn.slice(q_packed, [0, 0, r0, 0], [1, B, r1, head_dim])
        m_i = ttnn.slice(attn_mask, [0, 0, r0, 0], [B, 1, r1, s_k]) if attn_mask is not None else None
        out_i = _call(q_i, m_i)
        ttnn.deallocate(q_i)
        if m_i is not None:
            ttnn.deallocate(m_i)
        parts.append(out_i)
    out = ttnn.concat(parts, dim=2)  # [1, B, H_local*P, head_dim]
    for p in parts:
        ttnn.deallocate(p)
    return out


def _packed_fill_kv_loopfree(cache, staging, new_seq, merge_idx, hot_pt):
    """Loop-free packed KV-cache write via PERSISTENT STAGING — replaces the
    per-position ``paged_update_cache`` loop, and never reads the committed
    cache on the hot path.

    ``staging`` is this layer's resident hot-block copy
    ``[1, nkv, S2, head_dim]`` (S2 = max_batch_size · BLK · block_size),
    slot-indexed: slot ``s`` owns seq ``[s·BLK·bs, (s+1)·BLK·bs)`` and already
    holds the committed prefix of its hot block(s) — seeded at prefill,
    maintained step-to-step (see server ``_seed_staging`` / ``_refresh_loopfree_
    write_idx``). ``new_seq`` is the freshly projected/normed/RoPE'd K (or V)
    ``[1, nkv, B_v·P, head_dim]`` (user-major / position-minor rows).

    Per step (all fixed-shape ⇒ trace-safe; cost scales with S2, NEVER with
    max_num_blocks — that is the whole point vs reading the cache):
      1. merge: gather along the seq dim of ``concat([staging, new_seq])`` —
         committed positions copy from staging (identity, or shifted by one
         block on a rollover), the P new positions pull from ``new_seq``.
         ``merge_idx`` encodes the per-position source.
      2. ``assign`` the merged hot blocks back into ``staging`` (persists for
         next step — no cache read needed next time).
      3. one ``paged_fill_cache`` writes each slot's hot block(s) to its
         physical page(s) named by ``hot_pt`` ( -1 = skip ⇒ idle slots / the
         unused spill block are left untouched — writer_fill_cache_interleaved
         .cpp:22,95).

    ``merge_idx`` ([S2] u32) is layer-INDEPENDENT (it addresses staging
    positions, not physical pages) and shared across layers; its full-shape
    gather index ``[1, nkv, S2, hd]`` is rebuilt on device via ttnn.repeat (tiny
    H2D). ``hot_pt`` ([1, max_batch_size·BLK] i32) is the full- or sliding-cache
    physical destination, selected by the caller. The dim-2 gather transposes
    only the small ``concat`` (S2 + B_v·P positions), not the cache.
    """
    dram = ttnn.DRAM_MEMORY_CONFIG
    nkv = staging.shape[1]
    S2 = staging.shape[2]
    head_dim = staging.shape[3]

    # ① merge resident staging with this step's new K/V — no committed-cache read.
    new_dram = ttnn.to_memory_config(new_seq, dram)
    src = ttnn.concat([staging, new_dram], dim=2, memory_config=dram)  # [1, nkv, S2 + B_v*P, hd]
    ttnn.deallocate(new_dram)
    # ttnn.gather needs a TILE-layout index (it tile-pads internally; a ROW_MAJOR
    # index trips FillPad's "tile layout only" assert). Build full-shape in
    # ROW_MAJOR (cheap broadcast) then tilize once; S2/head_dim are tile-aligned.
    midx = ttnn.reshape(merge_idx, (1, 1, S2, 1))
    midx = ttnn.repeat(midx, [1, nkv, 1, head_dim])  # RM [1, nkv, S2, hd]
    midx = ttnn.to_layout(midx, ttnn.TILE_LAYOUT)
    merged = ttnn.gather(src, dim=2, index=midx, memory_config=dram)  # [1, nkv, S2, hd]
    ttnn.deallocate(midx)
    ttnn.deallocate(src)

    # ② persist the updated hot blocks back into staging for the next step.
    ttnn.assign(merged, staging)

    # ③ one launch: write each slot's hot block(s) to its physical page(s).
    ttnn.experimental.paged_fill_cache(cache, merged, hot_pt, batch_idx=0)
    ttnn.deallocate(merged)


def _packed_fill_kv_loopfree_embed(cache, staging, new_seq, embed_idx, hot_pt):
    """``ttnn.embedding`` row-gather variant of ``_packed_fill_kv_loopfree``.

    The dim-2 gather in the baseline transposes the small ``concat`` and needs a
    full ``[1, nkv, S2, hd]`` index materialized on device every call (a
    ``repeat`` + ``to_layout`` of ~0.5M elems, ×2 per layer). ``ttnn.embedding``
    instead gathers whole rows of the *flattened* ``concat`` — transpose-free,
    and the index is the tiny per-head-flattened ``embed_idx`` (``[1, nkv·S2]``),
    so nothing large is built on device. Same merge semantics as the baseline;
    this is the same trace-safe device-index row-gather already used for the
    on-device mask gather (server ``_build_packed_verify_fwd``).

    ``embed_idx[h·S2 + j] = h·(S2+B_v·P) + merge_idx[j]`` — the host bakes the
    per-head row offset in (server ``_refresh_loopfree_write_idx``) so the
    flattened ``[nkv·(S2+B_v·P), hd]`` view is gathered with one index. ``nkv``
    differs full vs sliding, so the caller selects the matching index.
    """
    dram = ttnn.DRAM_MEMORY_CONFIG
    nkv = staging.shape[1]
    S2 = staging.shape[2]
    head_dim = staging.shape[3]
    src_seq = S2 + new_seq.shape[2]  # = S2 + B_v*P

    # ① merge resident staging with this step's new K/V — no committed-cache read.
    new_dram = ttnn.to_memory_config(new_seq, dram)
    src = ttnn.concat([staging, new_dram], dim=2, memory_config=dram)  # [1, nkv, src_seq, hd] TILE
    ttnn.deallocate(new_dram)
    # Flatten (nkv, seq) into the row axis and gather rows by embed_idx. The
    # reshape is metadata-only (src_seq and S2 are tile-aligned), embedding reads
    # bf16/TILE weights (same as the mask table), output is [1, nkv*S2, hd].
    src2d = ttnn.reshape(src, (nkv * src_seq, head_dim))
    merged = ttnn.embedding(embed_idx, src2d, layout=ttnn.TILE_LAYOUT)  # [1, nkv*S2, hd]
    ttnn.deallocate(src)
    merged = ttnn.reshape(merged, (1, nkv, S2, head_dim))

    # ② persist updated hot blocks for next step, then ③ one fill launch.
    ttnn.assign(merged, staging)
    ttnn.experimental.paged_fill_cache(cache, merged, hot_pt, batch_idx=0)
    ttnn.deallocate(merged)


def packed_decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    kv_write_idxs,
    attn_mask,
    packed_p,
    page_table=None,
    page_table_sliding=None,
    ccl_manager=None,
    is_kv_shared=False,
    kv_write_idxs_sliding=None,
    rope_packed=None,
    up_kv_write=False,
    kv_write_up_masked_full=None,
    kv_write_up_masked_sliding=None,
    page_table_up_full=None,
    page_table_up_sliding=None,
    kv_staging=None,
    merge_idx=None,
    hot_pt=None,
    hot_pt_sliding=None,
    kv_merge="embedding",
    embed_idx=None,
    embed_idx_sliding=None,
):
    """Packed multi-token decode attention — P query positions/slot in one pass.

    Hoists QKV projection, prefill-style split, per-head norm, and RoPE OUT of
    the per-p loop: those run once on the full B*P tensor. The only per-p
    work is slice + reshard + two paged_update_cache calls (K then V).

    (We DON'T use ``paged_fused_update_cache`` because it requires K and V on
    disjoint core grids; setting that up via ``create_sharded_memory_config``
    is fragile layout plumbing.)

    Args:
        hidden_states: [1, 1, B*P, hidden_size]. Rows are user-major
            position-minor — row u*P+p is slot u's p-th packed token.
        cos_cache, sin_cache: 2D RoPE caches [max_seq_len, head_dim].
        position_idx: [1, B*P] uint32 — RoPE position per row (cur_pos_u + p).
        kv_write_idxs: list of P int32 [B] device tensors — the cache write
            position for each packed position p (raw cur_pos_u + p; used by
            full-attention layers).
        kv_write_idxs_sliding: optional list of P int32 [B] device tensors —
            the sliding ring write index ((cur_pos_u + p) % W) for each packed
            position. Sliding layers need this because their K/V cache is a
            W-token ring; without it raw cur_pos+p writes the wrong ring slot
            once cur_pos >= W. When None, sliding layers fall back to
            kv_write_idxs (correct only while every position < W).
        attn_mask: [B, 1, H_local*P, S_k] head-major mask baking in the causal
            upper bound and (sliding layers) the window lower bound.
        packed_p: P, the number of packed positions per slot.
        rope_packed: optional ``(cos_bp, sin_bp)`` pre-gathered for this
            layer-type. When provided, skips the per-layer embedding +
            unsqueeze gather. The gather depends only on ``position_idx``
            and the per-layer-type 2D RoPE cache — both identical across
            all layers of a given type — so the server computes it once per
            type per step and shares it.

    Returns:
        [1, 1, B*P, hidden_size] — attention output for every packed position.

    Layout flow
    -----------
      hidden_states [1, 1, B*P, hidden]
        ─► apply_qkv_projection  ───► xqkv     [1, 1, B*P, qkv_dim]   L1
        ─► split_qkv_heads_prefill ─► Q,K,V    [1, H_local, B*P, hd], [1, nkv, B*P, hd]  L1
        ─► apply_per_head_norm × 3 (Q, K, V on B*P)                                       L1
        ─► gather cos, sin from 2D tables once (one embedding each)
        ─► apply_rope (prefill mode, kernel iterates along S=B*P) on Q and K              L1
        ─► reshape K, V from [1, nkv, B*P, hd] to [1, nkv, B, P, hd]                      L1
        ─► for p in range(P):
              slice on dim 3 (P axis) → [1, nkv, B, hd] view per packed pos              L1
              reshard to L1_HEIGHT_SHARDED (paged_update_cache requirement)              L1
              paged_update_cache(K_cache, K_p) ; paged_update_cache(V_cache, V_p)
        ─► reshape+permute Q from [1, H_local, B*P, hd] to [1, B, H_local*P, hd]          L1
        ─► to DRAM (SDPA-mandated) ─► packed SDPA ─► DRAM out
        ─► unpack permute back to L1 ─► concat_heads + o_proj ─► allreduce

    L1 residency
    ------------
    The activation stream stays resident on L1 from the QKV projection all the
    way to the o_proj output. The split is forced to L1 (interleaved L1 in →
    interleaved L1 out, since ``nlp_create_qkv_heads`` only emits sharded
    output for sharded input), and every norm / RoPE / reshape / permute /
    slice inherits or is pinned to L1. The two — and only two — points the
    activation leaves L1:
      1. SDPA: the decode kernel requires Q in DRAM (or height-sharded) and
         emits its output to DRAM (sharded output is unsupported for GQA).
         The unpack permute immediately pulls the result back onto L1.
      2. The o_proj → all-reduce comms tail: ``all_reduce_async``'s fabric op
         is validated only on its current DRAM layout, so o_proj writes DRAM.

    The per-p reshard to ``L1_HEIGHT_SHARDED_MEMORY_CONFIG`` (an L1→L1 move)
    lets ttnn auto-determine the shard spec from the slice's shape; the decode
    split's layout requirement is preserved through this reshard.

    The user-major / position-minor row order of ``hidden_states`` (row
    ``u*P+p`` = slot ``u`` position ``p``) is what makes the reshape
    ``[1, nkv, B*P, hd] → [1, nkv, B, P, hd]`` correct: dim 2 becomes (B,P)
    with B outer, P inner, matching the row ordering.

    The head-major Q packing (h*P+t, not t*H_local+h) is load-bearing: the SDPA
    kernel maps packed query head i → KV head i//group, so a KV group must be a
    contiguous head block. Token-major breaks GQA on sliding layers (2 KV
    heads/device) — proven in tests/unit/test_packed_sliding_sdpa.py.

    Caveats
    -------
      * apply_rope with ``token_index=None`` uses the prefill rotary kernel,
        which iterates along the S dim of the tensor and applies the
        corresponding cos/sin row. We feed S=B*P and cos/sin of shape
        [1, 1, B*P, head_dim] gathered at the position_idx values — one cos
        per (slot, packed_pos).
      * ``paged_update_cache`` requires an L1 height-sharded input layout;
        we reshard each per-p slice with ``to_memory_config`` using the
        spec discovered via the one-time decode-split probe.
    """
    P = packed_p
    tp = mesh_config.tp if mesh_config else 1
    B = hidden_states.shape[2] // P
    H_local = config.num_attention_heads // tp
    head_dim = config.head_dim
    nkv_local = 1 if weights.kv_replicated else config.num_key_value_heads // tp
    layer_page_table = page_table_sliding if (config.is_sliding and page_table_sliding is not None) else page_table
    write_idxs = kv_write_idxs_sliding if (config.is_sliding and kv_write_idxs_sliding is not None) else kv_write_idxs

    _op = _PV_OPPROF and PV_OPPROF["active"]
    if _op:
        ttnn.synchronize_device(mesh_device)
        _op_t = time.perf_counter()

    # ─── ① QKV projection (one call on the full B*P) ──────────────────────
    xqkv = apply_qkv_projection(hidden_states, weights, memory_config=ttnn.L1_MEMORY_CONFIG)
    qkv_dim = xqkv.shape[-1]

    # ─── ② L1 height-sharded MemoryConfig for paged_update_cache ─────────
    # ``paged_update_cache`` requires its input in the same L1 height-sharded
    # layout that ``nlp_create_qkv_heads_decode`` produces — generic
    # ``L1_HEIGHT_SHARDED_MEMORY_CONFIG`` carries no shard_spec and trips
    # "bad optional access" inside ``to_memory_config``. The real spec depends
    # only on shape/flag constants (see ``_q_sharded_mem_key``) so we cache
    # across calls.
    #
    # First call (compile pass before trace capture): run one decode-split on
    # a B-sized slice of xqkv to learn the spec, stash, deallocate the probe
    # outputs. Subsequent calls (including the actual trace capture): cache
    # hit, ZERO probe ops recorded — no probe ops run inside trace replay.
    # Number of decode "users" the KV-write reshard targets: the per-p loop
    # writes B users per position, so it needs the B-user spec; the (u,p)
    # single-shot write treats all B*P packed rows as decode users, so it needs
    # the B*P-user spec (B*P <= 32 by construction when up_kv_write is set).
    n_decode_users = B * P if up_kv_write else B
    cache_key = _q_sharded_mem_key(n_decode_users, qkv_dim, config, weights, tp)
    q_sharded_mem = _Q_SHARDED_MEM_CACHE.get(cache_key)
    if q_sharded_mem is None:
        # When the probe spans the whole xqkv (the (u,p) path, n_decode_users ==
        # B*P), ttnn.slice returns an ALIAS of xqkv — deallocating it would free
        # xqkv before the prefill split below ("Tensor is not allocated"). So
        # probe on xqkv directly in that case and free only the probe OUTPUTS.
        probe_is_full = n_decode_users == xqkv.shape[2]
        xqkv_probe = xqkv if probe_is_full else ttnn.slice(xqkv, [0, 0, 0, 0], [1, 1, n_decode_users, qkv_dim])
        q_probe, k_probe, v_probe = split_qkv_heads_decode(
            xqkv_probe, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        q_sharded_mem = q_probe.memory_config()
        _Q_SHARDED_MEM_CACHE[cache_key] = q_sharded_mem
        ttnn.deallocate(q_probe)
        ttnn.deallocate(k_probe)
        ttnn.deallocate(v_probe)
        if not probe_is_full:
            ttnn.deallocate(xqkv_probe)

    # ─── ②b Prefill-style split → L1 Q/K/V on B*P ────────────────────────
    # L1 output keeps the activation stream resident: the split, per-head
    # norms, RoPE, and every reshape/permute/slice below stay on L1. The
    # only DRAM hops are the SDPA call (kernel-mandated, see ⑥) and the
    # o_proj → all-reduce comms tail (CCL fabric op layout requirement).
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # Q: [1, H_local, B*P, head_dim], K/V: [1, nkv_local, B*P, head_dim]
    # xqkv is dead after the split — free it now.
    ttnn.deallocate(xqkv)

    # ─── ③ Per-head norms (one call each on B*P, kept on L1) ─────────────
    l1 = ttnn.L1_MEMORY_CONFIG
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=l1)
    if kv_cache is not None:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=l1)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False, memory_config=l1)

    # ─── ④ RoPE on B*P, prefill mode (kernel iterates along S=B*P) ────────
    # cos_bp / sin_bp: [1, 1, B*P, head_dim] TILE bf16. token_index=None →
    # prefill rotary kernel: tensor [1, heads, S, hd] × cos/sin [1, 1, S, hd]
    # element-wise along S. One call per Q, one per K.
    #
    # The gather (embedding + unsqueeze_to_4D, 4 ops) depends only on
    # ``position_idx`` and the per-layer-type 2D RoPE cache — both identical
    # across all layers of a given type. When the caller pre-computes them
    # once per type (typical: server's _build_packed_verify_fwd), it passes
    # them in via ``rope_packed`` and we skip the gather entirely.
    if rope_packed is not None:
        cos_bp, sin_bp = rope_packed
        owns_rope = False
    else:
        cos_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        owns_rope = True
    if _op:
        ttnn.synchronize_device(mesh_device)
        _rope_t = time.perf_counter()
        PV_OPPROF["qkv_split_norm"] += _rope_t - _op_t

    tt_q = apply_rope(tt_q, cos_bp, sin_bp, token_index=None, memory_config=l1)
    if kv_cache is not None:
        tt_k = apply_rope(tt_k, cos_bp, sin_bp, token_index=None, memory_config=l1)
    if owns_rope:
        ttnn.deallocate(cos_bp)
        ttnn.deallocate(sin_bp)

    if _op:
        ttnn.synchronize_device(mesh_device)
        _now = time.perf_counter()
        PV_OPPROF["rope"] += _now - _rope_t
        PV_OPPROF["attn_prep"] += _now - _op_t
        _op_t = _now

    # ─── ⑤ Loop-free K/V write (persistent staging → one paged_fill_cache) ──
    # Active when the caller supplies the staging buffers + merge index (server
    # packed-verify path). Writes every row's P new tokens for the whole bucket
    # in ONE paged_fill_cache per K/V — no per-position loop, no reshard, and no
    # committed-cache read (the hot blocks live resident in ``kv_staging``).
    # ``merge_idx`` is shared full/sliding; only ``hot_pt`` (fill destination)
    # differs by layer type. See ``_packed_fill_kv_loopfree``.
    if kv_cache is not None and kv_staging is not None and merge_idx is not None:
        # Park Q off L1 (it is idle until the SDPA pack ⑥); the merge
        # intermediates live in DRAM, so this just preserves the existing
        # DRAM-resident Q the pack step expects on the packed path.
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        k_cache_w, v_cache_w = kv_cache
        k_stg, v_stg = kv_staging
        h_pt = hot_pt_sliding if (config.is_sliding and hot_pt_sliding is not None) else hot_pt
        if kv_merge == "embedding" and embed_idx is not None:
            # Transpose-free ttnn.embedding row-gather merge (per-head index;
            # nkv differs full vs sliding so select the matching index).
            e_idx = embed_idx_sliding if (config.is_sliding and embed_idx_sliding is not None) else embed_idx
            _packed_fill_kv_loopfree_embed(k_cache_w, k_stg, tt_k, e_idx, h_pt)
            _packed_fill_kv_loopfree_embed(v_cache_w, v_stg, tt_v, e_idx, h_pt)
        else:
            _packed_fill_kv_loopfree(k_cache_w, k_stg, tt_k, merge_idx, h_pt)
            _packed_fill_kv_loopfree(v_cache_w, v_stg, tt_v, merge_idx, h_pt)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)

    # ─── ⑤ FALLBACK per-p loop (direct/legacy callers without index tensors) ──
    elif kv_cache is not None:
        # Spill Q to DRAM for the duration of the KV-write loop. ``tt_q`` (the
        # full packed Q, H_local heads on B*P rows) is idle between RoPE (④) and
        # the SDPA pack (⑥) — it isn't read until step ⑥. Leaving it resident on
        # L1 here grows the interleaved-L1 region downward until it collides with
        # ``paged_update_cache``'s statically-allocated circular buffers (the CB
        # region clashes with an L1 buffer once B*P is large enough — e.g. P=8,
        # B=32). Parking Q in DRAM frees those L1 cores for the per-p reshards +
        # paged_update_cache CBs; the SDPA pack (⑥) then runs the ROW_MAJOR
        # untilize/reshape/permute on the DRAM tensor and tilizes straight back
        # into the DRAM q_packed the SDPA kernel wants, so Q never re-enters L1.
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        k_cache_w, v_cache_w = kv_cache
        # ``paged_fused_update_cache`` enforces
        # ``page_table.padded_shape()[0] == input_tensor.padded_shape()[1]`` —
        # the input must be in decode-style layout ``[1, B, nkv, head_dim]``
        # (B on dim 1), which is what ``nlp_create_qkv_heads_decode``
        # produces. Our prefill-style split outputs ``[1, nkv, B*P, head_dim]``
        # (heads on dim 1, B*P on dim 2). Convert by permuting dim 1 ↔ dim 2,
        # then reshape to expose the P axis for per-position slicing.
        #
        # Land the permuted K/V in DRAM (not L1). The full ``[1, B*P, nkv, hd]``
        # K and V stay resident across the whole per-p paged_update_cache loop;
        # at P=8/B=32 keeping them on L1 collides with paged_update_cache's
        # statically-allocated circular buffers — the same clash the Q spill
        # above only partially cleared. With K/V in DRAM, each per-p slice is
        # resharded DRAM→L1 height-sharded one position at a time
        # (``[1, B, nkv, hd]``), so only that single small reshard — not the
        # full B*P tensors — occupies L1 when the CBs are allocated.
        tt_k_bp = ttnn.permute(tt_k, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, B*P, nkv, hd] DRAM
        tt_v_bp = ttnn.permute(tt_v, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, B*P, nkv, hd] DRAM
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)
        if up_kv_write:
            # ── (u,p) masked KV write (B*P <= 32) ───────────────────────────
            # Reshard the whole [1, B*P, nkv, hd] tensor ONCE to the B*P-user
            # decode spec (q_sharded_mem probed at n_decode_users=B*P above — no
            # crash, no padding; B*P <= 32). Then write each packed position in a
            # SEPARATE sequential paged_update_cache, using a per-position masked
            # write-idx that is real only on that position's (u,p) rows (-1
            # elsewhere → skipped). Within a call the active rows are DISTINCT
            # users → distinct cache tiles, so there is no tile read-modify-write
            # race; across calls the writes to a user's tile are sequential, so
            # consecutive positions accumulate correctly (exactly the invariant
            # the classic per-p loop relies on). The reshard is identical in
            # shape/spec to the per-p reshard at B=32 ([1, 32, nkv, hd]).
            up_writes = (
                kv_write_up_masked_sliding
                if (config.is_sliding and kv_write_up_masked_sliding is not None)
                else kv_write_up_masked_full
            )
            up_pt = (
                page_table_up_sliding
                if (config.is_sliding and page_table_up_sliding is not None)
                else page_table_up_full
            )
            # Reshard FRESH per call: paged_update_cache does not tolerate its
            # L1 height-sharded input being reused by a second call (the CB is
            # consumed), so each of the P writes gets its own reshard of the
            # full [1, B*P, nkv, hd] tensor — then is deallocated, exactly like
            # the per-p loop's reshard→write→deallocate cycle.
            for p in range(P):
                k_sh = ttnn.to_memory_config(tt_k_bp, q_sharded_mem)
                v_sh = ttnn.to_memory_config(tt_v_bp, q_sharded_mem)
                if up_pt is not None:
                    ttnn.experimental.paged_update_cache(
                        k_cache_w, k_sh, update_idxs_tensor=up_writes[p], page_table=up_pt
                    )
                    ttnn.experimental.paged_update_cache(
                        v_cache_w, v_sh, update_idxs_tensor=up_writes[p], page_table=up_pt
                    )
                else:
                    ttnn.experimental.paged_update_cache(k_cache_w, k_sh, update_idxs_tensor=up_writes[p])
                    ttnn.experimental.paged_update_cache(v_cache_w, v_sh, update_idxs_tensor=up_writes[p])
                ttnn.deallocate(k_sh)
                ttnn.deallocate(v_sh)
            ttnn.deallocate(tt_k_bp)
            ttnn.deallocate(tt_v_bp)
        else:
            # ── per-p loop (B_v == 32: B*P > 32 doesn't fit the decode split) ──
            # Reshape to [1, B, P, nkv, head_dim]. Valid because hidden_states is
            # user-major / position-minor (row u*P+p = slot u position p) and the
            # permute above preserves that ordering on dim 1.
            tt_k_view = ttnn.reshape(tt_k_bp, (1, B, P, nkv_local, head_dim))
            tt_v_view = ttnn.reshape(tt_v_bp, (1, B, P, nkv_local, head_dim))

            for p in range(P):
                # Slice on dim 2 (P axis) — [1, B, 1, nkv, hd] view of position p.
                k_p = ttnn.slice(tt_k_view, [0, 0, p, 0, 0], [1, B, p + 1, nkv_local, head_dim])
                k_p = ttnn.reshape(k_p, (1, B, nkv_local, head_dim))
                v_p = ttnn.slice(tt_v_view, [0, 0, p, 0, 0], [1, B, p + 1, nkv_local, head_dim])
                v_p = ttnn.reshape(v_p, (1, B, nkv_local, head_dim))

                # Reshard DRAM slice → L1 height-sharded using the spec discovered
                # via the decode-split probe above.
                k_p = ttnn.to_memory_config(k_p, q_sharded_mem)
                v_p = ttnn.to_memory_config(v_p, q_sharded_mem)

                # K and V cache writes. ``paged_fused_update_cache`` would fuse
                # these into one launch but enforces "input_tensor1 and
                # input_tensor2 must not overlap" on core grids — fitting both
                # K and V into disjoint core ranges requires custom
                # ``create_sharded_memory_config`` plumbing (see
                # tt_transformers.attention.to_qk_fused_memory_config for the
                # recipe).
                if layer_page_table is not None:
                    ttnn.experimental.paged_update_cache(
                        k_cache_w,
                        k_p,
                        update_idxs_tensor=write_idxs[p],
                        page_table=layer_page_table,
                    )
                    ttnn.experimental.paged_update_cache(
                        v_cache_w,
                        v_p,
                        update_idxs_tensor=write_idxs[p],
                        page_table=layer_page_table,
                    )
                else:
                    ttnn.experimental.paged_update_cache(
                        k_cache_w,
                        k_p,
                        update_idxs_tensor=write_idxs[p],
                    )
                    ttnn.experimental.paged_update_cache(
                        v_cache_w,
                        v_p,
                        update_idxs_tensor=write_idxs[p],
                    )
                ttnn.deallocate(k_p)
                ttnn.deallocate(v_p)
            ttnn.deallocate(tt_k_bp)
            ttnn.deallocate(tt_v_bp)

    if _op:
        ttnn.synchronize_device(mesh_device)
        _now = time.perf_counter()
        PV_OPPROF["kv_write"] += _now - _op_t
        PV_OPPROF["attn_prep"] += _now - _op_t
        _op_t = _now

    k_cache_use, v_cache_use = kv_cache if kv_cache is not None else (None, None)

    # ─── ⑥ Head-major pack Q → [1, B, H_local*P, head_dim] → SDPA ─────────
    # tt_q is [1, H_local, B*P, head_dim]; reorder to head-major
    # [1, B, H_local*P, head_dim] (split B*P → B,P, swap H_local↔B, merge
    # H_local,P). Done in ROW_MAJOR on purpose: P (< 32 = tile height) never
    # lands on a tile axis, so the split/merge reshapes are free views and the
    # rank-5 permute is a single strided copy. In TILE these same reshapes
    # re-tile (untilize→retile) AND pad P up to 32 — an ~8× data blow-up at
    # P=4. Here one untilize + one tilize replace two padded re-tile cycles;
    # the logical result is identical (untilize/tilize are lossless for bf16).
    # Inherit tt_q's current memory domain — do NOT pin to L1. On the packed
    # (kv_cache) path tt_q was spilled to DRAM above, so this untilize and the
    # reshape/permute below run in DRAM, then tilize straight back to the
    # DRAM q_packed the SDPA kernel wants — Q never re-enters L1. (Forcing it
    # back to L1 here re-creates the very L1 collision the spill avoided and
    # adds a DRAM→L1 untilize the original never did; with fast-runtime-mode
    # validation off that surfaced as a SIGILL instead of a clean throw.) The
    # ROW_MAJOR rationale below is about layout, not memory domain, so all of
    # its benefits hold in DRAM. (kv_cache None ⇒ tt_q is still L1 TILE and
    # this stays an L1→L1 untilize, exactly as before.)
    tt_q = ttnn.to_layout(tt_q, ttnn.ROW_MAJOR_LAYOUT)  # RM [1, H_local, B*P, hd] (DRAM on packed path)
    tt_q = ttnn.reshape(tt_q, (1, H_local, B, P, head_dim))  # free view
    tt_q = ttnn.permute(tt_q, (0, 2, 1, 3, 4))  # [1, B, H_local, P, hd]
    tt_q = ttnn.reshape(tt_q, (1, B, H_local * P, head_dim))  # free view
    # SDPA decode kernel requires Q tilized in DRAM — tilize + place in one op.
    q_packed = ttnn.to_layout(tt_q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_q)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=_sdpa_grid(config),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )
    # Global-attention layers (head_dim 512 ⇒ DHt 16, single local KV head)
    # overflow L1 in one packed decode-SDPA at small B_v; split the packed
    # query-head dim into full-grid sub-ops + concat (no K/V-cache slicing —
    # all heads share the lone KV head). Mirrors the dflash propose head split;
    # sliding layers stay a single op. See `_verify_head_splits`.
    _grid = sdpa_program_config.compute_with_storage_grid_size
    n_sdpa_splits = _verify_head_splits(B, H_local, nkv_local, P, head_dim, grid=_grid.x * _grid.y)
    tt_sdpa = _packed_verify_sdpa(
        q_packed,
        k_cache_use,
        v_cache_use,
        layer_page_table,
        attn_mask,
        1.0,
        sdpa_program_config,
        n_sdpa_splits,
        B,
        H_local,
        P,
        head_dim,
    )

    # ─── ⑦ Unpack head-major SDPA output → concat heads + o_proj + AR ─────
    # SDPA emits TILE/DRAM (kernel-mandated). Untilize straight onto L1 (the
    # untilize doubles as the DRAM→L1 move), reorder in ROW_MAJOR — same
    # reasoning as the pack (⑥): with P off the tile axis the reshapes are
    # free views and the permute is one strided copy, no P→32 padding — then
    # tilize back to L1 for concat_heads. After this the activation never
    # touches DRAM again until the all-reduce tail.
    tt_sdpa = ttnn.to_layout(tt_sdpa, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)  # DRAM TILE → L1 RM
    tt_sdpa = ttnn.reshape(tt_sdpa, (1, B, H_local, P, head_dim))  # free view
    tt_sdpa = ttnn.permute(tt_sdpa, (0, 1, 3, 2, 4))  # [1, B, P, H_local, head_dim]
    tt_sdpa = ttnn.reshape(tt_sdpa, (1, B * P, H_local, head_dim))  # free view
    tt_sdpa = ttnn.to_layout(tt_sdpa, ttnn.TILE_LAYOUT, memory_config=l1)  # L1 TILE for concat_heads

    if _op:
        ttnn.synchronize_device(mesh_device)
        PV_OPPROF["sdpa"] += time.perf_counter() - _op_t

    tt_out = concat_heads(tt_sdpa, is_decode_mode=True, memory_config=l1)
    # o_proj output → DRAM: it feeds the all-reduce, whose fabric op is
    # validated only on its current (DRAM) layout. This + SDPA are the sole
    # points the activation leaves L1.
    tt_out = apply_output_projection(tt_out, weights)
    if tp > 1:
        tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)
    return tt_out
