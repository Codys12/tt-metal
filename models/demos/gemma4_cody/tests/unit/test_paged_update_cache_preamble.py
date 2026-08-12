# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone divergence test: verify_pre preamble → packed_decode_forward.

The unified spec-decode trace in ``server/server.py`` derives the per-step
inputs to ``paged_update_cache`` on-device:

    cur_pos [B] int32 RM (persistent, written by commit)
       ├─ reshape → cur_pos_col [B, 1]
       ├─ add(arange_P [1, P]) → position_idx_2d [B, P] int32
       ├─ reshape → position_idx_flat [1, B*P] int32
       ├─ maximum(0) → position_idx_safe_i32 [1, B*P] int32
       └─ typecast → _spec_position_idx_dev [1, B*P] uint32
                                  (consumed by RoPE embedding gather)

       position_idx_2d
       └─ for p in range(P):
           slice([0,p], [B,p+1]) → reshape((B,))
           → _spec_kv_write_idxs_dev[p] [B] int32 RM
           (consumed as update_idxs_tensor by paged_update_cache)

Scratch's tree (``/mnt/nas/scratch/models/demos/gemma4_cody``) builds these
host-side and H2Ds them each step. Empirically, scratch's verify trace works
and ours hangs inside ``paged_update_cache``.

Two tests:
  1. ``test_device_vs_host_index_derivation_matches`` — pure index-derivation
     check. Verifies the on-device chain produces values bit-identical to the
     host-equivalent computation. Empirically: PASSES (2026-05-19) — the
     indices are correct.
  2. ``test_server_attention_path_eager_vs_trace`` — faithful repro of the
     server's verify_model attention call site. Builds a real Gemma4Attention
     with random weights, allocates the same persistent buffers as
     ``_preallocate_packed_verify_buffers``, then runs
     ``preamble_fwd`` (the verify_pre index derivation) followed by
     ``packed_decode_forward`` (the layer's attention path: apply_qkv →
     split → DRAM round-trip → norm → rope → ``to_memory_config(q_sharded_mem)``
     → paged_update_cache → SDPA → out_proj → allreduce). Mode-eager mirrors
     the server's warmup compile; mode-trace captures the same body and
     replays it. If trace replay hangs while eager passes — exactly the
     server's failure mode — this test isolates the bug to those two ops
     interacting inside a trace.

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=4xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_paged_update_cache_preamble.py -k 1x4
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.demos.gemma4_cody.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.attention.decode import packed_decode_forward
from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4_cody.tt.ccl import CCLManager
from models.demos.gemma4_cody.tt.model import create_rope_caches

from ...tests.test_factory import TestFactory, parametrize_mesh_with_fabric

# Match the server: B=32 slots, P=5 packed positions (4 drafts + 1 bonus).
B_TEST = 32
P_TEST = 5
PREFILL_LEN = 64
BLOCK_SIZE = 64
# "Deeply-negative" idle sentinel matching what the server populates
# ``_spec_cur_pos_dev`` with for inactive slots. The server's design relies on
# ``paged_update_cache`` no-op'ing for negative write_idxs.
IDLE_CUR_POS = -10_000_000


def _alloc_dev(mesh_device, host_tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=None):
    """Replicate-mapper-aware H2D allocator that mirrors server.py:_alloc_device_tensor."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    return ttnn.from_torch(
        host_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=mapper,
        memory_config=memory_config,
    )


def _d2h(tensor):
    """First-shard D2H read — for replicated tensors all shards are identical."""
    if hasattr(tensor.device(), "get_num_devices") and tensor.device().get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


def _build_arange_P(mesh_device, P):
    return _alloc_dev(mesh_device, torch.arange(P, dtype=torch.int32).reshape(1, P), ttnn.int32)


def _derive_preamble_device(mesh_device, cur_pos_dev, arange_P_dev, B, P):
    """Replicate ``_build_verify_pre_fwd`` index-derivation block, with the
    proposed fix: clamp kv_write_idxs to >= -1.

    Rationale: ``paged_update_cache``'s ``update_idxs_tensor`` recognises -1
    as the skip sentinel. Any other negative value (e.g. our
    ``_SPEC_INACTIVE = -(1<<30)`` for idle slots, propagated through
    ``cur_pos + p``) is treated as a real (out-of-bounds) write index and
    hangs the kernel. Clamping to -1 collapses every deep-negative slot's
    derived index to the kernel's recognised skip sentinel.

    Returns:
        position_idx_dev: [1, B*P] uint32 RM — clamped to >=0 for RoPE.
        kv_write_idxs_dev: P × [B] int32 RM — clamped to >=-1 (skip-sentinel).
    """
    cur_pos_col = ttnn.reshape(cur_pos_dev, (B, 1))
    position_idx_2d = ttnn.add(cur_pos_col, arange_P_dev)
    position_idx_flat = ttnn.reshape(position_idx_2d, (1, B * P))

    # RoPE input: clamp >=0 → uint32.
    position_idx_safe_i32 = ttnn.maximum(position_idx_flat, 0)
    position_idx_safe = ttnn.typecast(position_idx_safe_i32, ttnn.uint32)
    ttnn.deallocate(position_idx_safe_i32)

    # paged_update_cache write indices: clamp >=-1. Idle slots whose
    # ``cur_pos + p`` came out as a deep-negative value collapse to -1,
    # which paged_update_cache treats as "skip this slot's write."
    write_idxs_2d_clamped = ttnn.maximum(position_idx_2d, -1)

    kv_write_idxs = []
    for p in range(P):
        col = ttnn.slice(write_idxs_2d_clamped, [0, p], [B, p + 1])
        col_b = ttnn.reshape(col, (B,))
        col_clone = ttnn.clone(col_b)
        ttnn.deallocate(col)
        kv_write_idxs.append(col_clone)
    ttnn.deallocate(write_idxs_2d_clamped)

    return position_idx_safe, kv_write_idxs


def _derive_preamble_host(cur_pos_torch, B, P):
    """Host-derived equivalent (matches scratch's _refresh_packed_verify_inputs).

    Mirrors _derive_preamble_device exactly, including the >=-1 clamp on
    kv_write_idxs.
    """
    arange_P = torch.arange(P, dtype=torch.int32).reshape(1, P)
    position_idx_2d = cur_pos_torch.reshape(B, 1) + arange_P  # [B, P] int32
    position_idx_flat = position_idx_2d.reshape(1, B * P)
    position_idx_safe = torch.clamp(position_idx_flat, min=0).to(torch.int32)
    write_idxs_2d_clamped = torch.clamp(position_idx_2d, min=-1)
    kv_write_idxs = [write_idxs_2d_clamped[:, p].clone().to(torch.int32) for p in range(P)]
    return position_idx_safe, kv_write_idxs


def _fill_kv_cache_random(mesh_device, kv_cache, page_table_tt, nkv_local, head_dim, prefill_len, block_size, B):
    """Populate the K/V caches with deterministic per-slot data so the test can
    detect overwrites by paged_update_cache."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    tp = mesh_device.shape[1] if is_mesh else 1
    padded = ((prefill_len + block_size - 1) // block_size) * block_size

    seed_k = (
        torch.arange(B * nkv_local * padded * head_dim, dtype=torch.float32)
        .reshape(1, nkv_local, B * padded, head_dim)
        .to(torch.bfloat16)
    )
    seed_v = (seed_k * 2.0).to(torch.bfloat16)

    for dev_idx in range(tp):
        dev_k = ttnn.get_device_tensors(kv_cache[0])[dev_idx] if is_mesh else kv_cache[0]
        dev_v = ttnn.get_device_tensors(kv_cache[1])[dev_idx] if is_mesh else kv_cache[1]
        dev_pt = ttnn.get_device_tensors(page_table_tt)[dev_idx] if is_mesh else page_table_tt
        dev = dev_k.device()
        for u in range(B):
            k_u = seed_k[:, :, u * padded : (u + 1) * padded, :]
            v_u = seed_v[:, :, u * padded : (u + 1) * padded, :]
            k_tt = ttnn.from_torch(k_u, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            v_tt = ttnn.from_torch(v_u, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            ttnn.experimental.paged_fill_cache(dev_k, k_tt, dev_pt, batch_idx=u)
            ttnn.experimental.paged_fill_cache(dev_v, v_tt, dev_pt, batch_idx=u)


# ───────────────────────────────────────────────────────────────────────────
# Test 1: pure preamble check (passes — proves indices are correct).
# ───────────────────────────────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [5], ids=["global"])
def test_device_vs_host_index_derivation_matches(layer_idx, mesh_device):
    """Pure preamble check — device-derived indices == host-derived."""
    torch.manual_seed(0)
    B, P = B_TEST, P_TEST

    cur_pos_host = torch.zeros(B, dtype=torch.int32)
    for u in range(B):
        cur_pos_host[u] = PREFILL_LEN + u if u % 2 == 0 else IDLE_CUR_POS

    cur_pos_dev = _alloc_dev(mesh_device, cur_pos_host, ttnn.int32)
    arange_P_dev = _build_arange_P(mesh_device, P)

    pos_idx_dev, kv_write_idxs_dev = _derive_preamble_device(mesh_device, cur_pos_dev, arange_P_dev, B, P)

    pos_idx_host_expected, kv_write_idxs_host_expected = _derive_preamble_host(cur_pos_host, B, P)

    pos_idx_got = _d2h(pos_idx_dev).to(torch.int64).reshape(B * P)
    pos_idx_exp = pos_idx_host_expected.to(torch.int64).reshape(B * P)
    assert torch.equal(
        pos_idx_got, pos_idx_exp
    ), f"position_idx_safe mismatch:\n  device : {pos_idx_got.tolist()}\n  host   : {pos_idx_exp.tolist()}"

    for p in range(P):
        got_p = _d2h(kv_write_idxs_dev[p]).to(torch.int64).reshape(B)
        exp_p = kv_write_idxs_host_expected[p].to(torch.int64)
        assert torch.equal(
            got_p, exp_p
        ), f"kv_write_idxs[{p}] mismatch:\n  device : {got_p.tolist()}\n  host   : {exp_p.tolist()}"


# ───────────────────────────────────────────────────────────────────────────
# Test 8: sliding-ring kv_write_idxs across the W boundary — exact iter-2
# hang the production server hits with --max-seq-len 4096 and speculative
# decode.
# ───────────────────────────────────────────────────────────────────────────


def _derive_sliding_write_idxs_device(mesh_device, write_idxs_2d_clamped, W, log_W):
    """Compute sliding-cache write indices on device.

    Mirrors the fixed-server derivation in ``_build_verify_pre_fwd``: starting
    from ``write_idxs_2d_clamped`` (= ``max(cur_pos + arange_P, -1)``, already
    -1 for idle rows), produce ``(cur_pos + p) % W`` for active rows while
    preserving -1 for idle rows.

    W is required to be a power of 2 (the bitwise-and trick is exact only in
    that case).

    Mapping per row::

                       X  bitwise_and(X, W-1)  rshift(X, 31)  lshift(., log_W)  sum
        active c+p >=0 → (c+p) & (W-1) = (c+p)%W         0                0  →  (c+p)%W
        idle    -1     → (W-1)                          -1               -W  →  -1
    """
    mod_W = ttnn.bitwise_and(write_idxs_2d_clamped, W - 1)
    sign_mask = ttnn.bitwise_right_shift(write_idxs_2d_clamped, 31)  # arithmetic
    sign_shift = ttnn.bitwise_left_shift(sign_mask, log_W)
    sliding_2d = ttnn.add(mod_W, sign_shift)
    ttnn.deallocate(sign_shift)
    ttnn.deallocate(sign_mask)
    ttnn.deallocate(mod_W)
    return sliding_2d


def _derive_full_and_sliding_preamble(mesh_device, cur_pos_dev, arange_P_dev, B, P, W, log_W):
    """End-to-end preamble for verify_pre: position_idx + kv_write_idxs (full)
    + kv_write_idxs_sliding ((c+p) % W).

    Returns:
        position_idx_dev [1, B*P] uint32 RM        — RoPE / mask gather input
        kv_write_idxs        : P × [B] int32 RM    — full-attention writes
        kv_write_idxs_sliding: P × [B] int32 RM    — sliding-attention writes
    """
    cur_pos_col = ttnn.reshape(cur_pos_dev, (B, 1))
    position_idx_2d = ttnn.add(cur_pos_col, arange_P_dev)
    position_idx_flat = ttnn.reshape(position_idx_2d, (1, B * P))

    position_idx_safe_i32 = ttnn.maximum(position_idx_flat, 0)
    position_idx_safe = ttnn.typecast(position_idx_safe_i32, ttnn.uint32)
    ttnn.deallocate(position_idx_safe_i32)

    # Full-attention writes (raw cur_pos + p clamped to >= -1 for the
    # idle skip sentinel).
    write_idxs_2d_clamped = ttnn.maximum(position_idx_2d, -1)
    kv_write_idxs = []
    for p in range(P):
        col = ttnn.slice(write_idxs_2d_clamped, [0, p], [B, p + 1])
        col_b = ttnn.reshape(col, (B,))
        col_clone = ttnn.clone(col_b)
        ttnn.deallocate(col)
        kv_write_idxs.append(col_clone)

    # Sliding-attention writes ((c+p) % W; -1 preserved for idle).
    sliding_2d = _derive_sliding_write_idxs_device(mesh_device, write_idxs_2d_clamped, W, log_W)
    ttnn.deallocate(write_idxs_2d_clamped)
    kv_write_idxs_sliding = []
    for p in range(P):
        col = ttnn.slice(sliding_2d, [0, p], [B, p + 1])
        col_b = ttnn.reshape(col, (B,))
        col_clone = ttnn.clone(col_b)
        ttnn.deallocate(col)
        kv_write_idxs_sliding.append(col_clone)
    ttnn.deallocate(sliding_2d)

    return position_idx_safe, kv_write_idxs, kv_write_idxs_sliding


# Pick a sliding layer for this test (the buggy path only fires on
# ``config.is_sliding`` layers).  Standard Gemma4 ``layer_types`` pattern is
# 5 sliding then 1 global per group of 6; layer 0 is the first sliding layer.
_SLIDING_LAYER_IDX = 0


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [_SLIDING_LAYER_IDX], ids=["sliding"])
def test_sliding_kv_write_idxs_device_correctness(layer_idx, mesh_device):
    """Pure preamble check for the on-device sliding-mod derivation.

    With ``cur_pos`` spanning the W boundary (so ``cur_pos + p`` crosses W
    for some rows), confirm that the device-derived kv_write_idxs_sliding
    matches the host-derived ``(c + p) % W`` for active rows and stays -1
    for idle rows.

    This is the equivalent of test 1 (``test_device_vs_host_index_derivation_matches``)
    but for the new sliding-ring index.  No SDPA, no paged_update_cache — just
    the index derivation, so it can never hang and runs in milliseconds.
    """
    torch.manual_seed(0)
    B, P = B_TEST, P_TEST
    W = 1024  # production sliding_cache_len; the test only needs a power of 2 <= max cur_pos
    log_W = 10

    # cur_pos pattern designed to exercise:
    #   - far from W (slot 0): cur_pos = 64,  c+p in [64,68]  → mod = c+p
    #   - crossing W (slot 2): cur_pos = 1022, c+p in [1022,1026] → mod = [1022,1023,0,1,2]
    #   - beyond W (slot 4):   cur_pos = 1500, c+p in [1500,1504] → mod = [476,477,478,479,480]
    #   - idle (odd slots): _SPEC_INACTIVE                  → mod = -1 (skip)
    active_positions = {0: 64, 2: 1022, 4: 1500, 6: 2047, 8: 4090}
    cur_pos_host = torch.full((B,), IDLE_CUR_POS, dtype=torch.int32)
    for u, c in active_positions.items():
        if u < B:
            cur_pos_host[u] = c

    cur_pos_dev = _alloc_dev(mesh_device, cur_pos_host, ttnn.int32)
    arange_P_dev = _build_arange_P(mesh_device, P)

    pos_idx_dev, kv_write_idxs_dev, kv_write_idxs_sliding_dev = _derive_full_and_sliding_preamble(
        mesh_device, cur_pos_dev, arange_P_dev, B, P, W, log_W
    )

    # Host reference.
    expected_full = []
    expected_sliding = []
    for p in range(P):
        row_full = torch.full((B,), -1, dtype=torch.int64)
        row_sliding = torch.full((B,), -1, dtype=torch.int64)
        for u, c in active_positions.items():
            if u < B:
                row_full[u] = c + p  # raw position; safe because we only set non-idle slots here
                row_sliding[u] = (c + p) % W
        expected_full.append(row_full)
        expected_sliding.append(row_sliding)

    for p in range(P):
        got_full = _d2h(kv_write_idxs_dev[p]).to(torch.int64).reshape(B)
        got_slid = _d2h(kv_write_idxs_sliding_dev[p]).to(torch.int64).reshape(B)
        assert torch.equal(got_full, expected_full[p]), (
            f"kv_write_idxs (full) mismatch p={p}:\n"
            f"  device : {got_full.tolist()}\n"
            f"  expect : {expected_full[p].tolist()}"
        )
        assert torch.equal(got_slid, expected_sliding[p]), (
            f"kv_write_idxs_sliding mismatch p={p}:\n"
            f"  device : {got_slid.tolist()}\n"
            f"  expect : {expected_sliding[p].tolist()}\n"
            "If active slots show c+p instead of (c+p) %% W, the bitwise_right_shift "
            "is logical (zero-extended) instead of arithmetic (sign-extended) — "
            "swap the op or compute the sign mask via ttnn.lt(X, 0)."
        )


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [_SLIDING_LAYER_IDX], ids=["sliding"])
@pytest.mark.parametrize("mode", ["eager", "trace"], ids=lambda v: f"mode-{v}")
def test_sliding_packed_decode_crosses_W_two_steps(layer_idx, mode, mesh_device):
    """End-to-end repro of the iter-2 hang on a sliding-attention layer.

    ``cur_pos`` is positioned so that iter 1 writes positions just below W and
    iter 2 writes positions crossing W. With the production server's bug
    (sliding layers fed RAW ``cur_pos + p`` instead of ``(c + p) % W``), iter 2
    would index ``page_table_sliding`` past its
    ``blocks_per_user_sliding = W / block_size`` entries → OOB read of the
    page table → ``paged_update_cache`` hangs at trace replay.

    PREFILL_LEN = W - P + 1 sits one short of the W boundary so iter 1's
    write positions are entirely inside [W - P + 1, W - 1] (block_idx
    W/block_size - 1, safely within the slot's
    ``blocks_per_user_sliding`` pages) and iter 2's positions are entirely
    above W (block_idx W/block_size, one past the slot's last sliding page).

    This test passes the device-derived ``(c + p) % W`` indices to the sliding
    layer (the FIX). If anyone reverts to feeding raw ``c + p`` (the BUG),
    iter 2's ``execute_trace`` would hang here — making this test the
    regression guard for the server fix.
    """
    from models.tt_transformers.tt.common import PagedAttentionConfig

    torch.manual_seed(0)
    B, P = B_TEST, P_TEST
    block_size = BLOCK_SIZE
    W = 1024
    log_W = 10

    # Position the canary's cur_pos so iter 0 stays *inside* the slot's sliding
    # cache (block_idx < W/block_size) and iter 1 (after one advance of P) has
    # at least one packed position that crosses W (block_idx == W/block_size,
    # one past the last entry of page_table_sliding → OOB without (c+p)%W).
    #
    # iter 0 positions: [cur_pos, cur_pos + P-1]   safe when cur_pos + P - 1 < W
    # iter 1 positions: [cur_pos + P, cur_pos + 2P - 1]  hangs when
    #                   cur_pos + 2P - 1 ≥ W
    # → choose cur_pos ∈ [W - 2P + 1, W - P]. Lower bound = 1015 for W=1024,P=5.
    prefill_len = W - 2 * P + 1  # 1015 for W=1024, P=5
    advance_per_iter = P  # steady-state full acceptance: bonus + T drafts

    logger.info(
        f"[SLIDE-W mode={mode}] starting" f" — prefill_len={prefill_len}, W={W}, P={P}, advance={advance_per_iter}"
    )

    # ── Build attention with synthetic weights so we don't load the real model.
    hf_text_config = TestFactory.create_hf_text_config(num_experts=1, top_k=1)
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    assert config.is_sliding, (
        f"layer_idx={layer_idx} resolved to layer_type={hf_text_config.layer_types[layer_idx]!r}; "
        "this test requires a sliding-attention layer to exercise the "
        "page_table_sliding write path."
    )

    hidden_size = hf_text_config.hidden_size
    nkv_global = config.num_key_value_heads
    head_dim = config.head_dim
    n_q_heads = config.num_attention_heads
    q_size = n_q_heads * head_dim
    kv_size = nkv_global * head_dim
    state_dict = {
        "q_proj.weight": torch.randn(q_size, hidden_size, dtype=torch.bfloat16),
        "k_proj.weight": torch.randn(kv_size, hidden_size, dtype=torch.bfloat16),
        "v_proj.weight": torch.randn(kv_size, hidden_size, dtype=torch.bfloat16),
        "o_proj.weight": torch.randn(hidden_size, q_size, dtype=torch.bfloat16),
        "q_norm.weight": torch.randn(head_dim, dtype=torch.bfloat16),
        "k_norm.weight": torch.randn(head_dim, dtype=torch.bfloat16),
    }

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    H_local = n_q_heads // tp
    kv_replicated = nkv_global < tp
    local_kv = 1 if kv_replicated else nkv_global // tp
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1

    # Sliding cache: blocks_per_user_sliding = W / block_size pages per slot.
    blocks_per_user_sliding = W // block_size
    max_num_blocks = B * blocks_per_user_sliding
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    kv_cache = init_kv_cache(mesh_device, config, paged_attention_config=paged_cfg, cache_dtype=ttnn.bfloat16)
    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        max_batch_size=B,
    )
    weights = tt_attn.weights

    # Sliding page table: one slot's worth of rings, replicated B-wise. Each
    # slot has exactly ``blocks_per_user_sliding`` ring blocks; the kernel
    # uses page_table[r, block_idx] to translate ring positions → physical
    # pages. With block_idx == blocks_per_user_sliding (one past), the lookup
    # reads past the row → OOB.
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B, blocks_per_user_sliding)
    page_table_tt = _alloc_dev(mesh_device, page_table, ttnn.int32)
    # Seed the K/V caches with deterministic data so the SDPA reads from
    # initialized memory (otherwise mask=NEG positions still get *read*
    # before being masked out, and reading from uninitialized device memory
    # can trip data-corruption guards).
    _fill_kv_cache_random(
        mesh_device,
        kv_cache,
        page_table_tt,
        local_kv,
        head_dim,
        prefill_len,
        block_size,
        B,
    )
    ttnn.synchronize_device(mesh_device)

    # 2D RoPE caches sized to cover iter-2's positions (which cross W).
    max_seq_len_rope = prefill_len + 2 * P  # well within the 2D table's range
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len_rope)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos_cache_2d, sin_cache_2d = rope_caches_2d[layer_type]

    # cur_pos pattern:
    #   - canary_slot (u=0):   prefill_len (= W - 2P + 1). iter-0 inside, iter-1
    #                          crosses W. This is the slot that would hang
    #                          without the (c+p)%W fix.
    #   - other even slots :   64, a safely small position. Iter 0 and iter 1
    #                          stay far below W. Exercises non-critical active
    #                          rows so the test isn't single-row trivial.
    #   - odd slots         :  IDLE_CUR_POS. The bitwise-and trick must
    #                          preserve -1 (skip sentinel) for these.
    cur_pos_host = torch.full((B,), IDLE_CUR_POS, dtype=torch.int32)
    canary_slot = 0
    cur_pos_host[canary_slot] = prefill_len
    safe_position = 64
    for u in range(2, B, 2):
        cur_pos_host[u] = safe_position
    assert (
        cur_pos_host[canary_slot] == prefill_len
    ), f"canary slot must be at prefill_len={prefill_len}, got {cur_pos_host[canary_slot]}"

    cur_pos_dev = _alloc_dev(mesh_device, cur_pos_host, ttnn.int32)
    arange_P_dev = _build_arange_P(mesh_device, P)

    # Persistent buffers (mirror server's _spec_*).
    position_idx_dev = _alloc_dev(
        mesh_device,
        torch.zeros(1, B * P, dtype=torch.int32),
        ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_write_idxs_dev = [
        _alloc_dev(mesh_device, torch.zeros(B, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for _ in range(P)
    ]
    kv_write_idxs_sliding_dev = [
        _alloc_dev(mesh_device, torch.zeros(B, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for _ in range(P)
    ]

    x_packed = torch.randn(1, 1, B * P, hidden_size, dtype=torch.float32).to(torch.bfloat16)
    hidden_states_dev = _alloc_dev(mesh_device, x_packed, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Causal mask covers W keys (sliding cache extent). For active rows we
    # only need a well-formed *shape* for SDPA to accept — the exact mask
    # values don't matter for hang detection (the real verify_pre derives
    # these from a precomputed mask_table_sliding gather; the kernel reads
    # all W keys per query regardless of the mask).
    NEG = float(-1e9)
    mask_torch = torch.zeros(B, 1, H_local * P, W, dtype=torch.float32)
    for u in range(B):
        c = int(cur_pos_host[u].item())
        if c < 0:
            continue  # idle: full-zero mask is fine (skip happens via -1 write_idx)
        for h in range(H_local):
            for p in range(P):
                ub = (c + p) % W
                mask_torch[u, 0, h * P + p, ub + 1 :] = NEG
    attn_mask_dev = _alloc_dev(mesh_device, mask_torch.to(torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    advance_host = torch.full((B,), advance_per_iter, dtype=torch.int32)
    advance_dev = _alloc_dev(mesh_device, advance_host, ttnn.int32)

    def _preamble_into_persistent():
        """Run the on-device preamble, persisting into the test's buffers.

        Mirrors the (fixed) server's ``_build_verify_pre_fwd``: derives
        both full and sliding kv_write_idxs on device — sliding gets
        ``(cur_pos + p) % W`` while full gets raw ``cur_pos + p`` (clamped
        to -1 for idle).
        """
        position_idx_local, kv_write_idxs_local, kv_write_idxs_sliding_local = _derive_full_and_sliding_preamble(
            mesh_device, cur_pos_dev, arange_P_dev, B, P, W, log_W
        )
        ttnn.copy(position_idx_local, position_idx_dev)
        ttnn.deallocate(position_idx_local)
        for p in range(P):
            ttnn.copy(kv_write_idxs_local[p], kv_write_idxs_dev[p])
            ttnn.deallocate(kv_write_idxs_local[p])
            ttnn.copy(kv_write_idxs_sliding_local[p], kv_write_idxs_sliding_dev[p])
            ttnn.deallocate(kv_write_idxs_sliding_local[p])

    def _attn_call():
        out = packed_decode_forward(
            hidden_states=hidden_states_dev,
            cos_cache=cos_cache_2d,
            sin_cache=sin_cache_2d,
            weights=weights,
            kv_cache=kv_cache,
            config=config,
            mesh_config=mesh_config,
            mesh_device=mesh_device,
            position_idx=position_idx_dev,
            kv_write_idxs=kv_write_idxs_dev,
            kv_write_idxs_sliding=kv_write_idxs_sliding_dev,
            attn_mask=attn_mask_dev,
            packed_p=P,
            page_table=None,
            page_table_sliding=page_table_tt,
            ccl_manager=ccl_manager,
        )
        ttnn.deallocate(out)

    def _advance_cur_pos():
        new_cur_pos = ttnn.add(cur_pos_dev, advance_dev)
        ttnn.copy(new_cur_pos, cur_pos_dev)
        ttnn.deallocate(new_cur_pos)

    if mode == "eager":
        for it in range(2):
            logger.info(f"[SLIDE-W eager iter {it}] preamble (cur_pos crosses W: {it == 1})")
            _preamble_into_persistent()
            ttnn.synchronize_device(mesh_device)
            logger.info(f"[SLIDE-W eager iter {it}] attn")
            _attn_call()
            ttnn.synchronize_device(mesh_device)
            logger.info(f"[SLIDE-W eager iter {it}] advance")
            _advance_cur_pos()
            ttnn.synchronize_device(mesh_device)
            logger.info(f"[SLIDE-W eager iter {it}] DONE")

    elif mode == "trace":
        # Warmup compile.
        logger.info("[SLIDE-W trace] warmup")
        _preamble_into_persistent()
        _attn_call()
        _advance_cur_pos()
        ttnn.synchronize_device(mesh_device)

        # Reset cur_pos (warmup advanced it).
        cur_pos_reset = _alloc_dev(mesh_device, cur_pos_host, ttnn.int32)
        ttnn.copy(cur_pos_reset, cur_pos_dev)
        ttnn.deallocate(cur_pos_reset)
        ttnn.synchronize_device(mesh_device)

        # Capture.
        logger.info("[SLIDE-W trace] begin_trace_capture")
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        _preamble_into_persistent()
        _attn_call()
        _advance_cur_pos()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        # Two replays. Iter 0 writes positions below W; iter 1 writes
        # positions crossing W. If the sliding kv_write_idxs were the raw
        # (c+p) instead of (c+p) % W, iter 1's paged_update_cache would index
        # past the slot's sliding page table and hang here.
        for it in range(2):
            logger.info(f"[SLIDE-W trace iter {it}] execute_trace (crosses W: {it == 1})")
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            logger.info(f"[SLIDE-W trace iter {it}] DONE")

    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Post-condition: read the on-device kv_write_idxs_sliding back and check
    # that the canary slot's iter-2 indices wrapped around to [1, P]
    # (W=1024 boundary crossed). The persistent buffer holds the last
    # preamble's values — iter 2 ran with cur_pos = prefill_len +
    # advance_per_iter (1025 here), so the stored indices are (1025+p) % W.
    # If the bitwise-and-and-sign-shift derivation is broken, these would be
    # the raw (c+p) values (1025..1029, OOB for the sliding cache), and the
    # earlier iter-2 packed_decode_forward would have hung — but we add this
    # explicit check anyway in case the kernel silently writes garbage.
    cur_pos_before_iter2_preamble = prefill_len + advance_per_iter
    final_sliding = [_d2h(t).to(torch.int64).reshape(B)[canary_slot].item() for t in kv_write_idxs_sliding_dev]
    expected_final_sliding = [(cur_pos_before_iter2_preamble + p) % W for p in range(P)]
    assert final_sliding == expected_final_sliding, (
        f"canary slot iter-2 sliding kv_write_idxs mismatch:\n"
        f"  got    : {final_sliding}\n"
        f"  expect : {expected_final_sliding}\n"
        f"  cur_pos at iter-2 preamble = {cur_pos_before_iter2_preamble}, W = {W}.\n"
        f"If got matches raw (c+p) instead of (c+p) %% W, the sliding-mod "
        f"derivation is broken."
    )

    logger.info(f"[SLIDE-W mode={mode} DONE] both iterations completed; canary sliding indices={final_sliding}")


def _find_layer_idx_of_type(layer_types, target_type):
    """Return the first index in ``layer_types`` whose entry equals
    ``target_type``. Raises if not found (the production Gemma4 config has
    both types, so a miss means the loaded config is wrong)."""
    for i, lt in enumerate(layer_types):
        if lt == target_type:
            return i
    raise AssertionError(
        f"layer_type {target_type!r} not present in layer_types={layer_types!r} — "
        "expected the real Gemma4 5-sliding-then-1-full pattern from config.json"
    )


def _build_one_tt_decoder_layer_real_weights(
    mesh_device, mesh_config, ccl_manager, model_args, state_dict, tensor_cache_path, layer_idx, max_seq_len_for_layer
):
    """Build one TT ``Gemma4DecoderLayer`` loaded with the REAL production
    weights for ``layer_idx`` from ``$HF_MODEL``.

    The TT layer's constructor calls ``substate(state_dict, "model.layers.{i}")``
    or ``substate(state_dict, "model.language_model.layers.{i}")`` internally
    to pick out just the layer's tensors — when ``state_dict`` is a
    ``LazyStateDict`` (mmap-backed), only the per-layer subset is actually
    read from the safetensors shards. The tt-metal as_tensor cache at
    ``tensor_cache_path`` further short-circuits re-conversion on repeat
    runs.
    """
    from models.demos.gemma4_cody.tt.layer import Gemma4DecoderLayer

    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=state_dict,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=tensor_cache_path,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len_for_layer,
        max_local_batch_size=B_TEST,
    )
    return tt_layer


def _load_embed_and_lm_head(mesh_device, mesh_config, model_args, state_dict, tensor_cache_path):
    """Load the production-tied embed_tokens + lm_head weights from the real
    state_dict, replicating ``Gemma4Model``'s sharding (embed_tokens
    column-parallel on hidden dim; lm_head column-parallel on vocab dim).

    Returns ``(embedding_weight, lm_head_weight, norm_weight_tensor,
    embed_scale, final_logit_softcapping)`` — enough to call ``embed_tokens``
    and the verify-head softcapped lm_head pipeline standalone.
    """
    from models.demos.gemma4_cody.utils.general_utils import cached_tensor_placeholder, get_cache_file_name

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tp = mesh_config.tp if mesh_config else 1
    tp_suffix = f"_tp{tp}" if tp > 1 else ""

    # Pick the same embed key the model uses.
    if "model.language_model.embed_tokens.weight" in state_dict:
        embed_key = "model.language_model.embed_tokens.weight"
    elif "model.embed_tokens.weight" in state_dict:
        embed_key = "model.embed_tokens.weight"
    else:
        raise KeyError("no embed_tokens.weight in state_dict")

    embed_cache_name = get_cache_file_name(tensor_cache_path, f"embed_tokens.weight{tp_suffix}")
    lm_head_cache_name = get_cache_file_name(tensor_cache_path, f"lm_head.weight{tp_suffix}")
    embed_weight = None
    embed_tensor = cached_tensor_placeholder(embed_cache_name, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    lm_head_tensor = cached_tensor_placeholder(lm_head_cache_name, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    if embed_tensor is None or lm_head_tensor is None:
        embed_weight = state_dict[embed_key]

    if tp > 1:
        embed_mapper = mesh_config.column_parallel(mesh_device)
    else:
        embed_mapper = replicate
    if embed_tensor is None:
        embed_tensor = embed_weight.unsqueeze(0).unsqueeze(0)
    embedding_weight = ttnn.as_tensor(
        embed_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=embed_mapper,
        cache_file_name=embed_cache_name,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if lm_head_tensor is None:
        lm_head_tensor = embed_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    if tp > 1:
        lm_mapper = mesh_config.column_parallel(mesh_device)
    else:
        lm_mapper = replicate
    lm_head_weight = ttnn.as_tensor(
        lm_head_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=lm_mapper,
        cache_file_name=lm_head_cache_name,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Final norm weight.
    if "model.language_model.norm.weight" in state_dict:
        norm_w = state_dict["model.language_model.norm.weight"]
    elif "model.norm.weight" in state_dict:
        norm_w = state_dict["model.norm.weight"]
    else:
        raise KeyError("no model.norm.weight in state_dict")
    norm_weight_tt = ttnn.from_torch(
        norm_w.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate,
    )

    embed_scale = float(model_args.hidden_size) ** 0.5
    final_logit_softcapping = getattr(model_args, "final_logit_softcapping", 30.0) or 30.0

    return embedding_weight, lm_head_weight, norm_weight_tt, embed_scale, final_logit_softcapping


# ───────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE SERVER SPEC_FUSED REPRO
#
# Status (after eliminations through 2026-05-20):
#   * Verify_model alone × N twice                    PASSES at N=60
#   * Verify_model with full chunks (chunks=v_only)    PASSES at N=60
#   * Chunks=all_chunks with 2 shared real layers     PASSES at N=60
# So whatever the server hangs on, it needs MORE than "real-weight target
# layers + spec_fused chunks + 2 replays". This test ramps the remaining
# axes:
#
#   distinct_layers  : False = 2 layer objects (one sliding + one full)
#                              reused 5:1, the cheap path tested above.
#                      True  = N TT layers built with each layer index's
#                              own real weights — matches the server's
#                              per-layer weight variation. Heavy: 60 layers
#                              of state_dict access and column-parallel
#                              weight conversion.
#
#   with_real_drafter: False = mock propose (embed_tokens + lm_head +
#                              global_argmax × T, exercising the CCL
#                              pattern but not running the real 4-layer
#                              drafter). Same as test 10's all_chunks.
#                      True  = load Gemma4AssistantModel from
#                              $GEMMA4_DRAFTER_PATH and call its real
#                              forward inside propose. Adds 4 sliding/full
#                              drafter layers per draft step + the
#                              drafter's own per-layer allgathers. Heavy:
#                              ~1 GB drafter weight load on top.
#
#   n_replays        : 2 = the server's failure point (step-0 OK, step-1
#                          hangs). Default.
#                      5 = in case the bug is latent and shows after a few
#                          replays.
#
# Suggested ramp order (most → least likely to reproduce):
#   distinct-True_drafter-True_reps5  ← full server fidelity
#   distinct-True_drafter-True_reps2
#   distinct-True_drafter-False_reps5
#   distinct-True_drafter-False_reps2
#   distinct-False_drafter-True_reps5
#   distinct-False_drafter-True_reps2
#   distinct-False_drafter-False_reps5
#   distinct-False_drafter-False_reps2 ← baseline (= test 10 all_chunks)
#
# The FIRST combination that hangs is the data point that tells us what
# the production server's spec_fused trace depends on.
# ───────────────────────────────────────────────────────────────────────────


def _build_n_distinct_tt_layers(
    mesh_device,
    mesh_config,
    ccl_manager,
    model_args,
    state_dict,
    tensor_cache_path,
    n_layers,
    max_seq_len,
    layer_types,
):
    """Build N TT decoder layers, one per layer index. Each layer has its
    OWN real production weights — the heavy escalation past the 2-layer
    shared-weights setup. Caches across runs via ``tensor_cache_path``.
    """
    tt_layers = []
    for layer_idx in range(n_layers):
        lt = layer_types[layer_idx]
        logger.info(f"[REPRO] building TT layer {layer_idx}/{n_layers} ({lt})…")
        tt_layer = _build_one_tt_decoder_layer_real_weights(
            mesh_device,
            mesh_config,
            ccl_manager,
            model_args,
            state_dict,
            tensor_cache_path,
            layer_idx=layer_idx,
            max_seq_len_for_layer=max_seq_len,
        )
        tt_layers.append(tt_layer)
    return tt_layers


def _build_real_drafter_with_rope(mesh_device, mesh_config, ccl_manager, assistant_path, max_seq_len):
    """Load the real Gemma4AssistantModel drafter from ``assistant_path``
    (= ``$GEMMA4_DRAFTER_PATH``) plus the per-layer-type 2D RoPE caches the
    server's propose loop reads via ``ttnn.embedding`` gathers.

    Mirrors ``server.py:_build_drafter_rope_caches_device`` exactly so the
    propose chunk inside this test takes the same shape inputs as the
    server's actual propose."""
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_cody.tt.assistant.config import Gemma4AssistantConfig
    from models.demos.gemma4_cody.tt.assistant.model import Gemma4AssistantModel

    logger.info(f"[REPRO] loading real drafter from {assistant_path}…")
    drafter_config = Gemma4AssistantConfig.from_hf_path(assistant_path)
    drafter_cache_dir = os.environ.get("GEMMA4_DRAFTER_CACHE_DIR") or os.path.join(
        os.environ.get("TT_CACHE_PATH", assistant_path),
        "drafter_tensor_cache_bf16",
    )
    drafter = Gemma4AssistantModel(
        mesh_device=mesh_device,
        config=drafter_config,
        cache_dir=drafter_cache_dir,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
    )

    # 2D RoPE caches per layer type (sliding / full). Server's
    # _build_drafter_rope_caches_device does exactly this.
    hf_cfg = AutoConfig.from_pretrained(assistant_path, trust_remote_code=True)
    text_cfg = hf_cfg.get_text_config() if hasattr(hf_cfg, "get_text_config") else hf_cfg
    position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    rope_module = Gemma4TextRotaryEmbedding(text_cfg)
    dummy = torch.zeros(1, 1, text_cfg.hidden_size, dtype=torch.float32)

    cos_cache_dev = {}
    sin_cache_dev = {}
    head_dim_per_type = {}
    for lt in dict.fromkeys(text_cfg.layer_types):
        cos, sin = rope_module(dummy, position_ids, layer_type=lt)
        cos_2d = cos[0].to(torch.bfloat16).contiguous()
        sin_2d = sin[0].to(torch.bfloat16).contiguous()
        cos_cache_dev[lt] = _alloc_dev(mesh_device, cos_2d, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        sin_cache_dev[lt] = _alloc_dev(mesh_device, sin_2d, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        head_dim_per_type[lt] = cos.shape[-1]

    return drafter, cos_cache_dev, sin_cache_dev, head_dim_per_type


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("n_layers_per_iter", [60], ids=lambda v: f"L{v}")
@pytest.mark.parametrize("distinct_layers", [False, True], ids=lambda v: f"distinct-{v}")
@pytest.mark.parametrize("n_replays", [2, 5], ids=lambda v: f"reps{v}")
@pytest.mark.parametrize("with_real_drafter", [False, True], ids=lambda v: f"drafter-{v}")
def test_server_spec_fused_repro(n_layers_per_iter, distinct_layers, n_replays, with_real_drafter, mesh_device):
    """Most server-faithful spec_fused trace repro. See the comment block
    above this function for the elimination history and the suggested ramp
    order over (distinct_layers, with_real_drafter, n_replays).

    Always includes:
      * Full spec_fused chunk sequence: propose + verify_pre + verify_model
        + verify_head + commit
      * Real production weights for target layers, embed_tokens, lm_head,
        and final norm (loaded from $HF_MODEL/*.safetensors via LazyStateDict)
      * 5:1 sliding-then-full layer rhythm from the real config
      * Real per-layer-type RoPE caches
      * In-place state advances (cur_pos, next_token, spec_hidden) in commit

    Per-parametrize add-ons:
      * distinct_layers=True  → N distinct TT layers, each with its own real
                                weights at the matching layer_idx.
      * with_real_drafter=True → Gemma4AssistantModel from $GEMMA4_DRAFTER_PATH
                                drives propose's T drafter forwards.
      * n_replays=5           → 5 trace replays after capture (vs server's 2).

    Mode is always trace (eager always passes; the bug is trace-replay specific).
    """
    from models.demos.gemma4_cody.tt.ccl import ccl_allgather
    from models.demos.gemma4_cody.tt.model_config import Gemma4ModelArgs
    from models.tt_transformers.tt.common import PagedAttentionConfig

    torch.manual_seed(0)
    B, P = B_TEST, P_TEST
    T_drafts = P - 1
    block_size = BLOCK_SIZE
    prefill_len = 64
    max_seq_len_test = 256
    W_test = 128
    log_W_test = int(math.log2(W_test))

    logger.info(
        f"[REPRO N={n_layers_per_iter} distinct={distinct_layers} "
        f"reps={n_replays} drafter={with_real_drafter}] starting"
    )

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    model_path = os.environ.get("HF_MODEL") or os.environ.get(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )
    if not os.path.isdir(model_path):
        pytest.skip(f"need $HF_MODEL ({model_path}) — real weights required")

    assistant_path = os.environ.get("GEMMA4_DRAFTER_PATH", "/mnt/nas/gemma-31b-assistant")
    if with_real_drafter and not os.path.isdir(assistant_path):
        pytest.skip(
            f"with_real_drafter=True needs $GEMMA4_DRAFTER_PATH ({assistant_path}) "
            "to be a readable drafter model directory."
        )

    hf_config_full = Gemma4ModelArgs.load_hf_config(model_path)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config_full)
    hf_text_config = getattr(hf_config_full, "text_config", hf_config_full)
    model_args._hf_text_config = hf_text_config
    layer_types = list(hf_text_config.layer_types)

    sliding_layer_idx = _find_layer_idx_of_type(layer_types, "sliding_attention")
    full_layer_idx = _find_layer_idx_of_type(layer_types, "full_attention")
    sliding_attn_cfg = Gemma4AttentionConfig(model_args, sliding_layer_idx)
    full_attn_cfg = Gemma4AttentionConfig(model_args, full_layer_idx)
    hidden_size = model_args.hidden_size
    n_q_heads = sliding_attn_cfg.num_attention_heads
    H_local = n_q_heads // tp

    logger.info(f"[REPRO] opening real state_dict from {model_path}")
    state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
    tensor_cache_path = str(model_args.weight_cache_path(model_path, ttnn.bfloat16))

    # ── Build TT target layers.
    if distinct_layers:
        # N distinct TT layers with real per-index weights — the heavy path.
        tt_layers = _build_n_distinct_tt_layers(
            mesh_device,
            mesh_config,
            ccl_manager,
            model_args,
            state_dict,
            tensor_cache_path,
            n_layers=n_layers_per_iter,
            max_seq_len=max_seq_len_test,
            layer_types=layer_types,
        )

        def _layer_for_call(k):
            return tt_layers[k]

    else:
        logger.info("[REPRO] building 2 TT layers (one per type, reused 5:1)…")
        tt_layer_sliding = _build_one_tt_decoder_layer_real_weights(
            mesh_device,
            mesh_config,
            ccl_manager,
            model_args,
            state_dict,
            tensor_cache_path,
            layer_idx=sliding_layer_idx,
            max_seq_len_for_layer=max_seq_len_test,
        )
        tt_layer_full = _build_one_tt_decoder_layer_real_weights(
            mesh_device,
            mesh_config,
            ccl_manager,
            model_args,
            state_dict,
            tensor_cache_path,
            layer_idx=full_layer_idx,
            max_seq_len_for_layer=max_seq_len_test,
        )

        def _layer_for_call(k):
            return tt_layer_sliding if layer_types[k] == "sliding_attention" else tt_layer_full

    # ── Load embed + lm_head + final norm.
    logger.info("[REPRO] loading embed_tokens + lm_head + final norm…")
    embedding_weight, lm_head_weight, final_norm_weight, embed_scale, final_logit_softcapping = _load_embed_and_lm_head(
        mesh_device, mesh_config, model_args, state_dict, tensor_cache_path
    )

    # ── Optionally load the real drafter.
    drafter = drafter_cos_cache = drafter_sin_cache = None
    if with_real_drafter:
        drafter, drafter_cos_cache, drafter_sin_cache, _ = _build_real_drafter_with_rope(
            mesh_device,
            mesh_config,
            ccl_manager,
            assistant_path,
            max_seq_len=max_seq_len_test,
        )

    # ── Per-call KV caches sized per-layer-type.
    blocks_per_user_full = (max_seq_len_test + block_size - 1) // block_size
    blocks_per_user_slide = W_test // block_size
    paged_cfg_full = PagedAttentionConfig(
        block_size=block_size,
        max_num_blocks=B * blocks_per_user_full,
    )
    paged_cfg_slide = PagedAttentionConfig(
        block_size=block_size,
        max_num_blocks=B * blocks_per_user_slide,
    )

    call_schedule = [layer_types[k % len(layer_types)] for k in range(n_layers_per_iter)]
    kv_caches = []
    for lt in call_schedule:
        if lt == "sliding_attention":
            kv_caches.append(
                init_kv_cache(
                    mesh_device, sliding_attn_cfg, paged_attention_config=paged_cfg_slide, cache_dtype=ttnn.bfloat16
                )
            )
        else:
            kv_caches.append(
                init_kv_cache(
                    mesh_device, full_attn_cfg, paged_attention_config=paged_cfg_full, cache_dtype=ttnn.bfloat16
                )
            )

    # Shared (drafter) KV caches — drafter reads the LAST layer of each type's
    # cache in the server, so we point its read at the kv_cache of the last
    # such layer in our schedule.
    last_idx_sliding = max(k for k, lt in enumerate(call_schedule) if lt == "sliding_attention")
    last_idx_full = max(k for k, lt in enumerate(call_schedule) if lt == "full_attention")
    drafter_shared_kv = {
        "sliding_attention": kv_caches[last_idx_sliding],
        "full_attention": kv_caches[last_idx_full],
    }

    page_table_full_np = torch.arange(B * blocks_per_user_full, dtype=torch.int32).reshape(B, blocks_per_user_full)
    page_table_full_tt = _alloc_dev(mesh_device, page_table_full_np, ttnn.int32)
    page_table_slide_np = torch.arange(B * blocks_per_user_slide, dtype=torch.int32).reshape(B, blocks_per_user_slide)
    page_table_slide_tt = _alloc_dev(mesh_device, page_table_slide_np, ttnn.int32)

    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len_test)
    cos_slide, sin_slide = rope_caches_2d["sliding_attention"]
    cos_full, sin_full = rope_caches_2d["full_attention"]

    # ── State buffers (persistent across iters; commit writes them, propose /
    # verify_pre read them).
    cur_pos_host = torch.full((B,), IDLE_CUR_POS, dtype=torch.int32)
    for u in range(B):
        if u % 2 == 0:
            cur_pos_host[u] = prefill_len + u
    cur_pos_dev = _alloc_dev(mesh_device, cur_pos_host, ttnn.int32)
    arange_P_dev = _build_arange_P(mesh_device, P)

    next_token_dev = _alloc_dev(
        mesh_device,
        torch.zeros(1, B, dtype=torch.int32),
        ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    spec_hidden_dev = _alloc_dev(
        mesh_device,
        torch.zeros(1, 1, B, hidden_size, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    drafts_2d_dev = _alloc_dev(
        mesh_device,
        torch.zeros(B, T_drafts, dtype=torch.int32),
        ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
    )
    pv_tgt_dev = _alloc_dev(
        mesh_device,
        torch.zeros(B, P, dtype=torch.int32),
        ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
    )
    v_hidden_dev = _alloc_dev(
        mesh_device,
        torch.zeros(1, 1, B * P, hidden_size, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    embeds_v_dev = _alloc_dev(
        mesh_device,
        torch.zeros(1, 1, B * P, hidden_size, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    position_idx_dev = _alloc_dev(
        mesh_device,
        torch.zeros(1, B * P, dtype=torch.int32),
        ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_write_idxs_dev = [
        _alloc_dev(mesh_device, torch.zeros(B, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for _ in range(P)
    ]
    kv_write_idxs_sliding_dev = [
        _alloc_dev(mesh_device, torch.zeros(B, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for _ in range(P)
    ]

    # ── Pre-baked mask tables (verify_pre gather inputs).
    cap_full = max_seq_len_test
    NEG = float(-1e9)
    mask_table_full_torch = torch.full((cap_full, cap_full), NEG, dtype=torch.bfloat16)
    for r in range(cap_full):
        mask_table_full_torch[r, : r + 1] = 0.0
    mask_table_full_dev = _alloc_dev(mesh_device, mask_table_full_torch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    mask_table_slide_torch = torch.full((W_test, W_test), NEG, dtype=torch.bfloat16)
    for r in range(W_test):
        mask_table_slide_torch[r, : r + 1] = 0.0
    mask_table_slide_dev = _alloc_dev(mesh_device, mask_table_slide_torch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    mask_full_dev = _alloc_dev(
        mesh_device,
        torch.zeros(B, 1, H_local * P, cap_full, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    mask_slide_dev = _alloc_dev(
        mesh_device,
        torch.zeros(B, 1, H_local * P, W_test, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    r_base_dev = _alloc_dev(
        mesh_device,
        (torch.arange(B, dtype=torch.int32) * P),
        ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ones_B_dev = _alloc_dev(
        mesh_device,
        torch.ones(B, dtype=torch.int32),
        ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Closures shared by propose and verify_head.
    def _embed_tokens(tokens_uint32):
        emb = ttnn.embedding(tokens_uint32, embedding_weight, dtype=ttnn.bfloat16)
        emb = ttnn.mul(emb, embed_scale)
        if tp > 1:
            emb = ttnn.unsqueeze_to_4D(emb)
            emb = ccl_allgather(emb, mesh_config, ccl_manager)
        return emb

    def _global_argmax(logits):
        if tp > 1:
            logits = ccl_allgather(logits, mesh_config, ccl_manager)
        _, idx = ttnn.topk(logits, k=1, dim=-1)
        return idx

    # ── Preamble: position_idx + kv_write_idxs (full & sliding) + mask gather.
    def _preamble():
        position_idx_local, kv_write_idxs_local, kv_write_idxs_sliding_local = _derive_full_and_sliding_preamble(
            mesh_device, cur_pos_dev, arange_P_dev, B, P, W_test, log_W_test
        )
        ttnn.copy(position_idx_local, position_idx_dev)
        for p in range(P):
            ttnn.copy(kv_write_idxs_local[p], kv_write_idxs_dev[p])
            ttnn.deallocate(kv_write_idxs_local[p])
            ttnn.copy(kv_write_idxs_sliding_local[p], kv_write_idxs_sliding_dev[p])
            ttnn.deallocate(kv_write_idxs_sliding_local[p])

        mask_full_rows = ttnn.embedding(position_idx_local, mask_table_full_dev, layout=ttnn.TILE_LAYOUT)
        mask_full = ttnn.reshape(mask_full_rows, (B, 1, P, cap_full))
        mask_full = ttnn.repeat(mask_full, [1, 1, H_local, 1])
        ttnn.deallocate(mask_full_rows)
        ttnn.assign(mask_full, mask_full_dev)
        ttnn.deallocate(mask_full)

        mask_slide_rows = ttnn.embedding(position_idx_local, mask_table_slide_dev, layout=ttnn.TILE_LAYOUT)
        mask_slide = ttnn.reshape(mask_slide_rows, (B, 1, P, W_test))
        mask_slide = ttnn.repeat(mask_slide, [1, 1, H_local, 1])
        ttnn.deallocate(mask_slide_rows)
        ttnn.assign(mask_slide, mask_slide_dev)
        ttnn.deallocate(mask_slide)
        ttnn.deallocate(position_idx_local)

    def _spec_dict():
        return {
            "p": P,
            "position_idx": position_idx_dev,
            "kv_write_idxs": kv_write_idxs_dev,
            "kv_write_idxs_sliding": kv_write_idxs_sliding_dev,
            "attn_mask": {
                "full_attention": mask_full_dev,
                "sliding_attention": mask_slide_dev,
            },
        }

    # ── Propose chunk: real drafter (if loaded) or mock.
    def _propose():
        if drafter is not None:
            # Real drafter — matches server.py:_build_propose_fwd exactly.
            cur_pos_sdpa = ttnn.maximum(cur_pos_dev, 0)
            emb0 = _embed_tokens(next_token_dev)
            emb0 = ttnn.reshape(emb0, (1, 1, B, hidden_size))
            emb0 = ttnn.to_layout(emb0, ttnn.TILE_LAYOUT)
            draft_input = ttnn.concat([emb0, spec_hidden_dev], dim=-1)
            ttnn.deallocate(emb0)

            drafts_cols = []
            for k in range(T_drafts):
                out_hidden, logits = drafter.forward(
                    target_last_hidden=draft_input,
                    shared_kv=drafter_shared_kv,
                    cos_cache_full=drafter_cos_cache["full_attention"],
                    sin_cache_full=drafter_sin_cache["full_attention"],
                    cos_cache_sliding=drafter_cos_cache["sliding_attention"],
                    sin_cache_sliding=drafter_sin_cache["sliding_attention"],
                    cur_pos_tensor=cur_pos_sdpa,
                    cur_pos_tensor_sliding=cur_pos_sdpa,
                    page_table_full=page_table_full_tt,
                    page_table_sliding=page_table_slide_tt,
                )
                ttnn.deallocate(draft_input)
                topk_idx = _global_argmax(logits)
                ttnn.deallocate(logits)
                draft_k = ttnn.reshape(topk_idx, (B, 1))
                draft_k_tile = ttnn.to_layout(draft_k, ttnn.TILE_LAYOUT)
                drafts_cols.append(draft_k_tile)
                if k < T_drafts - 1:
                    draft_k_1B = ttnn.reshape(draft_k, (1, B))
                    draft_k_1B_rm = ttnn.to_layout(draft_k_1B, ttnn.ROW_MAJOR_LAYOUT)
                    draft_k_1B_u = ttnn.typecast(draft_k_1B_rm, ttnn.uint32)
                    ttnn.deallocate(draft_k_1B_rm)
                    emb_k = _embed_tokens(draft_k_1B_u)
                    ttnn.deallocate(draft_k_1B_u)
                    emb_k = ttnn.reshape(emb_k, (1, 1, B, hidden_size))
                    emb_k = ttnn.to_layout(emb_k, ttnn.TILE_LAYOUT)
                    draft_input = ttnn.concat([emb_k, out_hidden], dim=-1)
                    ttnn.deallocate(emb_k)
                ttnn.deallocate(out_hidden)

            drafts_2d = ttnn.concat(drafts_cols, dim=-1)
            ttnn.copy(drafts_2d, drafts_2d_dev)
            ttnn.deallocate(drafts_2d)
            ttnn.deallocate(cur_pos_sdpa)
        else:
            # Mock propose — same as test 10 all_chunks. T embed_tokens + T
            # lm_head + T global_argmax, reading the same persistent state
            # the real drafter would.
            emb0 = _embed_tokens(next_token_dev)
            emb0 = ttnn.reshape(emb0, (1, 1, B, hidden_size))
            emb0 = ttnn.to_layout(emb0, ttnn.TILE_LAYOUT)
            proxy = ttnn.add(emb0, spec_hidden_dev)
            ttnn.deallocate(emb0)
            drafts_cols = []
            cur_proxy = proxy
            for k in range(T_drafts):
                logits = ttnn.linear(cur_proxy, lm_head_weight)
                topk_idx = _global_argmax(logits)
                ttnn.deallocate(logits)
                draft_k = ttnn.reshape(topk_idx, (B, 1))
                draft_k_tile = ttnn.to_layout(draft_k, ttnn.TILE_LAYOUT)
                drafts_cols.append(draft_k_tile)
                if k < T_drafts - 1:
                    next_uint = ttnn.reshape(draft_k, (1, B))
                    next_uint_rm = ttnn.to_layout(next_uint, ttnn.ROW_MAJOR_LAYOUT)
                    next_uint = ttnn.typecast(next_uint_rm, ttnn.uint32)
                    ttnn.deallocate(next_uint_rm)
                    emb_k = _embed_tokens(next_uint)
                    ttnn.deallocate(next_uint)
                    emb_k = ttnn.reshape(emb_k, (1, 1, B, hidden_size))
                    emb_k = ttnn.to_layout(emb_k, ttnn.TILE_LAYOUT)
                    ttnn.deallocate(cur_proxy)
                    cur_proxy = emb_k
            ttnn.deallocate(cur_proxy)
            drafts_2d = ttnn.concat(drafts_cols, dim=-1)
            ttnn.copy(drafts_2d, drafts_2d_dev)
            ttnn.deallocate(drafts_2d)

    # ── Verify_pre: build tokens_flat = [next_token, drafts], embed → embeds_v.
    def _verify_pre_tokens_embed():
        nxt_col_rm = ttnn.reshape(next_token_dev, (B, 1))
        nxt_col_tile = ttnn.to_layout(nxt_col_rm, ttnn.TILE_LAYOUT)
        tokens_per_slot = ttnn.concat([nxt_col_tile, drafts_2d_dev], dim=-1)
        tokens_flat_tile = ttnn.reshape(tokens_per_slot, (1, B * P))
        tokens_flat = ttnn.to_layout(tokens_flat_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(nxt_col_tile)
        ttnn.deallocate(tokens_flat_tile)
        ttnn.deallocate(tokens_per_slot)
        embeds_v = _embed_tokens(tokens_flat)
        embeds_v = ttnn.reshape(embeds_v, (1, 1, B * P, hidden_size))
        embeds_v = ttnn.to_layout(embeds_v, ttnn.TILE_LAYOUT)
        ttnn.assign(embeds_v, embeds_v_dev)
        ttnn.deallocate(embeds_v)

    # ── Verify_model: N layer calls reading from embeds_v_dev.
    def _verify_model():
        spec = _spec_dict()
        h = ttnn.clone(embeds_v_dev)
        for k, lt in enumerate(call_schedule):
            layer = _layer_for_call(k)
            if lt == "sliding_attention":
                h = layer(
                    h,
                    rope_mats=(cos_slide, sin_slide),
                    position_idx=None,
                    page_table=None,
                    kv_cache=kv_caches[k],
                    is_decode=True,
                    page_table_sliding=page_table_slide_tt,
                    packed=spec,
                )
            else:
                h = layer(
                    h,
                    rope_mats=(cos_full, sin_full),
                    position_idx=None,
                    page_table=page_table_full_tt,
                    kv_cache=kv_caches[k],
                    is_decode=True,
                    page_table_sliding=None,
                    packed=spec,
                )
        ttnn.assign(h, v_hidden_dev)
        ttnn.deallocate(h)

    # ── Verify_head: final norm + lm_head + softcap + global argmax → pv_tgt.
    def _verify_head():
        hidden_local = ttnn.clone(v_hidden_dev)
        hidden_local = ttnn.rms_norm(
            hidden_local,
            weight=final_norm_weight,
            epsilon=getattr(model_args, "rms_norm_eps", 1e-6),
        )
        logits = ttnn.linear(hidden_local, lm_head_weight)
        ttnn.deallocate(hidden_local)
        if final_logit_softcapping and final_logit_softcapping > 0:
            cap = final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)
        v_topk_idx = _global_argmax(logits)
        ttnn.deallocate(logits)
        pv_tgt = ttnn.reshape(v_topk_idx, (B, P))
        ttnn.copy(pv_tgt, pv_tgt_dev)

    # ── Commit: eq + cumprod + sum → n_acc; bonus + advance + state updates.
    def _commit():
        pv_tgt_first_T = ttnn.slice(pv_tgt_dev, [0, 0], [B, T_drafts])
        eq = ttnn.eq(drafts_2d_dev, pv_tgt_first_T)
        cum = ttnn.cumprod(eq, dim=-1)
        n_acc = ttnn.sum(cum, dim=-1, keepdim=True)
        ttnn.deallocate(eq)
        ttnn.deallocate(cum)
        ttnn.deallocate(pv_tgt_first_T)

        bonus = ttnn.gather(pv_tgt_dev, dim=1, index=n_acc)
        n_acc_flat_u = ttnn.reshape(n_acc, (B,))
        n_acc_flat = ttnn.typecast(n_acc_flat_u, ttnn.int32)

        advance = ttnn.add(n_acc_flat, ones_B_dev)
        new_cur_pos_tile = ttnn.add(cur_pos_dev, advance)
        new_cur_pos = ttnn.to_layout(new_cur_pos_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(new_cur_pos, cur_pos_dev)
        if new_cur_pos is not new_cur_pos_tile:
            ttnn.deallocate(new_cur_pos_tile)
        ttnn.deallocate(new_cur_pos)
        ttnn.deallocate(advance)

        bonus_1B_tile = ttnn.reshape(bonus, (1, B))
        bonus_1B_rm = ttnn.to_layout(bonus_1B_tile, ttnn.ROW_MAJOR_LAYOUT)
        bonus_1B_u32 = ttnn.typecast(bonus_1B_rm, ttnn.uint32)
        ttnn.copy(bonus_1B_u32, next_token_dev)
        if bonus_1B_rm is not bonus_1B_tile:
            ttnn.deallocate(bonus_1B_rm)
        ttnn.deallocate(bonus_1B_u32)

        v_hidden_2d = ttnn.reshape(v_hidden_dev, (B * P, hidden_size))
        gather_idx = ttnn.add(r_base_dev, n_acc_flat)
        gather_idx_2d = ttnn.typecast(ttnn.reshape(gather_idx, (1, B)), ttnn.uint32)
        ttnn.deallocate(gather_idx)
        new_hidden = ttnn.embedding(gather_idx_2d, v_hidden_2d, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(gather_idx_2d)
        new_hidden = ttnn.reshape(new_hidden, (1, 1, B, hidden_size))
        ttnn.assign(new_hidden, spec_hidden_dev)
        ttnn.deallocate(new_hidden)
        ttnn.deallocate(bonus)

    def _body():
        _propose()
        _preamble()
        _verify_pre_tokens_embed()
        _verify_model()
        _verify_head()
        _commit()

    # ── Trace capture + n_replays. The first replay number that hangs is the
    # parametrize answer. Eager mode is omitted — eager always passes, the
    # bug only fires under trace replay.
    logger.info(f"[REPRO] warmup body")
    _body()
    ttnn.synchronize_device(mesh_device)

    cur_pos_reset = _alloc_dev(mesh_device, cur_pos_host, ttnn.int32)
    ttnn.copy(cur_pos_reset, cur_pos_dev)
    ttnn.deallocate(cur_pos_reset)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"[REPRO] begin_trace_capture (distinct={distinct_layers} drafter={with_real_drafter})")
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    _body()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info("[REPRO] capture done")

    for it in range(n_replays):
        logger.info(f"[REPRO replay {it}/{n_replays}] execute_trace")
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[REPRO replay {it}/{n_replays}] DONE")

    logger.info(
        f"[REPRO DONE N={n_layers_per_iter} distinct={distinct_layers} "
        f"reps={n_replays} drafter={with_real_drafter}]"
    )


# ───────────────────────────────────────────────────────────────────────────
# PROFILING: per-step mask construction cost — host build + H2D vs on-device
# gather. Production dimensions, eager mode, percentile stats.
#
# Settles the central question of the on-device-everything refactor: does
# moving the verify masks on-device actually win meaningful wall-clock per
# step, given that this host is underpowered for the loops scratch's
# ``_refresh_packed_verify_inputs`` runs?
# ───────────────────────────────────────────────────────────────────────────


def _build_masks_host_scratch_style(host_mask_full, host_mask_slide, verify_slots_cur_pos, B, P, H_local, cap, W, NEG):
    """Replicate scratch's mask construction in ``_refresh_packed_verify_inputs``.

    ``verify_slots_cur_pos`` is a list ``[(slot_row, cur_pos), ...]`` of
    active rows. The function zeroes the host masks, then for each active
    row r at cur_pos c, fills row r's [H_local, P, cap] (full) and
    [H_local, P, W] (sliding) view with the causal pattern
    ``[0]*ub + [NEG]*(rest)`` per position p, where ub = c+p.

    Inactive rows (not in ``verify_slots_cur_pos``) keep the all-zero default
    — these rows' KV writes are skipped via kv_write_idxs=-1, so mask values
    are irrelevant.
    """
    host_mask_full.zero_()
    host_mask_slide.zero_()
    for r, c in verify_slots_cur_pos:
        fm = host_mask_full[r, 0].view(H_local, P, cap)
        sm = host_mask_slide[r, 0].view(H_local, P, W)
        for p in range(P):
            ub = c + p
            fm[:, p, : ub + 1] = 0.0
            fm[:, p, ub + 1 :] = NEG
            sm[:, p, : ub + 1] = 0.0
            sm[:, p, ub + 1 :] = NEG


def _h2d_masks(mesh_device, host_mask_full, host_mask_slide, dev_mask_full, dev_mask_slide, replicate):
    """Wrap masks as ttnn host tensors and push to pre-allocated device
    buffers — matches scratch's per-step push pattern."""
    host_full = ttnn.from_torch(host_mask_full, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    host_slide = ttnn.from_torch(host_mask_slide, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    ttnn.copy_host_to_device_tensor(host_full, dev_mask_full)
    ttnn.copy_host_to_device_tensor(host_slide, dev_mask_slide)


def _gather_masks_device(position_idx_safe_dev, mask_table_full_dev, mask_table_slide_dev, B, P, H_local, cap, W):
    """On-device gather mirroring the proposed verify_pre optimization.

    Returns new bf16 TILE tensors of shape [B, 1, H_local*P, cap] and
    [B, 1, H_local*P, W]. Caller assigns into persistent buffers.
    """
    mask_full_rows = ttnn.embedding(position_idx_safe_dev, mask_table_full_dev, layout=ttnn.TILE_LAYOUT)
    mask_full = ttnn.reshape(mask_full_rows, (B, 1, P, cap))
    mask_full = ttnn.repeat(mask_full, [1, 1, H_local, 1])
    ttnn.deallocate(mask_full_rows)

    mask_slide_rows = ttnn.embedding(position_idx_safe_dev, mask_table_slide_dev, layout=ttnn.TILE_LAYOUT)
    mask_slide = ttnn.reshape(mask_slide_rows, (B, 1, P, W))
    mask_slide = ttnn.repeat(mask_slide, [1, 1, H_local, 1])
    ttnn.deallocate(mask_slide_rows)

    return mask_full, mask_slide


def _stats(times_ms):
    """Return dict with min / median / mean / p95 / max in ms."""
    t = sorted(times_ms)
    n = len(t)
    return {
        "n": n,
        "min": t[0],
        "median": t[n // 2],
        "mean": sum(t) / n,
        "p95": t[min(n - 1, int(n * 0.95))],
        "max": t[-1],
    }


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("n_active_slots", [1, 8, 16, 32], ids=lambda v: f"slots{v}")
@pytest.mark.parametrize("n_iters", [50], ids=lambda v: f"iters{v}")
def test_mask_construction_profile(n_active_slots, n_iters, mesh_device):
    """Wall-clock profile: per-step verify mask construction.

    Compares two paths at production dimensions (B=32, H_local=8, P=5,
    cap=4096, W=1024):

    Path A (scratch baseline):
      1. Host-side build the full + sliding masks (the nested-loop
         ``host[mask_full][r,0].view(...)`` writes).
      2. ``ttnn.from_torch`` to wrap as host tensor (TILE bf16).
      3. ``copy_host_to_device_tensor`` to the pre-allocated device buffer.
      The total = the per-step H2D budget the user wants to know about.

    Path B (proposed optimization):
      1. Pre-baked mask tables stay resident on device.
      2. Per step: ``ttnn.embedding(position_idx_safe, table) → reshape →
         repeat`` — output materialised in device memory.
      3. ``ttnn.assign`` into the same persistent buffer Path A pushes to.

    Each phase is measured independently (build-only, H2D-only,
    build+H2D combined, on-device-gather) so the breakdown is visible.

    ``n_iters`` measurement runs after a 10-iter warmup; percentiles printed
    so a few host-stalls don't skew the mean.

    Parametrize ``n_active_slots`` to see how the host build scales (it does
    — N×B inner loop work) while the device gather doesn't.
    """
    torch.manual_seed(0)
    B, P = B_TEST, P_TEST  # 32 × 5
    cap = 4096  # _PV_SK_CAP (production)
    W = 1024  # sliding_cache_len (production)
    H_local = 8  # num_attention_heads / tp = 32 / 4 on 1x4
    NEG = float(-1e9)
    n_warmup = 10
    prefill_len = 64

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # Host tensors — sized to production.
    host_mask_full = torch.zeros(B, 1, H_local * P, cap, dtype=torch.bfloat16)
    host_mask_slide = torch.zeros(B, 1, H_local * P, W, dtype=torch.bfloat16)

    # Per-iter cur_pos: first ``n_active_slots`` slots are active at prefill_len + u,
    # rest are idle. Scratch's loop iterates verify_slots only.
    verify_slots_cur_pos = [(r, prefill_len + r) for r in range(n_active_slots)]

    # Pre-allocate device buffers (the persistent verify-mask buffers).
    dev_mask_full = _alloc_dev(mesh_device, host_mask_full, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    dev_mask_slide = _alloc_dev(mesh_device, host_mask_slide, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Pre-baked mask tables for the device gather path.
    mask_table_full_torch = torch.full((cap, cap), NEG, dtype=torch.bfloat16)
    for r in range(cap):
        mask_table_full_torch[r, : r + 1] = 0.0
    mask_table_full_dev = _alloc_dev(mesh_device, mask_table_full_torch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    mask_table_slide_torch = torch.full((W, W), NEG, dtype=torch.bfloat16)
    for r in range(W):
        mask_table_slide_torch[r, : r + 1] = 0.0
    mask_table_slide_dev = _alloc_dev(mesh_device, mask_table_slide_torch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # position_idx_safe for device gather — [1, B*P] uint32 RM. Reflects each
    # slot's c+p packed positions: active slots at prefill_len+r, idle slots
    # at 0 (safe-clamped from the deep-negative sentinel).
    pos_idx_host = torch.zeros(1, B * P, dtype=torch.int32)
    for r, c in verify_slots_cur_pos:
        for p in range(P):
            pos_idx_host[0, r * P + p] = c + p
    position_idx_safe_dev = _alloc_dev(mesh_device, pos_idx_host, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    logger.info(
        f"[MASK PROFILE] B={B} P={P} H_local={H_local} cap={cap} W={W} "
        f"n_active_slots={n_active_slots} n_iters={n_iters}"
    )
    mb_full = B * H_local * P * cap * 2 / (1024 * 1024)
    mb_slide = B * H_local * P * W * 2 / (1024 * 1024)
    logger.info(
        f"[MASK PROFILE] full mask {mb_full:.2f} MB, slide mask {mb_slide:.2f} MB, total {mb_full + mb_slide:.2f} MB / step"
    )

    # ─── Correctness check: device gather output matches host build for active
    # rows, BIT-FOR-BIT. (For idle rows the two paths diverge harmlessly: host
    # leaves them at 0; device gathers ``mask_table[0] = [0, NEG, NEG, …]``
    # because ``position_idx_safe`` clamps idle's deep-negative cur_pos to 0.
    # paged_update_cache skips idle rows via ``kv_write_idxs=-1`` so the
    # difference never reaches an output. We only check active rows.)
    #
    # If this assertion fails the timing numbers below are meaningless — they'd
    # be measuring a faster but WRONG path. Fail loudly here before the
    # 60-second timing loops.
    logger.info("[MASK PROFILE] correctness check: device gather vs host build (active rows only)…")

    # 1. Build the host masks once, capture for ground-truth comparison.
    _build_masks_host_scratch_style(host_mask_full, host_mask_slide, verify_slots_cur_pos, B, P, H_local, cap, W, NEG)
    host_full_truth = host_mask_full.clone()
    host_slide_truth = host_mask_slide.clone()

    # 2. Run the on-device gather once → assign → D2H. Same code as the timed
    #    loop below uses, so this is what the timing claims is "fast".
    mf_chk, ms_chk = _gather_masks_device(
        position_idx_safe_dev, mask_table_full_dev, mask_table_slide_dev, B, P, H_local, cap, W
    )
    ttnn.assign(mf_chk, dev_mask_full)
    ttnn.assign(ms_chk, dev_mask_slide)
    ttnn.deallocate(mf_chk)
    ttnn.deallocate(ms_chk)
    ttnn.synchronize_device(mesh_device)

    dev_full_torch = _d2h(dev_mask_full).to(torch.bfloat16)
    dev_slide_torch = _d2h(dev_mask_slide).to(torch.bfloat16)

    # 3. Bit-equality assertion per active (slot_row, head, packed_position).
    #    All rows for one (slot, p) share the same mask across heads; we still
    #    check every head explicitly to catch ttnn.repeat layout regressions.
    for r, c in verify_slots_cur_pos:
        for h in range(H_local):
            for p in range(P):
                idx = h * P + p
                host_row_full = host_full_truth[r, 0, idx, :]
                dev_row_full = dev_full_torch[r, 0, idx, :]
                if not torch.equal(host_row_full, dev_row_full):
                    diff_mask = host_row_full != dev_row_full
                    first_diff = int(torch.argmax(diff_mask.to(torch.int32)))
                    raise AssertionError(
                        f"[MASK PROFILE] FULL-mask BIT MISMATCH at slot_row={r} "
                        f"cur_pos={c} head={h} packed_pos={p} (row idx {idx}). "
                        f"First diverging key index: {first_diff}. "
                        f"host[{first_diff}]={host_row_full[first_diff].item()}, "
                        f"dev[{first_diff}]={dev_row_full[first_diff].item()}. "
                        f"Expected causal pattern: 0 for k<={c + p}, NEG for k>{c + p}."
                    )

                host_row_slide = host_slide_truth[r, 0, idx, :]
                dev_row_slide = dev_slide_torch[r, 0, idx, :]
                if not torch.equal(host_row_slide, dev_row_slide):
                    diff_mask = host_row_slide != dev_row_slide
                    first_diff = int(torch.argmax(diff_mask.to(torch.int32)))
                    raise AssertionError(
                        f"[MASK PROFILE] SLIDE-mask BIT MISMATCH at slot_row={r} "
                        f"cur_pos={c} head={h} packed_pos={p} (row idx {idx}). "
                        f"First diverging key index: {first_diff}. "
                        f"host[{first_diff}]={host_row_slide[first_diff].item()}, "
                        f"dev[{first_diff}]={dev_row_slide[first_diff].item()}."
                    )
    logger.info(
        f"[MASK PROFILE] correctness check PASSED: "
        f"{len(verify_slots_cur_pos)} active rows × {H_local} heads × {P} packed positions "
        f"bit-identical between host build and device gather (full + sliding masks)."
    )

    # ─── Measurement 1: host build only (no H2D) ───────────────────────────
    for _ in range(n_warmup):
        _build_masks_host_scratch_style(
            host_mask_full, host_mask_slide, verify_slots_cur_pos, B, P, H_local, cap, W, NEG
        )
    times_host_build = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _build_masks_host_scratch_style(
            host_mask_full, host_mask_slide, verify_slots_cur_pos, B, P, H_local, cap, W, NEG
        )
        t1 = time.perf_counter_ns()
        times_host_build.append((t1 - t0) / 1e6)

    # ─── Measurement 2: H2D only (re-use the already-built host masks) ─────
    for _ in range(n_warmup):
        _h2d_masks(mesh_device, host_mask_full, host_mask_slide, dev_mask_full, dev_mask_slide, replicate)
    ttnn.synchronize_device(mesh_device)

    times_h2d = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _h2d_masks(mesh_device, host_mask_full, host_mask_slide, dev_mask_full, dev_mask_slide, replicate)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter_ns()
        times_h2d.append((t1 - t0) / 1e6)

    # ─── Measurement 3: combined host build + H2D (= what scratch does) ────
    for _ in range(n_warmup):
        _build_masks_host_scratch_style(
            host_mask_full, host_mask_slide, verify_slots_cur_pos, B, P, H_local, cap, W, NEG
        )
        _h2d_masks(mesh_device, host_mask_full, host_mask_slide, dev_mask_full, dev_mask_slide, replicate)
    ttnn.synchronize_device(mesh_device)

    times_host_full = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _build_masks_host_scratch_style(
            host_mask_full, host_mask_slide, verify_slots_cur_pos, B, P, H_local, cap, W, NEG
        )
        _h2d_masks(mesh_device, host_mask_full, host_mask_slide, dev_mask_full, dev_mask_slide, replicate)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter_ns()
        times_host_full.append((t1 - t0) / 1e6)

    # ─── Measurement 4: on-device gather only (= the proposed optimization) ─
    # The gather allocates new device tensors; ttnn.assign copies into the
    # persistent buffer so the comparison is apples-to-apples (both paths
    # end with dev_mask_full / dev_mask_slide populated).
    for _ in range(n_warmup):
        mf, ms = _gather_masks_device(
            position_idx_safe_dev, mask_table_full_dev, mask_table_slide_dev, B, P, H_local, cap, W
        )
        ttnn.assign(mf, dev_mask_full)
        ttnn.assign(ms, dev_mask_slide)
        ttnn.deallocate(mf)
        ttnn.deallocate(ms)
    ttnn.synchronize_device(mesh_device)

    times_device_gather = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        mf, ms = _gather_masks_device(
            position_idx_safe_dev, mask_table_full_dev, mask_table_slide_dev, B, P, H_local, cap, W
        )
        ttnn.assign(mf, dev_mask_full)
        ttnn.assign(ms, dev_mask_slide)
        ttnn.deallocate(mf)
        ttnn.deallocate(ms)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter_ns()
        times_device_gather.append((t1 - t0) / 1e6)

    # ─── Report ────────────────────────────────────────────────────────────
    def _fmt(stats):
        return (
            f"min={stats['min']:.3f} median={stats['median']:.3f} "
            f"mean={stats['mean']:.3f} p95={stats['p95']:.3f} max={stats['max']:.3f}"
        )

    s_host_build = _stats(times_host_build)
    s_h2d = _stats(times_h2d)
    s_host_full = _stats(times_host_full)
    s_dev = _stats(times_device_gather)

    logger.info("=" * 80)
    logger.info(
        f"[MASK PROFILE n_active={n_active_slots}] All times in milliseconds. "
        f"Warmup={n_warmup} iters; measured={n_iters} iters."
    )
    logger.info("=" * 80)
    logger.info(f"  host_build_only          : {_fmt(s_host_build)}")
    logger.info(f"  h2d_only                 : {_fmt(s_h2d)}")
    logger.info(f"  host_full (build + h2d)  : {_fmt(s_host_full)}")
    logger.info(f"  device_gather            : {_fmt(s_dev)}")
    logger.info("-" * 80)
    speedup_median = s_host_full["median"] / s_dev["median"] if s_dev["median"] > 0 else float("inf")
    saving_median = s_host_full["median"] - s_dev["median"]
    logger.info(
        f"  ⇒ device_gather is {speedup_median:.2f}× faster than host_full at median; "
        f"saves {saving_median:.3f} ms/step ({saving_median * 1000:.0f} µs)"
    )
    logger.info("=" * 80)

    # Soft assertions — these aren't pass/fail conditions, they document the
    # measurement so the test result includes the numbers as part of pytest
    # captured stdout. Use `pytest -s` to see them inline.
    assert s_dev["median"] >= 0, "device gather median must be measurable"
    assert s_host_full["median"] >= 0, "host full median must be measurable"
