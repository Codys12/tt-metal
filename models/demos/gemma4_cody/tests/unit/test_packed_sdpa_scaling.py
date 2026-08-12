# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure how packed-Q SDPA decode cost scales with T (drafts per step).

Background: the speculative-decode speedup projections in
``test_packed_decode_compare.py`` and the n-gram demo assume the packed SDPA
call costs the same as the single-token SDPA call. That's optimistic — the
packed call processes T× more Q rows, which has to cost *something*. This
test measures the actual cost curve so we can plug a real number into the
speedup math instead of guessing.

What's varied:  T ∈ {1, 2, 4, 8, 16}
What's held:    B=32, KV cache length=prefill_len + T, mesh=1×8 P150

What's timed:
  paged_scaled_dot_product_attention_decode(packed_Q, K_cache, V_cache,
    page_table, attn_mask=per_row_mask, cur_pos_tensor=...)

Run:
    cd /mnt/nas/scratch
    source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_packed_sdpa_scaling.py -k 1x8
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.gemma4_cody.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache
from models.tt_transformers.tt.common import PagedAttentionConfig

from ...tests.test_factory import TestFactory, parametrize_mesh_with_fabric

B_TEST = 32
PREFILL_LEN = 128  # tokens in cache when each measurement runs
NUM_WARMUP = 3
NUM_TIMED = 10
T_VALUES = (1, 2, 4, 8, 16)


def _now() -> float:
    return time.perf_counter()


def _sync(mesh_device):
    ttnn.synchronize_device(mesh_device)


def _build_packed_q(B: int, T: int, H_local: int, head_dim: int, mesh_device, replicate):
    """Synthetic packed Q tensor: [1, B, T*H_local, head_dim] in TILE layout."""
    q = torch.randn(1, B, T * H_local, head_dim, dtype=torch.float32)
    return ttnn.from_torch(
        q.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )


def _build_per_row_mask(B: int, T: int, H_local: int, S_k: int, mesh_device, replicate):
    """Per-(token, head) causal mask: [B, 1, T*H_local, S_k].

    Row ``t*H_local + h`` (token t, head h) masks positions > PREFILL_LEN + t.
    """
    NEG = float(-1e9)
    mask = torch.zeros(B, 1, T * H_local, S_k, dtype=torch.float32)
    for t in range(T):
        for h in range(H_local):
            mask[:, 0, t * H_local + h, PREFILL_LEN + t + 1 :] = NEG
    return ttnn.from_torch(
        mask.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )


def _time_one_sdpa_call(
    q_packed, k_cache, v_cache, page_table_tt, mask_tt, cur_pos_tt, sdpa_program_config, mesh_device
):
    """One call to paged_scaled_dot_product_attention_decode; returns wall ms."""
    t0 = _now()
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_packed,
        k_cache,
        v_cache,
        page_table_tensor=page_table_tt,
        is_causal=False,
        attn_mask=mask_tt,
        cur_pos_tensor=cur_pos_tt,
        scale=1.0,
        sliding_window_size=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=sdpa_program_config,
    )
    _sync(mesh_device)
    return (_now() - t0) * 1e3


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [5], ids=["global"])
def test_packed_sdpa_cost_vs_t(layer_idx, mesh_device):
    """Time packed-Q SDPA decode for T ∈ {1, 2, 4, 8, 16}."""
    hf_text_config = TestFactory.create_hf_text_config()
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    assert not config.is_sliding, "Test targets global (non-sliding) layer"

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    head_dim = config.head_dim
    H = config.num_attention_heads
    H_local = H // tp
    nkv = config.num_key_value_heads
    kv_replicated = nkv < tp
    local_kv = 1 if kv_replicated else nkv // tp
    assert local_kv == 1, (
        f"Test currently requires local_kv==1 (got {local_kv}); "
        "needed for correct Q-head→KV-head mapping with the heads-packing trick."
    )

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # KV cache sized to the largest T we test (so all variants share the same cache shape).
    max_T = max(T_VALUES)
    block_size = 64
    S_k = ((PREFILL_LEN + max_T + block_size - 1) // block_size + 1) * block_size  # round up
    blocks_per_user = S_k // block_size
    max_num_blocks = B_TEST * blocks_per_user

    page_table_host = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B_TEST, blocks_per_user)
    page_table_tt = ttnn.from_torch(
        page_table_host,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    kv_cache = init_kv_cache(
        mesh_device,
        config,
        paged_attention_config=paged_cfg,
        cache_dtype=ttnn.bfloat16,
    )
    k_cache, v_cache = kv_cache

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4) if head_dim >= 512 else ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )

    results = []
    for T in T_VALUES:
        # Each measurement: fresh Q + mask, cur_pos lined up with the prefill state.
        q_tt = _build_packed_q(B_TEST, T, H_local, head_dim, mesh_device, replicate)
        mask_tt = _build_per_row_mask(B_TEST, T, H_local, S_k, mesh_device, replicate)
        cur_pos_tt = ttnn.from_torch(
            torch.full((B_TEST,), PREFILL_LEN + T - 1, dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )

        # Warmup
        for _ in range(NUM_WARMUP):
            _time_one_sdpa_call(
                q_tt,
                k_cache,
                v_cache,
                page_table_tt,
                mask_tt,
                cur_pos_tt,
                sdpa_program_config,
                mesh_device,
            )

        # Timed
        samples = []
        for _ in range(NUM_TIMED):
            ms = _time_one_sdpa_call(
                q_tt,
                k_cache,
                v_cache,
                page_table_tt,
                mask_tt,
                cur_pos_tt,
                sdpa_program_config,
                mesh_device,
            )
            samples.append(ms)

        mean_ms = sum(samples) / len(samples)
        min_ms = min(samples)
        max_ms = max(samples)
        results.append((T, mean_ms, min_ms, max_ms))

        q_tt.deallocate(True)
        mask_tt.deallocate(True)
        cur_pos_tt.deallocate(True)

    # ── Report
    print()
    print(f"=== Packed-Q SDPA cost vs T (B={B_TEST}, tp={tp}, H_local={H_local}, head_dim={head_dim}, S_k={S_k}) ===")
    print(f"Averaged over {NUM_TIMED} runs ({NUM_WARMUP} warmup), pipelined (sync at end of call).")
    print()
    print(f"{'T':>3} {'mean (ms)':>10} {'min (ms)':>10} {'max (ms)':>10} {'vs T=1':>8} {'per-token':>10}")
    print(f"{'-'*3:>3} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8} {'-'*10:>10}")
    base = results[0][1]
    for T, mean_ms, min_ms, max_ms in results:
        ratio = mean_ms / base
        per_tok = mean_ms / (T + 1)  # T drafts + 1 bonus from verifier
        print(f"{T:>3} {mean_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f} {ratio:>7.2f}x {per_tok:>10.3f}")
    print()

    # ── Plug back into the speedup math:
    # packed step = prep B (20.85 ms measured) + this SDPA cost
    # break-even avg accepted-per-step = (packed_step / single_step) - 1
    prep_b_ms = 20.85
    single_step_ms = 88.0
    print(f"Plugging measured packed SDPA into speedup math (using prep B = {prep_b_ms} ms):")
    print(f"{'T':>3} {'packed_step (ms)':>18} {'break-even acc':>16}")
    for T, mean_ms, _, _ in results:
        packed_step = prep_b_ms + mean_ms
        break_even = packed_step / single_step_ms - 1
        print(f"{T:>3} {packed_step:>18.2f} {break_even:>15.3f}")
    print()
    print("Interpretation: 'break-even acc' is the avg accepted-tokens-per-step you'd need")
    print(f"to NOT regress vs the baseline {single_step_ms} ms single-token decode.")
