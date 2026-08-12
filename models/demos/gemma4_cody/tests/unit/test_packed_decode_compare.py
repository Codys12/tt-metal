# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Microbenchmark: per-token-loop decode prep (Approach A) vs batched prep (Approach B).

Both approaches eventually call ONE packed-Q SDPA decode. The interesting
difference is the per-step prep cost:

  Approach A: T iterations of {QKV projection, split heads, q/k/v norm, RoPE,
              cache write} → 1 SDPA call.
  Approach B: 1 call to {QKV projection, split heads, q/k/v norm, RoPE} on a
              stacked [1, 1, B*T, hidden] input, then T cache writes, then 1
              SDPA call.

We do NOT compare against HF here — the existing test_packed_decode.py
verifies kernel correctness. The purpose of this file is wall-clock per-op
timing on the user's actual mesh shape, plus a sanity check that the Q
tensors produced by A and B match (so the approaches are mathematically
equivalent modulo numerical noise).

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_packed_decode_compare.py -k 1x8
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.demos.gemma4_cody.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.attention.operations import (
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    split_qkv_heads_decode,
)
from models.demos.gemma4_cody.tt.ccl import CCLManager
from models.demos.gemma4_cody.tt.model import create_rope_caches

from ...tests.test_factory import TestFactory, parametrize_mesh_with_fabric

B_TEST = 32
NUM_WARMUP = 3
NUM_TIMED = 10


def _now() -> float:
    return time.perf_counter()


def _sync(mesh_device):
    """Force the host to wait for device work. ttnn ops are dispatch-only;
    timing without sync measures dispatch latency, not work."""
    ttnn.synchronize_device(mesh_device)


def _build_x_decode(x_hidden: torch.Tensor, mesh_device, replicate) -> ttnn.Tensor:
    """Approach-A input: [1, 1, B, hidden] (one token slice, B users)."""
    assert x_hidden.shape[0] == 1 and x_hidden.shape[1] == 1  # [1, 1, B, hidden]
    return ttnn.from_torch(
        x_hidden.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )


def _build_x_batched(x_hidden_t_b: torch.Tensor, mesh_device, replicate) -> ttnn.Tensor:
    """Approach-B input: [1, 1, B*T, hidden]. Stacked users-outer, tokens-inner."""
    assert x_hidden_t_b.shape[0] == 1 and x_hidden_t_b.shape[1] == 1  # [1, 1, B*T, hidden]
    return ttnn.from_torch(
        x_hidden_t_b.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )


def _approach_a_pipelined(
    x_per_token_host: list,
    cur_pos: int,
    T: int,
    weights,
    config,
    tp: int,
    cos_cache_2d,
    sin_cache_2d,
    mesh_device,
    replicate,
):
    """Approach A end-to-end with NO internal sync — only one sync at the end,
    so the device pipelines all ops. Production-style measurement.

    Returns: (per_token_q_list, total_wall_ms).
    """
    t0 = _now()
    per_token_q = []
    for t in range(T):
        x_tt = _build_x_decode(x_per_token_host[t], mesh_device, replicate)
        xqkv = apply_qkv_projection(x_tt, weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_q, tt_k, tt_v = split_qkv_heads_decode(
            xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)
        positions = torch.full((B_TEST,), cur_pos + t, dtype=torch.int32)
        cos_g, sin_g = _gather_positions_2d(positions, cos_cache_2d, sin_cache_2d, mesh_device, replicate)
        batch_dim = tt_q.shape[1]
        if cos_g.shape[2] != batch_dim:
            cos_g = cos_g[:, :, :batch_dim, :]
            sin_g = sin_g[:, :, :batch_dim, :]
        tt_q = apply_rope(tt_q, cos_g, sin_g, token_index=0)
        tt_k = apply_rope(tt_k, cos_g, sin_g, token_index=0)
        per_token_q.append(tt_q)
    _sync(mesh_device)
    return per_token_q, _now() - t0


def _approach_b_pipelined(
    x_batched_host: torch.Tensor,
    cur_pos: int,
    T: int,
    weights,
    config,
    tp: int,
    cos_cache_2d,
    sin_cache_2d,
    mesh_device,
    replicate,
):
    """Approach B end-to-end with NO internal sync. Production-style."""
    t0 = _now()
    x_tt = _build_x_batched(x_batched_host, mesh_device, replicate)
    xqkv = apply_qkv_projection(x_tt, weights, memory_config=ttnn.L1_MEMORY_CONFIG)
    try:
        tt_q, tt_k, tt_v = split_qkv_heads_decode(
            xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
    except Exception:
        from models.demos.gemma4_cody.tt.attention.operations import split_qkv_heads_prefill

        tt_q, tt_k, tt_v = split_qkv_heads_prefill(
            xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        tt_q = ttnn.transpose(tt_q, 1, 2)
        tt_k = ttnn.transpose(tt_k, 1, 2)
        tt_v = ttnn.transpose(tt_v, 1, 2)

    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
    tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
    tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    positions = torch.tensor(
        [cur_pos + t for _ in range(B_TEST) for t in range(T)],
        dtype=torch.int32,
    )
    cos_g, sin_g = _gather_positions_2d(positions, cos_cache_2d, sin_cache_2d, mesh_device, replicate)
    batch_dim = tt_q.shape[1]
    if cos_g.shape[2] != batch_dim:
        cos_g = cos_g[:, :, :batch_dim, :]
        sin_g = sin_g[:, :, :batch_dim, :]
    tt_q = apply_rope(tt_q, cos_g, sin_g, token_index=0)
    tt_k = apply_rope(tt_k, cos_g, sin_g, token_index=0)
    _sync(mesh_device)
    return tt_q, _now() - t0


def _gather_positions_2d(
    positions_flat: torch.Tensor,
    cos_cache_2d: ttnn.Tensor,
    sin_cache_2d: ttnn.Tensor,
    mesh_device,
    replicate,
):
    """Look up cos/sin at the given flat positions, returning [1, 1, N, head_dim] tensors."""
    pos_uint = ttnn.from_torch(
        positions_flat.reshape(1, -1).to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    cos_g = ttnn.embedding(pos_uint, cos_cache_2d, layout=ttnn.TILE_LAYOUT)
    sin_g = ttnn.embedding(pos_uint, sin_cache_2d, layout=ttnn.TILE_LAYOUT)
    cos_g = ttnn.unsqueeze_to_4D(cos_g)
    sin_g = ttnn.unsqueeze_to_4D(sin_g)
    return cos_g, sin_g


def _approach_a_one_step(
    x_per_token: list,  # list of T host tensors, each [1, 1, B, hidden]
    cur_pos: int,
    T: int,
    weights,
    config,
    tp: int,
    cos_cache_2d,
    sin_cache_2d,
    mesh_device,
    replicate,
):
    """T iterations of full decode prep. Returns list of T Q tensors and phase timings."""
    phase = {"upload": 0.0, "qkv": 0.0, "split": 0.0, "norm": 0.0, "rope": 0.0}
    per_token_q = []

    for t in range(T):
        # Upload (counted because A pays it T times)
        t0 = _now()
        x_tt = _build_x_decode(x_per_token[t], mesh_device, replicate)
        _sync(mesh_device)
        phase["upload"] += _now() - t0

        # QKV
        t0 = _now()
        xqkv = apply_qkv_projection(x_tt, weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        _sync(mesh_device)
        phase["qkv"] += _now() - t0

        # Split
        t0 = _now()
        tt_q, tt_k, tt_v = split_qkv_heads_decode(
            xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        _sync(mesh_device)
        phase["split"] += _now() - t0

        # Norms (Q, K; V no-scale)
        t0 = _now()
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)
        _sync(mesh_device)
        phase["norm"] += _now() - t0

        # RoPE (per-slot positions = cur_pos + t for every batch slot)
        t0 = _now()
        positions = torch.full((B_TEST,), cur_pos + t, dtype=torch.int32)
        cos_g, sin_g = _gather_positions_2d(positions, cos_cache_2d, sin_cache_2d, mesh_device, replicate)
        batch_dim = tt_q.shape[1]
        if cos_g.shape[2] != batch_dim:
            cos_g = cos_g[:, :, :batch_dim, :]
            sin_g = sin_g[:, :, :batch_dim, :]
        tt_q = apply_rope(tt_q, cos_g, sin_g, token_index=0)
        tt_k = apply_rope(tt_k, cos_g, sin_g, token_index=0)
        _sync(mesh_device)
        phase["rope"] += _now() - t0

        per_token_q.append(tt_q)

    return per_token_q, phase


def _approach_b_one_step(
    x_batched_host: torch.Tensor,  # [1, 1, B*T, hidden] users-outer tokens-inner
    cur_pos: int,
    T: int,
    weights,
    config,
    tp: int,
    cos_cache_2d,
    sin_cache_2d,
    mesh_device,
    replicate,
):
    """ONE pass of decode prep on a stacked [1, 1, B*T, hidden] input.

    Layout: [u0_t0, u0_t1, …, u0_t{T-1}, u1_t0, …, u{B-1}_t{T-1}] in the "batch" slot.
    """
    phase = {"upload": 0.0, "qkv": 0.0, "split": 0.0, "norm": 0.0, "rope": 0.0}

    # Upload
    t0 = _now()
    x_tt = _build_x_batched(x_batched_host, mesh_device, replicate)
    _sync(mesh_device)
    phase["upload"] += _now() - t0

    # QKV (one call on a B*T-row input)
    t0 = _now()
    xqkv = apply_qkv_projection(x_tt, weights, memory_config=ttnn.L1_MEMORY_CONFIG)
    _sync(mesh_device)
    phase["qkv"] += _now() - t0

    # Split (one call). nlp_create_qkv_heads_decode treats dim 2 as batch; it'll
    # see B*T "slots." Fall back to the prefill split if the decode split rejects.
    t0 = _now()
    try:
        tt_q, tt_k, tt_v = split_qkv_heads_decode(
            xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
    except Exception:
        # Prefill split takes shape [1, 1, S, qkv_dim] and returns [1, H_local, S, head_dim].
        from models.demos.gemma4_cody.tt.attention.operations import split_qkv_heads_prefill

        tt_q, tt_k, tt_v = split_qkv_heads_prefill(
            xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        # Transpose to decode-style [1, B*T, H_local, head_dim] for downstream ops.
        tt_q = ttnn.transpose(tt_q, 1, 2)
        tt_k = ttnn.transpose(tt_k, 1, 2)
        tt_v = ttnn.transpose(tt_v, 1, 2)
    _sync(mesh_device)
    phase["split"] += _now() - t0

    # Norms (one call each)
    t0 = _now()
    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
    tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
    tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)
    _sync(mesh_device)
    phase["norm"] += _now() - t0

    # RoPE (one call with B*T positions). Layout is users-outer tokens-inner,
    # so position[u*T + t] = cur_pos + t. (Test uses uniform cur_pos across users.)
    t0 = _now()
    positions = torch.tensor(
        [cur_pos + t for _ in range(B_TEST) for t in range(T)],
        dtype=torch.int32,
    )
    cos_g, sin_g = _gather_positions_2d(positions, cos_cache_2d, sin_cache_2d, mesh_device, replicate)
    batch_dim = tt_q.shape[1]
    if cos_g.shape[2] != batch_dim:
        cos_g = cos_g[:, :, :batch_dim, :]
        sin_g = sin_g[:, :, :batch_dim, :]
    tt_q = apply_rope(tt_q, cos_g, sin_g, token_index=0)
    tt_k = apply_rope(tt_k, cos_g, sin_g, token_index=0)
    _sync(mesh_device)
    phase["rope"] += _now() - t0

    return tt_q, phase


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [5], ids=["global"])
@pytest.mark.parametrize("T", [4], ids=lambda v: f"T{v}")
def test_compare_prep(layer_idx, T, mesh_device):
    """Time per-token-loop prep (A) vs batched prep (B) for the same logical work.

    Sanity-check: Q tensors from A and B should agree at PCC > 0.99.
    """
    torch.manual_seed(0)

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    state_dict = {k: v.clone() for k, v in hf_layer.self_attn.state_dict().items() if not k.startswith("v_norm")}

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    layer_type = hf_text_config.layer_types[layer_idx]
    head_dim = config.head_dim
    H = config.num_attention_heads
    H_local = H // tp
    hidden_size = hf_text_config.hidden_size

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        max_batch_size=B_TEST,
    )
    weights = tt_attn.weights

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    max_seq_len = 4096
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
    cos_cache_2d, sin_cache_2d = rope_caches_2d[layer_type]

    cur_pos = 128

    # Build identical input data for A and B. One logical user replicated across B_TEST.
    x_user = torch.randn(1, T, hidden_size, dtype=torch.float32)  # [1, T, hidden]
    # Per-token slice for A: [1, 1, B, hidden]
    x_per_token_host = [
        x_user[:, t : t + 1, :].expand(1, B_TEST, hidden_size).contiguous().unsqueeze(0) for t in range(T)
    ]
    # Stacked for B: users-outer tokens-inner → [1, 1, B*T, hidden].
    # Index (u, t) → row u*T + t.
    x_batched_host = x_user.expand(B_TEST, T, hidden_size).contiguous().reshape(1, 1, B_TEST * T, hidden_size)

    # ── Warm-up (covers both modes)
    for _ in range(NUM_WARMUP):
        _ = _approach_a_one_step(
            x_per_token_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )
        _ = _approach_b_one_step(
            x_batched_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )
        _ = _approach_a_pipelined(
            x_per_token_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )
        _ = _approach_b_pipelined(
            x_batched_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )

    # ── Mode 1: per-phase timing (sync at every phase boundary for attribution).
    accum_a = {"upload": 0.0, "qkv": 0.0, "split": 0.0, "norm": 0.0, "rope": 0.0}
    accum_b = {"upload": 0.0, "qkv": 0.0, "split": 0.0, "norm": 0.0, "rope": 0.0}
    total_a = 0.0
    total_b = 0.0

    last_q_a = None
    last_q_b = None
    for _ in range(NUM_TIMED):
        t0 = _now()
        q_list_a, phase_a = _approach_a_one_step(
            x_per_token_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )
        total_a += _now() - t0
        for k, v in phase_a.items():
            accum_a[k] += v

        t0 = _now()
        q_b, phase_b = _approach_b_one_step(
            x_batched_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )
        total_b += _now() - t0
        for k, v in phase_b.items():
            accum_b[k] += v

        last_q_a = q_list_a
        last_q_b = q_b

    # ── Mode 2: pipelined timing (sync ONLY at end of step — production-style).
    pipelined_a = 0.0
    pipelined_b = 0.0
    for _ in range(NUM_TIMED):
        _, ms_a = _approach_a_pipelined(
            x_per_token_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )
        pipelined_a += ms_a

        _, ms_b = _approach_b_pipelined(
            x_batched_host, cur_pos, T, weights, config, tp, cos_cache_2d, sin_cache_2d, mesh_device, replicate
        )
        pipelined_b += ms_b

    # ── Report
    n = NUM_TIMED
    print()
    print(f"=== Prep timing: Approach A (T-loop) vs Approach B (batched) ===")
    print(f"T={T}, B={B_TEST}, tp={tp}, H_local={H_local}, head_dim={head_dim}, hidden={hidden_size}")
    print(f"Averaged over {n} runs ({NUM_WARMUP} warmup).")
    print()

    print(f"--- Mode 1: per-phase sync (attribution; INFLATES A because of fixed per-call sync) ---")
    print(f"{'Phase':<10} {'A (ms)':>10} {'B (ms)':>10} {'A/B':>8}  {'Note'}")
    print(f"{'-'*10:<10} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8}  {'-'*30}")
    for k in ("upload", "qkv", "split", "norm", "rope"):
        a_ms = accum_a[k] / n * 1e3
        b_ms = accum_b[k] / n * 1e3
        ratio = a_ms / b_ms if b_ms > 0 else float("inf")
        note = "T calls vs 1 call"
        print(f"{k:<10} {a_ms:>10.3f} {b_ms:>10.3f} {ratio:>8.2f}x  {note}")
    a_total = total_a / n * 1e3
    b_total = total_b / n * 1e3
    print(f"{'TOTAL':<10} {a_total:>10.3f} {b_total:>10.3f} {(a_total / b_total):>8.2f}x")
    print(
        f"Per-token: A {a_total / T:.3f} ms  /  B {b_total / T:.3f} ms  "
        f"=> Saved {(a_total - b_total):.3f} ms/step "
        f"({100 * (a_total - b_total) / a_total:.1f}%)"
    )
    print()

    print(f"--- Mode 2: pipelined (sync only at end of step; PRODUCTION-style) ---")
    a_pipe = pipelined_a / n * 1e3
    b_pipe = pipelined_b / n * 1e3
    print(f"{'TOTAL':<10} {a_pipe:>10.3f} {b_pipe:>10.3f} {(a_pipe / b_pipe):>8.2f}x")
    print(
        f"Per-token: A {a_pipe / T:.3f} ms  /  B {b_pipe / T:.3f} ms  "
        f"=> Saved {(a_pipe - b_pipe):.3f} ms/step "
        f"({100 * (a_pipe - b_pipe) / a_pipe:.1f}%)"
    )
    print()
    print(f"Mode 1 vs Mode 2 reveals: Mode 1 over-attributes overhead to A. The Mode 2")
    print(f"number ({100 * (a_pipe - b_pipe) / a_pipe:.1f}%) is the honest production gain estimate.")
    print()

    # ── Sanity check: Q tensors should match between A and B (modulo numerical noise).
    # A produced T tensors each [1, B, H_local, head_dim].
    # B produced one tensor [1, B*T, H_local, head_dim] in users-outer tokens-inner layout.
    # Reshape B → [1, B, T, H_local, head_dim] and compare slice-by-slice with A.

    def to_cpu(t):
        if is_mesh:
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        return ttnn.to_torch(t).float()

    q_b_cpu = to_cpu(last_q_b)  # [1, B*T, H_local, head_dim]
    # Layout: row u*T + t. Pull out the per-(u,t) slice.
    pccs = []
    for t in range(T):
        q_a_cpu = to_cpu(last_q_a[t])  # [1, B, H_local, head_dim]
        # Approach-B slice: every T-th row starting at offset t (u=0,1,2,...)
        q_b_slice = q_b_cpu[:, t::T, :, :]  # [1, B, H_local, head_dim]
        denom = (q_a_cpu.flatten().norm() * q_b_slice.flatten().norm()).item()
        cos_sim = (q_a_cpu.flatten() @ q_b_slice.flatten()).item() / max(denom, 1e-12)
        pccs.append(cos_sim)
        print(f"  Q PCC (token {t}, A vs B): {cos_sim:.6f}")

    assert all(p > 0.99 for p in pccs), f"Q tensors disagree between A and B: PCCs={pccs}"
