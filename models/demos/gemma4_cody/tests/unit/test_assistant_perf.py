# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drafter wall-clock timing on TT mesh.

Runs the assistant forward repeatedly with synthetic inputs and times the
pipelined wall-clock (sync at end). Plugs the result into the speedup math
from the n-gram demo so we can put real numbers on the question:
"is the drafter cheap enough to make speculation worth it?"

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_assistant_perf.py -k 1x8
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.demos.gemma4_cody.tt.assistant.config import Gemma4AssistantConfig
from models.demos.gemma4_cody.tt.assistant.model import Gemma4AssistantModel

from ...tests.test_factory import parametrize_mesh_with_fabric

B = 32  # cody's production batch
KV_LEN = 128  # synthetic shared-KV length
NUM_WARMUP = 5
NUM_TIMED = 20


def _now():
    return time.perf_counter()


def _sync(d):
    ttnn.synchronize_device(d)


@parametrize_mesh_with_fabric()
def test_drafter_forward_perf(mesh_device):
    """Time a drafter forward pass; report mean / min / max and project speedup."""
    cfg = Gemma4AssistantConfig.from_hf_path("/mnt/nas/gemma-assistant")

    print(f"Loading TT drafter ...")
    t0 = _now()
    model = Gemma4AssistantModel(mesh_device=mesh_device, config=cfg)
    print(f"  loaded in {_now() - t0:.1f}s")

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    def to_tt(t, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=layout,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    target_hidden = to_tt(torch.randn(1, 1, B, 2 * cfg.backbone_hidden_size))
    K_swa = to_tt(torch.randn(B, cfg.num_key_value_heads, KV_LEN, cfg.head_dim))
    V_swa = to_tt(torch.randn(B, cfg.num_key_value_heads, KV_LEN, cfg.head_dim))
    K_full = to_tt(torch.randn(B, cfg.num_global_key_value_heads, KV_LEN, cfg.global_head_dim))
    V_full = to_tt(torch.randn(B, cfg.num_global_key_value_heads, KV_LEN, cfg.global_head_dim))
    shared_kv = {
        "sliding_attention": (K_swa, V_swa),
        "full_attention": (K_full, V_full),
    }
    cos_full = to_tt(torch.randn(1, 1, B, cfg.global_head_dim))
    sin_full = to_tt(torch.randn(1, 1, B, cfg.global_head_dim))
    cos_swa = to_tt(torch.randn(1, 1, B, cfg.head_dim))
    sin_swa = to_tt(torch.randn(1, 1, B, cfg.head_dim))
    cur_pos = ttnn.from_torch(
        torch.full((B,), KV_LEN - 1, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    def call_forward():
        out_hidden, logits = model.forward(
            target_last_hidden=target_hidden,
            shared_kv=shared_kv,
            cos_pos_full=cos_full,
            sin_pos_full=sin_full,
            cos_pos_sliding=cos_swa,
            sin_pos_sliding=sin_swa,
            cur_pos_tensor=cur_pos,
        )
        _sync(mesh_device)
        return out_hidden, logits

    # Warmup
    for _ in range(NUM_WARMUP):
        call_forward()

    # Timed (pipelined: only sync at end of call, not between layers).
    samples_ms = []
    for _ in range(NUM_TIMED):
        t0 = _now()
        call_forward()
        samples_ms.append((_now() - t0) * 1e3)

    mean_ms = sum(samples_ms) / len(samples_ms)
    min_ms = min(samples_ms)
    max_ms = max(samples_ms)
    median_ms = sorted(samples_ms)[len(samples_ms) // 2]

    print()
    print(f"=== Drafter forward perf on 1x8 P150 (B={B}, KV_LEN={KV_LEN}) ===")
    print(f"  warmup={NUM_WARMUP}  timed={NUM_TIMED}  pipelined sync only at end")
    print(f"  mean   {mean_ms:6.2f} ms")
    print(f"  median {median_ms:6.2f} ms")
    print(f"  min    {min_ms:6.2f} ms")
    print(f"  max    {max_ms:6.2f} ms")
    print()

    # ── Speedup math
    BASELINE_MS = 88.0  # production single-token decode step
    VERIFIER_MS = 109.0  # packed-decode verifier (Approach B, prep + SDPA)

    print(f"=== Plug into speedup math (T draft tokens per step) ===")
    print(f"  baseline single-token step: {BASELINE_MS:.1f} ms")
    print(f"  packed-decode verifier:     {VERIFIER_MS:.1f} ms")
    print(f"  drafter forward (this run): {mean_ms:.2f} ms")
    print()

    def expected_accepts(p, T):
        if p >= 1.0:
            return float(T)
        return p * (1 - p**T) / (1 - p)

    for T in (2, 4, 8):
        drafter_cost_T = mean_ms * T  # T sequential drafter forwards per step
        # Parallel deploy (drafter || verifier — runs on shared mesh, same iteration):
        # effective step = max(verifier, drafter_cost_T) since they share devices.
        # Serial deploy (drafter then verifier, distinct passes):
        eff_parallel = max(VERIFIER_MS, drafter_cost_T)
        eff_serial = VERIFIER_MS + drafter_cost_T

        print(f"  -- T={T} drafts/step --")
        print(f"     drafter cost (T forwards): {drafter_cost_T:.1f} ms")
        print(f"     {'p':>5} | {'parallel':>14} | {'serial':>14}")
        for p in (0.70, 0.80, 0.85, 0.90):
            emit = expected_accepts(p, T) + 1
            sp_par = emit / eff_parallel * BASELINE_MS
            sp_ser = emit / eff_serial * BASELINE_MS
            print(f"     {p:>5.2f} | {sp_par:>11.2f}x   | {sp_ser:>11.2f}x")
        print()

    assert mean_ms > 0
    assert min_ms < 5 * mean_ms  # sanity: no extreme outliers
