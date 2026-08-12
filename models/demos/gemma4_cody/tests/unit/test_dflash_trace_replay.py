# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight reproducer for the dflash speculative-decode HANG.

Symptom (server): the first packed-verify step completes, then the run hangs
entering step 2. The dflash drafter proposes **eagerly** — allocating device
buffers (matmul outputs, a `from_torch` mask, concat/pad) every step — while the
decode + packed-verify traces are already captured ("active"). tt-metal warns
that allocating device buffers while a trace is active is unsafe; the hang is
the warning materializing — an eager allocation lands in the trace's memory and
the next `execute_trace` deadlocks on corrupted state.

This isolates that pattern with NO model/weights: capture a trivial trace once,
then replay it in a loop. Two variants:

  * ``control``  — pure replay loop (no eager work). Must always pass; proves
    trace replay itself is fine on this mesh.
  * ``eager``    — each iter does dflash-propose-shaped eager allocations
    (`from_torch` of a mask + a few matmuls + concat + pad + a D2H read) BEFORE
    the `execute_trace`. If THIS hangs (and control passes), the eager-alloc-
    during-active-trace hypothesis is confirmed and the fix is to trace the
    dflash propose / pre-allocate its buffers (Stage 5), not eager-interleave.

NOTE: this does not exercise CCL-semaphore sharing (the dflash drafter reuses
the target's ``ccl_manager`` eagerly, which is a second hang candidate on tp>1
meshes). That needs the real ccl_manager; see the server checkpoints
(GEMMA4_DFLASH_DEBUG=1) to distinguish.

Run
---

    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_dflash_trace_replay.py -k 1x8
"""

from __future__ import annotations

import time

import torch

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric

HIDDEN = 256
SEQ = 64
N_ITERS = 8
PER_ITER_TIMEOUT_S = 30.0  # a replay iter that takes longer than this ≈ a hang


def _mesh_info(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    return is_mesh, replicate


def _to_tt(mesh_device, replicate, t):
    return ttnn.from_torch(
        t.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
    )


def _read(mesh_device, is_mesh, t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0] if is_mesh else t).float()


def _eager_propose_shaped_work(mesh_device, is_mesh, replicate, w):
    """Mimic one dflash `decode_step`'s eager allocation profile: a `from_torch`
    mask (host→device alloc during active trace — the prime suspect), a few
    matmuls, a concat, a pad, and a D2H read. All freed before returning."""
    mask = _to_tt(mesh_device, replicate, torch.zeros(1, 1, SEQ, 2 * HIDDEN))  # like _build_noise_mask
    x = _to_tt(mesh_device, replicate, torch.randn(1, 1, SEQ, HIDDEN))
    for _ in range(5):  # ~5 dflash layers
        y = ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        x = y
    cat = ttnn.concat([x, x], dim=2)  # like anchor∥noise concat
    padded = ttnn.pad(cat, [(0, 0), (0, 0), (0, 8), (0, 0)], value=0.0)
    _ = _read(mesh_device, is_mesh, padded)  # like draft_ids_from_logits
    ttnn.deallocate(mask)
    ttnn.deallocate(x)
    ttnn.deallocate(cat)
    ttnn.deallocate(padded)


def _capture_matmul_trace(mesh_device, replicate, w):
    """Pre-allocate a persistent input + capture a trivial (input@w) trace."""
    in_buf = _to_tt(mesh_device, replicate, torch.randn(1, 1, SEQ, HIDDEN))
    # Warmup/compile outside the trace, then capture.
    out = ttnn.linear(in_buf, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out.deallocate(True)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = ttnn.linear(in_buf, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    return in_buf, out, tid


def _run_replay_loop(mesh_device, eager: bool):
    is_mesh, replicate = _mesh_info(mesh_device)
    w = _to_tt(mesh_device, replicate, torch.randn(1, 1, HIDDEN, HIDDEN))
    in_buf, out, tid = _capture_matmul_trace(mesh_device, replicate, w)

    try:
        for it in range(N_ITERS):
            t0 = time.time()
            if eager:
                _eager_propose_shaped_work(mesh_device, is_mesh, replicate, w)
            # Replay the captured trace (the server uses blocking=False).
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            res = _read(mesh_device, is_mesh, out)  # forces completion (would hang here)
            dt = time.time() - t0
            print(
                f"[trace-replay {'eager' if eager else 'control'}] iter {it} OK in {dt * 1e3:.0f}ms "
                f"(out.sum~{float(res.sum()):.1f})",
                flush=True,
            )
            assert dt < PER_ITER_TIMEOUT_S, f"iter {it} took {dt:.1f}s — likely hung"
    finally:
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(in_buf)
        ttnn.deallocate(w)

    print(f"[trace-replay {'eager' if eager else 'control'}] all {N_ITERS} iters completed.", flush=True)


@parametrize_mesh_with_fabric()
def test_trace_replay_control(mesh_device):
    """Pure trace replay loop — no eager allocations. Sanity baseline."""
    _run_replay_loop(mesh_device, eager=False)


@parametrize_mesh_with_fabric()
def test_trace_replay_eager_interleaved(mesh_device):
    """Replay loop with dflash-propose-shaped eager allocations each iter.

    If this hangs/times out while ``test_trace_replay_control`` passes, the
    server hang is the eager-alloc-during-active-trace pattern → trace the
    dflash propose (Stage 5) rather than eager-interleaving it.
    """
    _run_replay_loop(mesh_device, eager=True)
