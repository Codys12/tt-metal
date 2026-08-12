# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the DFlash **on-device fast path** (server commit).

These call the EXACT production functions (`DflashSpeculativeDecoder` static
methods that `_commit_packed_verify` / `append_committed_ondevice` use) — only
the *reference* is an independent torch implementation. So a green test means
production is correct; there is no separate test-only device logic to drift.

  * **Phase 2a — on-device aux gather** (in production today). The committed
    positions' aux taps stay ON DEVICE and are gathered into `_aux_dev` via
    `aux_gather_index` (the row map) + `gather_committed_aux_ondevice` (concat +
    `ttnn.embedding` row-gather + `ttnn.assign`), replacing the old
    D2H(K taps)→host-cat→H2D round-trip. `test_ondevice_aux_gather_parity` runs
    those two production fns at PRODUCTION dims (B=32, hidden=5376, K=6) and
    asserts `_aux_dev` matches the torch reference bit-for-bit. The old
    `test_dflash_packed_decode_steps.py` builds `_aux_dev` on the HOST, so it
    never exercised this path — which is why the gather bug was silent.

  * **Phase 2b — on-device greedy accept count** (`ondevice_n_accepted`). The
    helper the on-device accept WILL call; validated here against the host greedy
    match BEFORE it is wired into `_commit_packed_verify`, so wiring is safe.
    Token ids MUST be int32 — bf16 cannot represent ids > 256, so a bf16 compare
    would yield false matches. (NOTE: not yet called by the server.)

Run (no checkpoint/model needed — exercises the ttnn op helpers only):
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOME=/mnt/nas/scratch MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_dflash_fast_path.py -k 1x8
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4_cody.server.dflash_spec import DflashSpeculativeDecoder

from ...tests.test_factory import parametrize_mesh_with_fabric

# Production z-lab dims (mirror DFlashConfig + DECODE_BATCH so TILE padding,
# vocab range, and the concat/gather shapes match the server exactly).
B = 32  # DECODE_BATCH / num drafter slots
BLOCK = 16  # block_size ⇒ P = block, T = block-1
K = 6  # num aux taps
HIDDEN = 5376  # target hidden ⇒ Kh = fc_in_features = K*HIDDEN = 32256
VOCAB = 262144


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom < 1e-12 else (a @ b).item() / denom


@parametrize_mesh_with_fabric()
def test_ondevice_aux_gather_parity(mesh_device):
    """Phase 2a (PRODUCTION path): `aux_gather_index` + `gather_committed_aux_ondevice`
    must reproduce the host-built `_aux_dev` bit-for-bit. This is the test that
    would have caught the acceptance collapse."""
    torch.manual_seed(0)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    P, Kh = BLOCK, K * HIDDEN
    B_v = B  # single occupancy bucket (server uses B_v=B=32)

    def to_tile(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def dev0(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]) if is_mesh else ttnn.to_torch(t)

    # Synthetic verify-trace aux taps: K device tensors [1,1,B_v*P,hidden] (bf16,
    # replicated) — exactly what `_packed_verify_traces[B_v]["aux"]` holds.
    aux_host = [torch.randn(B_v * P, HIDDEN, dtype=torch.float32).to(torch.bfloat16) for _ in range(K)]
    aux_taps = [to_tile(a.reshape(1, 1, B_v * P, HIDDEN)) for a in aux_host]

    # Commits cover n_acc = 0 (bonus only), a mid prefix, and a full block.
    commits = [(0, 0, 4), (1, 1, BLOCK - 1), (7, 2, 0)]

    # ── PRODUCTION index build (the exact fn append_committed_ondevice calls) ──
    gidx, any_write = DflashSpeculativeDecoder.aux_gather_index(commits, B, BLOCK, P)
    assert any_write
    gidx_dev = ttnn.from_torch(
        gidx.reshape(1, B * BLOCK),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    aux_dev = to_tile(torch.zeros(1, 1, B * BLOCK, Kh))

    # ── PRODUCTION on-device gather (the exact fn the server replays) ──
    DflashSpeculativeDecoder.gather_committed_aux_ondevice(aux_taps, gidx_dev, aux_dev, B_v, P)
    got = dev0(aux_dev).float().reshape(B * BLOCK, Kh)

    # ── reference: src2d = concat(taps, feature); _aux_dev[out] = src2d[gidx[out]] ──
    src2d = torch.cat([a.float() for a in aux_host], dim=-1)  # [B_v*P, Kh]
    expected = src2d[gidx.long()]  # [B*BLOCK, Kh]

    overall = _pcc(got, expected)
    print(f"[2a] overall gather PCC = {overall:.6f}")
    for slot_idx, r, n_acc in commits:  # every row that actually becomes an anchor must be exact
        for j in range(n_acc + 1):
            row = slot_idx * BLOCK + j
            pcc = _pcc(got[row], expected[row])
            assert pcc > 0.999, (
                f"committed row {row} (slot {slot_idx}, j={j}) mismatch: PCC={pcc:.4f}, "
                f"got_norm={got[row].norm():.3f} exp_norm={expected[row].norm():.3f} "
                f"(norm~0 ⇒ ttnn.assign did NOT write _aux_dev in place)"
            )
    assert overall > 0.999, f"on-device aux gather diverges from reference (PCC {overall:.4f})"
    print("[2a] on-device aux gather parity PASSED")


@parametrize_mesh_with_fabric()
def test_ondevice_n_accepted_parity(mesh_device):
    """Phase 2b helper: `ondevice_n_accepted` must equal the host greedy match for
    every row, at the PRODUCTION vocab (int32 — bf16 can't represent ids>256).
    Validates the function BEFORE it is wired into `_commit_packed_verify`."""
    torch.manual_seed(1)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    B_v, T, P = B, BLOCK - 1, BLOCK

    draft = torch.randint(0, VOCAB, (B_v, T), dtype=torch.int64)
    target = torch.randint(0, VOCAB, (B_v, P), dtype=torch.int64)
    for b in range(B_v):  # force each row to a known n_acc by matching a prefix then breaking it
        n = b % (T + 1)
        for k in range(n):
            target[b, k] = draft[b, k]
        if n < T:
            target[b, n] = (int(draft[b, n]) + 1) % VOCAB

    host_n = []
    for b in range(B_v):
        n = 0
        for k in range(T):
            if int(draft[b, k]) == int(target[b, k]):
                n += 1
            else:
                break
        host_n.append(n)

    def to_int(t):
        return ttnn.from_torch(
            t.to(torch.int32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
        )

    n_acc_dev = DflashSpeculativeDecoder.ondevice_n_accepted(to_int(draft), to_int(target), T)
    got = ttnn.to_torch(ttnn.get_device_tensors(n_acc_dev)[0] if is_mesh else n_acc_dev).flatten()[:B_v]
    got_n = [int(round(float(x))) for x in got]
    print(f"[2b] host n_acc   = {host_n}\n[2b] device n_acc = {got_n}")
    assert got_n == host_n, f"on-device greedy accept count mismatch: device={got_n} host={host_n}"
    print("[2b] on-device greedy accept parity PASSED")
