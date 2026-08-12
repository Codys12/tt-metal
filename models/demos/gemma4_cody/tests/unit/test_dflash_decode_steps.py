# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash decode-time parity: own-anchor-KV `decode_step` vs PyTorch reference.

`test_dflash_parity.py` validates the one-shot `forward`. This validates the
**decode-time** path the server drives: `DFlashDrafter.decode_step` reading an
own anchor KV cache that grows by one anchor per step (`append_anchors`), versus
the lazy PyTorch reference re-run over the full (growing) context each step.

Per step `s` (B=1 logical user):

  1. append anchor `s` (its synthetic aux taps) at RoPE position `s`;
  2. `decode_step(noise_s)` drafts `block_size-1` tokens reading anchors [0..s];
  3. the reference runs with `aux_concat = anchors[0..s]` + the same `noise_s`;
  4. assert PCC ≥ 0.99 on the draft logits and report per-position argmax match.

This is the Stage-1 correctness gate; it must run on hardware. The cached
(incrementally normed/RoPE'd) anchors must equal the reference's recomputed
context K/V — the equivalence the decode design rests on.

Run
---

    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           TT_CACHE_PATH=/mnt/nas/gemma_cache DFLASH_PATH=/mnt/nas/dflash-gemma MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_dflash_decode_steps.py -k 1x8
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gemma4_cody.tt.dflash import DFlashConfig, DFlashDrafter
from models.demos.gemma4_cody.tt.dflash._reference import build_rope_cache, load_reference_from_safetensors

from ...tests.test_factory import parametrize_mesh_with_fabric

DFLASH_PATH = os.environ.get("DFLASH_PATH", "/mnt/nas/dflash-gemma")
DFLASH_CACHE = os.environ.get(
    "DFLASH_CACHE",
    os.path.join(os.environ.get("TT_CACHE_PATH", "/mnt/nas/gemma_cache"), "tensor_cache_dflash_bf16"),
)
N_STEPS = 6  # anchor_len grows 1..N_STEPS
PCC_LOGITS = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)
    denom = (a_flat.norm() * b_flat.norm()).item()
    if denom < 1e-12:
        return 0.0
    return (a_flat @ b_flat).item() / denom


@parametrize_mesh_with_fabric()
def test_dflash_decode_steps_against_reference(mesh_device):
    if not Path(DFLASH_PATH, "config.json").is_file():
        pytest.skip(
            f"DFlash checkpoint not found at {DFLASH_PATH}; "
            f"download with `huggingface-cli download RedHatAI/gemma-4-31B-it-speculator.dflash --local-dir {DFLASH_PATH}`"
        )
    if not Path(DFLASH_CACHE, "fc.weight.tensorbin").is_file():
        pytest.skip(
            f"DFlash TT cache not found at {DFLASH_CACHE}; "
            f"run `python -m models.demos.gemma4_cody.tt.dflash.convert_weights --src {DFLASH_PATH} --dst {DFLASH_CACHE}`"
        )

    torch.manual_seed(0)
    cfg = DFlashConfig.from_hf_path(DFLASH_PATH)
    block = cfg.block_size

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def tt_logits_to_torch(tt) -> torch.Tensor:
        # lm_head is column-parallel on the draft-vocab dim → reassemble shards.
        tp = mesh_device.shape[1] if is_mesh else 1
        if is_mesh and tp > 1:
            return torch.cat([ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(tt)], dim=-1)
        if is_mesh:
            return ttnn.to_torch(ttnn.get_device_tensors(tt)[0]).float()
        return ttnn.to_torch(tt).float()

    # Synthetic per-step inputs (B=1). Anchor `s` is one [K*target_hidden] tap row.
    aux_all = torch.randn(1, N_STEPS, cfg.fc_in_features, dtype=torch.float32)
    noise_all = torch.randn(1, N_STEPS, block, cfg.hidden_size, dtype=torch.float32)

    # Full RoPE table covering every anchor + the (padded-to-64) noise block.
    NOISE_PAD = 64
    max_pos = N_STEPS + NOISE_PAD + 1
    positions = torch.arange(max_pos).unsqueeze(0)
    cos_host, sin_host = build_rope_cache(positions, head_dim=cfg.head_dim, theta=cfg.rope_theta)  # [1, max_pos, hd]

    print("Loading lazy DFlash reference ...")
    ref = load_reference_from_safetensors(cfg, DFLASH_PATH)

    print("Building TT DFlashDrafter ...")
    model = DFlashDrafter(mesh_device=mesh_device, config=cfg)
    cache = model.init_anchor_cache()

    failures = []
    for s in range(N_STEPS):
        noise_len = s + 1  # anchor_len after this append (anchor s sits at cache index s)

        # ── append anchor s (stored un-roped) ────────────────────────────────
        aux_s = aux_all[:, s : s + 1, :]  # [1, 1, K*target_hidden]
        model.append_anchors(cache, to_tt(aux_s.unsqueeze(0)))
        assert cache.length == noise_len, f"cache.length {cache.length} != {noise_len}"

        # ── decode_step ──────────────────────────────────────────────────────
        # cos_full spans the full key range [0 : noise_len+64]; cos_q is the
        # noise query positions [noise_len : noise_len+64].
        noise_s = noise_all[:, s, :, :]  # [1, block, hidden]
        cos_full_tt = to_tt(cos_host[:, 0 : noise_len + 64, :].unsqueeze(0))  # [1,1,noise_len+64,hd]
        sin_full_tt = to_tt(sin_host[:, 0 : noise_len + 64, :].unsqueeze(0))
        cos_q = to_tt(cos_host[:, noise_len : noise_len + 64, :].unsqueeze(0))  # [1,1,64,hd]
        sin_q = to_tt(sin_host[:, noise_len : noise_len + 64, :].unsqueeze(0))
        tt_logits = model.decode_step(cache, to_tt(noise_s.unsqueeze(0)), cos_full_tt, sin_full_tt, cos_q, sin_q)
        tt_logits = tt_logits_to_torch(tt_logits).reshape(block - 1, cfg.draft_vocab_size)

        # ── reference over the same growing context ─────────────────────────
        aux_concat = aux_all[:, : s + 1, :]  # [1, s+1, K*target_hidden]
        ref_pos = torch.arange(noise_len + block).unsqueeze(0)
        cos_r, sin_r = build_rope_cache(ref_pos, head_dim=cfg.head_dim, theta=cfg.rope_theta)
        with torch.no_grad():
            ref_logits, _ = ref(aux_concat, noise_s, cos_r, sin_r)
        ref_logits = ref_logits.squeeze(0).float()  # [block-1, draft_vocab]

        pcc = _pcc(tt_logits, ref_logits)
        match = (tt_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        print(f"step {s} (anchor_len={noise_len}): logits PCC = {pcc:.6f}  argmax_match = {match:.3f}")
        if pcc < PCC_LOGITS:
            failures.append(f"step {s} PCC {pcc:.4f} < {PCC_LOGITS}")

    cache.reset()
    ref.lsd.close()
    assert not failures, "Decode-step parity failures:\n  " + "\n  ".join(failures)
    print("\nDFlash decode-step parity PASSED.")
