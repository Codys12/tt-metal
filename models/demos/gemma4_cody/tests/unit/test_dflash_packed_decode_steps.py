# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash **traced packed** decode parity: `decode_forward_packed` +
`write_anchors_packed` + `alloc_anchor_caches` vs the PyTorch reference.

`test_dflash_decode_steps.py` validates the *eager* `decode_step` (growing-concat
anchor cache). This validates the **traced** path the server actually drives —
the fixed-shape `[B, n_kv, MAX_ANCHORS, head_dim]` anchor cache written by
`paged_update_cache`, K stored **un-RoPE'd** and RoPE'd whole-cache on read, and
the head-major packed-SDPA-decode (mirrors `tt/attention/decode.py`
`packed_decode_forward`). These methods are exercised eagerly here (the trace is
just a replay wrapper over the same op sequence; trace mechanics are covered by
`test_dflash_trace_replay.py`).

The loop reproduces the server's real **propose → commit** cache evolution
(`server._step_decode`): at step `s` the propose writes this step's noise into
the cache at `[anchor_len : anchor_len+block]`, then the commit overwrites the
accepted position(s) with anchor K/V. We append exactly one anchor per step, so
`anchor_len == s` at decode time:

  step 0      decode reads 0 anchors (bootstrap) — NOT compared (the reference's
              ctx_len=0 path is degenerate); the call also teaches the drafter
              its `_q_sharded_mem_B` reshard spec so the first append works.
  step s≥1    decode reads anchors [0..s-1]; compared vs the reference run on
              `aux_concat = anchors[0..s-1]` + the same noise.
  (each step) append anchor s (its synthetic aux) at cache index s.

Anchor `s` sits at cache index `s` == its absolute RoPE position, which is what
makes the whole-cache RoPE-on-read identical to the reference's per-position
RoPE — the equivalence the un-RoPE'd cache design rests on.

Run
---

    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           TT_CACHE_PATH=/mnt/nas/gemma_cache DFLASH_PATH=/mnt/nas/dflash-gemma MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_dflash_packed_decode_steps.py -k 1x8
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
N_STEPS = 7  # anchor_len at decode time grows 0..N_STEPS-1; compared for s>=1
MAX_ANCHORS = 64  # fixed cache depth (a 64 multiple — the SDPA k_chunk); >= N_STEPS+block
PCC_LOGITS = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)
    denom = (a_flat.norm() * b_flat.norm()).item()
    if denom < 1e-12:
        return 0.0
    return (a_flat @ b_flat).item() / denom


@parametrize_mesh_with_fabric()
def test_dflash_packed_decode_steps_against_reference(mesh_device):
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
    hd = cfg.head_dim
    hidden = cfg.hidden_size
    fc_in = cfg.fc_in_features

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tp = mesh_device.shape[1] if is_mesh else 1
    n_heads_local = cfg.num_attention_heads // tp

    def to_tile(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def to_idx(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.int32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
        )

    def tt_logits_to_torch(tt) -> torch.Tensor:
        # lm_head is column-parallel on the draft-vocab dim → reassemble shards.
        if is_mesh and tp > 1:
            return torch.cat([ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(tt)], dim=-1)
        if is_mesh:
            return ttnn.to_torch(ttnn.get_device_tensors(tt)[0]).float()
        return ttnn.to_torch(tt).float()

    # Synthetic per-step inputs (B=1). Anchor `s` is one [fc_in] aux-tap row;
    # noise_all[s] is the [block, hidden] noise stream fed to step `s`'s propose.
    aux_all = torch.randn(1, N_STEPS, fc_in, dtype=torch.float32)
    noise_all = torch.randn(1, N_STEPS, block, hidden, dtype=torch.float32)

    # Full host RoPE table (anchor-relative: position i == cache index i).
    max_pos = MAX_ANCHORS + block + 1
    positions = torch.arange(max_pos).unsqueeze(0)
    cos_host, sin_host = build_rope_cache(positions, head_dim=hd, theta=cfg.rope_theta)  # [1, max_pos, hd]
    cos_host, sin_host = cos_host[0], sin_host[0]  # [max_pos, hd]

    print("Loading lazy DFlash reference ...")
    ref = load_reference_from_safetensors(cfg, DFLASH_PATH)

    print("Building TT DFlashDrafter ...")
    model = DFlashDrafter(mesh_device=mesh_device, config=cfg)
    caches = model.alloc_anchor_caches(1, MAX_ANCHORS)

    # Whole-cache RoPE (positions 0..MAX_ANCHORS-1) — constant across steps.
    fixed_cos = to_tile(cos_host[:MAX_ANCHORS].reshape(1, 1, MAX_ANCHORS, hd))
    fixed_sin = to_tile(sin_host[:MAX_ANCHORS].reshape(1, 1, MAX_ANCHORS, hd))

    failures = []
    for s in range(N_STEPS):
        anchor_len = s  # anchors [0..s-1] already in the cache at decode time

        # ── propose: decode_forward_packed (writes this step's noise at [s..s+block-1]) ──
        noise_embeds = to_tile(noise_all[:, s, :, :].reshape(1, 1, block, hidden))
        noise_write_idxs = [to_idx(torch.tensor([anchor_len + p])) for p in range(block)]
        cos_bp = to_tile(torch.stack([cos_host[anchor_len + p] for p in range(block)]).reshape(1, 1, block, hd))
        sin_bp = to_tile(torch.stack([sin_host[anchor_len + p] for p in range(block)]).reshape(1, 1, block, hd))
        # Non-causal additive mask: every noise query attends anchors [0..s-1]
        # ∥ this step's noise [s..s+block-1]; everything past s+block is masked.
        mask_t = torch.full((1, 1, n_heads_local * block, MAX_ANCHORS), -1e9, dtype=torch.float32)
        mask_t[0, 0, :, : anchor_len + block] = 0.0
        attn_mask = to_tile(mask_t)

        tt_logits = model.decode_forward_packed(
            noise_embeds, caches, noise_write_idxs, cos_bp, sin_bp, fixed_cos, fixed_sin, attn_mask, 1, block
        )
        tt_logits = tt_logits_to_torch(tt_logits).reshape(block - 1, cfg.draft_vocab_size)
        for t in noise_write_idxs:
            ttnn.deallocate(t)
        ttnn.deallocate(cos_bp)
        ttnn.deallocate(sin_bp)
        ttnn.deallocate(attn_mask)

        # ── compare vs reference over the same anchors + noise (skip bootstrap) ──
        if s >= 1:
            aux_concat = aux_all[:, :s, :]  # [1, s, fc_in] — anchors [0..s-1]
            noise_s = noise_all[:, s, :, :]  # [1, block, hidden]
            ref_pos = torch.arange(s + block).unsqueeze(0)
            cos_r, sin_r = build_rope_cache(ref_pos, head_dim=hd, theta=cfg.rope_theta)
            with torch.no_grad():
                ref_logits, _ = ref(aux_concat, noise_s, cos_r, sin_r)
            ref_logits = ref_logits.squeeze(0).float()  # [block-1, draft_vocab]
            pcc = _pcc(tt_logits, ref_logits)
            match = (tt_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
            print(f"step {s} (anchor_len={anchor_len}): logits PCC = {pcc:.6f}  argmax_match = {match:.3f}")
            if pcc < PCC_LOGITS:
                failures.append(f"step {s} PCC {pcc:.4f} < {PCC_LOGITS}")
        else:
            print(f"step {s} (anchor_len=0): bootstrap propose (learns _q_sharded_mem_B), not compared")

        # ── commit: append anchor s at cache index s (mirrors write_anchors_packed
        # with n_acc+1==1 committed position; widx[0]==s, the rest -1 → skipped) ──
        aux_app = torch.zeros(1, 1, block, fc_in, dtype=torch.float32)
        aux_app[0, 0, 0, :] = aux_all[0, s, :]
        anchor_widx = [to_idx(torch.tensor([s if j == 0 else -1])) for j in range(block)]
        model.write_anchors_packed(caches, to_tile(aux_app), anchor_widx, 1, block)
        for t in anchor_widx:
            ttnn.deallocate(t)

    ref.lsd.close()
    assert not failures, "Packed decode-step parity failures:\n  " + "\n  ".join(failures)
    print("\nDFlash traced-packed decode-step parity PASSED.")
