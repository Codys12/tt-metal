# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter parity test: TT vs PyTorch reference.

Both run with the SAME synthetic inputs (replicated across batch where TT
needs B=32 alignment, B=1 on the reference). PCC ≥ 0.99 on:

  * ``fc + hidden_norm`` output (the context projection)
  * Per-layer hidden states
  * Final norm output
  * Draft logits over the 32k draft vocab

This is the Phase-4 gate. If parity holds we proceed to server integration.

Prereqs
-------

1. Download the HF speculator checkpoint::

       huggingface-cli download RedHatAI/gemma-4-31B-it-speculator.dflash \\
           --local-dir /mnt/nas/dflash-gemma

2. Convert weights to the TT cache::

       export TT_CACHE_PATH=/mnt/nas/gemma_cache
       python -m models.demos.gemma4_cody.tt.dflash.convert_weights \\
           --src /mnt/nas/dflash-gemma \\
           --dst $TT_CACHE_PATH/tensor_cache_dflash_bf16

Run
---

    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           TT_CACHE_PATH=/mnt/nas/gemma_cache MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_dflash_parity.py -k 1x8
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
CTX_LEN = 64
PCC_LOGITS = 0.99
PCC_HIDDEN = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)
    denom = (a_flat.norm() * b_flat.norm()).item()
    if denom < 1e-12:
        return 0.0
    return (a_flat @ b_flat).item() / denom


def _tt_to_torch(t: ttnn.Tensor) -> torch.Tensor:
    """Pull device-0 shard from the mesh as a torch tensor."""
    shards = ttnn.get_device_tensors(t)
    return ttnn.to_torch(shards[0]).float()


def _tt_to_torch_full(t: ttnn.Tensor, gather_dim: int | None) -> torch.Tensor:
    """Reassemble a TP-sharded tensor by concatenating per-device shards.

    When the tensor is replicated (gather_dim=None), returns device 0's shard.
    """
    shards = ttnn.get_device_tensors(t)
    pieces = [ttnn.to_torch(s).float() for s in shards]
    if gather_dim is None or len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=gather_dim)


@parametrize_mesh_with_fabric()
def test_dflash_parity_against_reference(mesh_device):
    """One forward pass, compare TT vs the PyTorch reference."""
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

    # ─── Build inputs (B=1 logical user) ────────────────────────────────────
    H_target = cfg.target_hidden_size
    aux_concat_1 = torch.randn(1, CTX_LEN, cfg.fc_in_features, dtype=torch.float32)
    noise_1 = torch.randn(1, cfg.block_size, cfg.hidden_size, dtype=torch.float32)

    # Position ids cover (ctx_len + block_size). Use absolute positions
    # ``[0, ctx_len + block_size)``. The actual server choice in steady state
    # is ``[cached_seq_len, cached_seq_len + block_size + ctx_len)``; for
    # parity any consistent absolute span works as long as both sides agree.
    positions = torch.arange(CTX_LEN + cfg.block_size).unsqueeze(0)
    cos, sin = build_rope_cache(positions, head_dim=cfg.head_dim, theta=cfg.rope_theta)

    # ─── PyTorch reference (lazy, streamed from safetensors) ──────────────────
    # The reference reads each tensor on demand from `model.safetensors` and
    # frees per-layer weights between layers; peak RSS is ~1 layer fp32 + the
    # small top-level pieces (fc, lm_head, norms) ≈ ~2.5 GB. Eager loading the
    # full 4.3 B fp32 model would need 17 GB.
    print("Loading lazy DFlash reference from safetensors ...")
    ref = load_reference_from_safetensors(cfg, DFLASH_PATH)
    with torch.no_grad():
        ref_logits, ref_hidden, ref_inters = ref(aux_concat_1, noise_1, cos, sin, return_intermediates=True)
    print(f"  ref_logits {tuple(ref_logits.shape)} ref_hidden {tuple(ref_hidden.shape)}")

    # ─── TT forward ──────────────────────────────────────────────────────────
    print("Building TT DFlashDrafter ...")
    model = DFlashDrafter(mesh_device=mesh_device, config=cfg)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tp = mesh_device.shape[1] if is_mesh else 1

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    # Add a leading dim: TT operates on [1, B, S, F] shapes.
    aux_tt = to_tt(aux_concat_1.unsqueeze(0))  # [1, 1, ctx_len, K*target_hidden]
    noise_tt = to_tt(noise_1.unsqueeze(0))  # [1, 1, block_size, hidden]
    cos_tt = to_tt(cos.unsqueeze(0))  # [1, 1, ctx+block, head_dim]
    sin_tt = to_tt(sin.unsqueeze(0))

    tt_logits, tt_hidden, tt_inters = model.forward(
        aux_hiddens_concat=aux_tt,
        noise_embeddings=noise_tt,
        cos_full=cos_tt,
        sin_full=sin_tt,
        return_intermediates=True,
    )

    # ─── Compare ────────────────────────────────────────────────────────────
    # tt_inters / tt_hidden come from `ttnn.to_torch(ttnn.get_device_tensors(...)[0])`,
    # which is device-0's shard. The drafter's residual stream is replicated
    # (post-allgather), so device 0 carries the full feature dim.
    print("\nIntermediate parity:")
    failures = []

    name_to_ref = {
        "target_hidden_projected": ref_inters["target_hidden_projected"],
        **{f"layer_{i}": ref_inters[f"layer_{i}"] for i in range(cfg.num_hidden_layers)},
        "final_norm": ref_inters["final_norm"],
    }
    for name, tt_t in tt_inters:
        ref_t = name_to_ref.get(name)
        if ref_t is None:
            print(f"  {name}: <no reference>")
            continue
        # tt_t shape: [1, 1, S, hidden]  ;  ref_t shape: [B=1, S, hidden]
        tt_squeezed = tt_t.squeeze(0)
        pcc = _pcc(tt_squeezed.float(), ref_t.float())
        print(f"  {name}: PCC = {pcc:.6f}")
        if pcc < PCC_HIDDEN:
            failures.append(f"{name} PCC {pcc:.4f} < {PCC_HIDDEN}")

    # Logits: lm_head is column-parallel on draft_vocab dim → reassemble.
    tt_logits_full = _tt_to_torch_full(tt_logits, gather_dim=-1)  # [1, 1, block_size-1, draft_vocab]
    tt_logits_squeezed = tt_logits_full.squeeze(0).squeeze(0)  # [block_size-1, draft_vocab]
    ref_logits_squeezed = ref_logits.squeeze(0).float()  # [block_size-1, draft_vocab]
    print(f"\nlogits TT shape={tuple(tt_logits_squeezed.shape)}  ref shape={tuple(ref_logits_squeezed.shape)}")
    logits_pcc = _pcc(tt_logits_squeezed, ref_logits_squeezed)
    print(f"draft_logits PCC = {logits_pcc:.6f}  (threshold {PCC_LOGITS})")
    if logits_pcc < PCC_LOGITS:
        failures.append(f"draft_logits PCC {logits_pcc:.4f} < {PCC_LOGITS}")

    # Per-position argmax agreement (greedy verify uses argmax — this is the
    # downstream metric the server cares about).
    tt_argmax = tt_logits_squeezed.argmax(dim=-1)
    ref_argmax = ref_logits_squeezed.argmax(dim=-1)
    match = (tt_argmax == ref_argmax).float().mean().item()
    print(f"per-position argmax agreement = {match:.3f}  ({tt_argmax.tolist()} vs {ref_argmax.tolist()})")

    assert not failures, "Parity failures:\n  " + "\n  ".join(failures)
    print("\nDFlash parity test PASSED.")
