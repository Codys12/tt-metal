# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the DFlash drafter on TT.

Loads the converted DFlash cache, builds synthetic ``aux_hiddens_concat`` +
``noise_embeddings`` (the shapes the server would pass at decode time), and
runs one forward through the 5-layer drafter. Verifies the output tensors
have the expected shapes.

Does NOT verify correctness — that's `test_dflash_parity.py`. This catches
shape / dispatch regressions.

Prereqs
-------

1. The HF speculator checkpoint downloaded:

       huggingface-cli download RedHatAI/gemma-4-31B-it-speculator.dflash \\
           --local-dir /mnt/nas/dflash-gemma

2. Weight cache converted:

       export TT_CACHE_PATH=/mnt/nas/gemma_cache
       python -m models.demos.gemma4_cody.tt.dflash.convert_weights \\
           --src /mnt/nas/dflash-gemma \\
           --dst $TT_CACHE_PATH/tensor_cache_dflash_bf16

Run
---

    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           TT_CACHE_PATH=/mnt/nas/gemma_cache MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_dflash_forward.py -k 1x8
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.gemma4_cody.tt.dflash import DFlashConfig, DFlashDrafter

from ...tests.test_factory import parametrize_mesh_with_fabric

# Defaults — override via env vars for non-standard paths.
DFLASH_PATH = os.environ.get("DFLASH_PATH", "/mnt/nas/dflash-gemma")
CTX_LEN = 64  # parity-friendly: multiple of SDPA k_chunk=64


@parametrize_mesh_with_fabric()
def test_dflash_forward_runs(mesh_device):
    """One forward pass with synthetic inputs. Checks output shapes."""
    if not os.path.isfile(os.path.join(DFLASH_PATH, "config.json")):
        pytest.skip(
            f"DFlash checkpoint not found at {DFLASH_PATH}; "
            f"download with `huggingface-cli download RedHatAI/gemma-4-31B-it-speculator.dflash --local-dir {DFLASH_PATH}`"
        )

    cfg = DFlashConfig.from_hf_path(DFLASH_PATH)
    model = DFlashDrafter(mesh_device=mesh_device, config=cfg)

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

    # Aux hiddens concat'd along feature dim — caller's responsibility.
    # Shape: [1, 1, ctx_len, K * target_hidden].
    aux_concat = torch.randn(1, 1, CTX_LEN, cfg.fc_in_features)
    # Noise embeddings — [1, 1, block_size, hidden]. block_size=8 for Gemma.
    noise = torch.randn(1, 1, cfg.block_size, cfg.hidden_size)
    # RoPE cos/sin spanning (ctx + noise) positions.
    cos = torch.randn(1, 1, CTX_LEN + cfg.block_size, cfg.head_dim)
    sin = torch.randn(1, 1, CTX_LEN + cfg.block_size, cfg.head_dim)

    draft_logits, draft_hidden = model.forward(
        aux_hiddens_concat=to_tt(aux_concat),
        noise_embeddings=to_tt(noise),
        cos_full=to_tt(cos),
        sin_full=to_tt(sin),
    )

    # draft_hidden: [1, 1, block_size, hidden]
    # draft_logits: [1, 1, block_size - 1, draft_vocab]  (HF reference slices the bonus)
    print(f"draft_hidden shape: {draft_hidden.shape}")
    print(f"draft_logits shape: {draft_logits.shape}")

    assert (
        int(draft_hidden.shape[-1]) == cfg.hidden_size
    ), f"draft_hidden last dim {draft_hidden.shape[-1]} ≠ hidden_size {cfg.hidden_size}"
    assert (
        int(draft_hidden.shape[-2]) == cfg.block_size
    ), f"draft_hidden seq dim {draft_hidden.shape[-2]} ≠ block_size {cfg.block_size}"
    assert (
        int(draft_logits.shape[-2]) == cfg.block_size - 1
    ), f"draft_logits seq dim {draft_logits.shape[-2]} ≠ block_size-1 {cfg.block_size - 1}"
    # draft_vocab dim is column-parallel on the lm_head: each device holds vocab/tp cols.
    tp = mesh_device.shape[1] if is_mesh else 1
    expected_vocab_local = cfg.draft_vocab_size // tp
    assert (
        int(draft_logits.shape[-1]) == expected_vocab_local
    ), f"draft_logits last dim {draft_logits.shape[-1]} ≠ draft_vocab/tp {expected_vocab_local}"

    print("Smoke test passed.")
