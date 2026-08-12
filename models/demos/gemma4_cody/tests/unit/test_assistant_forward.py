# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Gemma4 MTP assistant (drafter) forward pass on TT.

Loads the converted drafter weights, builds synthetic ``target_last_hidden`` +
``shared_kv`` (matching the shapes the target would produce), and runs one
forward through the 4-layer drafter. Verifies it returns tensors of the
expected shape.

Doesn't verify correctness against the HF reference — that needs the target
model running too. This is a "does the TT graph dispatch without shape
errors" smoke test.

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_assistant_forward.py -k 1x8
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4_cody.tt.assistant.config import Gemma4AssistantConfig
from models.demos.gemma4_cody.tt.assistant.model import Gemma4AssistantModel

from ...tests.test_factory import parametrize_mesh_with_fabric

B = 32  # cody's production batch
KV_LEN = 128  # synthetic KV history length


@parametrize_mesh_with_fabric()
def test_assistant_forward_runs(mesh_device):
    """One forward pass with synthetic target hidden + shared_kv. Checks shapes."""
    cfg = Gemma4AssistantConfig.from_hf_path("/mnt/nas/gemma-assistant")

    model = Gemma4AssistantModel(
        mesh_device=mesh_device,
        config=cfg,
        cache_dir=None,  # uses $TT_CACHE_PATH/tensor_cache_assistant_bf16
    )

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    def to_tt(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=layout,
            dtype=dtype,
            mesh_mapper=replicate,
        )

    # Target's last-layer hidden, concat with next-token embedding: width = 2 * backbone_hidden.
    target_hidden = torch.randn(1, 1, B, 2 * cfg.backbone_hidden_size)
    target_hidden_tt = to_tt(target_hidden)

    # Synthetic shared_kv. The target's KV at the deepest sliding / full layer.
    # SDPA decode expects K, V shape: [B, nkv_local, kv_len, head_dim_for_layer_type].
    K_swa = to_tt(torch.randn(B, cfg.num_key_value_heads, KV_LEN, cfg.head_dim))
    V_swa = to_tt(torch.randn(B, cfg.num_key_value_heads, KV_LEN, cfg.head_dim))
    K_full = to_tt(torch.randn(B, cfg.num_global_key_value_heads, KV_LEN, cfg.global_head_dim))
    V_full = to_tt(torch.randn(B, cfg.num_global_key_value_heads, KV_LEN, cfg.global_head_dim))
    shared_kv = {
        "sliding_attention": (K_swa, V_swa),
        "full_attention": (K_full, V_full),
    }

    # Synthetic per-slot RoPE (cos/sin already gathered for each slot's cur_pos).
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

    out_hidden, logits = model.forward(
        target_last_hidden=target_hidden_tt,
        shared_kv=shared_kv,
        cos_pos_full=cos_full,
        sin_pos_full=sin_full,
        cos_pos_sliding=cos_swa,
        sin_pos_sliding=sin_swa,
        cur_pos_tensor=cur_pos,
    )

    # Validate output shapes.
    print(f"out_hidden shape: {out_hidden.shape}")
    print(f"logits shape: {logits.shape}")

    # out_hidden should be [1, 1, B, backbone_hidden].
    # logits should be [1, 1, B, vocab_size].
    assert (
        out_hidden.shape[-1] == cfg.backbone_hidden_size
    ), f"out_hidden width mismatch: {out_hidden.shape} vs expected last dim {cfg.backbone_hidden_size}"
    assert (
        logits.shape[-1] == cfg.vocab_size
    ), f"logits width mismatch: {logits.shape} vs expected last dim {cfg.vocab_size}"
    print("Smoke test passed.")
