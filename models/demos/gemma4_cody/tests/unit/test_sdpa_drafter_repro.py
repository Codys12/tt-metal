# SPDX-License-Identifier: Apache-2.0
"""Focused repro: scaled_dot_product_attention_decode at the drafter's shapes.

The 31b drafter (32 attention heads) fails parity; the 2816 drafter (16 heads)
passes. q into the SDPA is verified correct (PCC 0.9998) yet the SDPA output is
garbage (PCC 0.006). This isolates the SDPA op alone, parametrized over head
count, to confirm whether the decode SDPA mis-handles 32 heads.

Run:
    pytest -s models/demos/gemma4_cody/tests/unit/test_sdpa_drafter_repro.py -k 1x8
"""

import pytest
import torch

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    d = (a.norm() * b.norm()).item()
    return 0.0 if d < 1e-12 else (a @ b).item() / d


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("nh", [16, 32], ids=["nh16", "nh32"])
def test_sdpa_decode_head_count(mesh_device, nh):
    torch.manual_seed(0)
    B, KV_LEN, hd = 32, 64, 256
    nkv = nh // 2  # GQA ratio 2, as in both drafters

    q = torch.randn(1, B, nh, hd)
    K = torch.randn(B, nkv, KV_LEN, hd)
    V = torch.randn(B, nkv, KV_LEN, hd)
    cur = torch.full((B,), KV_LEN - 1, dtype=torch.int32)

    # Torch reference for slot 0 (scale=1.0, attend to all KV).
    g = nh // nkv
    Kf = K[0].repeat_interleave(g, dim=0)  # [nh, KV_LEN, hd]
    Vf = V[0].repeat_interleave(g, dim=0)
    scores = q[0, 0].unsqueeze(1) @ Kf.transpose(-1, -2)  # [nh, 1, KV_LEN]
    ref = (scores.softmax(dim=-1) @ Vf).squeeze(1)  # [nh, hd]

    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    def tt(t, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.to(torch.bfloat16), device=mesh_device, layout=layout, dtype=ttnn.bfloat16, mesh_mapper=repl
        )

    qd, Kd, Vd = tt(q), tt(K), tt(V)
    curd = ttnn.from_torch(cur, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=repl)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        qd,
        Kd,
        Vd,
        cur_pos_tensor=curd,
        scale=1.0,
        program_config=pc,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_t = ttnn.to_torch(ttnn.get_device_tensors(out)[0])  # [1, nh, B, hd]
    print(f"\n  nh={nh}: SDPA out shape={tuple(out_t.shape)}")
    tt_slot0 = out_t[0, :, 0, :].float()  # [nh, hd]
    pcc = _pcc(ref, tt_slot0)
    print(f"  nh={nh}: PCC(torch vs TT decode-SDPA) = {pcc:.6f}")
    assert pcc > 0.99, f"nh={nh}: SDPA decode PCC {pcc} too low"
