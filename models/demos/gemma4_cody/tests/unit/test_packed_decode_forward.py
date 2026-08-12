# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify `packed_decode_forward` (the full packed attention) for one layer.

`test_packed_sliding_sdpa.py` proved the packed-Q SDPA kernel call in
isolation. This test exercises the whole `packed_decode_forward` function —
QKV projection, head split, per-head norms, per-row RoPE, the P
`paged_update_cache` writes, the head-major packed SDPA, unpack, and o_proj —
for one layer, against P sequential single-token HF decodes.

Parametrized over a sliding layer (head_dim 256, 16 KV heads) and a global
layer (head_dim 512, 4 KV heads) so both the GQA layouts are covered.

The ``kv_write`` axis selects how the P new positions land in the paged cache:
  - ``loop``      — the per-position ``paged_update_cache`` fallback (legacy).
  - ``gather``    — the loop-free persistent-staging write, dim-2 ``ttnn.gather``
                    merge (``_packed_fill_kv_loopfree``).
  - ``embedding`` — the loop-free write, transpose-free ``ttnn.embedding``
                    row-gather merge (``_packed_fill_kv_loopfree_embed``).
All three must produce the same PCC vs HF: the write path is an
implementation detail, not an output change. The test builds the staging +
``merge_idx`` / ``hot_pt`` (and, for embedding, the per-head ``embed_idx``)
exactly as the server's ``_refresh_loopfree_write_idx`` does for the clean
case (``cur_pos`` block-aligned ⇒ no committed prefix, no rollover, no spill).

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    # loop-free embedding write, B=32 P=16 (both layer types), 4-device mesh:
    pytest -s models/demos/gemma4_cody/tests/unit/test_packed_decode_forward.py -k "1x4 and P16 and embedding"
"""

import pytest
import torch

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.demos.gemma4_cody.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.attention.decode import packed_decode_forward
from models.demos.gemma4_cody.tt.attention.kv_cache import PV_HOT_BLOCKS, init_kv_cache, init_kv_staging
from models.demos.gemma4_cody.tt.ccl import CCLManager
from models.demos.gemma4_cody.tt.model import create_rope_caches

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_mesh_with_fabric

B_TEST = 32


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("packed_p", [5, 16], ids=lambda v: f"P{v}")
@pytest.mark.parametrize("kv_write", ["loop", "gather", "embedding"], ids=lambda v: v)
@pytest.mark.parametrize("prefill_len", [128], ids=lambda v: f"pre{v}")
def test_packed_decode_forward_matches_hf(layer_idx, packed_p, kv_write, prefill_len, mesh_device):
    """packed_decode_forward (P positions, one pass) ≡ P sequential HF decodes.

    Must pass PCC vs the HF reference.
    """
    impl = packed_decode_forward
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.tt_transformers.tt.common import PagedAttentionConfig

    P = packed_p
    torch.manual_seed(0)

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    H = config.num_attention_heads
    H_local = H // tp
    nkv = config.num_key_value_heads
    head_dim = config.head_dim
    hidden_size = hf_text_config.hidden_size
    layer_type = hf_text_config.layer_types[layer_idx]

    block_size = 64
    blocks_per_user = (prefill_len + P + block_size - 1) // block_size + 1
    max_num_blocks = B_TEST * blocks_per_user
    max_seq_len = blocks_per_user * block_size
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    kv_cache = init_kv_cache(mesh_device, config, paged_attention_config=paged_cfg, cache_dtype=ttnn.bfloat16)
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

    # ── One logical user: P input positions + prefill_len of history K/V.
    x_user = torch.randn(P, hidden_size, dtype=torch.float32)
    k_init = torch.randn(1, nkv, prefill_len, head_dim)
    v_init = k_init.clone() if config.use_kv_tying else torch.randn(1, nkv, prefill_len, head_dim)

    # ── Page table: B_TEST users, blocks_per_user consecutive blocks each.
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B_TEST, blocks_per_user)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )

    # ── Fill the prefill history into the paged KV cache.
    prefill_padded = ((prefill_len + block_size - 1) // block_size) * block_size
    if prefill_padded > prefill_len:
        pad = torch.zeros(1, nkv, prefill_padded - prefill_len, head_dim)
        k_fill = torch.cat([k_init, pad], dim=2)
        v_fill = torch.cat([v_init, pad.clone()], dim=2)
    else:
        k_fill, v_fill = k_init, v_init
    kv_replicated = nkv < tp
    local_kv = 1 if kv_replicated else nkv // tp
    for dev_idx in range(tp):
        if kv_replicated:
            kv_idx = (dev_idx * H_local) * nkv // H
            k_local = k_fill[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
            v_local = v_fill[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
        else:
            ks = dev_idx * local_kv
            k_local = k_fill[:, ks : ks + local_kv].to(torch.bfloat16)
            v_local = v_fill[:, ks : ks + local_kv].to(torch.bfloat16)
        dev_k = ttnn.get_device_tensors(kv_cache[0])[dev_idx] if is_mesh else kv_cache[0]
        dev_v = ttnn.get_device_tensors(kv_cache[1])[dev_idx] if is_mesh else kv_cache[1]
        dev_pt = ttnn.get_device_tensors(page_table_tt)[dev_idx] if is_mesh else page_table_tt
        dev = dev_k.device()
        k_local_tt = ttnn.from_torch(k_local, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        v_local_tt = ttnn.from_torch(v_local, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for user_idx in range(B_TEST):
            ttnn.experimental.paged_fill_cache(dev_k, k_local_tt, dev_pt, batch_idx=user_idx)
            ttnn.experimental.paged_fill_cache(dev_v, v_local_tt, dev_pt, batch_idx=user_idx)

    # ── HF reference: P sequential single-token decodes (user 0), causal.
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    hf_cache = DynamicCache()
    hf_cache.update(k_init.clone(), v_init.clone(), layer_idx=layer_idx)
    ref_outputs = []
    for p in range(P):
        pos = prefill_len + p
        x_p = x_user[p : p + 1].reshape(1, 1, hidden_size)
        cos, sin = rope(x_p, torch.tensor([[pos]]), layer_type=layer_type)
        mask = torch.zeros(1, 1, 1, pos + 1)
        with torch.no_grad():
            ref_out, _ = hf_attn(
                x_p,
                position_embeddings=(cos, sin),
                past_key_values=hf_cache,
                attention_mask=mask,
                shared_kv_states=None,
            )
        ref_outputs.append(ref_out.reshape(hidden_size).float())
    ref_stacked = torch.stack(ref_outputs, dim=0)  # [P, hidden_size]

    # ── TT: build packed inputs and call packed_decode_forward.
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
    cos_cache_2d, sin_cache_2d = rope_caches_2d[layer_type]

    # hidden_states [1, 1, B*P, hidden] — rows user-major position-minor.
    x_packed = x_user.unsqueeze(0).expand(B_TEST, P, hidden_size).reshape(1, 1, B_TEST * P, hidden_size)
    hidden_tt = ttnn.from_torch(
        x_packed.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    # position_idx [1, B*P] uint32 — row u*P+p → position prefill_len+p.
    pos_torch = torch.tensor([prefill_len + p for _ in range(B_TEST) for p in range(P)], dtype=torch.int32)
    position_idx = ttnn.from_torch(
        pos_torch.reshape(1, B_TEST * P),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    # kv_write_idxs — one int32 [B] tensor per packed position p.
    kv_write_idxs = [
        ttnn.from_torch(
            torch.full((B_TEST,), prefill_len + p, dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )
        for p in range(P)
    ]
    # attn_mask [B, 1, H_local*P, S_k] head-major — row h*P+p masks key > prefill_len+p.
    NEG = float(-1e9)
    S_k = max_seq_len
    mask_torch = torch.zeros(B_TEST, 1, H_local * P, S_k, dtype=torch.float32)
    for h in range(H_local):
        for p in range(P):
            mask_torch[:, 0, h * P + p, prefill_len + p + 1 :] = NEG
    attn_mask = ttnn.from_torch(
        mask_torch.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )

    # ── Loop-free persistent-staging KV-write inputs (gather / embedding modes).
    # Mirrors server._refresh_loopfree_write_idx for the clean case: cur_pos is
    # block-aligned (prefill_len % block_size == 0) so block-slot 0 of each slot
    # is a fresh block — no committed prefix to seed, no rollover, and the P-tail
    # fits one block (no spill). merge_idx is identity except the P new tokens;
    # hot_pt names each slot's cur_pos physical page (block-slot 1 stays -1).
    loopfree_kwargs = {}
    if kv_write != "loop":
        BLK = PV_HOT_BLOCKS
        S2 = B_TEST * BLK * block_size
        n_slots_all = B_TEST * BLK
        a_blk = prefill_len // block_size  # cur_pos block index within a user's pages
        off = prefill_len % block_size
        assert off == 0 and off + P <= block_size, "test assumes a fresh, single-block P-tail"

        m = torch.arange(S2, dtype=torch.int32)  # identity ⇒ keep (zero) resident staging
        hp = torch.full((1, n_slots_all), -1, dtype=torch.int32)
        for r in range(B_TEST):
            base = r * BLK * block_size
            for p in range(P):  # the P new tokens at staging block-slot 0, offset [off, off+P)
                m[base + off + p] = S2 + (r * P + p)
            hp[0, r * BLK] = int(page_table[r, a_blk])  # fill cur_pos block to its physical page

        merge_idx_tt = ttnn.from_torch(
            m, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
        )
        hot_pt_tt = ttnn.from_torch(
            hp, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
        )
        k_stg, v_stg = init_kv_staging(
            mesh_device, config, max_batch_size=B_TEST, block_size=block_size, blk=BLK, cache_dtype=ttnn.bfloat16
        )
        nkv_local = k_stg.shape[1]  # per-device local KV heads (matches the cache shard)

        embed_idx_tt = None
        if kv_write == "embedding":
            # Per-head-flattened gather index: row h*S2+j ← concat pos h*src_seq+m[j],
            # over the [nkv_local*(S2+B*P), hd] flattened concat view.
            src_seq = S2 + B_TEST * P
            off_h = (torch.arange(nkv_local, dtype=torch.int32) * src_seq).unsqueeze(1)  # [nkv,1]
            e = (m.unsqueeze(0) + off_h).reshape(1, nkv_local * S2)
            embed_idx_tt = ttnn.from_torch(
                e, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
            )

        # hot_pt / embed_idx are selected by config.is_sliding inside the helper;
        # this test drives one layer (one cache, one page table) so pass the same
        # tensor for both the full and sliding slots.
        loopfree_kwargs = dict(
            kv_staging=[k_stg, v_stg],
            merge_idx=merge_idx_tt,
            hot_pt=hot_pt_tt,
            hot_pt_sliding=hot_pt_tt,
            kv_merge=("embedding" if kv_write == "embedding" else "gather"),
            embed_idx=embed_idx_tt,
            embed_idx_sliding=embed_idx_tt,
        )

    tt_out = impl(
        hidden_states=hidden_tt,
        cos_cache=cos_cache_2d,
        sin_cache=sin_cache_2d,
        weights=weights,
        kv_cache=kv_cache,
        config=config,
        mesh_config=mesh_config,
        mesh_device=mesh_device,
        position_idx=position_idx,
        kv_write_idxs=kv_write_idxs,
        attn_mask=attn_mask,
        packed_p=P,
        page_table=page_table_tt,
        ccl_manager=ccl_manager,
        **loopfree_kwargs,
    )

    # tt_out [1, 1, B*P, hidden] — extract user 0's P rows (rows 0..P-1).
    out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]) if is_mesh else ttnn.to_torch(tt_out)
    out_torch = out_torch.float().reshape(B_TEST * P, hidden_size)[:P, :]  # [P, hidden]

    passing, pcc = compare_tensors(out_torch, ref_stacked, pcc_threshold=0.95)
    assert passing, (
        f"packed_decode_forward mismatch (layer={layer_idx}, P={P}, kv_write={kv_write}, "
        f"prefill_len={prefill_len}, tp={tp}): PCC vs HF too low: {pcc}"
    )
