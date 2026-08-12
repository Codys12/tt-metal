# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify the full-model packed decode path (the speculative-decode verify).

`packed_decode_forward` is proven per-layer in `test_packed_decode_forward.py`.
This test exercises the *threaded* packed path — `model(..., packed=...)` runs
all layers' attention through `packed_decode_forward` — for a 6-layer model
(layers 0-4 sliding, layer 5 global), comparing one packed forward over P
positions against P sequential single-token decodes.

prefill_len is 0: the P packed positions are 0..P-1, decoded from empty caches,
so no prefill machinery is needed. Writes are idempotent (same model, same
tokens) so the packed and sequential runs share the caches.

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_packed_model_decode.py -k 1x8
"""

import pytest
import torch

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.demos.gemma4_cody.tt.ccl import CCLManager
from models.demos.gemma4_cody.tt.model import Gemma4Model
from models.demos.gemma4_cody.tt.model_config import Gemma4ModelArgs

from ...tests.test_factory import compare_tensors, parametrize_mesh_with_fabric
from .test_model import _create_hf_model, _create_hf_text_config, _hf_model_state_to_tt_state

B_TEST = 32
NUM_LAYERS = 6  # layers 0-4 sliding, layer 5 global


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("packed_p", [5], ids=lambda v: f"P{v}")
def test_packed_model_decode_matches_sequential(packed_p, mesh_device):
    """model(packed=...) over P positions ≡ P sequential ttnn decode steps."""
    from models.tt_transformers.tt.common import PagedAttentionConfig

    P = packed_p
    torch.manual_seed(0)

    hf_text_config = _create_hf_text_config(vocab_size=256, num_layers=NUM_LAYERS)
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    tt_state = _hf_model_state_to_tt_state(hf_model)

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    block_size = 64
    blocks_per_user = (P + block_size - 1) // block_size + 1
    max_num_blocks = B_TEST * blocks_per_user
    max_seq_len = blocks_per_user * block_size
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    tt_model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=B_TEST,
        num_layers=NUM_LAYERS,
        paged_attention_config=paged_cfg,
        paged_attention_config_sliding=paged_cfg,
        create_kv_cache=True,
    )
    hidden_size = model_args.hidden_size
    H_local = model_args.num_attention_heads // tp

    # P input tokens, one logical user replicated across B_TEST.
    tok_user = torch.randint(0, model_args.vocab_size, (P,), dtype=torch.int32)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B_TEST, blocks_per_user)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )

    def _u32(t):
        return ttnn.from_torch(
            t, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
        )

    def _i32(t):
        return ttnn.from_torch(
            t, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
        )

    def _embed(tokens_1d):
        """tokens_1d: torch [N] → TT [1, 1, N, hidden] TILE."""
        N = tokens_1d.shape[0]
        tt_tok = _u32(tokens_1d.reshape(1, N).to(torch.int32))
        emb = tt_model.embed_tokens(tt_tok)
        emb = ttnn.reshape(emb, (1, 1, N, hidden_size))
        return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

    # ── PACKED run: one forward over B*P positions. ──────────────────────
    tok_packed = tok_user.unsqueeze(0).expand(B_TEST, P).reshape(B_TEST * P)  # u-major p-minor
    packed_embeds = _embed(tok_packed)

    position_idx_packed = _u32(
        torch.tensor([p for _ in range(B_TEST) for p in range(P)], dtype=torch.int32).reshape(1, B_TEST * P)
    )
    kv_write_idxs = [_i32(torch.full((B_TEST,), p, dtype=torch.int32)) for p in range(P)]
    # Head-major causal mask [B,1,H_local*P,S_k] — row h*P+p masks key > p.
    NEG = float(-1e9)
    S_k = max_seq_len
    mask_torch = torch.zeros(B_TEST, 1, H_local * P, S_k, dtype=torch.float32)
    for h in range(H_local):
        for p in range(P):
            mask_torch[:, 0, h * P + p, p + 1 :] = NEG
    attn_mask = ttnn.from_torch(
        mask_torch.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    packed_spec = {
        "p": P,
        "position_idx": position_idx_packed,
        "kv_write_idxs": kv_write_idxs,
        "attn_mask": {"full_attention": attn_mask, "sliding_attention": attn_mask},
    }
    packed_logits = tt_model(
        packed_embeds,
        rope_mats=None,
        position_idx=None,
        page_table=page_table_tt,
        kv_caches=None,
        is_decode=True,
        page_table_sliding=page_table_tt,
        packed=packed_spec,
    )

    def _read_logits(tt):
        """Read logits to torch [rows, vocab]. The decode path leaves the
        lm_head output vocab-sharded across TP (the all-gather is skipped) —
        concat the per-device shards along the vocab dim."""
        if not is_mesh:
            return ttnn.to_torch(tt).float()
        shards = [ttnn.to_torch(ttnn.get_device_tensors(tt)[d]).float() for d in range(tp)]
        if shards[0].shape[-1] == model_args.vocab_size:
            return shards[0]  # already gathered / replicated
        return torch.cat(shards, dim=-1)  # vocab-sharded

    pl = _read_logits(packed_logits).reshape(B_TEST * P, model_args.vocab_size)[:P, :]

    # ── HF reference: one causal forward over the P tokens. With causal
    # attention this is exactly P sequential single-token decodes — position p
    # attends 0..p. (P < sliding_window so sliding ≡ causal here.) Inlined
    # rather than calling HFRefModel.forward so we can pass a `shared_kv_states`
    # dict (the store_full_length_kv layers require it). Avoids the TT
    # decode_forward fused-CCL path, isolating the packed forward.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    causal_mask = torch.triu(torch.full((1, 1, P, P), float("-inf")), diagonal=1)
    with torch.no_grad():
        x = hf_model.embed_tokens(tok_user.reshape(1, P).long())
        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        pos_ids = torch.arange(P).unsqueeze(0)
        x_dummy = torch.randn(1, P, hf_text_config.hidden_size)
        rope_cache = {lt: rope(x_dummy, pos_ids, layer_type=lt) for lt in set(hf_text_config.layer_types[:NUM_LAYERS])}
        shared_kv = {}
        for i, layer in enumerate(hf_model.layers):
            lt = hf_text_config.layer_types[i]
            x = layer(
                x,
                per_layer_input=None,
                shared_kv_states=shared_kv,
                position_embeddings=rope_cache[lt],
                attention_mask=causal_mask,
            )
        x = hf_model.norm(x)
        hf_logits = hf_model.lm_head(x)
        cap = hf_text_config.final_logit_softcapping
        if cap and cap > 0:
            hf_logits = torch.tanh(hf_logits / cap) * cap
    hf_stacked = hf_logits[0].float()  # [P, vocab]

    passing, pcc = compare_tensors(pl, hf_stacked, pcc_threshold=0.90)
    assert passing, f"packed model decode mismatch (P={P}, tp={tp}): PCC vs HF too low: {pcc}"
