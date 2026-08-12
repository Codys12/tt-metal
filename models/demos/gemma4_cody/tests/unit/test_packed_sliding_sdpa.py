# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prove packed-Q SDPA decode works for a SLIDING-window attention layer.

`test_packed_decode.py` proves the packed-Q trick for a *global* layer. The
speculative-decode packed verify (Item 1) must also run the target's sliding
layers (50 of 60). This test isolates that:

  - Pick a sliding layer.
  - Force the sliding window to actually CLIP: window W_TEST < prefill_len, so
    each query attends only a strict sub-range of the cache (not all of it) —
    otherwise the test degenerates to the global case.
  - HF reference: T sequential single-token decodes, each with an explicit
    `attention_mask` = the per-position sliding window. HF's
    `eager_attention_forward` ignores `sliding_window` and applies ONLY the
    passed mask, so this is an exact reference.
  - TT: stack T per-token Q along the heads dim, ONE
    `paged_scaled_dot_product_attention_decode` with `is_causal=False`,
    `sliding_window_size=None`, and a per-(token,head) `attn_mask` that bakes
    in BOTH the causal upper bound AND the sliding-window lower bound.

This deliberately uses the SAME SDPA call shape as the proven global test —
only the mask content differs (extra masked entries below the window). If it
passes, packed sliding SDPA needs no new kernel behaviour: a fully-explicit
mask + `sliding_window_size=None` covers both layer types uniformly.

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_packed_sliding_sdpa.py -k 1x8
"""

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_mesh_with_fabric

B_TEST = 32  # production decode batch padding


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0], ids=["sliding"])
@pytest.mark.parametrize("packed_tokens", [4], ids=lambda v: f"T{v}")
@pytest.mark.parametrize("prefill_len", [128], ids=lambda v: f"pre{v}")
@pytest.mark.parametrize("window", [96], ids=lambda v: f"W{v}")
def test_packed_sliding_decode_matches_hf(layer_idx, packed_tokens, prefill_len, window, mesh_device):
    """T tokens packed in heads dim of one sliding-window SDPA ≡ T HF decodes."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.demos.gemma4.tt.attention.operations import (
        apply_per_head_norm,
        apply_qkv_projection,
        apply_rope,
        split_qkv_heads_decode,
    )
    from models.demos.gemma4.tt.model import create_rope_caches
    from models.tt_transformers.tt.common import PagedAttentionConfig

    T = packed_tokens
    W = window
    torch.manual_seed(0)

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    assert config.is_sliding, "This test targets a SLIDING layer"
    # Force the window to clip: every query at pos>=W must drop the oldest keys.
    assert W < prefill_len, "window must be smaller than prefill_len so it clips"

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    # ── Mesh / layout
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    H = config.num_attention_heads
    H_local = H // tp
    nkv = config.num_key_value_heads
    head_dim = config.head_dim
    hidden_size = hf_text_config.hidden_size
    layer_type = hf_text_config.layer_types[layer_idx]

    k_chunk_size = 64
    block_size = 64
    blocks_per_user = (prefill_len + T + block_size - 1) // block_size + 1
    max_num_blocks = B_TEST * blocks_per_user
    max_seq_len = blocks_per_user * block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    kv_cache = init_kv_cache(
        mesh_device, config, paged_attention_config=paged_attention_config, cache_dtype=ttnn.bfloat16
    )
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
    tt_attn.kv_cache = kv_cache

    # ── One logical user's random data; replicated across B_TEST for TT.
    x_torch_single = torch.randn(1, T, hidden_size, dtype=torch.float32)
    k_init_single = torch.randn(1, nkv, prefill_len, head_dim)
    v_init_single = k_init_single.clone() if config.use_kv_tying else torch.randn(1, nkv, prefill_len, head_dim)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B_TEST, blocks_per_user)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )

    # ── Initial KV cache fill (prefill_len positions, same for every user).
    prefill_padded = ((prefill_len + block_size - 1) // block_size) * block_size
    if prefill_padded > prefill_len:
        pad = torch.zeros(1, nkv, prefill_padded - prefill_len, head_dim)
        k_fill_single = torch.cat([k_init_single, pad], dim=2)
        v_fill_single = torch.cat([v_init_single, pad.clone()], dim=2)
    else:
        k_fill_single, v_fill_single = k_init_single, v_init_single

    kv_replicated = nkv < tp
    local_kv = 1 if kv_replicated else nkv // tp
    for dev_idx in range(tp):
        if kv_replicated:
            kv_idx = (dev_idx * H_local) * nkv // H
            k_local = k_fill_single[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
            v_local = v_fill_single[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
        else:
            kv_start = dev_idx * local_kv
            k_local = k_fill_single[:, kv_start : kv_start + local_kv].to(torch.bfloat16)
            v_local = v_fill_single[:, kv_start : kv_start + local_kv].to(torch.bfloat16)
        dev_k = ttnn.get_device_tensors(kv_cache[0])[dev_idx] if is_mesh else kv_cache[0]
        dev_v = ttnn.get_device_tensors(kv_cache[1])[dev_idx] if is_mesh else kv_cache[1]
        dev_pt = ttnn.get_device_tensors(page_table_tt)[dev_idx] if is_mesh else page_table_tt
        dev = dev_k.device()
        k_local_tt = ttnn.from_torch(k_local, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        v_local_tt = ttnn.from_torch(v_local, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for user_idx in range(B_TEST):
            ttnn.experimental.paged_fill_cache(dev_k, k_local_tt, dev_pt, batch_idx=user_idx)
            ttnn.experimental.paged_fill_cache(dev_v, v_local_tt, dev_pt, batch_idx=user_idx)

    # ── HF reference: T sequential single-token decodes (user 0). Each gets an
    # explicit sliding-window mask: query at pos attends keys (pos-W, pos].
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    hf_cache = DynamicCache()
    hf_cache.update(k_init_single.clone(), v_init_single.clone(), layer_idx=layer_idx)
    NEG = float(-1e9)

    ref_outputs = []
    for t in range(T):
        pos = prefill_len + t
        x_t = x_torch_single[:, t : t + 1, :]
        cos, sin = rope(x_t, torch.tensor([[pos]]), layer_type=layer_type)
        lo = max(0, pos - W + 1)  # oldest key still inside the window
        mask = torch.zeros(1, 1, 1, pos + 1)
        mask[:, :, :, :lo] = NEG  # keys before the window are masked out
        with torch.no_grad():
            ref_out, _ = hf_attn(
                x_t,
                position_embeddings=(cos, sin),
                past_key_values=hf_cache,
                attention_mask=mask,
                shared_kv_states=None,
            )
        ref_outputs.append(ref_out.squeeze(0).squeeze(0).float())
    ref_stacked = torch.stack(ref_outputs, dim=0)  # [T, hidden_size]

    # ── TT packed path: T per-token decode-preps, then ONE packed SDPA.
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
    cos_cache_2d, sin_cache_2d = rope_caches_2d[layer_type]
    weights = tt_attn.weights

    per_token_q_dram = []
    for t in range(T):
        pos = prefill_len + t
        x_t_single = x_torch_single[:, t : t + 1, :]
        x_t_batched = x_t_single.expand(1, B_TEST, hidden_size).contiguous()
        x_tt = ttnn.from_torch(
            x_t_batched.unsqueeze(0).to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        xqkv = apply_qkv_projection(x_tt, weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_q, tt_k, tt_v = split_qkv_heads_decode(
            xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        q_sharded_mem = tt_q.memory_config()
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

        position_idx_uint = ttnn.from_torch(
            torch.full((1, B_TEST), pos, dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx_uint, cos_cache_2d, layout=ttnn.TILE_LAYOUT))
        sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx_uint, sin_cache_2d, layout=ttnn.TILE_LAYOUT))
        batch = tt_q.shape[1]
        if cos_pos.shape[2] != batch:
            cos_pos = cos_pos[:, :, :batch, :]
            sin_pos = sin_pos[:, :, :batch, :]
        tt_q = apply_rope(tt_q, cos_pos, sin_pos, token_index=0)
        tt_k = apply_rope(tt_k, cos_pos, sin_pos, token_index=0)

        tt_k_cache_write = ttnn.to_memory_config(tt_k, q_sharded_mem)
        tt_v_cache_write = ttnn.to_memory_config(tt_v, q_sharded_mem)
        position_idx_cache_attn = ttnn.from_torch(
            torch.full((B_TEST,), pos, dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )
        ttnn.experimental.paged_update_cache(
            kv_cache[0], tt_k_cache_write, update_idxs_tensor=position_idx_cache_attn, page_table=page_table_tt
        )
        ttnn.experimental.paged_update_cache(
            kv_cache[1], tt_v_cache_write, update_idxs_tensor=position_idx_cache_attn, page_table=page_table_tt
        )
        per_token_q_dram.append(tt_q)

    # Pack Q HEAD-MAJOR along the heads dim → [1, B, H_local*T, head_dim],
    # row index = h*T + t. Token-major packing (t*H_local+h, as in the proven
    # *global* test) breaks GQA on sliding layers: the SDPA kernel maps packed
    # query head i → KV head i//group_size, so a KV group must be a CONTIGUOUS
    # block of packed heads. Token-major interleaves heads across tokens, which
    # is fine only when there is 1 KV head/device (global, kv_replicated).
    # Sliding has 2 KV heads/device — head-major keeps each KV group contiguous.
    q_tok_major = ttnn.concat(per_token_q_dram, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q5 = ttnn.reshape(q_tok_major, (1, B_TEST, T, H_local, head_dim))
    q5 = ttnn.permute(q5, (0, 1, 3, 2, 4))  # [1, B, H_local, T, head_dim]
    tt_q_packed = ttnn.reshape(q5, (1, B_TEST, H_local * T, head_dim))

    # Per-(head, token) mask [B, 1, H_local*T, S_k]. Row h*T+t (head h, token t)
    # is masked for key > prefill_len+t (causal) AND key <= prefill_len+t-W
    # (sliding window).
    S_k = max_seq_len
    mask_torch = torch.zeros(B_TEST, 1, H_local * T, S_k, dtype=torch.float32)
    for h in range(H_local):
        for t in range(T):
            pos = prefill_len + t
            lo = max(0, pos - W + 1)
            row = h * T + t
            mask_torch[:, 0, row, pos + 1 :] = NEG  # causal upper bound
            mask_torch[:, 0, row, :lo] = NEG  # sliding-window lower bound
    mask_tt = ttnn.from_torch(
        mask_torch.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    cur_pos_tt = ttnn.from_torch(
        torch.full((B_TEST,), prefill_len + T - 1, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4) if head_dim >= 512 else ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )
    # sliding_window_size=None: the explicit mask carries the whole window.
    tt_packed_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q_packed,
        kv_cache[0],
        kv_cache[1],
        page_table_tensor=page_table_tt,
        is_causal=False,
        attn_mask=mask_tt,
        cur_pos_tensor=cur_pos_tt,
        scale=1.0,
        sliding_window_size=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=sdpa_program_config,
    )

    # ── Reassemble user 0 across devices, apply o_proj in torch.
    # SDPA output matches the head-major packed Q layout: [1, B, H_local*T, hd],
    # row h*T+t. Device d holds heads [d*H_local : (d+1)*H_local].
    if is_mesh:
        per_dev_sdpa = [ttnn.to_torch(ttnn.get_device_tensors(tt_packed_sdpa)[d]).float() for d in range(tp)]
    else:
        per_dev_sdpa = [ttnn.to_torch(tt_packed_sdpa).float()]
    full_sdpa = torch.zeros(T, H, head_dim, dtype=torch.float32)
    for d in range(tp):
        dev_out = per_dev_sdpa[d][0, 0, : H_local * T, :].reshape(H_local, T, head_dim)
        for h in range(H_local):
            full_sdpa[:, d * H_local + h, :] = dev_out[h, :, :]
    sdpa_flat = full_sdpa.reshape(T, H * head_dim)
    o_proj_weight = state_dict["o_proj.weight"].float()
    tt_packed_outputs = torch.matmul(sdpa_flat, o_proj_weight.T)  # [T, hidden_size]

    passing, pcc = compare_tensors(tt_packed_outputs, ref_stacked, pcc_threshold=0.95)
    assert passing, (
        f"Packed sliding-decode SDPA mismatch (layer={layer_idx}, T={T}, "
        f"prefill_len={prefill_len}, W={W}, tp={tp}): PCC vs HF too low: {pcc}"
    )
