# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify that packing T tokens into the heads dim of one non-causal
``paged_scaled_dot_product_attention_decode`` call produces the same result as
T sequential single-token decodes against HuggingFace Gemma4.

Mechanism under test:
  - Paged KV cache, mirroring the production decode setup.
  - Decode batch = 32 (matches the production tile-padded batch); all 32 batch
    slots carry copies of one logical user, and only user 0 is checked against HF.
  - Pre-write KV cache entries for T positions ``prefill_len .. prefill_len+T-1``
    via the production ``tt_attn(...)`` decode wrapper (same code path as the demo).
  - Stack T per-token Q tensors along the heads dim (heads-major) so the
    SDPA decode call sees PNH = T * H_local.
  - Provide a per-(token, head) attention mask ``[B_test, 1, T*H_local, S_k]``
    where row ``t*H_local + h`` masks positions ``> prefill_len + t`` to -inf.
  - SDPA decode with ``is_causal=False`` consumes the mask; the kernel treats
    each row of dim 2 as an independent attention computation.

Targets the global-attention layer (no sliding window) on Gemma4.

Run:
    pytest models/demos/gemma4/tests/unit/test_packed_decode.py -k 1x4
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
@pytest.mark.parametrize("layer_idx", [5], ids=["global"])
@pytest.mark.parametrize("packed_tokens", [4], ids=lambda v: f"T{v}")
@pytest.mark.parametrize("prefill_len", [128], ids=lambda v: f"pre{v}")
def test_packed_decode_matches_hf(layer_idx, packed_tokens, prefill_len, mesh_device):
    """T tokens packed in heads dim of one SDPA call ≡ T sequential HF decodes."""
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
    torch.manual_seed(0)

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    assert not config.is_sliding, "Packed decode test only targets non-sliding (global) layers"

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
    max_seq_len = blocks_per_user * block_size  # per-user, multiple of k_chunk_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    kv_cache = init_kv_cache(
        mesh_device,
        config,
        paged_attention_config=paged_attention_config,
        cache_dtype=ttnn.bfloat16,
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

    # ── One logical user's random data; replicated across B_TEST batch slots for TT.
    x_torch_single = torch.randn(1, T, hidden_size, dtype=torch.float32)  # used by HF
    k_init_single = torch.randn(1, nkv, prefill_len, head_dim)
    v_init_single = k_init_single.clone() if config.use_kv_tying else torch.randn(1, nkv, prefill_len, head_dim)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # ── Page table: each of B_TEST users owns `blocks_per_user` consecutive global blocks.
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B_TEST, blocks_per_user)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    # ── Initial KV cache fill — same data for every user, per-user paged_fill_cache call.
    prefill_padded = ((prefill_len + block_size - 1) // block_size) * block_size
    if prefill_padded > prefill_len:
        pad = torch.zeros(1, nkv, prefill_padded - prefill_len, head_dim)
        k_fill_single = torch.cat([k_init_single, pad], dim=2)
        v_fill_single = torch.cat([v_init_single, pad.clone()], dim=2)
    else:
        k_fill_single = k_init_single
        v_fill_single = v_init_single

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

    # ── HF reference: T sequential single-token decodes through hf_attn (user 0 only).
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    hf_cache = DynamicCache()
    hf_cache.update(k_init_single.clone(), v_init_single.clone(), layer_idx=layer_idx)

    ref_outputs = []
    for t in range(T):
        pos = prefill_len + t
        x_t = x_torch_single[:, t : t + 1, :]
        cos, sin = rope(x_t, torch.tensor([[pos]]), layer_type=layer_type)
        # Global layer: no sliding window → attend to everything ≤ pos.
        mask = torch.zeros(1, 1, 1, pos + 1)
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

    # ── TT packed path. Use the 2D rope cache (production decode RoPE path).
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
    cos_cache_2d, sin_cache_2d = rope_caches_2d[layer_type]
    weights = tt_attn.weights

    per_token_q_dram = []
    for t in range(T):
        pos = prefill_len + t

        # Replicate user 0's row across all B_TEST batch slots.
        x_t_single = x_torch_single[:, t : t + 1, :]  # [1, 1, hidden_size]
        x_t_batched = x_t_single.expand(1, B_TEST, hidden_size).contiguous()  # [1, B_TEST, hidden_size]
        x_tt = ttnn.from_torch(
            x_t_batched.unsqueeze(0).to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

        # Inline the production decode_forward pipeline up to (but not including) SDPA:
        # apply_qkv_projection → split → q_sharded_mem capture → DRAM norm/RoPE → cast back → paged_update_cache.
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

        # Position-aware RoPE via 2D-cache embedding lookup. uint32 [1, B_TEST].
        position_idx_uint = ttnn.from_torch(
            torch.full((1, B_TEST), pos, dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        cos_pos = ttnn.embedding(position_idx_uint, cos_cache_2d, layout=ttnn.TILE_LAYOUT)
        sin_pos = ttnn.embedding(position_idx_uint, sin_cache_2d, layout=ttnn.TILE_LAYOUT)
        cos_pos = ttnn.unsqueeze_to_4D(cos_pos)
        sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
        batch = tt_q.shape[1]
        if cos_pos.shape[2] != batch:
            cos_pos = cos_pos[:, :, :batch, :]
            sin_pos = sin_pos[:, :, :batch, :]
        tt_q = apply_rope(tt_q, cos_pos, sin_pos, token_index=0)
        tt_k = apply_rope(tt_k, cos_pos, sin_pos, token_index=0)

        # Cast K, V back to Q's HEIGHT_SHARDED layout, then write the paged cache.
        # At B=32 this round-trip fits q_sharded_mem's 32-core grid.
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

    # Pack Q along heads dim: [1, B_TEST, T*H_local, head_dim].
    tt_q_packed = ttnn.concat(per_token_q_dram, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Per-(token, head) mask: [B_TEST, 1, T*H_local, S_k]. Same causal pattern for every user.
    S_k = max_seq_len
    NEG = float(-1e9)
    mask_torch = torch.zeros(B_TEST, 1, T * H_local, S_k, dtype=torch.float32)
    for t in range(T):
        for h in range(H_local):
            mask_torch[:, 0, t * H_local + h, prefill_len + t + 1 :] = NEG
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

    # ── Reassemble across-device SDPA output for batch row 0, apply o_proj in torch.
    if is_mesh:
        per_dev_sdpa = [ttnn.to_torch(ttnn.get_device_tensors(tt_packed_sdpa)[d]).float() for d in range(tp)]
    else:
        per_dev_sdpa = [ttnn.to_torch(tt_packed_sdpa).float()]

    full_sdpa = torch.zeros(T, H, head_dim, dtype=torch.float32)
    for d in range(tp):
        dev_out = per_dev_sdpa[d]  # [1, B_TEST, padded(T*H_local), head_dim]
        # Take user 0, then the T*H_local real rows (no padding when TP*T*H_local % 32 == 0).
        dev_out = dev_out[0, 0, : T * H_local, :].reshape(T, H_local, head_dim)
        full_sdpa[:, d * H_local : (d + 1) * H_local, :] = dev_out

    sdpa_flat = full_sdpa.reshape(T, H * head_dim)
    o_proj_weight = state_dict["o_proj.weight"].float()  # [hidden_size, H*head_dim]
    tt_packed_outputs = torch.matmul(sdpa_flat, o_proj_weight.T)  # [T, hidden_size]

    passing, pcc = compare_tensors(tt_packed_outputs, ref_stacked, pcc_threshold=0.95)
    assert passing, (
        f"Packed-decode SDPA mismatch (layer={layer_idx}, T={T}, prefill_len={prefill_len}, tp={tp}): "
        f"PCC vs HF reference too low: {pcc}"
    )
