# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 Attention — uses HF Gemma4TextAttention as reference.

Tests prefill and decode for both sliding and global layers, across all TP factors.

    pytest -k "1x1"              # single card
    pytest -k "1x8"              # T3K
    pytest -k "sliding"          # sliding attention only
    pytest -k "global"           # global attention only
    pytest -k "prefill"          # prefill only
    pytest -k "decode"           # decode only
"""

import pytest
import torch

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.demos.gemma4_cody.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.ccl import CCLManager

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_mesh_with_fabric


def _skip_if_l1_overflow(config, mesh_device):
    """Skip if global attention head_dim overflows L1 on this mesh config."""
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    # Global layers with large head_dim (512) overflow L1 when hidden_size > 4096 on single device
    if not config.is_sliding and config.head_dim >= 512 and tp == 1:
        hf_config = TestFactory.create_hf_config()
        if hf_config.hidden_size > 4096:
            pytest.skip("Global attention head_dim=512 overflows L1 on single device for large models")


def _setup_attention(mesh_device, layer_idx, create_kv_cache=False, max_seq_len=128):
    """Create HF reference and TT attention module for a given mesh."""
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=create_kv_cache,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )

    return hf_text_config, hf_attn, config, tt_attn, mesh_config


def _to_device(tensor, mesh_device):
    """Send tensor to mesh device with appropriate mapper."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _from_device(tensor, mesh_device):
    """Read tensor back from device 0."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    if is_mesh:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


# ── Prefill PCC Test ──────────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", [32], ids=["seq32"])
def test_attention_prefill(layer_idx, seq_len, mesh_device):
    """Test prefill attention against HF reference with PCC >= 0.95."""
    hf_text_config, hf_attn, config, tt_attn, mesh_config = _setup_attention(mesh_device, layer_idx)
    _skip_if_l1_overflow(config, mesh_device)

    x_torch = torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32)

    # HF reference
    hf_rope = TestFactory.create_hf_rope(hf_text_config, seq_len, layer_idx)
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=hf_rope, attention_mask=causal_mask, shared_kv_states=None)

    # TT forward
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)
    x_tt = _to_device(x_torch.unsqueeze(0).to(torch.bfloat16), mesh_device)
    tt_output = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False)
    tt_output_torch = _from_device(tt_output, mesh_device).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention prefill (layer={layer_idx}, seq={seq_len}, tp={mesh_config.tp}) PCC too low: {pcc_msg}"


# ── RoPE PCC Test at high positions ──────────────────────────────────────


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("position", [0, 32, 512, 1023, 1024, 1500, 2047], ids=lambda p: f"pos{p}")
def test_rope_pcc(layer_idx, position, mesh_device):
    """Test RoPE cos/sin values and apply_rope PCC at various decode positions up to 2k.

    Compares TT rotary embedding output against HF reference at each position.
    This catches bfloat16 precision loss or indexing bugs at high positions.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_cody.tt.attention.operations import apply_rope

    hf_text_config = TestFactory.create_hf_text_config()
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    head_dim = config.head_dim
    max_seq_len = 2048

    # HF reference rope at the target position
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    x_dummy = torch.randn(1, 1, hf_text_config.hidden_size)
    cos_ref, sin_ref = rope(x_dummy, torch.tensor([[position]]), layer_type=layer_type)
    # cos_ref, sin_ref: [1, 1, head_dim]

    # Random Q-like tensor [1, num_heads, 1, head_dim]
    num_heads = config.num_attention_heads
    q_torch = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float32)

    # HF-style manual RoPE application (reference)
    cos_expanded = cos_ref.unsqueeze(0).expand(1, num_heads, 1, head_dim)  # [1, heads, 1, head_dim]
    sin_expanded = sin_ref.unsqueeze(0).expand(1, num_heads, 1, head_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    ref_rotated = (q_torch * cos_expanded) + (rotate_half(q_torch) * sin_expanded)

    # TT RoPE — build 4D cache [1, 1, max_seq_len, head_dim] and index at position
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)
    q_tt = ttnn.from_torch(q_torch.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_rotated = apply_rope(q_tt, cos_tt, sin_tt, token_index=position)
    tt_rotated_torch = ttnn.to_torch(tt_rotated).float()

    # Also verify the cos/sin cache values themselves at this position
    cos_cache_torch = ttnn.to_torch(cos_tt).float()  # [1, 1, max_seq_len, head_dim]
    sin_cache_torch = ttnn.to_torch(sin_tt).float()
    cos_at_pos = cos_cache_torch[0, 0, position, :]
    sin_at_pos = sin_cache_torch[0, 0, position, :]
    cos_ref_flat = cos_ref[0, 0, :]
    sin_ref_flat = sin_ref[0, 0, :]

    # Check cos/sin cache PCC at this position
    cos_pcc = torch.corrcoef(torch.stack([cos_at_pos, cos_ref_flat]))[0, 1].item()
    sin_pcc = torch.corrcoef(torch.stack([sin_at_pos, sin_ref_flat]))[0, 1].item()

    # Check rotated output PCC
    passing, pcc_msg = compare_tensors(tt_rotated_torch, ref_rotated, pcc_threshold=0.98)
    assert passing, (
        f"RoPE output PCC too low at position {position} (layer={layer_idx}): {pcc_msg}\n"
        f"  cos PCC: {cos_pcc:.6f}, sin PCC: {sin_pcc:.6f}"
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("position", [32, 1500], ids=lambda p: f"pos{p}")
def test_rope_cache_builder_matches_hf(layer_idx, position, mesh_device):
    """Compare Gemma4Model's local RoPE cache builder against HF cos/sin."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_cody.tt.model import _create_rope_cache_tensors

    hf_text_config = TestFactory.create_hf_text_config()
    layer_type = hf_text_config.layer_types[layer_idx]
    max_seq_len = 2048

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    x_dummy = torch.randn(1, 1, hf_text_config.hidden_size)
    cos_ref, sin_ref = rope(x_dummy, torch.tensor([[position]]), layer_type=layer_type)

    cos, sin = _create_rope_cache_tensors(hf_text_config, max_seq_len, layer_type)
    cos_at_pos = cos[0, position, :]
    sin_at_pos = sin[0, position, :]

    passing_cos, cos_msg = compare_tensors(cos_at_pos, cos_ref[0, 0, :], pcc_threshold=0.999)
    passing_sin, sin_msg = compare_tensors(sin_at_pos, sin_ref[0, 0, :], pcc_threshold=0.999)
    assert passing_cos and passing_sin, (
        f"Gemma4 RoPE cache builder mismatch (layer_idx={layer_idx}, layer_type={layer_type}, position={position}): "
        f"cos={cos_msg}, sin={sin_msg}"
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("cache_source", ["hf", "gemma4"])
def test_rope_decode_embedding_lookup_pcc(layer_idx, cache_source, mesh_device):
    """Exercise the production decode RoPE path: 2D cache + ttnn.embedding + token_index=0."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_cody.tt.attention.operations import apply_rope
    from models.demos.gemma4_cody.tt.model import create_rope_caches

    hf_text_config = TestFactory.create_hf_text_config()
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    layer_type = hf_text_config.layer_types[layer_idx]
    position = 32
    max_seq_len = 128
    head_dim = config.head_dim
    num_heads = config.num_attention_heads

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    x_dummy = torch.randn(1, 1, hf_text_config.hidden_size)
    cos_ref, sin_ref = rope(x_dummy, torch.tensor([[position]]), layer_type=layer_type)

    q_torch = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float32)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos_expanded = cos_ref.unsqueeze(2).expand(1, 1, num_heads, head_dim)
    sin_expanded = sin_ref.unsqueeze(2).expand(1, 1, num_heads, head_dim)
    ref_rotated = (q_torch * cos_expanded) + (rotate_half(q_torch) * sin_expanded)

    if cache_source == "hf":
        rope_hf = Gemma4TextRotaryEmbedding(hf_text_config)
        cache_dummy = torch.randn(1, max_seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(max_seq_len).unsqueeze(0)
        cos_cache, sin_cache = rope_hf(cache_dummy, pos_ids, layer_type=layer_type)
        cos_cache = ttnn.from_torch(
            cos_cache.squeeze(0), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        sin_cache = ttnn.from_torch(
            sin_cache.squeeze(0), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
    else:
        _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
        cos_cache, sin_cache = rope_caches_2d[layer_type]

    position_idx = ttnn.from_torch(
        torch.nn.functional.pad(torch.tensor([position], dtype=torch.int32).reshape(1, 1), (0, 31)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    cos_pos = ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)
    sin_pos = ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
    cos_pos = ttnn.unsqueeze_to_4D(cos_pos)[:, :, :1, :]
    sin_pos = ttnn.unsqueeze_to_4D(sin_pos)[:, :, :1, :]

    q_tt = ttnn.from_torch(q_torch.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_rotated = apply_rope(q_tt, cos_pos, sin_pos, token_index=0)
    tt_rotated_torch = ttnn.to_torch(tt_rotated).float()

    passing, pcc_msg = compare_tensors(tt_rotated_torch, ref_rotated, pcc_threshold=0.98)
    assert passing, (
        f"Decode RoPE embedding lookup PCC too low "
        f"(layer_idx={layer_idx}, layer_type={layer_type}, cache_source={cache_source}): {pcc_msg}"
    )


# ── Decode PCC Test (paged attention, high positions) ────────────────────


def _build_sliding_window_mask(cache_len, sliding_window):
    """Build HF-compatible decode attention mask with sliding window.

    Returns [1, 1, 1, cache_len+1] mask where positions outside the window are -inf.
    For global layers (sliding_window=None), all positions are visible (mask=0).
    """
    total_len = cache_len + 1  # cache entries + current query token
    mask = torch.zeros(1, 1, 1, total_len)
    if sliding_window is not None:
        current_pos = cache_len
        for j in range(total_len):
            if j < current_pos - sliding_window + 1:
                mask[0, 0, 0, j] = float("-inf")
    return mask


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("cache_len", [32, 512, 1023, 1500], ids=lambda c: f"cache{c}")
def test_attention_decode_paged(layer_idx, cache_len, mesh_device):
    """Test decode attention with paged KV cache against HF reference.

    Tests at various cache lengths including positions beyond the sliding window (1024).
    At cache_len=1500 with sliding_window=1024, SDPA must correctly mask out old entries.
    Global layers (no sliding window) should attend to all cache positions.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache
    from models.tt_transformers.tt.common import PagedAttentionConfig

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    _skip_if_l1_overflow(config, mesh_device)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    # Paged attention: enough blocks to hold cache_len + some headroom
    block_size = 64
    max_num_blocks = (cache_len + block_size) // block_size + 1
    max_seq_len = max_num_blocks * block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
    kv_cache = init_kv_cache(
        mesh_device=mesh_device, config=config, paged_attention_config=paged_attention_config, cache_dtype=ttnn.bfloat16
    )

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    tt_attn.kv_cache = kv_cache

    # Random KV cache data [1, num_kv_heads, cache_len, head_dim]
    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    # HF cache
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    # TT paged cache fill
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, max_num_blocks)
    page_table_tt = ttnn.from_torch(page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    k_cache_tt, v_cache_tt = kv_cache
    k_fill = ttnn.from_torch(
        k_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    v_fill = ttnn.from_torch(
        v_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, page_table_tt, batch_idx=0)
    ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, page_table_tt, batch_idx=0)

    # Decode input
    x_torch = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)

    # HF reference with proper sliding-window mask
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    sliding_window = config.sliding_window if config.is_sliding else None
    mask = _build_sliding_window_mask(cache_len, sliding_window)
    with torch.no_grad():
        ref_output, _ = hf_attn(
            x_torch,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            attention_mask=mask,
            shared_kv_states=None,
        )

    # TT decode with paged attention
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    tt_output = tt_attn(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=cache_len,
        page_table=page_table_tt,
    )
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, (
        f"Attention paged decode (layer={layer_idx}, cache_len={cache_len}, "
        f"sliding_window={sliding_window}) PCC too low: {pcc_msg}"
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 8)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("rope_mode", ["hf_4d", "gemma4_4d", "gemma4_2d"])
def test_attention_decode_rope_mode_pcc(layer_idx, rope_mode, mesh_device):
    """Attention-only decode PCC while changing only the RoPE cache source/path."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache
    from models.demos.gemma4_cody.tt.model import create_rope_caches

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    cache_len = 32
    max_seq_len = cache_len + 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    kv_cache = init_kv_cache(mesh_device, config, max_batch_size=1, max_seq_len=max_seq_len, cache_dtype=ttnn.bfloat16)
    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    tt_attn.kv_cache = kv_cache

    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    if tp > 1:
        q_per_device = config.num_attention_heads // tp
        kv_replicated = config.num_key_value_heads < tp
        local_kv = 1 if kv_replicated else config.num_key_value_heads // tp
        for dev_idx in range(tp):
            if kv_replicated:
                kv_idx = (dev_idx * q_per_device) * config.num_key_value_heads // config.num_attention_heads
                k_local = k_data[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
                v_local = v_data[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
            else:
                start = dev_idx * local_kv
                k_local = k_data[:, start : start + local_kv].to(torch.bfloat16)
                v_local = v_data[:, start : start + local_kv].to(torch.bfloat16)

            dev_k = ttnn.get_device_tensors(kv_cache[0])[dev_idx]
            dev_v = ttnn.get_device_tensors(kv_cache[1])[dev_idx]
            ttnn.fill_cache(
                dev_k,
                ttnn.from_torch(k_local, device=dev_k.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                batch_idx=0,
            )
            ttnn.fill_cache(
                dev_v,
                ttnn.from_torch(v_local, device=dev_v.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                batch_idx=0,
            )
    else:
        k_fill = ttnn.from_torch(
            k_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        v_fill = ttnn.from_torch(
            v_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        ttnn.fill_cache(kv_cache[0], k_fill, batch_idx=0)
        ttnn.fill_cache(kv_cache[1], v_fill, batch_idx=0)

    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    x_torch = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)
    layer_type = hf_text_config.layer_types[layer_idx]
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    mask = _build_sliding_window_mask(cache_len, config.sliding_window if config.is_sliding else None)
    with torch.no_grad():
        ref_output, _ = hf_attn(
            x_torch,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            attention_mask=mask,
            shared_kv_states=None,
        )

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )

    if rope_mode == "hf_4d":
        rope_mats = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)
        position_idx_tt = ttnn.from_torch(
            torch.tensor([[cache_len]], dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )
        position_idx_cache_tt = None
        token_index = cache_len
    else:
        rope_caches_4d, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
        rope_mats = rope_caches_4d[layer_type] if rope_mode == "gemma4_4d" else rope_caches_2d[layer_type]
        if rope_mode == "gemma4_4d":
            position_idx_tt = ttnn.from_torch(
                torch.tensor([[cache_len]], dtype=torch.int32),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            )
            position_idx_cache_tt = None
            token_index = cache_len
        else:
            position_idx_tt = ttnn.from_torch(
                torch.nn.functional.pad(torch.tensor([cache_len], dtype=torch.int32).reshape(1, 1), (0, 31)),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            )
            position_idx_cache_tt = ttnn.from_torch(
                torch.tensor([cache_len], dtype=torch.int32),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            )
            token_index = None

    tt_output = tt_attn(
        x_tt,
        rope_mats=rope_mats,
        position_idx=position_idx_tt,
        position_idx_cache=position_idx_cache_tt,
        is_decode=True,
        token_index=token_index,
    )
    tt_output_torch = _from_device(tt_output, mesh_device).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.90)
    assert passing, (
        f"Attention decode PCC too low for rope_mode={rope_mode} "
        f"(layer_idx={layer_idx}, layer_type={layer_type}, tp={tp}): {pcc_msg}"
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 8)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
def test_attention_decode_component_pcc(layer_idx, mesh_device):
    """Pinpoint the first production decode component whose PCC diverges.

    This test intentionally mirrors the production gemma4_2d decode path and
    compares local per-device Q/K/V placement before checking SDPA and output.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache
    from models.demos.gemma4_cody.tt.attention.operations import (
        apply_allreduce,
        apply_per_head_norm,
        apply_qkv_projection,
        apply_rope,
        concat_heads,
        split_qkv_heads_decode,
    )
    from models.demos.gemma4_cody.tt.model import create_rope_caches

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    cache_len = 32
    max_seq_len = cache_len + 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    layer_type = hf_text_config.layer_types[layer_idx]

    kv_cache = init_kv_cache(mesh_device, config, max_batch_size=1, max_seq_len=max_seq_len, cache_dtype=ttnn.bfloat16)
    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    tt_attn.kv_cache = kv_cache

    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    q_per_device = config.num_attention_heads // tp
    kv_replicated = config.num_key_value_heads < tp
    local_kv = 1 if kv_replicated else config.num_key_value_heads // tp

    for dev_idx in range(tp):
        if kv_replicated:
            kv_idx = (dev_idx * q_per_device) * config.num_key_value_heads // config.num_attention_heads
            k_local = k_data[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
            v_local = v_data[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
        else:
            kv_start = dev_idx * local_kv
            k_local = k_data[:, kv_start : kv_start + local_kv].to(torch.bfloat16)
            v_local = v_data[:, kv_start : kv_start + local_kv].to(torch.bfloat16)

        dev_k = ttnn.get_device_tensors(kv_cache[0])[dev_idx]
        dev_v = ttnn.get_device_tensors(kv_cache[1])[dev_idx]
        ttnn.fill_cache(
            dev_k,
            ttnn.from_torch(k_local, device=dev_k.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            batch_idx=0,
        )
        ttnn.fill_cache(
            dev_v,
            ttnn.from_torch(v_local, device=dev_v.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            batch_idx=0,
        )

    x_torch = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    cos_expanded = cos.unsqueeze(2)
    sin_expanded = sin.unsqueeze(2)

    hidden_shape = (*x_torch.shape[:-1], -1, config.head_dim)
    q_split_ref = hf_attn.q_proj(x_torch).view(hidden_shape)
    k_split_ref = hf_attn.k_proj(x_torch).view(hidden_shape)
    if config.use_kv_tying or getattr(hf_attn, "v_proj", None) is None:
        v_split_ref = k_split_ref
    else:
        v_split_ref = hf_attn.v_proj(x_torch).view(hidden_shape)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_norm_ref = hf_attn.q_norm(q_split_ref)
    k_norm_ref = hf_attn.k_norm(k_split_ref)
    v_norm_ref = hf_attn.v_norm(v_split_ref)
    q_rope_ref = (q_norm_ref * cos_expanded) + (rotate_half(q_norm_ref) * sin_expanded)
    k_rope_ref = (k_norm_ref * cos_expanded) + (rotate_half(k_norm_ref) * sin_expanded)

    q_attn_ref = q_rope_ref.transpose(1, 2)
    k_current_ref = k_rope_ref.transpose(1, 2)
    v_current_ref = v_norm_ref.transpose(1, 2)
    k_full_ref = torch.cat([k_data, k_current_ref], dim=2)
    v_full_ref = torch.cat([v_data, v_current_ref], dim=2)

    def repeat_kv(hidden_states, n_rep):
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

    k_sdpa_ref = repeat_kv(k_full_ref, config.num_key_value_groups)
    v_sdpa_ref = repeat_kv(v_full_ref, config.num_key_value_groups)
    attn_weights_ref = torch.matmul(q_attn_ref, k_sdpa_ref.transpose(2, 3))
    attn_weights_ref = attn_weights_ref + _build_sliding_window_mask(
        cache_len, config.sliding_window if config.is_sliding else None
    )
    attn_weights_ref = torch.nn.functional.softmax(attn_weights_ref, dim=-1, dtype=torch.float32)
    sdpa_ref = torch.matmul(attn_weights_ref, v_sdpa_ref)
    decode_heads_ref = sdpa_ref.transpose(1, 2)

    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)
    with torch.no_grad():
        ref_output, _ = hf_attn(
            x_torch,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            attention_mask=_build_sliding_window_mask(cache_len, config.sliding_window if config.is_sliding else None),
            shared_kv_states=None,
        )

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
    cos_cache, sin_cache = rope_caches_2d[layer_type]
    position_idx_tt = ttnn.from_torch(
        torch.nn.functional.pad(torch.tensor([cache_len], dtype=torch.int32).reshape(1, 1), (0, 31)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    position_idx_cache_tt = ttnn.from_torch(
        torch.tensor([cache_len], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    weights = tt_attn.weights
    xqkv = apply_qkv_projection(x_tt, weights)
    tt_q_split, tt_k_split, tt_v_split = split_qkv_heads_decode(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

    q_sharded_mem = tt_q_split.memory_config()
    tt_q_norm = ttnn.to_memory_config(tt_q_split, ttnn.DRAM_MEMORY_CONFIG)
    tt_q_norm = apply_per_head_norm(tt_q_norm, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_k_norm = ttnn.to_memory_config(tt_k_split, ttnn.DRAM_MEMORY_CONFIG)
    tt_k_norm = apply_per_head_norm(tt_k_norm, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_v_norm = ttnn.to_memory_config(tt_v_split, ttnn.DRAM_MEMORY_CONFIG)
    tt_v_norm = apply_per_head_norm(tt_v_norm, None, config.rms_norm_eps, with_scale=False)

    cos_pos = ttnn.embedding(position_idx_tt, cos_cache, layout=ttnn.TILE_LAYOUT)
    sin_pos = ttnn.embedding(position_idx_tt, sin_cache, layout=ttnn.TILE_LAYOUT)
    cos_pos = ttnn.unsqueeze_to_4D(cos_pos)
    sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
    batch = tt_q_norm.shape[1]
    if cos_pos.shape[2] != batch:
        cos_pos = cos_pos[:, :, :batch, :]
        sin_pos = sin_pos[:, :, :batch, :]

    tt_q_rope = apply_rope(tt_q_norm, cos_pos, sin_pos, token_index=0)
    tt_k_rope = apply_rope(tt_k_norm, cos_pos, sin_pos, token_index=0)
    tt_k_cache = ttnn.to_memory_config(tt_k_rope, q_sharded_mem)
    tt_v_cache = ttnn.to_memory_config(tt_v_norm, q_sharded_mem)
    ttnn.experimental.paged_update_cache(kv_cache[0], tt_k_cache, update_idxs_tensor=position_idx_cache_tt)
    ttnn.experimental.paged_update_cache(kv_cache[1], tt_v_cache, update_idxs_tensor=position_idx_cache_tt)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4) if config.head_dim >= 512 else ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )
    tt_sdpa = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q_rope,
        kv_cache[0],
        kv_cache[1],
        cur_pos_tensor=position_idx_cache_tt,
        scale=1.0,
        sliding_window_size=config.sliding_window if config.is_sliding else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=sdpa_program_config,
    )
    tt_concat = concat_heads(tt_sdpa, is_decode_mode=True)
    tt_proj = ttnn.linear(tt_concat, weights.o_proj)

    def device_torch(tensor, dev_idx):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[dev_idx]).float()

    component_results = []

    def record(name, actual, expected, threshold=0.90):
        passing, pcc = compare_tensors(actual.float().detach(), expected.float().detach(), pcc_threshold=threshold)
        component_results.append((name, passing, pcc, tuple(actual.shape), tuple(expected.shape)))

    o_proj_weight = state_dict["o_proj.weight"]
    for dev_idx in range(tp):
        q_start = dev_idx * q_per_device
        q_end = q_start + q_per_device
        if kv_replicated:
            kv_start = q_start * config.num_key_value_heads // config.num_attention_heads
            kv_end = kv_start + 1
        else:
            kv_start = dev_idx * local_kv
            kv_end = kv_start + local_kv

        q_slice = slice(q_start, q_end)
        kv_slice = slice(kv_start, kv_end)
        q_dim_slice = slice(q_start * config.head_dim, q_end * config.head_dim)

        q_label = f"q_heads[{q_start}:{q_end}]"
        kv_label = f"kv_heads[{kv_start}:{kv_end}]"

        record(f"dev{dev_idx}.q_split.{q_label}", device_torch(tt_q_split, dev_idx), q_split_ref[:, :, q_slice, :])
        record(f"dev{dev_idx}.k_split.{kv_label}", device_torch(tt_k_split, dev_idx), k_split_ref[:, :, kv_slice, :])
        record(f"dev{dev_idx}.v_split.{kv_label}", device_torch(tt_v_split, dev_idx), v_split_ref[:, :, kv_slice, :])
        record(f"dev{dev_idx}.q_norm.{q_label}", device_torch(tt_q_norm, dev_idx), q_norm_ref[:, :, q_slice, :])
        record(f"dev{dev_idx}.k_norm.{kv_label}", device_torch(tt_k_norm, dev_idx), k_norm_ref[:, :, kv_slice, :])
        record(f"dev{dev_idx}.v_norm.{kv_label}", device_torch(tt_v_norm, dev_idx), v_norm_ref[:, :, kv_slice, :])
        record(f"dev{dev_idx}.q_rope.{q_label}", device_torch(tt_q_rope, dev_idx), q_rope_ref[:, :, q_slice, :])
        record(f"dev{dev_idx}.k_rope.{kv_label}", device_torch(tt_k_rope, dev_idx), k_rope_ref[:, :, kv_slice, :])

        if config.head_dim >= 512:
            q_split_actual = device_torch(tt_q_split, dev_idx)
            q_norm_actual = device_torch(tt_q_norm, dev_idx)
            q_rope_actual = device_torch(tt_q_rope, dev_idx)
            half_dim = config.head_dim // 2
            first_half = slice(0, half_dim)
            second_half = slice(half_dim, config.head_dim)
            record(
                f"dev{dev_idx}.q_split.{q_label}.dims[0:{half_dim}]",
                q_split_actual[..., first_half],
                q_split_ref[:, :, q_slice, first_half],
            )
            record(
                f"dev{dev_idx}.q_split.{q_label}.dims[{half_dim}:{config.head_dim}]",
                q_split_actual[..., second_half],
                q_split_ref[:, :, q_slice, second_half],
            )
            record(
                f"dev{dev_idx}.q_norm.{q_label}.dims[0:{half_dim}]",
                q_norm_actual[..., first_half],
                q_norm_ref[:, :, q_slice, first_half],
            )
            record(
                f"dev{dev_idx}.q_norm.{q_label}.dims[{half_dim}:{config.head_dim}]",
                q_norm_actual[..., second_half],
                q_norm_ref[:, :, q_slice, second_half],
            )
            record(
                f"dev{dev_idx}.q_rope.{q_label}.dims[0:{half_dim}]",
                q_rope_actual[..., first_half],
                q_rope_ref[:, :, q_slice, first_half],
            )
            record(
                f"dev{dev_idx}.q_rope.{q_label}.dims[{half_dim}:{config.head_dim}]",
                q_rope_actual[..., second_half],
                q_rope_ref[:, :, q_slice, second_half],
            )

        cache_k_actual = device_torch(kv_cache[0], dev_idx)[:, :, cache_len : cache_len + 1, :]
        cache_v_actual = device_torch(kv_cache[1], dev_idx)[:, :, cache_len : cache_len + 1, :]
        record(f"dev{dev_idx}.cache_k_write.{kv_label}", cache_k_actual, k_current_ref[:, kv_slice, :, :])
        record(f"dev{dev_idx}.cache_v_write.{kv_label}", cache_v_actual, v_current_ref[:, kv_slice, :, :])

        local_sdpa_ref = decode_heads_ref[:, :, q_slice, :]
        local_concat_ref = local_sdpa_ref.reshape(1, 1, 1, q_per_device * config.head_dim)
        local_proj_ref = torch.matmul(local_concat_ref, o_proj_weight[:, q_dim_slice].transpose(0, 1))

        record(f"dev{dev_idx}.sdpa.{q_label}", device_torch(tt_sdpa, dev_idx), local_sdpa_ref)
        record(f"dev{dev_idx}.concat.{q_label}", device_torch(tt_concat, dev_idx), local_concat_ref)
        record(f"dev{dev_idx}.o_proj_local.{q_label}", device_torch(tt_proj, dev_idx), local_proj_ref)

    tt_output = apply_allreduce(tt_proj, mesh_config, ccl_manager, config.hidden_size)
    record("final_allreduce", _from_device(tt_output, mesh_device).squeeze(0).float(), ref_output)

    failures = [result for result in component_results if not result[1]]
    assert not failures, "Decode component PCC divergence:\n" + "\n".join(
        f"{name}: pcc={pcc} actual_shape={actual_shape} expected_shape={expected_shape}"
        for name, _, pcc, actual_shape, expected_shape in failures
    )
