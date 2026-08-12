# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC coverage for the batch-1 packed target verification path."""

import pytest
import torch
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer as ReferenceDecoderLayer
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer.tt.attention import MuseGlimmerAttentionConfig
from models.demos.muse_glimmer.tt.attention.kv_cache import init_kv_cache
from models.demos.muse_glimmer.tt.layer import MuseGlimmerDecoderLayer
from models.demos.muse_glimmer.tt.model import _create_rope_cache_tensors
from models.demos.muse_glimmer.tt.model_config import MuseGlimmerModelArgs


def _to_tt(mesh_device, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, device=mesh_device, layout=layout, dtype=dtype)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.DISABLED}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,layer_type,layer_theta",
    [(0, "sliding_attention", 500000.0), (3, "full_attention", 0.0)],
)
@pytest.mark.parametrize("context_len", [32, 48, 256], ids=["one_page", "page_crossing", "later_page"])
@pytest.mark.parametrize("real_p", [16, 32], ids=["real16", "real32"])
def test_target_layer_packed_verify_pcc(mesh_device, tmp_path, layer_idx, layer_type, layer_theta, context_len, real_p):
    torch.manual_seed(0)
    packed_p, max_seq_len = 32, 320
    layer_types = ["sliding_attention"] * 4
    layer_types[layer_idx] = layer_type
    layer_thetas = [500000.0] * 4
    layer_thetas[layer_idx] = layer_theta
    ref_config = MuseGlimmerTextConfig(
        vocab_size=320,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=max_seq_len,
        sliding_window=max_seq_len,
        layer_types=layer_types,
        layer_rope_theta=layer_thetas,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        bos_token_id=1,
        eos_token_id=2,
    )
    reference = ReferenceDecoderLayer(ref_config, layer_idx).eval().to(torch.bfloat16)
    state_dict = {f"model.language_model.layers.{layer_idx}.{k}": v for k, v in reference.state_dict().items()}
    tt_config = MuseGlimmerModelArgs.from_hf_config(ref_config)
    tt_layer = MuseGlimmerDecoderLayer(
        device=mesh_device,
        hf_config=tt_config,
        state_dict=state_dict,
        layer_idx=layer_idx,
        attention_dtype=ttnn.bfloat16,
        mlp_dtype=ttnn.bfloat16,
        tensor_cache_path=tmp_path / f"layer_{layer_idx}",
        mlp_cache_path=tmp_path / f"layer_{layer_idx}",
        max_seq_len=max_seq_len,
    )
    kv_cache = init_kv_cache(
        mesh_device,
        MuseGlimmerAttentionConfig(tt_config, layer_idx),
        max_seq_len=max_seq_len,
        cache_dtype=ttnn.bfloat16,
    )

    context = torch.randn(1, context_len, ref_config.hidden_size, dtype=torch.bfloat16)
    packed_hidden = torch.randn(1, real_p, ref_config.hidden_size, dtype=torch.bfloat16)
    full_hidden = torch.cat([context, packed_hidden], dim=1)
    positions = torch.arange(context_len + real_p).unsqueeze(0)
    position_embeddings = MuseGlimmerTextRotaryEmbedding(ref_config)(full_hidden, positions) if layer_theta else None
    causal_mask = torch.full(
        (1, 1, context_len + real_p, context_len + real_p),
        torch.finfo(torch.bfloat16).min,
        dtype=torch.bfloat16,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    with torch.no_grad():
        expected = reference(
            full_hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )[:, -real_p:].float()

    cos_host, sin_host = _create_rope_cache_tensors(tt_config, max_seq_len, layer_type)
    # Cover both page zero and the identity-page-table mapping at a later page.
    # Short page-crossing cases use packed warmup; the later-page case can use
    # the faster aligned prefill directly.
    prefill_len = context_len if context_len >= 64 else 32
    context_tt = _to_tt(mesh_device, context[:, :prefill_len].unsqueeze(0))
    tt_layer(
        context_tt,
        rope_mats=(
            _to_tt(mesh_device, cos_host[:, :prefill_len].unsqueeze(0)),
            _to_tt(mesh_device, sin_host[:, :prefill_len].unsqueeze(0)),
        ),
        kv_cache=kv_cache,
        is_decode=False,
    )

    page_ids = torch.arange(max_seq_len // 64, dtype=torch.int32).reshape(1, -1)

    def packed_metadata(start, count):
        positions = torch.zeros(packed_p, dtype=torch.int32)
        positions[:count] = torch.arange(start, start + count, dtype=torch.int32)
        cur_positions = torch.full((packed_p,), -1, dtype=torch.int32)
        cur_positions[:count] = positions[:count]
        first_page = start // 64
        return {
            "p": packed_p,
            "real_p": count,
            "position_idx": _to_tt(
                mesh_device, positions.unsqueeze(0), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "cur_pos": _to_tt(mesh_device, cur_positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            "page_index": first_page,
            "page_offset": start % 64,
            "page_table": _to_tt(
                mesh_device,
                page_ids.repeat(packed_p, 1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        }

    if context_len > prefill_len:
        warm_count = context_len - prefill_len
        warm_tt = tt_layer(
            _to_tt(
                mesh_device,
                torch.nn.functional.pad(context[:, prefill_len:], (0, 0, 0, packed_p - warm_count)).unsqueeze(0),
            ),
            rope_mats=(
                _to_tt(mesh_device, cos_host.squeeze(0), layout=ttnn.TILE_LAYOUT),
                _to_tt(mesh_device, sin_host.squeeze(0), layout=ttnn.TILE_LAYOUT),
            ),
            kv_cache=kv_cache,
            is_decode=True,
            packed=packed_metadata(prefill_len, warm_count),
        )
        ttnn.deallocate(warm_tt)
        tt_layer.self_attn.commit_pending_tail(warm_count)

    packed = packed_metadata(context_len, real_p)
    actual_tt = tt_layer(
        _to_tt(
            mesh_device,
            torch.nn.functional.pad(packed_hidden, (0, 0, 0, packed_p - real_p)).unsqueeze(0),
        ),
        rope_mats=(
            _to_tt(mesh_device, cos_host.squeeze(0), layout=ttnn.TILE_LAYOUT),
            _to_tt(mesh_device, sin_host.squeeze(0), layout=ttnn.TILE_LAYOUT),
        ),
        kv_cache=kv_cache,
        is_decode=True,
        packed=packed,
    )
    actual = ttnn.to_torch(ttnn.get_device_tensors(actual_tt)[0]).squeeze(0)[:, :real_p].float()
    passing, pcc = comp_pcc(expected, actual, 0.999)
    print(f"packed verify layer {layer_idx} ({layer_type}) PCC: {float(pcc):.6f}")
    assert passing, pcc
