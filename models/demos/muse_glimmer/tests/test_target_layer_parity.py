# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer as ReferenceDecoderLayer
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer.tt.layer import MuseGlimmerDecoderLayer
from models.demos.muse_glimmer.tt.model import _create_rope_cache_tensors
from models.demos.muse_glimmer.tt.model_config import MuseGlimmerModelArgs


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.DISABLED}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,layer_type,layer_theta",
    [(0, "sliding_attention", 500000.0), (3, "full_attention", 0.0)],
)
def test_target_layer_prefill_pcc(mesh_device, layer_idx, layer_type, layer_theta):
    torch.manual_seed(0)
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
        max_position_embeddings=128,
        sliding_window=128,
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
        tensor_cache_path=None,
        mlp_cache_path=None,
        max_seq_len=128,
    )

    hidden = torch.randn(1, 32, ref_config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(32).unsqueeze(0)
    if layer_theta:
        cos, sin = MuseGlimmerTextRotaryEmbedding(ref_config)(hidden, positions)
        position_embeddings = (cos, sin)
    else:
        position_embeddings = None
    causal_mask = torch.full((1, 1, 32, 32), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    with torch.no_grad():
        expected = reference(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        ).float()

    cos_host, sin_host = _create_rope_cache_tensors(tt_config, 32, layer_type)
    cos_tt = ttnn.from_torch(cos_host.unsqueeze(0), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    sin_tt = ttnn.from_torch(sin_host.unsqueeze(0), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    hidden_tt = ttnn.from_torch(hidden.unsqueeze(0), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    actual_tt = tt_layer(
        hidden_tt,
        rope_mats=(cos_tt, sin_tt),
        kv_cache=None,
        is_decode=False,
    )
    actual = ttnn.to_torch(ttnn.get_device_tensors(actual_tt)[0]).squeeze(0).float()
    passing, pcc = comp_pcc(expected, actual, 0.999)
    print(f"target layer {layer_idx} ({layer_type}) PCC: {float(pcc):.6f}")
    assert passing, pcc


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.DISABLED}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,layer_type,layer_theta",
    [(0, "sliding_attention", 500000.0), (3, "full_attention", 0.0)],
)
def test_target_layer_two_chunk_prefill_pcc(mesh_device, layer_idx, layer_type, layer_theta):
    """Compare two 256-token paged chunks with one Torch reference pass."""
    torch.manual_seed(1)
    sequence_length, max_seq_len, chunk_size, sliding_window = 256, 512, 128, 64
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
        sliding_window=sliding_window,
        layer_types=layer_types,
        layer_rope_theta=layer_thetas,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        bos_token_id=1,
        eos_token_id=2,
    )
    reference = ReferenceDecoderLayer(ref_config, layer_idx).eval().to(torch.bfloat16)
    state_dict = {
        f"model.language_model.layers.{layer_idx}.{key}": value for key, value in reference.state_dict().items()
    }
    tt_config = MuseGlimmerModelArgs.from_hf_config(ref_config)
    tt_layer = MuseGlimmerDecoderLayer(
        device=mesh_device,
        hf_config=tt_config,
        state_dict=state_dict,
        layer_idx=layer_idx,
        attention_dtype=ttnn.bfloat16,
        mlp_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mlp_cache_path=None,
        max_seq_len=max_seq_len,
        create_kv_cache=True,
    )

    hidden = torch.randn(1, sequence_length, ref_config.hidden_size, dtype=torch.bfloat16)
    query_positions = torch.arange(sequence_length).unsqueeze(1)
    key_positions = torch.arange(sequence_length).unsqueeze(0)
    allowed = key_positions <= query_positions
    if layer_type == "sliding_attention":
        allowed &= key_positions >= query_positions - sliding_window + 1
    attention_mask = torch.where(allowed, 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[None, None]
    positions = torch.arange(sequence_length).unsqueeze(0)
    position_embeddings = MuseGlimmerTextRotaryEmbedding(ref_config)(hidden, positions) if layer_theta else None
    with torch.no_grad():
        expected = reference(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        ).float()

    cos_host, sin_host = _create_rope_cache_tensors(tt_config, max_seq_len, layer_type)
    actual_chunks = []
    for start in range(0, sequence_length, chunk_size):
        end = start + chunk_size
        to_tt = lambda tensor: ttnn.from_torch(tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        actual_tt = tt_layer(
            to_tt(hidden[:, start:end].unsqueeze(0)),
            rope_mats=(
                to_tt(cos_host[:, start:end].unsqueeze(0)),
                to_tt(sin_host[:, start:end].unsqueeze(0)),
            ),
            kv_cache=tt_layer.self_attn.kv_cache,
            is_decode=False,
            packed={
                "page_table": tt_layer.self_attn.page_table,
                "chunk_start": start,
                "logical_length": chunk_size,
            },
        )
        actual_chunks.append(ttnn.to_torch(ttnn.get_device_tensors(actual_tt)[0]).squeeze(0))

    actual = torch.cat(actual_chunks, dim=1).float()
    passing, pcc = comp_pcc(expected, actual, 0.999)
    print(f"two-chunk prefill layer {layer_idx} ({layer_type}) PCC: {float(pcc):.6f}")
    assert passing, pcc
