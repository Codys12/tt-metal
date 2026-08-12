# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC checks using the real Muse-Glimmer-30B checkpoint weights."""

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer as ReferenceDecoderLayer
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer.tt.common import DEFAULT_MODEL_PATH
from models.demos.muse_glimmer.tt.layer import MuseGlimmerDecoderLayer
from models.demos.muse_glimmer.tt.model import _create_rope_cache_tensors
from models.demos.muse_glimmer.tt.model_config import MuseGlimmerModelArgs
from models.demos.muse_glimmer.utils.lazy_state_dict import LazyStateDict

MODEL_PATH = Path(os.getenv("MUSE_TARGET_DIR", DEFAULT_MODEL_PATH))


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.DISABLED}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding", "full-nope"])
def test_checkpoint_layer_pcc(mesh_device, layer_idx):
    if not (MODEL_PATH / "model.safetensors.index.json").exists():
        pytest.skip(f"Muse Glimmer checkpoint is not available at {MODEL_PATH}")

    torch.manual_seed(0)
    hf_text = AutoConfig.from_pretrained(MODEL_PATH).text_config
    reference = ReferenceDecoderLayer(hf_text, layer_idx).eval().to(torch.bfloat16)
    state = LazyStateDict(MODEL_PATH)
    prefix = f"model.language_model.layers.{layer_idx}."
    reference.load_state_dict({key[len(prefix) :]: state[key] for key in state if key.startswith(prefix)})

    config = MuseGlimmerModelArgs.from_hf_config(hf_text)
    cache_path = config.weight_cache_path(MODEL_PATH, ttnn.bfloat8_b)
    mlp_cache_path = config.weight_cache_path(MODEL_PATH, ttnn.bfloat4_b)
    tt_layer = MuseGlimmerDecoderLayer(
        device=mesh_device,
        hf_config=config,
        state_dict=state,
        layer_idx=layer_idx,
        attention_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat4_b,
        tensor_cache_path=cache_path,
        mlp_cache_path=mlp_cache_path,
        max_seq_len=128,
    )

    hidden = torch.randn(1, 32, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(32).unsqueeze(0)
    theta = config.layer_rope_theta[layer_idx]
    position_embeddings = MuseGlimmerTextRotaryEmbedding(hf_text)(hidden, positions) if theta else None
    causal_mask = torch.full((1, 1, 32, 32), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    with torch.no_grad():
        expected = reference(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        ).float()

    layer_type = config.layer_types[layer_idx]
    cos, sin = _create_rope_cache_tensors(config, 32, layer_type)
    to_tt = lambda tensor: ttnn.from_torch(tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    actual_tt = tt_layer(
        to_tt(hidden.unsqueeze(0)),
        rope_mats=(to_tt(cos.unsqueeze(0)), to_tt(sin.unsqueeze(0))),
        kv_cache=None,
        is_decode=False,
    )
    actual = ttnn.to_torch(ttnn.get_device_tensors(actual_tt)[0]).squeeze(0).float()
    passing, pcc = comp_pcc(expected, actual, 0.995)
    print(f"checkpoint layer {layer_idx} ({layer_type}) PCC: {float(pcc):.6f}")
    state.close()
    assert passing, pcc
