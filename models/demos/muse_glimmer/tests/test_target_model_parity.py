# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end target-model parity using the official Transformers implementation."""

import pytest
import torch
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer.tt.model import MuseGlimmerModel
from models.demos.muse_glimmer.tt.model_config import MuseGlimmerModelArgs


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.DISABLED}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "mlp_dtype,min_pcc,min_top1",
    [
        (ttnn.bfloat8_b, 0.99, 0.75),
        (ttnn.bfloat4_b, 0.90, 0.40),
    ],
    ids=["bfp8-correctness", "production-bfp4-mlp"],
)
def test_target_model_prefill_logits_pcc(mesh_device, tmp_path, mlp_dtype, min_pcc, min_top1):
    torch.manual_seed(0)
    config = MuseGlimmerTextConfig(
        vocab_size=320,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        sliding_window=128,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        layer_rope_theta=[500000.0] * 3 + [0.0],
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        bos_token_id=1,
        eos_token_id=2,
    )
    reference = MuseGlimmerTextModel(config).eval().to(torch.bfloat16)
    reference_taps = []
    hooks = [
        layer.register_forward_hook(lambda _m, _i, out: reference_taps.append(out.detach()))
        for layer in reference.layers
    ]
    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(torch.bfloat16)
    state_dict = {f"model.language_model.{key}": value for key, value in reference.state_dict().items()}
    state_dict["lm_head.weight"] = lm_head.weight

    tt_config = MuseGlimmerModelArgs.from_hf_config(config)
    tt_config._hf_text_config = config
    model = MuseGlimmerModel(
        device=mesh_device,
        hf_config=tt_config,
        state_dict=state_dict,
        attention_dtype=ttnn.bfloat8_b,
        mlp_dtype=mlp_dtype,
        lm_head_dtype=ttnn.bfloat8_b,
        tensor_cache_path=str(tmp_path / "target_cache"),
        mlp_cache_path=str(tmp_path / "target_mlp_cache"),
        max_seq_len=128,
        create_kv_cache=False,
    )

    input_ids = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.int64)
    with torch.no_grad():
        expected_hidden = reference(input_ids=input_ids, use_cache=False).last_hidden_state
        expected_logits = lm_head(expected_hidden) * config.output_multiplier
        expected_logits = config.final_logit_softcapping * torch.tanh(expected_logits / config.final_logit_softcapping)

    for hook in hooks:
        hook.remove()
    model.configure_aux_taps(range(config.num_hidden_layers))
    actual_tt, _, actual_taps_tt = model(
        model.embed_input_ids(input_ids),
        is_decode=False,
        return_aux_hidden=True,
    )
    actual = ttnn.to_torch(ttnn.get_device_tensors(actual_tt)[0]).squeeze(0).float()
    passing, pcc = comp_pcc(expected_logits.float(), actual, min_pcc)
    top1_agreement = (expected_logits.argmax(-1) == actual.argmax(-1)).float().mean().item()
    for layer_idx, (expected_tap, actual_tap_tt) in enumerate(zip(reference_taps, actual_taps_tt)):
        actual_tap = ttnn.to_torch(ttnn.get_device_tensors(actual_tap_tt)[0]).squeeze(0).float()
        _, tap_pcc = comp_pcc(expected_tap.float(), actual_tap, 0.0)
        print(f"target model layer {layer_idx} PCC: {float(tap_pcc):.6f}")
    print(f"target model logits PCC: {float(pcc):.6f}")
    print(f"target model top-1 agreement: {top1_agreement:.2%}")
    assert passing, pcc
    assert top1_agreement >= min_top1
