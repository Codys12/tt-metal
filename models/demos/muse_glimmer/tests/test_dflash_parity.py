# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.muse_glimmer_assistant.configuration_muse_glimmer_assistant import MuseGlimmerAssistantConfig
from transformers.models.muse_glimmer_assistant.modeling_muse_glimmer_assistant import MuseGlimmerAssistantModel

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer.tt.dflash import DFlashConfig, DFlashDrafter
from models.demos.muse_glimmer.tt.dflash.rope import build_rope_cache


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.DISABLED}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "attention_dtype,mlp_dtype,fc_dtype,min_pcc",
    [
        (ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, 0.9999),
        (ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat8_b, 0.9998),
    ],
    ids=["bf16-reference", "production-mixed"],
)
def test_native_dflash_decode_pcc(mesh_device, tmp_path, attention_dtype, mlp_dtype, fc_dtype, min_pcc):
    torch.manual_seed(0)
    ref_config = MuseGlimmerAssistantConfig(
        block_size=16,
        mask_token_id=318,
        target_layer_ids=[0, 1],
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=2,
        max_position_embeddings=128,
        sliding_window=128,
        layer_types=["sliding_attention", "sliding_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    reference = MuseGlimmerAssistantModel(ref_config).eval().to(torch.bfloat16)
    save_file({k: v.contiguous() for k, v in reference.state_dict().items()}, tmp_path / "model.safetensors")
    config = DFlashConfig(
        block_size=16,
        mask_token_id=318,
        target_layer_ids=(0, 1),
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        max_position_embeddings=128,
        rope_theta=500000.0,
        rope_type="default",
        sliding_window=128,
        layer_types=("sliding_attention", "sliding_attention"),
        target_vocab_size=320,
    )

    context = torch.randn(1, 17, config.fc_in_features, dtype=torch.bfloat16)
    noise = torch.randn(1, config.block_size, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(context.shape[1] + config.block_size).unsqueeze(0)
    with torch.no_grad():
        expected_hidden = reference(
            noise_embeds=noise,
            context_hidden_states=context,
            position_ids=positions,
        ).last_hidden_state
        output_head = torch.nn.Linear(config.hidden_size, config.target_vocab_size, bias=False).to(torch.bfloat16)
        expected_logits = output_head(expected_hidden[:, 1:])

    target_lm_head = ttnn.from_torch(
        output_head.weight.transpose(-2, -1).unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    drafter = DFlashDrafter(
        mesh_device=mesh_device,
        config=config,
        safetensors_dir=str(tmp_path),
        cache_dir=tmp_path / "tt_cache",
        target_lm_head=target_lm_head,
        attention_dtype=attention_dtype,
        mlp_dtype=mlp_dtype,
        fc_dtype=fc_dtype,
    )

    def to_tt(tensor):
        return ttnn.from_torch(tensor.unsqueeze(0), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    cos, sin = build_rope_cache(positions, config.head_dim, config.rope_theta)
    cache = drafter.init_anchor_cache()
    drafter.append_anchors(cache, to_tt(context))
    decode_logits_tt = drafter.decode_step(
        cache,
        to_tt(noise),
        to_tt(cos),
        to_tt(sin),
    )
    decode_logits = ttnn.to_torch(ttnn.get_device_tensors(decode_logits_tt)[0]).squeeze(0).squeeze(0)
    decode_pass, decode_pcc = comp_pcc(expected_logits.squeeze(0).float(), decode_logits.float(), min_pcc)
    print(f"native DFlash cached-decode logits PCC: {float(decode_pcc):.6f}")
    assert decode_pass, decode_pcc

    fixed_cache = drafter.alloc_fixed_anchor_caches(64)
    valid, absolute_start = drafter.prepare_fixed_anchor_caches(cache, fixed_cache, max_new_tokens=16)
    assert (valid, absolute_start) == (17, 0)
    fixed_positions = torch.arange(64).unsqueeze(0)
    fixed_cos, fixed_sin = build_rope_cache(fixed_positions, config.head_dim, config.rope_theta)
    fixed_cos_tt = to_tt(fixed_cos)
    fixed_sin_tt = to_tt(fixed_sin)
    drafter.rotate_fixed_anchor_caches(fixed_cache, fixed_cos_tt, fixed_sin_tt)
    noise_cos = cos[:, context.shape[1] :]
    noise_sin = sin[:, context.shape[1] :]
    write_control_host = torch.zeros(1, 32, dtype=torch.int32)
    write_control_host[0, :2] = torch.tensor([valid, config.block_size], dtype=torch.int32)
    write_control = ttnn.from_torch(
        write_control_host,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    mask = torch.full((1, 1, config.num_attention_heads * config.block_size, 64), -1e9)
    mask[..., : valid + config.block_size] = 0
    fixed_logits_tt = drafter.fixed_propose_forward(
        to_tt(noise),
        fixed_cache,
        write_control,
        to_tt(noise_cos),
        to_tt(noise_sin),
        to_tt(mask.to(torch.bfloat16).squeeze(0)),
    )
    fixed_logits = ttnn.to_torch(ttnn.get_device_tensors(fixed_logits_tt)[0]).squeeze(0).squeeze(0)
    fixed_pass, fixed_pcc = comp_pcc(expected_logits.squeeze(0).float(), fixed_logits.float(), min_pcc)
    print(f"fixed-cache DFlash logits PCC: {float(fixed_pcc):.6f}")
    assert fixed_pass, fixed_pcc

    # Exercise the server's repeated propose -> partial commit -> propose
    # transition. The second proposal overwrites rejected noise rows while a
    # traced append writes only the committed prefix into the fixed cache.
    committed = 7
    new_context = torch.randn(1, committed, config.fc_in_features, dtype=torch.bfloat16)
    drafter.append_anchors(cache, to_tt(new_context))
    padded_context = torch.nn.functional.pad(new_context, (0, 0, 0, config.block_size - committed))
    append_control_host = torch.zeros(1, 32, dtype=torch.int32)
    append_control_host[0, :2] = torch.tensor([valid, committed], dtype=torch.int32)
    append_control = ttnn.from_torch(
        append_control_host,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    drafter.fixed_append_forward(
        fixed_cache,
        to_tt(padded_context),
        append_control,
        to_tt(cos[:, context.shape[1] : context.shape[1] + config.block_size]),
        to_tt(sin[:, context.shape[1] : context.shape[1] + config.block_size]),
    )

    noise_2 = torch.randn(1, config.block_size, config.hidden_size, dtype=torch.bfloat16)
    context_2 = torch.cat([context, new_context], dim=1)
    positions_2 = torch.arange(context_2.shape[1] + config.block_size).unsqueeze(0)
    with torch.no_grad():
        expected_hidden_2 = reference(
            noise_embeds=noise_2,
            context_hidden_states=context_2,
            position_ids=positions_2,
        ).last_hidden_state
        expected_logits_2 = output_head(expected_hidden_2[:, 1:])
    cos_2, sin_2 = build_rope_cache(positions_2, config.head_dim, config.rope_theta)
    decode_logits_2_tt = drafter.decode_step(cache, to_tt(noise_2), to_tt(cos_2), to_tt(sin_2))
    decode_logits_2 = ttnn.to_torch(ttnn.get_device_tensors(decode_logits_2_tt)[0]).squeeze(0).squeeze(0)
    decode_2_pass, decode_2_pcc = comp_pcc(expected_logits_2.squeeze(0).float(), decode_logits_2.float(), min_pcc)
    print(f"second cached-decode logits PCC: {float(decode_2_pcc):.6f}")
    assert decode_2_pass, decode_2_pcc

    valid_2 = valid + committed
    write_control_2_host = torch.zeros(1, 32, dtype=torch.int32)
    write_control_2_host[0, :2] = torch.tensor([valid_2, config.block_size], dtype=torch.int32)
    write_control_2 = ttnn.from_torch(
        write_control_2_host,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    mask_2 = torch.full((1, 1, config.num_attention_heads * config.block_size, 64), -1e9)
    mask_2[..., : valid_2 + config.block_size] = 0
    fixed_logits_2_tt = drafter.fixed_propose_forward(
        to_tt(noise_2),
        fixed_cache,
        write_control_2,
        to_tt(cos_2[:, context_2.shape[1] :]),
        to_tt(sin_2[:, context_2.shape[1] :]),
        to_tt(mask_2.to(torch.bfloat16).squeeze(0)),
    )
    fixed_logits_2 = ttnn.to_torch(ttnn.get_device_tensors(fixed_logits_2_tt)[0]).squeeze(0).squeeze(0)
    fixed_2_pass, fixed_2_pcc = comp_pcc(expected_logits_2.squeeze(0).float(), fixed_logits_2.float(), min_pcc)
    print(f"second fixed-cache DFlash logits PCC: {float(fixed_2_pcc):.6f}")
    assert fixed_2_pass, fixed_2_pcc
