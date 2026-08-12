# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Construction helper for the single-device Muse Glimmer text model."""

from __future__ import annotations

import os

import ttnn
from models.demos.muse_glimmer.tt.model import MuseGlimmerModel
from models.demos.muse_glimmer.tt.model_config import MuseGlimmerModelArgs

DEFAULT_MODEL_PATH = (
    "/home/user/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/"
    "snapshots/f84ecc3a0ea984a4c04542a84269e3d065350a6e"
)


def create_tt_model(
    mesh_device,
    max_seq_len=2048,
    model_path=None,
):
    if mesh_device.get_num_devices() != 1:
        raise ValueError(f"Muse Glimmer supports one device only, got {mesh_device.get_num_devices()}")

    model_path = model_path or os.getenv("HF_MODEL") or os.getenv("MUSE_GLIMMER_MODEL_PATH") or DEFAULT_MODEL_PATH
    hf_config = MuseGlimmerModelArgs.load_hf_config(model_path)
    model_args = MuseGlimmerModelArgs.from_hf_config(hf_config)
    model_args._hf_text_config = hf_config.text_config
    state_dict = MuseGlimmerModelArgs.load_state_dict(model_path)

    model = MuseGlimmerModel(
        device=mesh_device,
        hf_config=model_args,
        state_dict=state_dict,
        attention_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat4_b,
        lm_head_dtype=ttnn.bfloat8_b,
        tensor_cache_path=str(model_args.weight_cache_path(model_path, ttnn.bfloat8_b)),
        mlp_cache_path=str(model_args.weight_cache_path(model_path, ttnn.bfloat4_b)),
        max_seq_len=max_seq_len,
        create_kv_cache=True,
        kv_cache_dtype=ttnn.bfloat16,
        load_qkv_prefetch=os.getenv("MUSE_QKV_PREFETCH", "0") == "1",
    )
    return model_args, model, model.tt_kv_cache, state_dict
