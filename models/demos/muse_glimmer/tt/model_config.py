# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and lazy checkpoint loading for Muse Glimmer 30B."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig

import ttnn


@dataclass
class MuseGlimmerModelArgs:
    """Text-only Muse Glimmer parameters used by the TT implementation."""

    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    sliding_window: int = 2048
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    layer_types: tuple[str, ...] = ()
    layer_rope_theta: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = tuple(
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            )
        if not self.layer_rope_theta:
            self.layer_rope_theta = tuple(
                0.0 if layer_type == "full_attention" else self.rope_theta for layer_type in self.layer_types
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per decoder layer")
        if len(self.layer_rope_theta) != self.num_hidden_layers:
            raise ValueError("layer_rope_theta must contain one entry per decoder layer")

    @classmethod
    def from_hf_config(cls, hf_config) -> "MuseGlimmerModelArgs":
        text = getattr(hf_config, "text_config", hf_config)
        rope_parameters = getattr(text, "rope_parameters", {}) or {}
        return cls(
            hidden_size=text.hidden_size,
            intermediate_size=text.intermediate_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            vocab_size=text.vocab_size,
            sliding_window=text.sliding_window,
            max_position_embeddings=text.max_position_embeddings,
            rope_theta=float(rope_parameters.get("rope_theta", 500000.0)),
            rms_norm_eps=text.rms_norm_eps,
            post_norm_eps=text.post_norm_eps,
            qk_scale_factor=text.qk_scale_factor,
            output_multiplier=text.output_multiplier,
            final_logit_softcapping=text.final_logit_softcapping,
            layer_types=tuple(text.layer_types),
            layer_rope_theta=tuple(float(theta) for theta in text.layer_rope_theta),
        )

    @staticmethod
    def load_hf_config(model_path: str | Path):
        return AutoConfig.from_pretrained(model_path, trust_remote_code=False)

    @staticmethod
    def load_state_dict(weights_path: str | Path):
        from models.demos.muse_glimmer.utils.lazy_state_dict import LazyStateDict

        return LazyStateDict(weights_path)

    def weight_cache_path(self, model_path: str | Path, dtype) -> Path:
        root = Path(os.environ.get("TT_CACHE_PATH", model_path))
        dtype_name = {
            ttnn.bfloat16: "bf16",
            ttnn.bfloat8_b: "bfp8",
            ttnn.bfloat4_b: "bfp4",
        }[dtype]
        path = root / f"tensor_cache_muse_glimmer_single_device_{dtype_name}"
        path.mkdir(parents=True, exist_ok=True)
        return path
