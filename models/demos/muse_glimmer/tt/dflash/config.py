# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the native Muse Glimmer DFlash assistant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DFlashConfig:
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    rms_norm_eps: float
    hidden_act: str
    max_position_embeddings: int
    rope_theta: float
    rope_type: str
    sliding_window: int
    layer_types: tuple[str, ...]
    target_vocab_size: int = 202048

    @classmethod
    def from_hf_path(cls, path: str | Path) -> "DFlashConfig":
        raw = json.loads(Path(path, "config.json").read_text())
        if raw.get("model_type") != "muse_glimmer_assistant":
            raise ValueError(f"Expected muse_glimmer_assistant, got {raw.get('model_type')!r}")
        rope = raw.get("rope_parameters") or {}
        return cls(
            block_size=raw["block_size"],
            mask_token_id=raw["mask_token_id"],
            target_layer_ids=tuple(raw["target_layer_ids"]),
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            num_hidden_layers=raw["num_hidden_layers"],
            rms_norm_eps=raw["rms_norm_eps"],
            hidden_act=raw["hidden_act"],
            max_position_embeddings=raw["max_position_embeddings"],
            rope_theta=float(rope["rope_theta"]),
            rope_type=rope.get("rope_type", "default"),
            sliding_window=raw["sliding_window"],
            layer_types=tuple(raw["layer_types"]),
        )

    @property
    def num_aux_layers(self) -> int:
        return len(self.target_layer_ids)

    @property
    def fc_in_features(self) -> int:
        return self.num_aux_layers * self.hidden_size
