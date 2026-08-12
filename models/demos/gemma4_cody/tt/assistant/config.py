# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drafter (Gemma4Assistant) configuration extraction from the HF config.json.

The drafter is structurally similar to a 4-layer Gemma4 with two notable
differences:

  1. Q-only attention. The drafter has no K/V projections; it consumes the
     TARGET model's per-layer-type KV cache via `shared_kv_states`.
  2. Per-layer-type head dims. Layers 0..2 are sliding (head_dim=256);
     layer 3 is full-attention (head_dim=512). Same Q-output width
     (`num_attention_heads * head_dim`) — fewer heads at full-attention
     layers, same per-head compute width.

Plus two top-level projections that bridge the drafter's residual stream
to the target's:

  - pre_projection: [2 * backbone_hidden_size, hidden_size]  ←  target → drafter
  - post_projection: [hidden_size, backbone_hidden_size]    ←  drafter → target
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Gemma4AssistantConfig:
    """Subset of HF Gemma4AssistantConfig we need on the TT side."""

    # Top-level
    backbone_hidden_size: int
    num_centroids: int  # for use_ordered_embeddings (currently false on 26B-A4B drafter)
    use_ordered_embeddings: bool

    # text_config — drafter's own residual stream
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int  # for sliding layers
    num_global_key_value_heads: int  # for full-attention layers
    head_dim: int  # sliding
    global_head_dim: int  # full
    num_hidden_layers: int
    layer_types: List[str]  # ["sliding_attention"] * 3 + ["full_attention"]
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool

    # RoPE params per layer type. Each dict has the rope settings (theta,
    # type, partial_rotary_factor).
    rope_full: dict
    rope_sliding: dict

    @classmethod
    def from_hf_path(cls, path: str | Path) -> "Gemma4AssistantConfig":
        cfg = json.loads(Path(path, "config.json").read_text())
        tc = cfg["text_config"]
        rope = tc.get("rope_parameters", {})
        return cls(
            backbone_hidden_size=cfg["backbone_hidden_size"],
            num_centroids=cfg.get("num_centroids", 0),
            use_ordered_embeddings=cfg.get("use_ordered_embeddings", False),
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            num_global_key_value_heads=tc.get("num_global_key_value_heads", tc["num_key_value_heads"]),
            head_dim=tc["head_dim"],
            global_head_dim=tc.get("global_head_dim", tc["head_dim"]),
            num_hidden_layers=tc["num_hidden_layers"],
            layer_types=list(tc["layer_types"]),
            sliding_window=tc.get("sliding_window", 1024),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            vocab_size=tc["vocab_size"],
            max_position_embeddings=tc.get("max_position_embeddings", 4096),
            tie_word_embeddings=tc.get("tie_word_embeddings", True),
            rope_full=rope.get("full_attention", {}),
            rope_sliding=rope.get("sliding_attention", {}),
        )

    def layer_head_dim(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] == "full_attention":
            return self.global_head_dim
        return self.head_dim

    def layer_num_kv_heads(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] == "full_attention":
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    def layer_q_dim(self, layer_idx: int) -> int:
        """Output width of q_proj for layer ``layer_idx``."""
        return self.num_attention_heads * self.layer_head_dim(layer_idx)
