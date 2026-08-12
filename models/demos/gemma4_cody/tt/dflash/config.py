# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter configuration parsed from the HF ``config.json``.

Two checkpoint *formats* are supported, auto-detected by config shape:

speculators (``variant == "speculators"``)
    e.g. ``RedHatAI/gemma-4-31B-it-speculator.dflash`` (``DFLASH_PATH=
    /mnt/nas/dflash-gemma``). Two top-level groupings:

      * top-level DFlash fields — ``block_size``, ``draft_vocab_size``,
        ``mask_token_id``, ``aux_hidden_state_layer_ids``, ``max_anchors``,
        ``target_hidden_size`` (``None`` means "match the target");
      * ``transformer_layer_config`` — the per-layer transformer params
        (Llama-shaped, ``model_type: llama``).

    Owns its own narrow ``lm_head`` (``draft_vocab_size`` rows) + ``d2t`` /
    ``t2d`` vocab remap; ``block_size == 8``; head_dim 256, 32/16 heads.

z-lab native (``variant == "zlab"``)
    e.g. ``z-lab/gemma-4-31B-it-DFlash`` (``DFLASH_PATH=
    /mnt/nas/zlab-dflash-gemma``). A *flat* Qwen3 config with a nested
    ``dflash_config`` (``mask_token_id``, ``target_layer_ids``) and
    ``architectures: ["DFlashDraftModel"]``. Reference modeling code:
    ``/srv/nas/spec-decoding/model/dflash.py``.

    ``tie_word_embeddings: true`` ⇒ **no** own ``embed_tokens`` / ``lm_head`` /
    ``d2t`` in the checkpoint: the drafter reuses the *target's* embedding for
    noise and the *target's* ``lm_head`` for draft logits over the FULL target
    vocab (no remap). ``block_size == 16``; head_dim 128, 64/8 heads; 6 aux
    taps; ``rope_theta == 1e6``; sliding-window layer_types.

Aux-layer offset note
---------------------

The two formats index aux taps differently relative to the TT target's tap
point (``model.py`` clones the **output** of layer ``i`` when ``i`` is in
``configure_aux_taps``):

  * speculators — vLLM ``update_dflash()`` applies ``[i - 1 for i in
    aux_hidden_state_layer_ids]`` (DFlash trains against pre-attention
    residuals), so the stored ids are shifted ``-1`` to become TT
    layer-output indices: ``[1,17,29,47,58] → [0,16,28,46,57]``.
  * z-lab — ``extract_context_feature`` taps ``hidden_states[layer_id + 1]``
    (the HF output-hidden-states tuple, index 0 == embeddings), i.e. the
    **output of layer ``layer_id``** directly, so NO shift:
    ``[1,12,23,35,46,57]`` are already TT layer-output indices.

:attr:`aux_hidden_layers_raw` keeps the as-written ids; :attr:`aux_hidden_layers`
is post-shift (what the server hooks). Override the per-variant shift with
``GEMMA4_DFLASH_AUX_SHIFT`` (int) when tuning acceptance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DFlashConfig:
    # Which checkpoint format this came from: "speculators" | "zlab".
    variant: str

    # Top-level DFlash fields
    block_size: int
    draft_vocab_size: int  # rows of the lm_head used for drafts; == vocab_size when tied.
    mask_token_id: int
    aux_hidden_layers_raw: Tuple[int, ...]
    aux_hidden_layers: Tuple[int, ...]
    max_anchors: int
    target_hidden_size: int  # resolved: explicit or inherited from drafter hidden_size
    tie_word_embeddings: bool

    # When True the checkpoint ships NO own lm_head — draft logits come from the
    # TARGET's lm_head over the full target vocab, and there is no d2t remap
    # (draft ids ARE target ids). Set for the z-lab (tied-embedding) variant.
    uses_target_head: bool

    # transformer_layer_config — Llama/Qwen3-style decoder block
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    rms_norm_eps: float
    hidden_act: str  # "silu"
    max_position_embeddings: int
    vocab_size: int  # target's full vocab — for remap-table sizing / tied head
    attention_bias: bool
    mlp_bias: bool

    # RoPE
    rope_theta: float
    rope_type: str

    # Gemma/Qwen3 extras (z-lab). Carried for completeness; greedy drafting does
    # not need softcapping (monotonic ⇒ argmax-invariant), and sliding_window is
    # NOT yet enforced in the TT propose SDPA (see model.py / dflash_spec.py).
    final_logit_softcapping: float = 0.0
    sliding_window: int = 0
    layer_types: Tuple[str, ...] = field(default_factory=tuple)

    # ── format-dispatching loader ─────────────────────────────────────────────

    @classmethod
    def from_hf_path(cls, path: str | Path) -> "DFlashConfig":
        cfg = json.loads(Path(path, "config.json").read_text())
        if "transformer_layer_config" in cfg:
            return cls._from_speculators(cfg)
        return cls._from_zlab(cfg)

    @staticmethod
    def _aux_shift(default: int) -> int:
        env = os.environ.get("GEMMA4_DFLASH_AUX_SHIFT")
        return int(env) if env is not None and env != "" else default

    @classmethod
    def _from_speculators(cls, cfg: dict) -> "DFlashConfig":
        tlc = cfg["transformer_layer_config"]
        rope = tlc.get("rope_parameters", {})
        raw_aux = tuple(cfg["aux_hidden_state_layer_ids"])
        shift = cls._aux_shift(-1)
        target_hidden_size = cfg.get("target_hidden_size") or tlc["hidden_size"]
        return cls(
            variant="speculators",
            block_size=cfg["block_size"],
            draft_vocab_size=cfg["draft_vocab_size"],
            mask_token_id=cfg["mask_token_id"],
            aux_hidden_layers_raw=raw_aux,
            aux_hidden_layers=tuple(i + shift for i in raw_aux),
            max_anchors=cfg.get("max_anchors", 256),
            target_hidden_size=target_hidden_size,
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            uses_target_head=False,
            hidden_size=tlc["hidden_size"],
            intermediate_size=tlc["intermediate_size"],
            num_attention_heads=tlc["num_attention_heads"],
            num_key_value_heads=tlc["num_key_value_heads"],
            head_dim=tlc["head_dim"],
            num_hidden_layers=tlc["num_hidden_layers"],
            rms_norm_eps=tlc.get("rms_norm_eps", 1e-6),
            hidden_act=tlc.get("hidden_act", "silu"),
            max_position_embeddings=tlc.get("max_position_embeddings", 4096),
            vocab_size=tlc["vocab_size"],
            attention_bias=tlc.get("attention_bias", False),
            mlp_bias=tlc.get("mlp_bias", False),
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            rope_type=rope.get("rope_type", "default"),
            final_logit_softcapping=float(tlc.get("final_logit_softcapping") or 0.0),
            sliding_window=int(tlc.get("sliding_window") or 0),
            layer_types=tuple(tlc.get("layer_types") or ()),
        )

    @classmethod
    def _from_zlab(cls, cfg: dict) -> "DFlashConfig":
        """z-lab native ``DFlashDraftModel`` — flat Qwen3 config + ``dflash_config``.

        The drafter is a Qwen3 decoder stack; DFlash-specific fields live under
        ``dflash_config`` (``mask_token_id``, ``target_layer_ids``). Tied word
        embeddings ⇒ no own embed/lm_head/d2t (see module docstring).
        """
        dfc = cfg.get("dflash_config", {})
        raw_aux = tuple(dfc["target_layer_ids"])
        shift = cls._aux_shift(0)
        hidden_size = cfg["hidden_size"]
        head_dim = cfg.get("head_dim", hidden_size // cfg["num_attention_heads"])
        vocab_size = cfg["vocab_size"]
        # No max_anchors in the z-lab config (a training-time field). Default to
        # the speculators value (3072) so the inference anchor cache is usable;
        # GEMMA4_DFLASH_MAX_ANCHORS can override at the server.
        return cls(
            variant="zlab",
            block_size=cfg["block_size"],
            # Tied head ⇒ draft logits span the full target vocab; "draft ids"
            # are target ids (no d2t). draft_vocab_size == vocab_size drives the
            # per-shard topk reconstruction in dflash_spec.read_drafts.
            draft_vocab_size=vocab_size,
            mask_token_id=dfc["mask_token_id"],
            aux_hidden_layers_raw=raw_aux,
            aux_hidden_layers=tuple(i + shift for i in raw_aux),
            max_anchors=cfg.get("max_anchors", 3072),
            target_hidden_size=hidden_size,
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            uses_target_head=True,
            hidden_size=hidden_size,
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=head_dim,
            num_hidden_layers=cfg["num_hidden_layers"],
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            hidden_act=cfg.get("hidden_act", "silu"),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            vocab_size=vocab_size,
            attention_bias=cfg.get("attention_bias", False),
            mlp_bias=cfg.get("mlp_bias", False),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_type=(cfg.get("rope_scaling") or {}).get("rope_type", "default")
            if isinstance(cfg.get("rope_scaling"), dict)
            else "default",
            final_logit_softcapping=float(cfg.get("final_logit_softcapping") or 0.0),
            sliding_window=int(cfg.get("sliding_window") or 0),
            layer_types=tuple(cfg.get("layer_types") or ()),
        )

    @property
    def num_aux_layers(self) -> int:
        return len(self.aux_hidden_layers)

    @property
    def fc_in_features(self) -> int:
        """`fc` projects concat(K aux hiddens) → hidden_size."""
        return self.num_aux_layers * self.target_hidden_size

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim
