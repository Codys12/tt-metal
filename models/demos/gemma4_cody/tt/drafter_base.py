# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drafter base interface shared by the MTP assistant and DFlash drafters.

The server picks a concrete `Drafter` at startup; the rest of the pipeline talks
to the abstract API. Each drafter declares — via `required_target_outputs` —
what target-side tensors it needs per step (which hidden-state layers to tap,
whether it shares the target's KV cache, etc.), and then implements `propose()`
to emit T draft tokens per call.

Decoupling target outputs from drafter inputs lets us host two very different
drafters in one server:

  * The Gemma-4 MTP assistant (4 Gemma-shaped layers, Q-only attention reading
    the target's KV, autoregressive over T steps internally).
  * DFlash (5 Llama-shaped layers, full Q/K/V with its own KV, single-shot
    block-diffusion draft of `block_size` tokens, conditioning on 5 target
    hidden states fused through `fc`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RequiredTargetOutputs:
    """Static contract — what the drafter needs the target to expose.

    The server reads this once at startup to plan target hooks and per-step
    plumbing. Two drafters with different requirements coexist without the
    target needing to know about either of them.

    Attributes
    ----------
    aux_hidden_layers:
        Indices of target layers whose hidden states the drafter wants. ``-1``
        is shorthand for "last layer" (used by the current MTP drafter).
        DFlash declares e.g. ``(1, 17, 29, 47, 58)``.
    shared_kv_layer_types:
        Layer-types of the target whose K/V the drafter reads directly (Q-only
        attention). The MTP drafter declares ``("full_attention",
        "sliding_attention")`` — it reads the target's deepest layer of each
        type. DFlash declares ``()`` — it owns its KV.
    needs_scaled_embedding / needs_next_token_embedding:
        Whether the target must surface the Gemma-scaled embedding /
        next-token embedding alongside the hidden state. MTP needs the
        concatenated ``(hidden, next_emb)`` doubled-width input; DFlash
        embeds the mask token internally.
    block_size:
        Number of draft tokens the drafter emits per `propose()` call.
        MTP = 1 (the server loops T times); DFlash = 8 (single shot).
    """

    aux_hidden_layers: Tuple[int, ...] = (-1,)
    shared_kv_layer_types: Tuple[str, ...] = ()
    needs_scaled_embedding: bool = False
    needs_next_token_embedding: bool = False
    block_size: int = 1


@dataclass
class TargetOutputs:
    """Runtime payload the target produces for the drafter on each decode step.

    Field population is gated by the drafter's `RequiredTargetOutputs`: the
    server only fills what the drafter declared it needs. Extra fields are
    present for drafter convenience (RoPE caches, page tables) but those are
    drafter-supplied / pre-built, not target-produced.
    """

    aux_hidden: Dict[int, "ttnn.Tensor"] = field(default_factory=dict)
    shared_kv: Dict[str, Tuple["ttnn.Tensor", "ttnn.Tensor"]] = field(default_factory=dict)
    cur_pos_tensor: Optional["ttnn.Tensor"] = None
    cur_pos_tensor_sliding: Optional["ttnn.Tensor"] = None
    cos_pos_full: Optional["ttnn.Tensor"] = None
    sin_pos_full: Optional["ttnn.Tensor"] = None
    cos_pos_sliding: Optional["ttnn.Tensor"] = None
    sin_pos_sliding: Optional["ttnn.Tensor"] = None
    page_table_full: Optional["ttnn.Tensor"] = None
    page_table_sliding: Optional["ttnn.Tensor"] = None
    next_token_embedding: Optional["ttnn.Tensor"] = None


@dataclass
class DrafterOutput:
    """Result of one `propose()` call.

    Attributes
    ----------
    drafts:
        Proposed token ids in the **target's vocabulary space**. Shape
        ``[B, T]`` int32 where T = `RequiredTargetOutputs.block_size`. Drafters
        with a narrower internal vocab (e.g. DFlash's 32k subset) must remap
        before returning.
    draft_hidden:
        Optional — the drafter's last hidden state, fed back on the next step
        by drafters that condition on their own prior output (MTP). Shape
        ``[1, 1, B, drafter_hidden_size]``.
    draft_logits:
        Optional — full logits for the proposed positions, exposed for parity
        tests and for the rejection-sampling verify path (deferred). Shape
        ``[1, 1, B*T, vocab_size]``.
    """

    drafts: "ttnn.Tensor"
    draft_hidden: Optional["ttnn.Tensor"] = None
    draft_logits: Optional["ttnn.Tensor"] = None


class Drafter(ABC):
    """Abstract drafter. Concrete implementations live in `tt/assistant/` and
    `tt/dflash/`.

    Lifecycle:
        1. Server constructs the drafter once at startup (weights loaded into
           replicated mesh tensors).
        2. Server reads `required_target_outputs` once and plans target hooks.
        3. Per decode step the server fills a `TargetOutputs` and calls
           `propose()`, then verifies the returned drafts against a target
           forward.
    """

    @property
    @abstractmethod
    def required_target_outputs(self) -> RequiredTargetOutputs:
        ...

    @abstractmethod
    def propose(self, target_outputs: TargetOutputs) -> DrafterOutput:
        ...
