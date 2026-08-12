# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding orchestration for cody.

Pairs the MTP drafter (``models.demos.gemma4_cody.tt.assistant.model.Gemma4AssistantModel``)
with cody's target model verifier (the existing single-token or packed-decode
``_step_decode`` in ``server.py``).

Architecture
------------

For each step:

  1. Target completes a forward pass on the current token, producing logits
     AND exposing the last-layer hidden state + the deepest-of-each-layer-type
     K/V slices. (The hidden state + KV extraction is a separate change to
     cody's `Gemma4Model.forward` — see `INTEGRATION.md` adjacent to this file.)

  2. ``SpeculativeDecoder.propose(target_hidden, shared_kv, positions, T)``
     runs the drafter T times autoregressively, producing T candidate tokens
     per slot.

  3. Target runs its NEXT step using the drafter's tokens as input. (For
     greedy verification, target's outputs at each draft position tell us
     whether each draft was correct.)

  4. ``SpeculativeDecoder.verify(slot_idx, drafts, target_outputs)``
     finds the first mismatch per slot and reports the accepted prefix
     length.

  5. ``SpeculativeDecoder.commit(slot_idx, accepted_count)`` records the
     accepted tokens, advances cur_pos, and prepares state for the next
     step. Rejected drafts' KV entries must be invalidated by the caller
     (write-over on next step is the simplest path).

Per-slot state
--------------
A slot in this orchestrator mirrors a slot in cody's server: it has a
running token history, a current position, and (optionally) per-slot
draft buffer + acceptance stats.

What this file does NOT do
--------------------------
- Modify cody's ``_step_decode``. The orchestrator is standalone; integration
  is documented in INTEGRATION.md.
- Manage paged KV directly. That's the verifier's responsibility; this class
  only signals "first N drafts accepted" so the caller can advance cache
  positions accordingly.
- Implement packed-decode verification. The current verify() expects the
  target to produce per-position next-token predictions for each draft
  position — that's what packed decode gives. With single-token verify the
  caller runs the target T times in sequence (slower but works).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import ttnn
from models.demos.gemma4_cody.tt.assistant.config import Gemma4AssistantConfig
from models.demos.gemma4_cody.tt.assistant.model import Gemma4AssistantModel

_DEFAULT_NUM_DRAFTS = int(os.environ.get("GEMMA4_NUM_DRAFTS", "4"))


@dataclass
class _SlotState:
    """Per-slot speculative-decode state."""

    cur_pos: int = 0
    history: List[int] = field(default_factory=list)
    pending_drafts: List[int] = field(default_factory=list)
    n_proposed: int = 0
    n_accepted: int = 0
    n_steps_with_drafts: int = 0
    # Ring of recent per-draft outcomes (1=accepted, 0=rejected) — lets us see
    # acceptance *drift* across a generation (Item 3.2) rather than just the
    # cumulative mean, which masks late-context decay.
    recent_outcomes: Deque[int] = field(default_factory=lambda: deque(maxlen=256))

    def reset(self) -> None:
        self.cur_pos = 0
        self.history.clear()
        self.pending_drafts.clear()
        self.n_proposed = 0
        self.n_accepted = 0
        self.n_steps_with_drafts = 0
        self.recent_outcomes.clear()

    def acceptance_rate(self) -> float:
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    def record_outcomes(self, n_accepted: int, n_drafts: int) -> None:
        """Log this step's per-draft accept/reject into the recent-outcomes ring.

        The first ``n_accepted`` drafts matched; the rest (if any) did not.
        """
        for i in range(n_drafts):
            self.recent_outcomes.append(1 if i < n_accepted else 0)

    def windowed_acceptance(self) -> float:
        if not self.recent_outcomes:
            return 0.0
        return sum(self.recent_outcomes) / len(self.recent_outcomes)


class SpeculativeDecoder:
    """Orchestrates MTP drafter + target verifier for speculative decoding.

    Usage:

        spec = SpeculativeDecoder(mesh_device=mesh, num_slots=32, num_drafts=4)

        for step in loop:
            # 1) Target runs its forward pass on the current token. Caller
            # extracts target_last_hidden and per-layer-type shared_kv.
            target_hidden, shared_kv, positions = target_step(...)

            # 2) Drafter proposes T candidates per slot.
            drafts_per_slot = spec.propose(target_hidden, shared_kv, positions)

            # 3) Target verifies the drafts. The caller passes the per-position
            # target outputs back to verify() for each slot.
            for slot_idx, drafts in enumerate(drafts_per_slot):
                target_top1_per_position = target_verify_drafts(slot_idx, drafts)
                n_acc = spec.verify(slot_idx, drafts, target_top1_per_position)
                spec.commit(slot_idx, n_acc)

            # KV cache rollback for rejected drafts: caller invalidates
            # positions slot.cur_pos + n_acc .. slot.cur_pos + T - 1 in the
            # target's KV cache. Easiest path: those positions get overwritten
            # by the next step's writes since cur_pos rewinds.
    """

    def __init__(
        self,
        mesh_device,
        drafter_config: Gemma4AssistantConfig | None = None,
        drafter_cache_dir: str | None = None,
        num_slots: int = 32,
        num_drafts: int = _DEFAULT_NUM_DRAFTS,
        assistant_path: str = "/mnt/nas/gemma-assistant",
        mesh_config=None,
        ccl_manager=None,
    ):
        self.mesh_device = mesh_device
        self.num_slots = num_slots
        self.num_drafts = num_drafts

        if drafter_config is None:
            drafter_config = Gemma4AssistantConfig.from_hf_path(assistant_path)
        self.drafter_config = drafter_config

        # mesh_config / ccl_manager let the drafter's attention run column-
        # parallel (q heads sharded over TP) so its per-device SDPA matches
        # the target's per-device KV-cache shard, all-gathering before o_proj.
        self.drafter = Gemma4AssistantModel(
            mesh_device=mesh_device,
            config=drafter_config,
            cache_dir=drafter_cache_dir,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )

        self.slots: List[_SlotState] = [_SlotState() for _ in range(num_slots)]

    def reset_slot(self, slot_idx: int) -> None:
        """Reset a slot's state — call when a request finishes or a slot frees."""
        self.slots[slot_idx].reset()

    def propose(
        self,
        target_last_hidden,
        shared_kv: Dict[str, Tuple],
        cos_pos_full,
        sin_pos_full,
        cos_pos_sliding,
        sin_pos_sliding,
        cur_pos_tensor,
        page_table_full=None,
        page_table_sliding=None,
    ) -> List[List[int]]:
        """Run the drafter num_drafts times autoregressively per slot.

        Returns:
            ``[num_slots][num_drafts]`` of token IDs.

        The drafter shares state across the autoregressive loop: for draft 0
        we use the target's last hidden state. For drafts 1..T-1, the drafter
        uses its OWN previous hidden state (post_projection output) as the
        next input — same as how target_last_hidden is structured.

        NOTE: the second-input concat (the doubled-hidden-state convention
        the drafter's pre_projection requires) is currently not implemented
        for drafts 1..T-1 — we just pass the projected output of step n-1
        as the first half and zeros as the second half. This is one of the
        places the HF reference does something specific (likely the predicted
        token's embedding) that we'd want to mirror for best acceptance. For
        the proof-of-concept loop, the simpler pad-zero approach exercises
        the API end-to-end.
        """
        all_drafts: List[List[int]] = [[] for _ in range(self.num_slots)]

        current_hidden = target_last_hidden
        current_cos_full, current_sin_full = cos_pos_full, sin_pos_full
        current_cos_swa, current_sin_swa = cos_pos_sliding, sin_pos_sliding

        for draft_step in range(self.num_drafts):
            out_hidden, logits = self.drafter.forward(
                target_last_hidden=current_hidden,
                shared_kv=shared_kv,
                cos_pos_full=current_cos_full,
                sin_pos_full=current_sin_full,
                cos_pos_sliding=current_cos_swa,
                sin_pos_sliding=current_sin_swa,
                cur_pos_tensor=cur_pos_tensor,
                page_table_full=page_table_full,
                page_table_sliding=page_table_sliding,
            )

            # Argmax per slot from drafter's logits.
            # logits shape: [1, 1, B, vocab]
            logits_torch = self._read_logits(logits)  # [B, vocab]
            top1_per_slot = logits_torch.argmax(dim=-1).tolist()
            for slot_idx in range(self.num_slots):
                all_drafts[slot_idx].append(int(top1_per_slot[slot_idx]))

            # Prepare next step's input. The drafter's post_projection output is
            # [1, 1, B, backbone_hidden] = half the width pre_projection expects.
            # We pad with zeros for the second half. (See docstring NOTE.)
            current_hidden = self._pad_hidden_to_doubled(out_hidden)
            # cos/sin for next position would shift forward by one — caller
            # should ideally provide a sequence of positions. For POC, reuse
            # the same RoPE matrices.

        # Record proposals.
        for slot_idx in range(self.num_slots):
            self.slots[slot_idx].n_proposed += self.num_drafts
            self.slots[slot_idx].n_steps_with_drafts += 1
            self.slots[slot_idx].pending_drafts = list(all_drafts[slot_idx])

        return all_drafts

    def verify(
        self,
        slot_idx: int,
        draft_tokens: Sequence[int],
        target_top1_per_position: Sequence[int],
    ) -> int:
        """Find first mismatch between drafts and target's predictions.

        Args:
            slot_idx: which slot.
            draft_tokens: the T tokens this slot's drafter proposed.
            target_top1_per_position: target's argmax for each of the T
                draft positions. For greedy verification, this should be
                a list of length T (one prediction per draft position).

        Returns:
            Number of drafts accepted (0..T). +1 bonus token is the target's
            prediction at the (accepted+1)-th position, which always counts.
        """
        n_accepted = 0
        for i, draft_tok in enumerate(draft_tokens):
            if i >= len(target_top1_per_position):
                break
            if target_top1_per_position[i] == draft_tok:
                n_accepted += 1
            else:
                break

        self.slots[slot_idx].n_accepted += n_accepted
        self.slots[slot_idx].record_outcomes(n_accepted, len(draft_tokens))
        return n_accepted

    def commit(self, slot_idx: int, n_accepted: int, bonus: Optional[int] = None) -> None:
        """Record the accepted prefix (+ bonus token) in the slot's history.

        With ``bonus`` given (the packed-verify path), one convention and no
        caller-side fixup: ``cur_pos`` advances by ``n_accepted + 1`` (the
        accepted drafts plus the always-correct bonus) and ``history`` is
        extended with all ``n_accepted + 1`` tokens. The server owns the
        authoritative position bookkeeping (its ``_Slot.cur_pos``); this
        ``_SlotState`` mirror is for stats/inspection.

        With ``bonus`` omitted (the legacy ``test_speculative_loop.py`` path)
        only the accepted drafts are committed and the caller advances past
        the bonus itself.
        """
        slot = self.slots[slot_idx]
        accepted_tokens = slot.pending_drafts[:n_accepted]
        slot.history.extend(accepted_tokens)
        if bonus is not None:
            slot.history.append(int(bonus))
            slot.cur_pos += n_accepted + 1
        else:
            slot.cur_pos += n_accepted
        slot.pending_drafts = []

    def verify_sampled(self, slot_idx, draft_tokens, target_logits, proposal_logits):
        """Speculative rejection sampling for non-greedy requests — NOT YET
        IMPLEMENTED (greedy-first milestone; sampled requests currently route
        through the server's exact single-token decode path).

        The algorithm, for follow-on work: with ``p_i`` = softmax of the
        target logits at packed position i (under the request's temperature /
        top-p) and ``q_i`` = the drafter's proposal distribution, accept draft
        ``d_i`` with probability ``min(1, p_i(d_i) / q_i(d_i))``; on the first
        rejection sample the bonus from the normalized residual
        ``(p_i - q_i)_+``; on full acceptance sample the bonus from ``p_T``.
        This keeps the output distribution exact.
        """
        raise NotImplementedError("speculative rejection sampling — greedy-first milestone")

    def per_slot_stats(self, slot_idx: int) -> dict:
        slot = self.slots[slot_idx]
        return {
            "cur_pos": slot.cur_pos,
            "history_len": len(slot.history),
            "n_proposed": slot.n_proposed,
            "n_accepted": slot.n_accepted,
            "acceptance_rate": slot.acceptance_rate(),
            "windowed_acceptance": slot.windowed_acceptance(),
            "n_steps_with_drafts": slot.n_steps_with_drafts,
        }

    def aggregate_stats(self) -> dict:
        total_prop = sum(s.n_proposed for s in self.slots)
        total_acc = sum(s.n_accepted for s in self.slots)
        total_steps = sum(s.n_steps_with_drafts for s in self.slots)
        # Windowed acceptance across all slots' recent-outcome rings — surfaces
        # late-context drift that the cumulative rate hides (Item 3.2).
        recent = [o for s in self.slots for o in s.recent_outcomes]
        return {
            "total_proposed": total_prop,
            "total_accepted": total_acc,
            "total_steps": total_steps,
            # acceptance_rate = accepted / proposed (low pre-Item-1: only draft 0
            # is verified out of T proposed). mean_accepted_per_step = accepted /
            # steps — pre-Item-1 this is the draft-0 acceptance (~0.25); post
            # packed-verify it becomes mean accepted tokens per step (>1).
            "acceptance_rate": (total_acc / total_prop) if total_prop > 0 else 0.0,
            "mean_accepted_per_step": (total_acc / total_steps) if total_steps > 0 else 0.0,
            # Each packed-verify step emits n_accepted drafts + 1 bonus token.
            # Summed over steps that is total_acc + total_steps — so this is
            # the real throughput multiplier vs. one token / single-token step.
            "mean_tokens_per_step": ((total_acc + total_steps) / total_steps) if total_steps > 0 else 0.0,
            "windowed_acceptance": (sum(recent) / len(recent)) if recent else 0.0,
            "windowed_n": len(recent),
            "active_slots": sum(1 for s in self.slots if s.n_proposed > 0),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _read_logits(self, logits_tt) -> "torch.Tensor":
        """Pull logits to host as torch tensor. Returns shape [B, vocab].

        The drafter lm-head is column-parallel, so on a mesh the logits are
        TP-sharded over the vocab dim — concat the per-device shards (unless
        they are already full-vocab, i.e. replicated on a tp=1 mesh).
        """
        import torch

        is_mesh = hasattr(self.mesh_device, "shape") and self.mesh_device.get_num_devices() > 1
        if is_mesh:
            shards = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(logits_tt)]
            if shards[0].shape[-1] == self.drafter_config.vocab_size:
                t = shards[0]
            else:
                t = torch.cat(shards, dim=-1)
        else:
            t = ttnn.to_torch(logits_tt).float()
        # Shape is [1, 1, B, vocab] — squeeze leading dims.
        return t[0, 0]

    def _pad_hidden_to_doubled(self, hidden_tt):
        """Pad the drafter's [1, 1, B, backbone_hidden] output to [1, 1, B, 2*backbone_hidden].

        The drafter's pre_projection expects 2*backbone_hidden width. The first
        half is the target's last-layer hidden state (or the drafter's previous
        post_projection output, here). The second half is — per HF reference —
        some other slice of state. For this POC we pad with zeros.

        TODO: figure out what the second half is supposed to be (likely the
        next-token embedding from embed_tokens). When done, the drafter's
        proposals 1..T-1 will track the HF reference more closely.
        """
        import torch

        is_mesh = hasattr(self.mesh_device, "shape") and self.mesh_device.get_num_devices() > 1
        # Read out, double, write back.
        if is_mesh:
            h = ttnn.to_torch(ttnn.get_device_tensors(hidden_tt)[0])
            replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        else:
            h = ttnn.to_torch(hidden_tt)
            replicate = None
        zeros = torch.zeros_like(h)
        doubled = torch.cat([h, zeros], dim=-1).to(torch.bfloat16)
        return ttnn.from_torch(
            doubled,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
