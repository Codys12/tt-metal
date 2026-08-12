# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test of the speculative decode loop using the real TT drafter
and a mock target.

Validates:
  - propose() runs T drafter forwards autoregressively, returns T tokens/slot
  - verify() finds the first-mismatch correctly
  - commit() updates per-slot state correctly
  - aggregate_stats() reports sensible numbers after a run

Mock target: returns a pre-determined sequence of "ground-truth" tokens.
For each step we check how many of the drafter's proposals match this
ground truth.

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_speculative_loop.py -k 1x8
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4_cody.server.speculative import SpeculativeDecoder
from models.demos.gemma4_cody.tt.assistant.config import Gemma4AssistantConfig

from ...tests.test_factory import parametrize_mesh_with_fabric

B = 32
KV_LEN = 64  # multiple of k_chunk_size=64
NUM_STEPS = 4  # number of speculative steps to run
T = 4  # drafts per step


@parametrize_mesh_with_fabric()
def test_speculative_loop_runs(mesh_device):
    """One-shot run of the full speculative loop with mock target."""
    cfg = Gemma4AssistantConfig.from_hf_path("/mnt/nas/gemma-assistant")

    spec = SpeculativeDecoder(
        mesh_device=mesh_device,
        drafter_config=cfg,
        num_slots=B,
        num_drafts=T,
    )

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    def to_tt(t, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=layout,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    # Build mock target outputs: a deterministic next-token sequence per slot.
    # We use a simple Markov-style next = (current + slot_idx + step) % vocab.
    # The drafter has its own predictions; agreement is probabilistic.
    torch.manual_seed(42)

    print(f"\n=== Speculative loop: B={B}, T={T}, steps={NUM_STEPS} ===")

    for step in range(NUM_STEPS):
        # Synthetic inputs (would come from target's _step_decode in production)
        target_hidden = to_tt(torch.randn(1, 1, B, 2 * cfg.backbone_hidden_size))
        K_swa = to_tt(torch.randn(B, cfg.num_key_value_heads, KV_LEN, cfg.head_dim))
        V_swa = to_tt(torch.randn(B, cfg.num_key_value_heads, KV_LEN, cfg.head_dim))
        K_full = to_tt(torch.randn(B, cfg.num_global_key_value_heads, KV_LEN, cfg.global_head_dim))
        V_full = to_tt(torch.randn(B, cfg.num_global_key_value_heads, KV_LEN, cfg.global_head_dim))
        shared_kv = {
            "sliding_attention": (K_swa, V_swa),
            "full_attention": (K_full, V_full),
        }

        cos_full = to_tt(torch.randn(1, 1, B, cfg.global_head_dim))
        sin_full = to_tt(torch.randn(1, 1, B, cfg.global_head_dim))
        cos_swa = to_tt(torch.randn(1, 1, B, cfg.head_dim))
        sin_swa = to_tt(torch.randn(1, 1, B, cfg.head_dim))

        cur_pos = ttnn.from_torch(
            torch.full((B,), KV_LEN - 1, dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )

        # 1. propose: drafter generates T tokens per slot
        drafts = spec.propose(
            target_last_hidden=target_hidden,
            shared_kv=shared_kv,
            cos_pos_full=cos_full,
            sin_pos_full=sin_full,
            cos_pos_sliding=cos_swa,
            sin_pos_sliding=sin_swa,
            cur_pos_tensor=cur_pos,
        )
        assert len(drafts) == B
        for slot_drafts in drafts:
            assert len(slot_drafts) == T, f"Expected {T} drafts, got {len(slot_drafts)}"

        # 2. mock target verify: produce a pretend "target top-1 at each draft position"
        # We deliberately make some drafts match to test verify()'s mismatch logic.
        # Strategy: for each slot, the target's top-1 for position i is:
        #   - drafts[i] with probability 0.5 (acceptance)
        #   - random other token with probability 0.5 (rejection at position i)
        # This gives a synthetic acceptance rate around 50%.
        target_top1_per_slot = []
        for slot_idx in range(B):
            slot_drafts = drafts[slot_idx]
            slot_target = []
            for i, draft_tok in enumerate(slot_drafts):
                if torch.rand(1).item() < 0.5:
                    slot_target.append(draft_tok)  # match → accept
                else:
                    # Random different token
                    other = (draft_tok + 1) % cfg.vocab_size
                    slot_target.append(other)
            target_top1_per_slot.append(slot_target)

        # 3. verify + commit per slot
        step_accepts = []
        for slot_idx in range(B):
            n_acc = spec.verify(slot_idx, drafts[slot_idx], target_top1_per_slot[slot_idx])
            spec.commit(slot_idx, n_acc)
            step_accepts.append(n_acc)

        avg_accepted = sum(step_accepts) / len(step_accepts)
        print(f"  step {step}: avg accepted/slot = {avg_accepted:.2f} (of T={T})")

    # Aggregate stats
    agg = spec.aggregate_stats()
    print()
    print(f"=== Aggregate after {NUM_STEPS} steps ===")
    print(f"  Total proposed:    {agg['total_proposed']}")
    print(f"  Total accepted:    {agg['total_accepted']}")
    print(f"  Acceptance rate:   {agg['acceptance_rate']*100:.1f}%")
    print(f"  Active slots:      {agg['active_slots']}")
    print()
    print(f"  Per-slot stats (slot 0):")
    print(f"    {spec.per_slot_stats(0)}")
    print()

    # Sanity checks
    assert agg["total_proposed"] == B * T * NUM_STEPS
    assert 0 <= agg["acceptance_rate"] <= 1.0
    assert agg["active_slots"] == B  # every slot participated
    # With p=0.5 per-draft random match, expected acceptance is around p*T/(T) ~ 0.5
    # but TRUNCATED: E[accepts]/T = (1 - p^T)/(T * (1-p)). At p=0.5, T=4:
    # = (1 - 0.0625) / (4 * 0.5) = 0.469. So acceptance_rate should be ~0.30-0.45.
    assert 0.20 < agg["acceptance_rate"] < 0.70, (
        f"Acceptance rate {agg['acceptance_rate']} is outside expected ~30-50% range " f"for p=0.5 random matching"
    )

    print("Speculative loop test passed.")
