# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regressions for the server decode-step counter invariant."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import transformers

from models.demos.gemma4_cody.server import server as server_module


class _FakeTokenizer:
    eos_token_id = None
    unk_token_id = -1

    @staticmethod
    def convert_tokens_to_ids(_token):
        return -1


class _FakeThread:
    """Record worker startup without running the engine's infinite loop."""

    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


def _make_non_spec_engine(monkeypatch):
    """Run the real Engine initializer with all device/model work stubbed."""

    fake_model = SimpleNamespace(sampling=None, _per_layer_input_weight_keys=None)
    fake_args = SimpleNamespace(layer_types=[])
    monkeypatch.setattr(server_module, "_SPECULATIVE_DECODE", False)
    monkeypatch.setattr(
        server_module,
        "create_tt_model",
        lambda **_kwargs: (fake_args, fake_model, [], {}),
    )
    monkeypatch.setattr(server_module.Engine, "_disable_fused_reduce_scatter_buffers", lambda self: None)
    monkeypatch.setattr(server_module.Engine, "_push_sampling_params", lambda self: None)
    monkeypatch.setattr(server_module.Engine, "_capture_traces", lambda self: ([], {"trace_id": 1}))
    monkeypatch.setattr(server_module.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        transformers.GenerationConfig,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(eos_token_id=None)),
    )

    return server_module.Engine(
        mesh_device=object(),
        model_path="unused-host-only-model",
        max_seq_len=1024,
        max_prefill_bucket=1024,
        batch=1,
    )


def test_non_spec_engine_initializes_counter_and_reaches_decode(monkeypatch):
    engine = _make_non_spec_engine(monkeypatch)
    assert engine._spec is None
    assert engine._spec_steps_called == 0

    calls = []
    engine.slots = [
        server_module._Slot(
            rid="request-0",
            cur_pos=7,
            next_token=11,
            request=SimpleNamespace(cancelled=False),
        )
    ]
    engine._compact_slots = lambda: None
    engine._refresh_decode_page_tables = lambda active: calls.append(("refresh", active))
    engine._run_decode_trace = lambda slots: (calls.append(("decode", slots)) or [23], None)
    engine._emit_tokens = lambda slot, tokens: calls.append(("emit", slot, tokens))
    engine._dflash_ckpt = lambda *_args, **_kwargs: None

    engine._step_decode()

    assert calls == [("refresh", [0]), ("decode", [0]), ("emit", 0, [23])]
    assert engine.slots[0].cur_pos == 8
    assert engine.slots[0].next_token == 23
    assert engine._spec_steps_called == 0


class _FakeSpec:
    num_drafts = 1

    def __init__(self):
        self.aggregate_calls = 0

    def aggregate_stats(self):
        self.aggregate_calls += 1
        return {
            "mean_accepted_per_step": 1.0,
            "mean_tokens_per_step": 2.0,
            "windowed_acceptance": 1.0,
        }


def _make_spec_step_engine(monkeypatch, *, commit_raises=False):
    engine = server_module.Engine.__new__(server_module.Engine)
    spec = _FakeSpec()
    commits = []
    engine.batch = 1
    engine.slots = [server_module._Slot(rid="request-0", request=SimpleNamespace(cancelled=False))]
    engine._spec = spec
    engine._spec_steps_called = 2
    engine._drafter_kind = "mtp"
    engine._packed_verify_traces = {1: {"trace_id": 17}}
    engine.mesh_device = object()
    engine._compact_slots = lambda: None
    engine._slot_uses_packed_verify = lambda _slot, _idx: True
    engine._refresh_decode_page_tables = lambda _active: None
    engine._propose_drafts = lambda _slots: {0: [31]}
    engine._pick_pv_bucket = lambda _occupancy: 1
    engine._refresh_packed_verify_inputs = lambda _bucket, _slots, _drafts: None
    engine._read_packed_verify = lambda _bucket: ([31, 37], torch.empty(0))
    engine._dflash_ckpt = lambda *_args, **_kwargs: None

    def commit(*args, **kwargs):
        commits.append((args, kwargs))
        if commit_raises:
            raise RuntimeError("synthetic commit failure")

    engine._commit_packed_verify = commit
    monkeypatch.setattr(server_module.ttnn, "execute_trace", lambda *_args, **_kwargs: None)
    return engine, spec, commits


def test_spec_counter_counts_only_successful_packed_verify(monkeypatch):
    engine, spec, commits = _make_spec_step_engine(monkeypatch)

    engine._step_decode()

    assert len(commits) == 1
    assert engine._spec_steps_called == 3
    assert spec.aggregate_calls == 1


def test_spec_counter_does_not_count_failed_commit(monkeypatch):
    engine, spec, commits = _make_spec_step_engine(monkeypatch, commit_raises=True)

    engine._step_decode()

    assert len(commits) == 1
    assert engine._spec_steps_called == 2
    assert spec.aggregate_calls == 0
    assert engine._spec is None
    assert engine._packed_verify_traces is None
