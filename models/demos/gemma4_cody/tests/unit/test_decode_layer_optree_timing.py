# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Capture the ttnn op tree of one decode layer + attribute device op timings to it.

This runs a single Gemma4 decoder-layer DECODE step (seq_len=1, fully on device)
under BOTH:

  1. ``ttnn.graph.begin_graph_capture(RunMode.NORMAL)`` — gives the *op tree*: an
     ordered list of ttnn function_start/function_end nodes (op name + graph
     counter + host-side duration). NORMAL mode actually dispatches/executes on
     device, so the profiler below sees the same execution.

  2. The device profiler — ``ttnn.ReadDeviceProfiler`` +
     ``ttnn.get_latest_programs_perf_data()`` — gives the *op timings*: per-program
     device durations keyed by ``program_execution_uid.runtime_id``.

The catch this test exists to expose: the two come from disjoint keyspaces. The
tree is keyed by graph-node ``counter`` + op name; the timings are keyed by
``runtime_id``. **There is no shared id** (see conftest.py:ttnn_graph_report and
ttnn/ttnn/graph_report.py, which only ever store host-side durations). So the
only way to say "this timing belongs to that op" is execution order + op name —
which is ambiguous the moment a layer fires several identically-named matmuls.

This test does that order+name attribution and *flags every ambiguous match* so
the gap is explicit rather than silently wrong.

Run (needs a PROFILER-ENABLED build):
    TT_METAL_DEVICE_PROFILER=1 \
    TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
    HF_MODEL=/mnt/nas/gemma \
    pytest -s models/demos/gemma4_cody/tests/unit/test_decode_layer_optree_timing.py

If the profiler bindings/state aren't available the test SKIPS (it still logs the
captured op tree, which needs no profiler).
"""

import os

# Device-profiler env must be set before the device is opened by the mesh_device
# fixture (which runs before the test body). Setting it at import time — pytest
# imports this module at collection, before any fixture executes — guarantees it
# is in effect for the device init. Don't clobber a value the caller already set.
for _var, _val in {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
}.items():
    os.environ.setdefault(_var, _val)

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_cody.tt.layer import Gemma4DecoderLayer

from ...tests.test_factory import parametrize_mesh_with_fabric

# Reuse the exact decode setup helpers from the PCC decode test so the op tree we
# capture is the production decode path, not a bespoke one.
from .test_layer import (
    _create_gemma4_model_args,
    _create_hf_reference_layer,
    _create_hf_text_config,
    _fill_decode_kv_cache,
    _hf_state_to_tt_state,
)

# ── op-tree extraction ─────────────────────────────────────────────────────


def _walk_optree(captured_graph):
    """Flatten a captured graph into an ordered list of op records.

    Returns one dict per top-level (depth==1) ttnn op, in execution order:
        {counter, name, depth, host_duration_ns}
    Nested ops (a composite op's internals) are folded into their parent — we
    only attribute device time at the top level, which is what the profiler's
    program records line up with.
    """
    ops = []
    stack = []  # (counter, name) of currently-open function_start nodes
    for node in captured_graph:
        nt = node.get("node_type")
        if nt == "function_start":
            name = node.get("params", {}).get("name", "unknown")
            stack.append((node.get("counter"), name))
        elif nt == "function_end":
            if not stack:
                continue
            counter, name = stack.pop()
            depth = len(stack) + 1  # depth at which this op lived
            if depth == 1:
                ops.append(
                    {
                        "counter": counter,
                        "name": name,
                        "depth": depth,
                        "host_duration_ns": node.get("duration_ns", 0),
                    }
                )
    return ops


# ── device-timing extraction ───────────────────────────────────────────────


def _collect_device_programs(mesh_device):
    """Read the device profiler and return ordered per-program timing records.

    Returns (records, skip_reason). records is a list, ordered by
    (runtime_id, trace_id, trace_id_counter), of:
        {runtime_id, trace_id, trace_id_counter, device_duration_ns, analyses}
    skip_reason is set (and records empty) when the profiler is unavailable.
    """
    if not hasattr(getattr(ttnn, "_ttnn", None), "profiler"):
        return [], "profiler bindings not in this build"

    ttnn.synchronize_device(mesh_device)
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
        latest = ttnn.get_latest_programs_perf_data()
    except RuntimeError as exc:
        if "profiler_state_manager is nullptr" in str(exc):
            return [], "profiler state manager not initialized (profiling disabled in build?)"
        raise

    if not latest:
        return [], "profiler returned no program data (mid-run dump/post-process disabled?)"

    # One mesh device → take the first device id present. (Per-device timings of a
    # replicated decode layer are equivalent for attribution purposes.)
    device_id = next(iter(latest))
    programs = list(latest[device_id])

    def _sort_key(p):
        uid = p.program_execution_uid
        return (uid.runtime_id, uid.trace_id, uid.trace_id_counter)

    records = []
    for p in sorted(programs, key=_sort_key):
        uid = p.program_execution_uid
        analyses = {name: res.duration for name, res in p.program_analyses_results.items()}
        # No single canonical "device duration" field is exposed, so take the max
        # analysis duration as the representative kernel time and keep the rest.
        device_duration_ns = max(analyses.values(), default=0)
        records.append(
            {
                "runtime_id": uid.runtime_id,
                "trace_id": uid.trace_id,
                "trace_id_counter": uid.trace_id_counter,
                "device_duration_ns": device_duration_ns,
                "analyses": analyses,
            }
        )
    return records, None


# ── correlation (order + name; the ambiguous part) ──────────────────────────


def _correlate(ops, programs):
    """Attribute device-program timings to op-tree nodes by execution order.

    There is no shared key between the op tree and the profiler programs, so this
    is purely positional. Each pairing is annotated with whether it is AMBIGUOUS:
      - duplicate name : the op's name occurs >1x at top level, so order is the
                         ONLY thing distinguishing its timing row from a sibling.
      - count mismatch : len(ops) != len(programs) — host-only ops (no program)
                         or composite ops (multiple programs) break 1:1 order
                         alignment, so EVERY pairing past the first divergence is
                         suspect.
    Returns (annotated, summary).
    """
    name_counts = {}
    for op in ops:
        name_counts[op["name"]] = name_counts.get(op["name"], 0) + 1

    count_mismatch = len(ops) != len(programs)
    annotated = []
    for i, op in enumerate(ops):
        prog = programs[i] if i < len(programs) else None
        dup = name_counts[op["name"]] > 1
        annotated.append(
            {
                **op,
                "matched_program": prog,
                "device_duration_ns": prog["device_duration_ns"] if prog else None,
                "ambiguous": bool(prog is not None and (dup or count_mismatch)),
                "ambiguity_reasons": (
                    ([f"name x{name_counts[op['name']]}"] if dup else [])
                    + (["count_mismatch (order alignment unreliable)"] if count_mismatch else [])
                ),
            }
        )

    summary = {
        "num_ops": len(ops),
        "num_programs": len(programs),
        "count_mismatch": count_mismatch,
        "num_ambiguous": sum(1 for a in annotated if a["ambiguous"]),
        "num_unmatched_ops": sum(1 for a in annotated if a["matched_program"] is None),
        "num_unmatched_programs": max(0, len(programs) - len(ops)),
    }
    return annotated, summary


def _log_annotated_tree(annotated, summary):
    logger.info(
        f"[optree↔timing] ops={summary['num_ops']} programs={summary['num_programs']} "
        f"ambiguous={summary['num_ambiguous']} "
        f"unmatched_ops={summary['num_unmatched_ops']} "
        f"unmatched_programs={summary['num_unmatched_programs']}"
    )
    if summary["count_mismatch"]:
        logger.warning(
            "op count != program count — order-based attribution is unreliable past "
            "the first host-only/composite op. Pairings below are best-effort."
        )
    for i, a in enumerate(annotated):
        dev = f"{a['device_duration_ns']:>8} ns" if a["device_duration_ns"] is not None else "   (no pgm)"
        flag = "  ⚠ " + ", ".join(a["ambiguity_reasons"]) if a["ambiguous"] else ""
        logger.info(f"  [{i:>3}] counter={a['counter']:<6} {dev}  {a['name']}{flag}")


# ── the test ────────────────────────────────────────────────────────────────


def _build_decode_layer_and_fwd(mesh_device, layer_idx):
    """Construct the decoder layer + a no-arg fwd() closure for the decode step.

    Mirrors test_layer_forward_decode_with_gemma4_rope_cache (the production decode
    path: 2D RoPE cache + on-device position gather).
    """
    from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
    from models.demos.gemma4_cody.tt.attention import Gemma4AttentionConfig
    from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache
    from models.demos.gemma4_cody.tt.ccl import CCLManager
    from models.demos.gemma4_cody.tt.model import create_rope_caches

    hf_text_config = _create_hf_text_config(num_experts=4, top_k=2)
    hf_layer = _create_hf_reference_layer(hf_text_config, layer_idx)
    tt_state = _hf_state_to_tt_state(hf_layer.state_dict(), layer_idx)
    model_args = _create_gemma4_model_args(hf_text_config)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)

    cache_len = 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=cache_len + 32,
        max_local_batch_size=1,
    )

    k_data = torch.randn(1, attn_cfg.num_key_value_heads, cache_len, attn_cfg.head_dim)
    v_data = torch.randn(1, attn_cfg.num_key_value_heads, cache_len, attn_cfg.head_dim)
    kv_cache = init_kv_cache(
        mesh_device, attn_cfg, max_batch_size=1, max_seq_len=cache_len + 32, cache_dtype=ttnn.bfloat16
    )
    _fill_decode_kv_cache(mesh_device, kv_cache, attn_cfg, k_data, v_data)
    tt_layer.self_attn.kv_cache = kv_cache

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, cache_len + 32)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos_tt, sin_tt = rope_caches_2d[layer_type]

    x_torch = torch.randn(1, 1, model_args.hidden_size, dtype=torch.float32)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    position_idx_tt = ttnn.from_torch(
        torch.nn.functional.pad(torch.tensor([cache_len], dtype=torch.int32).reshape(1, 1), (0, 31)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    position_idx_cache_tt = ttnn.from_torch(
        torch.tensor([cache_len], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    def fwd():
        return tt_layer(
            x_tt,
            rope_mats=(cos_tt, sin_tt),
            position_idx=position_idx_tt,
            position_idx_cache=position_idx_cache_tt,
            page_table=None,
            kv_cache=kv_cache,
            is_decode=True,
            token_index=None,
        )

    return fwd


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
def test_decode_layer_optree_timing(layer_idx, mesh_device):
    """Capture one decode layer's op tree and attribute device timings to it.

    Asserts the op tree was captured (no profiler needed). If the profiler is
    available, it additionally attributes device durations to tree nodes by
    order+name and logs every ambiguous match; if not, it SKIPS the timing half.
    """
    fwd = _build_decode_layer_and_fwd(mesh_device, layer_idx)

    # Compile/warm run OUTSIDE capture so the captured tree is steady-state and
    # the profiler programs we read back correspond to the captured execution.
    out = fwd()
    out.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    # 1) Op tree — RunMode.NORMAL executes on device so the profiler sees it.
    assert not ttnn.graph.is_graph_capture_active(), "a graph capture is unexpectedly already active"
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        out = fwd()
    finally:
        captured_graph = ttnn.graph.end_graph_capture()
    out.deallocate(True)

    ops = _walk_optree(captured_graph)
    logger.info(f"[optree] captured {len(ops)} top-level ttnn ops for layer_idx={layer_idx}")
    assert len(ops) > 0, "op tree capture produced no ops"

    # 2) Op timings — device profiler (may be unavailable → skip the timing half).
    programs, skip_reason = _collect_device_programs(mesh_device)
    if skip_reason is not None:
        for i, op in enumerate(ops):
            logger.info(f"  [{i:>3}] counter={op['counter']:<6}  {op['name']}  (host {op['host_duration_ns']} ns)")
        pytest.skip(f"op tree captured ({len(ops)} ops) but device timings unavailable: {skip_reason}")

    # 3) Attribute timings to the tree by order+name, flagging ambiguity.
    annotated, summary = _correlate(ops, programs)
    _log_annotated_tree(annotated, summary)

    # Sanity: at least one op got a device timing. The interesting output is the
    # logged annotated tree + ambiguity flags, not a pass/fail on correctness of
    # the (inherently ambiguous) attribution.
    assert any(a["device_duration_ns"] is not None for a in annotated), "no device timing attributed to any op"
