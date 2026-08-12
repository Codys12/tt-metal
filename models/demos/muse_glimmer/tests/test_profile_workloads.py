# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, bounded workloads for Tracy and device-counter profiling."""

import os
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.muse_glimmer.server.server import DEFAULT_ASSISTANT_PATH, Engine
from models.demos.muse_glimmer.tt.common import DEFAULT_MODEL_PATH

WORKLOAD = os.getenv("MUSE_PROFILE_WORKLOAD")


@pytest.mark.skipif(WORKLOAD not in {"chunk", "packed", "decode"}, reason="set MUSE_PROFILE_WORKLOAD")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.DISABLED,
            "trace_region_size": 256_000_000,
            "num_command_queues": 2 if os.getenv("MUSE_QKV_PREFETCH") == "1" else 1,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_profile_server_phase(mesh_device):
    """Keep captures below the device profiler's finite marker capacity."""
    target = Path(os.getenv("MUSE_TARGET_DIR", DEFAULT_MODEL_PATH))
    assistant = Path(os.getenv("MUSE_ASSISTANT_DIR", DEFAULT_ASSISTANT_PATH))
    if not (target / "model.safetensors.index.json").exists() or not (assistant / "model.safetensors").exists():
        pytest.skip("Muse Glimmer checkpoints are unavailable")

    max_seq_len = int(os.getenv("MUSE_PROFILE_MAX_SEQ_LEN", "4096"))
    engine = Engine(mesh_device, str(target), str(assistant), max_seq_len=max_seq_len)
    # Weight/cache construction emits enough programs to consume the device
    # marker ring. Drain it before the bounded phase so no phase markers drop.
    if os.getenv("MUSE_TRACY_SIGNPOSTS") == "1":
        ttnn.ReadDeviceProfiler(mesh_device)
    token = int(engine.tokenizer.encode(" glimmer", add_special_tokens=False)[0])
    if WORKLOAD == "chunk":
        prompt_length = int(os.getenv("MUSE_PROFILE_PROMPT_TOKENS", "2048"))
    else:
        context_tokens = int(os.getenv("MUSE_PROFILE_CONTEXT_TOKENS", "64"))
        bonus, _, _ = engine._append_prompt_tokens([token] * context_tokens)
        if os.getenv("MUSE_TRACY_SIGNPOSTS") == "1":
            ttnn.ReadDeviceProfiler(mesh_device)
        if WORKLOAD == "decode":
            decode_tokens = int(os.getenv("MUSE_PROFILE_DECODE_TOKENS", "17"))
            engine._prepare_fixed_decode(decode_tokens)
            engine._capture_decode_traces()
            # Capture warmup emits thousands of markers. Drain them so the
            # bounded report contains replay only.
            if os.getenv("MUSE_TRACY_SIGNPOSTS") == "1":
                ttnn.ReadDeviceProfiler(mesh_device)
    started = time.perf_counter()
    if WORKLOAD == "chunk":
        bonus, chunked, packed = engine._append_prompt_tokens([token] * prompt_length)
    elif WORKLOAD == "packed":
        bonus, chunked, packed = engine._append_prompt_tokens([token] * 16)
    else:
        chunked = packed = 0
        try:
            engine._speculative_decode(bonus, decode_tokens)
        finally:
            engine._release_decode_traces()
    ttnn.synchronize_device(mesh_device)
    if os.getenv("MUSE_TRACY_SIGNPOSTS") == "1":
        ttnn.ReadDeviceProfiler(mesh_device)
    elapsed = time.perf_counter() - started
    print(f"profile workload={WORKLOAD}; seconds={elapsed:.6f}; chunked={chunked}; packed={packed}")
