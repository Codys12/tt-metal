# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in 128K paged-cache and multi-turn tool stress test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ttnn
from models.demos.muse_glimmer.server.server import DEFAULT_ASSISTANT_PATH, Engine
from models.demos.muse_glimmer.tt.common import DEFAULT_MODEL_PATH

TARGET_PATH = Path(os.getenv("MUSE_TARGET_DIR", DEFAULT_MODEL_PATH))
ASSISTANT_PATH = Path(os.getenv("MUSE_ASSISTANT_DIR", DEFAULT_ASSISTANT_PATH))
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "weather.get_forecast",
        "description": "Get the weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"],
        },
    },
}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.DISABLED, "trace_region_size": 256_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_128k_prefill_decode_and_cached_tool_round_trip(mesh_device):
    if os.getenv("MUSE_RUN_128K") != "1":
        pytest.skip("Set MUSE_RUN_128K=1 to run the long 128K stress test")
    if not (TARGET_PATH / "model.safetensors.index.json").exists():
        pytest.skip(f"Muse Glimmer checkpoint is not available at {TARGET_PATH}")
    if not (ASSISTANT_PATH / "model.safetensors").exists():
        pytest.skip(f"Muse Glimmer assistant checkpoint is not available at {ASSISTANT_PATH}")

    engine = Engine(mesh_device, str(TARGET_PATH), str(ASSISTANT_PATH), max_seq_len=131072)
    # With this checkpoint template and tool schema, 63,788 repetitions render
    # to exactly 128,000 input tokens.  Assert that invariant before touching
    # the device so tokenizer/template changes fail explicitly.
    long_user = {
        "role": "user",
        "content": (" glimmer" * 63788) + " What word was repeated? Answer with one word and do not call a tool.",
    }
    prompt_tokens = int(engine._tokenize([long_user], tools=[WEATHER_TOOL], current_date="2026-08-10").shape[1])
    assert prompt_tokens == 128000

    long_result = engine.generate(
        [long_user],
        64,
        tools=[WEATHER_TOOL],
        current_date="2026-08-10",
    )
    print(
        f"128K answer: {long_result.text!r}; prompt={long_result.prompt_tokens}; "
        f"prefill={long_result.prefill_tokens_per_second:.2f} tok/s; "
        f"decode={long_result.decode_tokens_per_second:.2f} tok/s"
    )
    assert long_result.prompt_tokens == 128000
    assert long_result.prefilled_prompt_tokens == 128000
    assert long_result.chunked_prefill_tokens == 128000
    assert long_result.packed_prefill_tokens == 0
    assert long_result.prefill_tokens_per_second > 0
    assert long_result.decode_tokens_per_second > 0
    assert "glimmer" in (long_result.text + " " + (long_result.reasoning_content or "")).lower()

    history = [
        long_user,
        {
            "role": "assistant",
            "content": long_result.text or None,
            "reasoning_content": long_result.reasoning_content,
        },
        {
            "role": "user",
            "content": "Now call weather.get_forecast for Tokyo for 2 days. You must use the tool.",
        },
    ]
    tool_result = engine.generate(history, 160, tools=[WEATHER_TOOL], current_date="2026-08-10")
    print(
        f"128K cached tool call: {tool_result.tool_calls}; cached={tool_result.cached_prompt_tokens}; "
        f"prefilled={tool_result.prefilled_prompt_tokens}; chunked={tool_result.chunked_prefill_tokens}; "
        f"prefill={tool_result.prefill_tokens_per_second:.2f} tok/s; "
        f"decode={tool_result.decode_tokens_per_second:.2f} tok/s"
    )
    assert tool_result.cached_prompt_tokens == 128000
    assert tool_result.prefilled_prompt_tokens < 256
    assert tool_result.chunked_prefill_tokens >= 64
    assert tool_result.chunked_prefill_tokens + tool_result.packed_prefill_tokens == tool_result.prefilled_prompt_tokens
    assert tool_result.prefill_tokens_per_second > 0
    assert tool_result.finish_reason == "tool_calls"
    assert tool_result.tool_calls
    call = tool_result.tool_calls[0]
    assert call["function"]["name"] == "weather.get_forecast"
    assert json.loads(call["function"]["arguments"]) == {"city": "Tokyo", "days": 2}

    history.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": tool_result.reasoning_content,
                "tool_calls": tool_result.tool_calls,
            },
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": '{"city":"Tokyo","forecast":["windy","sunny"],"unit":"celsius"}',
            },
        ]
    )
    answer = engine.generate(history, 128, tools=[WEATHER_TOOL], current_date="2026-08-10")
    print(
        f"128K cached tool answer: {answer.text!r}; cached={answer.cached_prompt_tokens}; "
        f"prefilled={answer.prefilled_prompt_tokens}; chunked={answer.chunked_prefill_tokens}; "
        f"prefill={answer.prefill_tokens_per_second:.2f} tok/s; "
        f"decode={answer.decode_tokens_per_second:.2f} tok/s"
    )
    assert answer.cached_prompt_tokens == tool_result.prompt_tokens
    assert answer.prefilled_prompt_tokens < 256
    assert answer.chunked_prefill_tokens >= 64
    assert answer.chunked_prefill_tokens + answer.packed_prefill_tokens == answer.prefilled_prompt_tokens
    assert answer.finish_reason == "stop"
    assert "windy" in answer.text.lower()
