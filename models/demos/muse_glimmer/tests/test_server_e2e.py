# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Live batch-1 server regression for tools and long-position DFlash decode."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ttnn
from models.demos.muse_glimmer.server.server import DEFAULT_ASSISTANT_PATH, Engine, build_app
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


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.DISABLED, "trace_region_size": 256_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_server_tool_round_trip_and_long_sequence(mesh_device):
    if not (TARGET_PATH / "model.safetensors.index.json").exists():
        pytest.skip(f"Muse Glimmer checkpoint is not available at {TARGET_PATH}")
    if not (ASSISTANT_PATH / "model.safetensors").exists():
        pytest.skip(f"Muse Glimmer assistant checkpoint is not available at {ASSISTANT_PATH}")

    engine = Engine(mesh_device, str(TARGET_PATH), str(ASSISTANT_PATH), max_seq_len=4096)
    user = {
        "role": "user",
        "content": (
            "Call weather.get_forecast for Paris for 3 days. " "You must use the tool and must not answer directly."
        ),
    }
    tool_result = engine.generate([user], 160, tools=[WEATHER_TOOL], current_date="2026-08-10")
    print(
        f"tool call: {tool_result.tool_calls}; tokens={len(tool_result.token_ids)}; "
        f"prefill={tool_result.prefill_tokens_per_second:.2f} tok/s; "
        f"decode={tool_result.decode_tokens_per_second:.2f} tok/s"
    )
    assert tool_result.finish_reason == "tool_calls"
    assert tool_result.cached_prompt_tokens == 0
    assert tool_result.prefilled_prompt_tokens == tool_result.prompt_tokens
    assert tool_result.chunked_prefill_tokens > 0
    assert tool_result.prefill_tokens_per_second > 0
    assert tool_result.decode_tokens_per_second > 0
    assert tool_result.tool_calls
    call = tool_result.tool_calls[0]
    assert call["function"]["name"] == "weather.get_forecast"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}

    history = [
        user,
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": tool_result.reasoning_content,
            "tool_calls": tool_result.tool_calls,
        },
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": '{"city":"Paris","forecast":["sunny","cloudy","rain"],"unit":"celsius"}',
        },
    ]
    history_tokens = int(engine._tokenize(history, tools=[WEATHER_TOOL]).shape[1])
    final_result = engine.generate(history, 192, tools=[WEATHER_TOOL], current_date="2026-08-10")
    print(
        f"tool result answer ({history_tokens}-token prompt): {final_result.text!r}; "
        f"prefill={final_result.prefill_tokens_per_second:.2f} tok/s; "
        f"decode={final_result.decode_tokens_per_second:.2f} tok/s"
    )
    assert history_tokens > 512
    assert final_result.cached_prompt_tokens > 0
    assert final_result.prefilled_prompt_tokens < final_result.prompt_tokens
    assert final_result.chunked_prefill_tokens > 0
    assert final_result.chunked_prefill_tokens + final_result.packed_prefill_tokens == (
        final_result.prefilled_prompt_tokens
    )
    assert final_result.finish_reason == "stop"
    assert "sunny" in final_result.text.lower()
    assert not final_result.tool_calls

    # Continue the same cached conversation through a second tool invocation
    # and response.  This proves that serialization of prior assistant/tool
    # messages remains token-identical enough to reuse the paged prefix.
    history.extend(
        [
            {
                "role": "assistant",
                "content": final_result.text,
                "reasoning_content": final_result.reasoning_content,
            },
            {
                "role": "user",
                "content": "Now call weather.get_forecast for Tokyo for 2 days. You must use the tool again.",
            },
        ]
    )
    second_call = engine.generate(history, 160, tools=[WEATHER_TOOL], current_date="2026-08-10")
    print(
        f"second tool call: {second_call.tool_calls}; cached={second_call.cached_prompt_tokens}; "
        f"prefilled={second_call.prefilled_prompt_tokens}; chunked={second_call.chunked_prefill_tokens}; "
        f"prefill={second_call.prefill_tokens_per_second:.2f} tok/s"
    )
    assert second_call.finish_reason == "tool_calls"
    assert second_call.cached_prompt_tokens > 0
    assert second_call.prefilled_prompt_tokens < second_call.prompt_tokens
    assert second_call.chunked_prefill_tokens + second_call.packed_prefill_tokens == second_call.prefilled_prompt_tokens
    # This suffix is only ~100 tokens and begins mid-page; depending on the
    # exact prior completion length it may be entirely packed or contain one
    # aligned 64-token chunk. Both exercise valid paged cache reuse.
    assert second_call.packed_prefill_tokens > 0
    assert second_call.tool_calls
    second = second_call.tool_calls[0]
    assert second["function"]["name"] == "weather.get_forecast"
    assert json.loads(second["function"]["arguments"]) == {"city": "Tokyo", "days": 2}

    history.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": second_call.reasoning_content,
                "tool_calls": second_call.tool_calls,
            },
            {
                "role": "tool",
                "tool_call_id": second["id"],
                "content": '{"city":"Tokyo","forecast":["windy","sunny"],"unit":"celsius"}',
            },
        ]
    )
    second_answer = engine.generate(history, 192, tools=[WEATHER_TOOL], current_date="2026-08-10")
    print(
        f"second tool answer: {second_answer.text!r}; cached={second_answer.cached_prompt_tokens}; "
        f"prefilled={second_answer.prefilled_prompt_tokens}; chunked={second_answer.chunked_prefill_tokens}; "
        f"prefill={second_answer.prefill_tokens_per_second:.2f} tok/s"
    )
    assert second_answer.finish_reason == "stop"
    assert second_answer.cached_prompt_tokens > 0
    assert second_answer.prefilled_prompt_tokens < second_answer.prompt_tokens
    assert second_answer.chunked_prefill_tokens + second_answer.packed_prefill_tokens == (
        second_answer.prefilled_prompt_tokens
    )
    assert "windy" in second_answer.text.lower()
    assert not second_answer.tool_calls

    sustained_result = engine.generate(
        "Write a numbered list from 1 through 100. On every line, write the number followed by ': glimmer'. "
        "Do not stop before line 100.",
        256,
    )
    print(
        f"sustained decode: tokens={len(sustained_result.token_ids)}; "
        f"finish={sustained_result.finish_reason}; "
        f"AR={sustained_result.ar_decode_tokens_per_second:.2f} tok/s; "
        f"text={sustained_result.text!r}"
    )
    assert len(sustained_result.token_ids) == 256
    assert sustained_result.finish_reason == "length"
    assert "glimmer" in sustained_result.text.lower()

    # Exercise real device decode from the server's background streaming
    # worker, and verify Muse's addressed wire tokens never leak into SSE.
    from fastapi.testclient import TestClient

    with TestClient(build_app(engine)).stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Write one coherent sentence containing glimmer."}],
            "max_tokens": 96,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.status_code == 200
        stream_events = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    streamed_text = "".join(
        event["choices"][0]["delta"].get("content", "") for event in stream_events if event.get("choices")
    )
    streamed_metrics = next(event["dflash"] for event in stream_events if event.get("dflash"))
    print(f"streamed answer: {streamed_text!r}; " f"AR={streamed_metrics['ar_decode_tokens_per_second']:.2f} tok/s")
    assert "glimmer" in streamed_text.lower()
    assert "<|" not in streamed_text
    assert streamed_metrics["ar_decode_tokens"] > 0

    fact = "The observatory ledger records that the copper beacon is codeword glimmer. "
    long_user = {
        "role": "user",
        "content": fact * 210 + "What codeword is assigned to the copper beacon? Answer in one sentence.",
    }
    long_prompt_tokens = int(engine._tokenize([long_user]).shape[1])
    long_result = engine.generate([long_user], 96)
    print(
        f"long answer ({long_prompt_tokens}-token prompt): {long_result.text!r}; "
        f"tok/s={long_result.tokens_per_second:.2f}"
    )
    assert long_prompt_tokens >= 3200
    assert long_result.finish_reason == "stop"
    assert "glimmer" in long_result.text.lower()

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        engine.generate([long_user], 1024)
