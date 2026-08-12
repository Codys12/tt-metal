# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fast, device-free checks for Muse's tokenizer and ATEM wire protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from models.demos.muse_glimmer.server.protocol import (
    IncrementalMuseResponse,
    normalize_messages,
    parse_tokenizer_response,
)
from models.demos.muse_glimmer.server.server import Engine, GenerationResult, build_app
from models.demos.muse_glimmer.tt.common import DEFAULT_MODEL_PATH

MODEL_PATH = Path(os.getenv("MUSE_TARGET_DIR", DEFAULT_MODEL_PATH))
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "weather.get_forecast",
        "description": "Get a weather forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "details": {"type": "object"},
            },
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="module")
def tokenizer():
    if not (MODEL_PATH / "tokenizer_config.json").exists():
        pytest.skip(f"Muse Glimmer tokenizer is not available at {MODEL_PATH}")
    return AutoTokenizer.from_pretrained(MODEL_PATH)


def test_parse_reasoning_and_schema_typed_atem_call(tokenizer):
    raw = (
        " to=self<|message|>I need the forecast.<|eom|>"
        "<|start|>assistant to=weather.get_forecast<|message|>"
        "<atem:function_calls>\n"
        '<atem:invoke name="weather.get_forecast">\n'
        '<atem:parameter name="city">New York</atem:parameter>\n'
        '<atem:parameter name="days">3</atem:parameter>\n'
        '<atem:parameter name="details">{"hourly":true}</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )

    parsed = parse_tokenizer_response(tokenizer, raw, tools=[WEATHER_TOOL], reached_stop_token=True)

    assert parsed.content is None
    assert parsed.reasoning_content == "I need the forecast."
    assert parsed.finish_reason == "tool_calls"
    assert parsed.tool_calls[0]["function"]["name"] == "weather.get_forecast"
    assert json.loads(parsed.tool_calls[0]["function"]["arguments"]) == {
        "city": "New York",
        "days": 3,
        "details": {"hourly": True},
    }


def test_parse_multiple_calls_and_final_answer(tokenizer):
    raw = (
        " to=weather.get_forecast<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get_forecast">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
        "<|start|>assistant to=calendar.lookup<|message|><atem:function_calls>\n"
        '<atem:invoke name="calendar.lookup">\n'
        '<atem:parameter name="date">tomorrow</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )

    parsed = parse_tokenizer_response(tokenizer, raw, tools=[WEATHER_TOOL], reached_stop_token=True)

    assert [call["id"] for call in parsed.tool_calls] == ["call_0", "call_1"]
    assert [call["function"]["name"] for call in parsed.tool_calls] == [
        "weather.get_forecast",
        "calendar.lookup",
    ]

    final = parse_tokenizer_response(
        tokenizer, " to=user<|message|>It will be sunny.<|eot|>", tools=None, reached_stop_token=True
    )
    assert final.content == "It will be sunny."
    assert final.finish_reason == "stop"


def test_truncated_atem_is_visible_and_reports_length(tokenizer):
    raw = (
        " to=weather.get_forecast<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get_forecast">\n'
        '<atem:parameter name="city">Paris</atem:parameter>'
    )
    parsed = parse_tokenizer_response(tokenizer, raw, tools=[WEATHER_TOOL], reached_stop_token=False)
    assert json.loads(parsed.tool_calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert parsed.finish_reason == "length"


def test_incremental_response_exposes_only_addressed_text(tokenizer):
    raw = (
        " to=self<|message|>I should answer clearly.<|eom|>"
        "<|start|>assistant to=user<|message|>Hello from Muse.<|eot|>"
    )
    token_ids = tokenizer.encode(raw, add_special_tokens=False)
    stream = IncrementalMuseResponse(tokenizer)
    events = []
    for end in range(1, len(token_ids) + 1):
        events.extend(stream.update(token_ids[:end]))

    assert "".join(text for kind, text in events if kind == "reasoning") == "I should answer clearly."
    assert "".join(text for kind, text in events if kind == "content") == "Hello from Muse."
    assert "to=user" not in "".join(text for _, text in events)


def test_normalize_openai_tool_history_without_mutating_input():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather.get_forecast", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": '{"temperature":21}'},
    ]

    normalized = normalize_messages(messages)

    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_reject_non_text_content_and_invalid_tool_history():
    with pytest.raises(ValueError, match="does not support content types: image"):
        normalize_messages([{"role": "user", "content": [{"type": "image", "url": "unused"}]}])
    with pytest.raises(ValueError, match="invalid JSON"):
        normalize_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "broken", "arguments": "{"}}],
                }
            ]
        )
    with pytest.raises(ValueError, match="Message role must be one of"):
        normalize_messages([{"role": "developer", "content": "unsupported"}])


def test_checkpoint_template_renders_tools_reasoning_and_tool_results(tokenizer):
    engine = Engine.__new__(Engine)
    engine.tokenizer = tokenizer
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I will check.",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather.get_forecast", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": '{"temperature":21}'},
        {"role": "user", "content": "Summarize that."},
    ]

    token_ids = engine._tokenize(
        messages,
        tools=[WEATHER_TOOL],
        reasoning_strength="xhigh",
        tool_namespace_descriptions={"weather": "Weather data."},
        current_date="2030-05-06",
        knowledge_cutoff="2029-12-31",
    )
    rendered = engine.tokenizer.decode(token_ids[0], skip_special_tokens=False)

    assert "Reasoning strength: xhigh." in rendered
    assert "Current date: 2030-05-06." in rendered
    assert "Knowledge cutoff: 2029-12-31." in rendered
    assert '"name": "weather.get_forecast"' in rendered
    assert "<|start|>assistant to=self<|message|>I will check.<|eom|>" in rendered
    assert '<atem:parameter name="city">Paris</atem:parameter>' in rendered
    assert '<tool_output name="weather.get_forecast">' in rendered
    assert rendered.endswith("<|start|>assistant")


def test_openai_response_contains_tool_calls_and_finish_reason():
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from fastapi.testclient import TestClient

    class FakeEngine:
        kwargs = None

        def generate(self, *args, **kwargs):
            self.kwargs = kwargs
            return GenerationResult(
                text="",
                token_ids=[1, 2],
                accepted_drafts=[1],
                elapsed_seconds=0.5,
                reasoning_content="Use the tool.",
                tool_calls=[
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "weather.get_forecast", "arguments": '{"city":"Paris"}'},
                    }
                ],
                recipient="weather.get_forecast",
                finish_reason="tool_calls",
                prompt_tokens=12,
                cached_prompt_tokens=8,
                prefilled_prompt_tokens=4,
            )

    engine = FakeEngine()
    client = TestClient(build_app(engine))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Weather?"}], "tools": [WEATHER_TOOL]},
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] == "Use the tool."
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "weather.get_forecast"
    assert response.json()["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 2,
        "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    assert response.json()["dflash"]["prefilled_prompt_tokens"] == 4
    assert response.json()["dflash"]["prefill_tokens_per_second"] == 0.0
    assert response.json()["dflash"]["decode_tokens_per_second"] == 0.0
    assert response.json()["dflash"]["ar_decode_tokens"] == 1
    assert response.json()["dflash"]["ar_decode_tokens_per_second"] == 0.0

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Do not use tools."}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        },
    )
    assert response.status_code == 200
    assert engine.kwargs["tools"] is None

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 400
    assert "only 'auto' or 'none'" in response.json()["detail"]


def test_openai_sse_streaming_contract():
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from fastapi.testclient import TestClient

    class FakeTokenizer:
        def decode(self, token_ids, **kwargs):
            del kwargs
            return {
                (1,): " to=user<|message|>Hello",
                (1, 2): " to=user<|message|>Hello world<|eot|>",
            }[tuple(token_ids)]

    class FakeEngine:
        tokenizer = FakeTokenizer()

        def generate(self, *args, **kwargs):
            del args
            callback = kwargs["on_token_ids"]
            callback([1])
            callback([1, 2])
            return GenerationResult(
                text="Hello world",
                token_ids=[1, 2],
                accepted_drafts=[1],
                elapsed_seconds=0.5,
                finish_reason="stop",
                prompt_tokens=12,
                cached_prompt_tokens=8,
                prefilled_prompt_tokens=4,
                decode_seconds=0.25,
            )

    response = TestClient(build_app(FakeEngine())).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    data_lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"
    events = [json.loads(line) for line in data_lines[:-1]]
    assert events[0]["object"] == "chat.completion.chunk"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert (
        "".join(event["choices"][0]["delta"].get("content", "") for event in events if event.get("choices"))
        == "Hello world"
    )
    assert any(event.get("choices") and event["choices"][0]["finish_reason"] == "stop" for event in events)
    usage_event = next(event for event in events if event.get("choices") == [])
    assert usage_event["usage"]["completion_tokens"] == 2
    assert usage_event["dflash"]["ar_decode_tokens_per_second"] == 4.0


def test_openai_sse_streams_indexed_tool_call():
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from fastapi.testclient import TestClient

    class FakeTokenizer:
        def decode(self, token_ids, **kwargs):
            del token_ids, kwargs
            return ""

    class FakeEngine:
        tokenizer = FakeTokenizer()

        def generate(self, *args, **kwargs):
            del args, kwargs
            return GenerationResult(
                text="",
                token_ids=[1, 2],
                accepted_drafts=[],
                elapsed_seconds=0.5,
                tool_calls=[
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "time.get_current_time", "arguments": '{"timezone":"UTC"}'},
                    }
                ],
                finish_reason="tool_calls",
            )

    response = TestClient(build_app(FakeEngine())).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Time?"}], "stream": True},
    )
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_delta = next(
        event["choices"][0]["delta"]["tool_calls"][0]
        for event in events
        if event.get("choices") and event["choices"][0]["delta"].get("tool_calls")
    )
    assert tool_delta["index"] == 0
    assert tool_delta["function"]["name"] == "time.get_current_time"
    assert any(event.get("choices") and event["choices"][0]["finish_reason"] == "tool_calls" for event in events)
