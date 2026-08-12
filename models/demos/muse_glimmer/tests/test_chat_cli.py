# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the streaming interactive client and local tools."""

from __future__ import annotations

import json

import pytest

from models.demos.muse_glimmer import chat


def test_local_tools_are_bounded_and_json_serializable():
    assert chat.execute_tool("calculator.calculate", {"expression": "(2 + 3) * 4"})["result"] == 20
    assert chat.execute_tool("text.get_statistics", {"text": "one two\nthree"}) == {
        "characters": 13,
        "words": 3,
        "lines": 2,
    }
    current = chat.execute_tool("time.get_current_time", {"timezone": "UTC"})
    assert current["timezone"] == "UTC"
    assert current["datetime"].endswith("+00:00")
    with pytest.raises(chat.ToolError, match="unsupported"):
        chat.execute_tool("calculator.calculate", {"expression": "__import__('os')"})
    with pytest.raises(chat.ToolError, match="unknown tool"):
        chat.execute_tool("shell.run", {"command": "true"})


def test_streaming_chat_executes_tool_and_preserves_history(monkeypatch, capsys):
    rounds = 0

    def fake_stream(server, messages, **kwargs):
        nonlocal rounds
        del server, kwargs
        rounds += 1
        if rounds == 1:
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {"name": "calculator.calculate", "arguments": '{"expression":"6*7"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        else:
            tool_result = json.loads(messages[-1]["content"])
            assert tool_result["result"] == 42
            yield {"choices": [{"delta": {"content": "The result "}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": "is 42."}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield {
            "choices": [],
            "usage": {"prompt_tokens_details": {"cached_tokens": 10}},
            "dflash": {
                "ar_decode_tokens": 4,
                "ar_decode_tokens_per_second": 100.0,
                "prefill_tokens_per_second": 200.0,
            },
        }

    monkeypatch.setattr(chat, "stream_completion", fake_stream)
    messages = [{"role": "user", "content": "What is 6*7?"}]
    answer = chat.chat_turn(
        "http://unused",
        messages,
        max_tokens=64,
        reasoning_strength="low",
        current_date="2026-08-10",
        timeout=1.0,
    )

    assert answer == "The result is 42."
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    captured = capsys.readouterr()
    assert "assistant> The result is 42." in captured.out
    assert "calculator.calculate" in captured.err
    assert "AR 4 tokens @ 100.00 tok/s" in captured.err
