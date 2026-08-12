# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse conversation helpers around the checkpoint's native response parser."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

_VALID_REASONING_STRENGTHS = {"low", "medium", "high", "xhigh"}
_VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class ParsedAssistantMessage:
    content: str | None
    reasoning_content: str | None
    tool_calls: list[dict[str, Any]]
    recipient: str | None
    finish_reason: str


class IncrementalMuseResponse:
    """Extract stable user-visible deltas from Muse's addressed token stream."""

    _segment = re.compile(
        r"(?:^|<\|start\|>)assistant\s+to=(?P<recipient>[^<\s]+)<\|message\|>"
        r"(?P<body>.*?)(?=<\|(?:eom|eot|start)\|>|$)",
        re.DOTALL,
    )

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.content = ""
        self.reasoning = ""

    @staticmethod
    def _without_partial_control_token(value: str) -> str:
        return re.sub(r"<\|[^|]*$", "", value)

    def _sections(self, token_ids: list[int]) -> tuple[str, str]:
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # Generation begins immediately after this prefix, so it is not part
        # of the returned token IDs.
        wire = "<|start|>assistant" + raw
        content = []
        reasoning = []
        for match in self._segment.finditer(wire):
            body = self._without_partial_control_token(match.group("body"))
            if match.group("recipient") == "user":
                content.append(body)
            elif match.group("recipient") == "self":
                reasoning.append(body)
        return "".join(content), "".join(reasoning)

    @staticmethod
    def _delta(previous: str, current: str) -> str:
        # Full tokenizer decoding is stable for this BPE. If a malformed or
        # incomplete control token temporarily changes old text, wait for the
        # next block instead of emitting text that cannot be retracted in SSE.
        return current[len(previous) :] if current.startswith(previous) else ""

    def update(self, token_ids: list[int]) -> list[tuple[str, str]]:
        content, reasoning = self._sections(token_ids)
        events = []
        reasoning_delta = self._delta(self.reasoning, reasoning)
        if reasoning_delta:
            self.reasoning = reasoning
            events.append(("reasoning", reasoning_delta))
        content_delta = self._delta(self.content, content)
        if content_delta:
            self.content = content
            events.append(("content", content_delta))
        return events

    def finish(self, parsed: ParsedAssistantMessage) -> list[tuple[str, str]]:
        """Flush parser-normalized suffixes that were not visible mid-token."""
        events = []
        final_reasoning = parsed.reasoning_content or ""
        reasoning_delta = self._delta(self.reasoning, final_reasoning)
        if reasoning_delta:
            self.reasoning = final_reasoning
            events.append(("reasoning", reasoning_delta))
        final_content = getattr(parsed, "content", None) or getattr(parsed, "text", "") or ""
        content_delta = self._delta(self.content, final_content)
        if content_delta:
            self.content = final_content
            events.append(("content", content_delta))
        return events


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool history into the checkpoint chat template's form."""

    normalized = copy.deepcopy(messages)
    for message in normalized:
        role = message.get("role")
        if role not in _VALID_ROLES:
            choices = ", ".join(sorted(_VALID_ROLES))
            raise ValueError(f"Message role must be one of: {choices}")

        content = message.get("content")
        if isinstance(content, list):
            unsupported = [part.get("type") for part in content if part.get("type") != "text"]
            if unsupported:
                kinds = ", ".join(str(kind) for kind in unsupported)
                raise ValueError(f"This text-only server does not support content types: {kinds}")

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ValueError("Tool-call history contains invalid JSON arguments") from error
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call history arguments must decode to a JSON object")
            function["arguments"] = arguments
    return normalized


def validate_reasoning_strength(reasoning_strength: str) -> str:
    if reasoning_strength not in _VALID_REASONING_STRENGTHS:
        choices = ", ".join(sorted(_VALID_REASONING_STRENGTHS))
        raise ValueError(f"reasoning_strength must be one of: {choices}")
    return reasoning_strength


def _fallback_content(raw_generation: str) -> str | None:
    visible = re.sub(r"<\|[^|]+\|>", "", raw_generation).strip()
    return visible or None


def parse_tokenizer_response(
    tokenizer,
    response: str | list[int],
    *,
    tools: list[dict[str, Any]] | None = None,
    reached_stop_token: bool,
) -> ParsedAssistantMessage:
    """Adapt Transformers' checkpoint-defined parser to the chat API shape."""

    raw_generation = response if isinstance(response, str) else tokenizer.decode(response, skip_special_tokens=False)
    try:
        parsed = tokenizer.parse_response(response, prefix="<|start|>assistant", tools=tools)
    except (AttributeError, TypeError, ValueError):
        parsed = {"role": "assistant", "content": _fallback_content(raw_generation)}

    tool_calls = []
    for index, tool_call in enumerate(parsed.get("tool_calls") or []):
        function = tool_call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        tool_calls.append(
            {
                "id": tool_call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )

    content = parsed.get("content") or None
    reasoning = parsed.get("reasoning_content") or None
    if not content and not reasoning and not tool_calls:
        content = _fallback_content(raw_generation)
    finish_reason = "length" if not reached_stop_token else ("tool_calls" if tool_calls else "stop")
    recipient = None
    if tool_calls:
        recipient = tool_calls[0]["function"]["name"]
    elif content:
        recipient = "user"
    elif reasoning:
        recipient = "self"
    return ParsedAssistantMessage(content, reasoning, tool_calls, recipient, finish_reason)
