# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tiny interactive client for the Muse Glimmer OpenAI-compatible server."""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import sys
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "time.get_current_time",
            "description": "Get the current date and time in an IANA timezone such as UTC, America/New_York, or Asia/Tokyo.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "IANA timezone name; defaults to UTC."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator.calculate",
            "description": "Evaluate arithmetic using numbers, parentheses, +, -, *, /, //, %, and **.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text.get_statistics",
            "description": "Count the characters, words, and lines in text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]


class ToolError(ValueError):
    pass


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calculate(expression: str) -> int | float:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 256:
        raise ToolError("expression must be a non-empty string of at most 256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ToolError("invalid arithmetic expression") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ToolError("expression is too complex")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("absolute exponent must not exceed 100")
            result = _BINARY_OPERATORS[type(node.op)](left, right)
            if isinstance(result, complex) or abs(result) > 1e100:
                raise ToolError("result is outside the supported range")
            return result
        raise ToolError("expression contains an unsupported operation")

    try:
        result = evaluate(tree)
    except (ArithmeticError, OverflowError) as error:
        raise ToolError(str(error)) from error
    if isinstance(result, float) and not math.isfinite(result):
        raise ToolError("result must be finite")
    return result


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one allow-listed local tool and return JSON-serializable output."""
    if not isinstance(arguments, dict):
        raise ToolError("tool arguments must be a JSON object")
    if name == "time.get_current_time":
        timezone = arguments.get("timezone", "UTC")
        if not isinstance(timezone, str):
            raise ToolError("timezone must be a string")
        try:
            now = datetime.now(ZoneInfo(timezone))
        except ZoneInfoNotFoundError as error:
            raise ToolError(f"unknown IANA timezone: {timezone}") from error
        return {
            "timezone": timezone,
            "datetime": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "utc_offset": now.strftime("%z"),
        }
    if name == "calculator.calculate":
        return {"expression": arguments.get("expression"), "result": _calculate(arguments.get("expression"))}
    if name == "text.get_statistics":
        value = arguments.get("text")
        if not isinstance(value, str):
            raise ToolError("text must be a string")
        return {
            "characters": len(value),
            "words": len(value.split()),
            "lines": len(value.splitlines()) if value else 0,
        }
    raise ToolError(f"unknown tool: {name}")


def _completion_url(server: str) -> str:
    server = server.rstrip("/")
    return server if server.endswith("/v1/chat/completions") else server + "/v1/chat/completions"


def stream_completion(
    server: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    reasoning_strength: str,
    current_date: str,
    timeout: float,
):
    """Yield decoded OpenAI SSE events until the server's [DONE] marker."""
    payload = {
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "reasoning_strength": reasoning_strength,
        "current_date": current_date,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        _completion_url(server),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise RuntimeError("server returned malformed SSE JSON") from error
                if "error" in event:
                    detail = event["error"]
                    message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                    raise RuntimeError(f"streaming server error: {message}")
                yield event
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"cannot reach {_completion_url(server)}: {error.reason}") from error


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    value = call.get("function", {}).get("arguments", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ToolError(f"invalid JSON arguments: {error.msg}") from error
    if not isinstance(value, dict):
        raise ToolError("tool arguments must be a JSON object")
    return value


def _print_metrics(response: dict[str, Any]) -> None:
    metrics = response.get("dflash", {})
    ar_tokens = metrics.get("ar_decode_tokens", 0)
    ar_rate = metrics.get("ar_decode_tokens_per_second", 0.0)
    cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prefill_rate = metrics.get("prefill_tokens_per_second", 0.0)
    print(
        f"[AR {ar_tokens} tokens @ {ar_rate:.2f} tok/s; prefill {prefill_rate:.2f} tok/s; cached {cached}]",
        file=sys.stderr,
    )


def chat_turn(
    server: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    reasoning_strength: str,
    current_date: str,
    timeout: float,
    max_tool_rounds: int = 8,
    show_reasoning: bool = False,
) -> str:
    """Run one user turn, including any model-requested local tool rounds."""
    for _ in range(max_tool_rounds + 1):
        content_parts = []
        reasoning_parts = []
        calls_by_index: dict[int, dict[str, Any]] = {}
        printed_content = False
        for event in stream_completion(
            server,
            messages,
            max_tokens=max_tokens,
            reasoning_strength=reasoning_strength,
            current_date=current_date,
            timeout=timeout,
        ):
            if event.get("dflash") is not None:
                _print_metrics(event)
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if not printed_content:
                    print("assistant> ", end="", flush=True)
                    printed_content = True
                print(content, end="", flush=True)
                content_parts.append(content)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
                if show_reasoning:
                    print(reasoning, end="", flush=True, file=sys.stderr)
            for call_delta in delta.get("tool_calls") or []:
                index = int(call_delta.get("index", 0))
                call = calls_by_index.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call_delta.get("id"):
                    call["id"] = call_delta["id"]
                if call_delta.get("type"):
                    call["type"] = call_delta["type"]
                function_delta = call_delta.get("function") or {}
                if function_delta.get("name"):
                    call["function"]["name"] += function_delta["name"]
                if function_delta.get("arguments"):
                    call["function"]["arguments"] += function_delta["arguments"]
        if printed_content:
            print()
        calls = [calls_by_index[index] for index in sorted(calls_by_index)]
        message = {"role": "assistant", "content": "".join(content_parts) or None}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if calls:
            message["tool_calls"] = calls
        messages.append(message)
        if not calls:
            if not printed_content:
                print("assistant> ")
            return message.get("content") or ""
        for call in calls:
            name = call.get("function", {}).get("name", "")
            try:
                arguments = _arguments(call)
                output = execute_tool(name, arguments)
                print(f"tool> {name}({json.dumps(arguments, ensure_ascii=False)})", file=sys.stderr)
            except ToolError as error:
                output = {"error": str(error)}
                print(f"tool> {name} failed: {error}", file=sys.stderr)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", "call_0"),
                    "content": json.dumps(output, ensure_ascii=False),
                }
            )
    raise RuntimeError(f"model exceeded the limit of {max_tool_rounds} consecutive tool rounds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="send one prompt and exit instead of starting an interactive chat")
    parser.add_argument("--server", default=os.getenv("MUSE_GLIMMER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-strength", choices=("low", "medium", "high", "xhigh"), default="low")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--show-reasoning", action="store_true")
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    messages: list[dict[str, Any]] = []
    session_date = datetime.now(ZoneInfo("UTC")).date().isoformat()

    def send(text: str) -> None:
        messages.append({"role": "user", "content": text})
        chat_turn(
            args.server,
            messages,
            max_tokens=args.max_tokens,
            reasoning_strength=args.reasoning_strength,
            current_date=session_date,
            timeout=args.timeout,
            show_reasoning=args.show_reasoning,
        )

    try:
        if args.prompt:
            send(args.prompt)
            return 0
        print("Muse Glimmer chat. Commands: /clear, /help, /quit")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                print()
                return 0
            if not text:
                continue
            if text in ("/quit", "/exit"):
                return 0
            if text == "/clear":
                messages.clear()
                print("Conversation cleared.")
                continue
            if text == "/help":
                print("Ask normally; time, calculator, and text-statistics tools run automatically.")
                continue
            send(text)
    except (RuntimeError, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            print(file=sys.stderr)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
