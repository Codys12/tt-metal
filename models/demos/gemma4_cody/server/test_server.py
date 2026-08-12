# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-user correctness harness for the Gemma4 OpenAI-compatible server.

Uses context-DEPENDENT secret-extraction prompts: each request embeds a
unique code in its prompt body and asks the model to echo it back. With
correct KV the answer contains the secret verbatim; with corrupted KV
the model usually emits a near-miss (a couple of extra/wrong characters)
or falls into a repetition loop on its prior.

Originally written to repro the device-kernel ``idle→active`` slot
transition bug. With ``--decode-steps-per-admit 8`` (the server's
default) this harness should pass on every scenario; set it to 0 via
``POST /admin/config`` to reproduce the bug.

Usage:
    # default: hit the running backend, run the bug-suspect set
    BASE_URL=http://api.dreamcatcher.co \\
        python -m models.demos.gemma4_cody.server.test_server

    python -m models.demos.gemma4_cody.server.test_server --scenario three_4k
    python -m models.demos.gemma4_cody.server.test_server --scenario all --flush-before

    # bypass the proxy (LAN only)
    BASE_URL=http://192.168.1.189:8001 \\
        python -m models.demos.gemma4_cody.server.test_server --scenario three_4k
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx

DEFAULT_BASE_URL = os.environ.get("BASE_URL", "http://api.dreamcatcher.co")


# Stable lorem-ipsum block — keeps token counts deterministic across runs.
_LOREM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 8000


def _make_prompt(secret: str, target_chars: int) -> str:
    """Context-dependent prompt: the secret is only in the body.

    A corrupted KV cache produces an obvious near-miss (wrong char) or
    a Lorem-ipsum repetition loop, depending on severity.
    """
    if target_chars <= 0:
        return f"The one-time code is {secret}. Reply with just the code."
    body = _LOREM[:target_chars]
    return (
        "You are given a long passage. Embedded somewhere in the passage "
        "is a one-time code. Read carefully.\n\n"
        f"PASSAGE:\n{body}\n"
        f"The one-time code is: {secret}\n"
        f"{body}\n\n"
        "Question: report the exact one-time code from the passage, "
        "character-for-character, and nothing else."
    )


def _new_secret() -> str:
    return f"ZPLX-{uuid.uuid4().hex[:8].upper()}"


# Size tiers chosen to land in each power-of-two prefill bucket in server.py:
#   PREFILL_BUCKET_LENS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
# Lorem ipsum tokenizes to roughly 5 chars/token, so target_chars ≈ 5 * bucket.
_TIER_CHARS = {
    "tiny": 0,  # ~30 tokens     → 1k bucket
    "1k": 800,  # ~250 tokens    → 1k bucket
    "2k": 6_000,  # ~1.5k tokens   → 2k bucket
    "4k": 15_000,  # ~3.5k tokens   → 4k bucket
    "8k": 33_000,  # ~7.5k tokens   → 8k bucket
    "16k": 70_000,  # ~15k tokens    → 16k bucket
    "32k": 140_000,  # ~30k tokens    → 32k bucket
}


@dataclass
class Result:
    label: str
    tier: str
    secret: str
    elapsed: float = 0.0
    first_token_at: Optional[float] = None
    content: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    chunks: int = 0

    @property
    def found_secret(self) -> bool:
        return self.secret in self.content

    @property
    def repetitiveness(self) -> float:
        text = self.content
        if len(text) < 16:
            return 0.0
        grams = [text[i : i + 4] for i in range(len(text) - 3)]
        return 1.0 - (len(set(grams)) / max(len(grams), 1))

    @property
    def top_4gram(self) -> Tuple[str, int]:
        text = self.content
        if len(text) < 16:
            return ("", 0)
        grams = [text[i : i + 4] for i in range(len(text) - 3)]
        return Counter(grams).most_common(1)[0]

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERR "
        if self.found_secret:
            return "OK  "
        if self.repetitiveness > 0.7:
            return "LOOP"
        # Secret not found but no loop — usually a near-miss (off-by-N chars)
        return "MISS"


async def _stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    label: str,
    tier: str,
    secret: str,
    prompt: str,
    max_tokens: int,
) -> Result:
    res = Result(label=label, tier=tier, secret=secret)
    start = time.time()
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Skip the thought channel so 100% of streamed tokens are content.
        "verbosity": "low",
        "stream": True,
    }
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(None, connect=15.0),
        ) as resp:
            if resp.status_code != 200:
                res.error = f"HTTP {resp.status_code}: {(await resp.aread())[:200]}"
                res.elapsed = time.time() - start
                return res
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta") or {}
                if "content" in delta and delta["content"]:
                    if res.first_token_at is None:
                        res.first_token_at = time.time() - start
                    res.content += delta["content"]
                    res.chunks += 1
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    res.reasoning += delta["reasoning_content"]
                fr = ch.get("finish_reason")
                if fr:
                    res.finish_reason = fr
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    res.elapsed = time.time() - start
    return res


async def _flush(client: httpx.AsyncClient, base_url: str, zero_kv: bool = False) -> dict:
    """POST /admin/flush — reset all slots and (optionally) zero KV cache.

    Useful between scenarios so each repro starts from a clean engine state.
    Server must be running with admin endpoints exposed (no auth needed unless
    ENGINE_ADMIN_TOKEN is set on the server side).
    """
    try:
        resp = await client.post(
            f"{base_url}/admin/flush",
            json={"zero_kv": zero_kv},
            timeout=httpx.Timeout(200.0),
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _status(client: httpx.AsyncClient, base_url: str) -> dict:
    try:
        resp = await client.get(f"{base_url}/admin/status", timeout=httpx.Timeout(15.0))
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _config(client: httpx.AsyncClient, base_url: str, **changes) -> dict:
    """POST /admin/config — change runtime knobs.

    Pass any subset of: decode_steps_per_admit, decode_warmup_after_admit,
    decode_log_enabled, completion_archive_enabled. Returns the changed map
    plus a fresh /admin/status snapshot.
    """
    try:
        resp = await client.post(f"{base_url}/admin/config", json=changes, timeout=httpx.Timeout(15.0))
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _decode_log_snapshot(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    limit: int = 5000,
    since: Optional[float] = None,
    slot: Optional[int] = None,
    rid: Optional[str] = None,
    admit_id: Optional[int] = None,
) -> dict:
    params: dict = {"limit": limit}
    if since is not None:
        params["since"] = since
    if slot is not None:
        params["slot"] = slot
    if rid is not None:
        params["rid"] = rid
    if admit_id is not None:
        params["admit_id"] = admit_id
    try:
        resp = await client.get(
            f"{base_url}/admin/decode-log/snapshot",
            params=params,
            timeout=httpx.Timeout(30.0),
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _completions_archive(client: httpx.AsyncClient, base_url: str, limit: int = 50) -> dict:
    try:
        resp = await client.get(f"{base_url}/admin/completions", params={"limit": limit}, timeout=httpx.Timeout(30.0))
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _admits(client: httpx.AsyncClient, base_url: str, limit: int = 100) -> dict:
    try:
        resp = await client.get(f"{base_url}/admin/admits", params={"limit": limit}, timeout=httpx.Timeout(15.0))
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _run(
    base_url: str,
    *,
    label: str,
    specs: List[Tuple[str, str]],  # list of (sub_label, tier)
    max_tokens: int = 100,
    stagger_s: float = 0.0,
    flush_before: bool = False,
    flush_zero_kv: bool = False,
) -> List[Result]:
    print(f"\n=== {label}  (stagger={stagger_s}s, max_tokens={max_tokens}) ===")
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15.0)) as client:
        if flush_before:
            flush_result = await _flush(client, base_url, zero_kv=flush_zero_kv)
            print(f"  [flush] {flush_result}")
        tasks = []
        for i, (sub_label, tier) in enumerate(specs):
            if i and stagger_s:
                await asyncio.sleep(stagger_s)
            secret = _new_secret()
            prompt = _make_prompt(secret, _TIER_CHARS[tier])
            tasks.append(
                asyncio.create_task(_stream_one(client, base_url, sub_label, tier, secret, prompt, max_tokens))
            )
        results = await asyncio.gather(*tasks)

    for r in results:
        ttft = f"{r.first_token_at:.2f}s" if r.first_token_at is not None else "  n/a"
        rep = f"{r.repetitiveness:.2f}"
        head = r.content[:140].replace("\n", " ")
        if len(r.content) > 140:
            head += "…"
        top, n = r.top_4gram
        top_str = f"top={top!r}x{n}" if n > 5 else ""
        if r.error:
            print(f"  [{r.label:<4}] ERR: {r.error}")
            continue
        print(
            f"  [{r.label:<4}] {r.verdict} {r.tier:<6} "
            f"ttft={ttft:>6}  total={r.elapsed:5.1f}s  toks={r.chunks:>4}  "
            f"rep={rep}  finish={r.finish_reason}  {top_str}\n"
            f"     secret={r.secret}\n"
            f"     content: {head}"
        )
    return results


async def _run_late_admit(
    base_url: str,
    *,
    label: str,
    a_max_tokens: int = 240,
    later_max_tokens: int = 60,
    later_count: int = 2,
    a_prefill_s: float = 5.5,  # approx prefill time for a 4k bucket
    delay_after_first_token_s: float = 4.0,
    flush_before: bool = True,
    flush_zero_kv: bool = False,
) -> List[Result]:
    """Production-pattern repro: A is mid-decode when B (and C) admit.

    A is submitted with a large max_tokens so its decode runs for several
    seconds. We wait ``a_prefill_s + delay_after_first_token_s`` and then
    submit B (and C, one more delay later). Each later admission therefore
    happens deep inside A's decode loop, well past any
    --decode-steps-per-admit window, so the admission throttle does not
    gate it.

    This is the production traffic pattern. The lab repros (e.g. three_4k
    with stagger=0.5s) admit B and C within A's first few decode steps,
    which the admission throttle catches. The late-admit case is where
    the throttle workaround leaks.
    """
    print(
        f"\n=== {label}  late-admit (a_max={a_max_tokens}, later_max={later_max_tokens}, "
        f"+{later_count} after prefill≈{a_prefill_s}s + decode {delay_after_first_token_s}s) ==="
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15.0)) as client:
        if flush_before:
            flush_result = await _flush(client, base_url, zero_kv=flush_zero_kv)
            print(f"  [flush] {flush_result}")

        tasks: List[asyncio.Task] = []

        def _spawn(sub_label: str, max_toks: int) -> None:
            secret = _new_secret()
            prompt = _make_prompt(secret, _TIER_CHARS["4k"])
            tasks.append(asyncio.create_task(_stream_one(client, base_url, sub_label, "4k", secret, prompt, max_toks)))

        # A: long-decode, submit immediately.
        _spawn("A", a_max_tokens)
        # Wait past A's prefill + a chunk of decode so B's admission lands
        # well inside A's decode loop (past the throttle window).
        await asyncio.sleep(a_prefill_s + delay_after_first_token_s)
        for i in range(later_count):
            label_later = "B" if i == 0 else ("C" if i == 1 else f"D{i - 1}")
            _spawn(label_later, later_max_tokens)
            # Space subsequent admissions out by the same delay so each
            # also lands mid-decode.
            await asyncio.sleep(delay_after_first_token_s)

        results = await asyncio.gather(*tasks)

    for r in results:
        ttft = f"{r.first_token_at:.2f}s" if r.first_token_at is not None else "  n/a"
        rep = f"{r.repetitiveness:.2f}"
        head = r.content[:140].replace("\n", " ")
        if len(r.content) > 140:
            head += "…"
        top, n = r.top_4gram
        top_str = f"top={top!r}x{n}" if n > 5 else ""
        if r.error:
            print(f"  [{r.label:<4}] ERR: {r.error}")
            continue
        print(
            f"  [{r.label:<4}] {r.verdict} {r.tier:<6} "
            f"ttft={ttft:>6}  total={r.elapsed:5.1f}s  toks={r.chunks:>4}  "
            f"rep={rep}  finish={r.finish_reason}  {top_str}\n"
            f"     secret={r.secret}\n"
            f"     content: {head}"
        )
    return results


# Each scenario is a list of (sub_label, tier). All should pass when the
# server's --decode-steps-per-admit is at its default (8); the three_*,
# five_*, and mixed_* scenarios fail when it's set to 0.
SCENARIOS = {
    # Baselines.
    "single_1k": [("S0", "1k")],
    "single_4k": [("S0", "4k")],
    "single_16k": [("S0", "16k")],
    "two_tiny": [("T0", "tiny"), ("T1", "tiny")],
    "two_1k": [("S0", "1k"), ("S1", "1k")],
    # 2-user pairs at progressively larger prefill buckets.
    "two_4k": [("A", "4k"), ("B", "4k")],
    "two_16k": [("A", "16k"), ("B", "16k")],
    "two_32k": [("A", "32k"), ("B", "32k")],
    # The bug-repro set: 3+ slots, large enough that prefill straddles
    # multiple decode steps of the prior slot.
    "three_4k": [("A", "4k"), ("B", "4k"), ("C", "4k")],
    "three_16k": [("A", "16k"), ("B", "16k"), ("C", "16k")],
    "five_4k": [("R0", "4k"), ("R1", "4k"), ("R2", "4k"), ("R3", "4k"), ("R4", "4k")],
    # Mixed bucket sizes — different prefill traces for adjacent admissions.
    "mixed_small": [("S", "1k"), ("M", "4k"), ("L", "16k")],
    "mixed_large": [("S", "1k"), ("L1", "16k"), ("L2", "16k")],
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS.keys()) + ("all", "smoke", "repro", "late_admit"),
        default="repro",
        help="Which scenario(s) to run. 'repro' runs the bug-trigger set. "
        "'late_admit' is the production-pattern repro (A mid-decode, B/C "
        "admit after A's first-token + delay).",
    )
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument(
        "--stagger-s",
        type=float,
        default=2.0,
        help="Delay between launching concurrent requests, in seconds. "
        "Critical for repro — admission ordering matters.",
    )
    parser.add_argument(
        "--flush-before",
        action="store_true",
        help="POST /admin/flush before each scenario so every run starts "
        "from a fresh engine state (cancels in-flight, frees pages, "
        "resets sampling). Requires server-side admin endpoints.",
    )
    parser.add_argument(
        "--flush-zero-kv",
        action="store_true",
        help="When flushing, also zero every KV cache buffer. Heavy but "
        "rules out stale-cache-data as the source of corruption.",
    )
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="GET /admin/status before each scenario and dump the snapshot.",
    )
    # Diagnostic knobs — set via /admin/config before scenarios run. These let
    # us A/B the warmup hypothesis and record forensic traces without
    # restarting the server. All optional; leaving them unset preserves the
    # server's existing config.
    parser.add_argument(
        "--throttle",
        type=int,
        default=None,
        help="Set decode_steps_per_admit before running. Server-side default "
        "is 8. Use 0 to reproduce the bug, higher (16/32/64) to test if a "
        "longer cooldown alone fixes it.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Set decode_warmup_after_admit before running. The slot replays "
        "the last prompt token N extra times before user-visible decode. "
        "Tests the cold-L1 hypothesis: if a small N eliminates corruption, "
        "the bug is in idle→active kernel state.",
    )
    parser.add_argument(
        "--compact-mode",
        choices=("single", "full", "none"),
        default=None,
        help="Set compact_mode before running. 'single' (server default) "
        "moves one slot per compact call. 'full' restores the cascade "
        "behavior that corrupts 3+ slot scenarios. 'none' disables compaction.",
    )
    parser.add_argument(
        "--enable-decode-log",
        action="store_true",
        help="Turn on /admin/decode-log before running. Per-step per-slot "
        "(input_token, output_token, cur_pos, is_warmup, ...) for forensic "
        "analysis. Use with --dump-decode-log to write the snapshot to disk.",
    )
    parser.add_argument(
        "--dump-decode-log",
        default=None,
        help="After running scenarios, GET /admin/decode-log/snapshot and " "write the entries to this path (NDJSON).",
    )
    parser.add_argument(
        "--enable-archive",
        action="store_true",
        help="Turn on /admin/completions archive before running. Captures "
        "the full output token sequence + admit metadata for each finished "
        "request.",
    )
    parser.add_argument(
        "--dump-archive",
        default=None,
        help="After running, GET /admin/completions and write the archive " "to this path (NDJSON).",
    )
    parser.add_argument(
        "--dump-admits",
        default=None,
        help="After running, GET /admin/admits and write the admit history " "to this path (NDJSON).",
    )
    args = parser.parse_args()

    # Apply diagnostic config changes once up-front. Toggles persist on the
    # server across this whole invocation (and beyond — caller's responsibility
    # to reset them via another /admin/config if they want clean prod state).
    pending_config: dict = {}
    if args.throttle is not None:
        pending_config["decode_steps_per_admit"] = int(args.throttle)
    if args.warmup is not None:
        pending_config["decode_warmup_after_admit"] = int(args.warmup)
    if args.compact_mode is not None:
        pending_config["compact_mode"] = args.compact_mode
    if args.enable_decode_log:
        pending_config["decode_log_enabled"] = True
    if args.enable_archive:
        pending_config["completion_archive_enabled"] = True
    if pending_config:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15.0)) as c:
            resp = await _config(c, args.base_url, **pending_config)
            print(f"[config] applied={pending_config} response={resp}")

    print(f"base_url   = {args.base_url}")
    print(f"max_tokens = {args.max_tokens}")
    print(f"stagger_s  = {args.stagger_s}")
    print("tier sizes:")
    for tier, chars in _TIER_CHARS.items():
        sample = _make_prompt("ZPLX-XXXXXXXX", chars)
        print(f"  '{tier}': ~{len(sample)} chars (~{len(sample)//5} tokens)")

    if args.scenario == "late_admit":
        # Production-pattern repro: A mid-decode, B/C admit after A's TTFT.
        await _run_late_admit(
            args.base_url,
            label="late_admit",
            a_max_tokens=240,
            later_max_tokens=args.max_tokens,
            later_count=2,
            delay_after_first_token_s=4.0,
            flush_before=args.flush_before,
            flush_zero_kv=args.flush_zero_kv,
        )
        scenarios = []
    elif args.scenario == "all":
        scenarios = list(SCENARIOS.keys())
    elif args.scenario == "smoke":
        scenarios = ["single_1k", "two_1k", "two_4k", "three_4k"]
    elif args.scenario == "repro":
        # The set that reliably triggers the bug on the production server.
        scenarios = ["three_4k", "three_16k", "five_4k", "mixed_large"]
    else:
        scenarios = [args.scenario]

    for s in scenarios:
        if args.show_status:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15.0)) as c:
                print(f"\n[status before {s}]: {await _status(c, args.base_url)}")
        # Single-user scenarios run with no stagger.
        stag = 0.0 if len(SCENARIOS[s]) == 1 else args.stagger_s
        await _run(
            args.base_url,
            label=s,
            specs=SCENARIOS[s],
            max_tokens=args.max_tokens,
            stagger_s=stag,
            flush_before=args.flush_before,
            flush_zero_kv=args.flush_zero_kv,
        )

    # Post-run diagnostic dumps. Written as NDJSON so each line is a complete
    # entry — pipes cleanly through `jq -c` for ad-hoc filtering.
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15.0)) as c:
        if args.dump_decode_log:
            snap = await _decode_log_snapshot(c, args.base_url, limit=200_000)
            entries = snap.get("entries", []) if isinstance(snap, dict) else []
            with open(args.dump_decode_log, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            print(f"[dump] decode-log: {len(entries)} entries → {args.dump_decode_log}")
        if args.dump_archive:
            snap = await _completions_archive(c, args.base_url, limit=200)
            entries = snap.get("entries", []) if isinstance(snap, dict) else []
            with open(args.dump_archive, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            print(f"[dump] completions: {len(entries)} entries → {args.dump_archive}")
        if args.dump_admits:
            snap = await _admits(c, args.base_url, limit=1000)
            entries = snap.get("entries", []) if isinstance(snap, dict) else []
            with open(args.dump_admits, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            print(f"[dump] admits: {len(entries)} entries → {args.dump_admits}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
