# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""N-gram speculator for Gemma4_cody packed-decode integration.

Tracks recent n-grams over a request's token history and proposes draft
continuations when the most-recent ``ngram_size`` tokens match a previously
seen prefix. This is the cheapest speculator that can plug into the
packed-decode verifier (Approach B from
``models/demos/gemma4_cody/tests/unit/test_packed_decode_compare.py``).

Algorithm
---------

Given a sliding window of seen tokens ``... t_{n-k+1}, t_{n-k+2}, ..., t_n``,
look up whether the prefix ``[t_{n-k+1}, ..., t_n]`` has occurred earlier in
the history. If it has, the tokens that followed it last time are the
draft proposal:

    history:        t_0, t_1, ..., t_n
    last k tokens:  t_{n-k+1}, ..., t_n
    if this k-gram appeared before at index i:
        draft = t_{i+k}, t_{i+k+1}, ..., t_{i+k+T-1}

The speculator stores a dict ``(t_{i-k+1}, ..., t_i) -> [tokens that followed]``
and updates it incrementally as new tokens get accepted.

Cost
----
- Lookup: O(1) hashmap probe per step.
- Update: O(k) per accepted token (build new k-gram key).
- Memory: O(unique k-grams) per request.

For natural language at k=3, T=4 the table fits comfortably in tens of MB
per request even for long contexts.

Tuning
------
- ``ngram_size`` (k): larger = pickier matching, higher per-match
  acceptance but lower hit rate. 2-4 typical.
- ``num_draft_tokens`` (T): how many tokens to propose when a match hits.
  Must match the verifier's packed batch size.
- ``min_hits``: only propose drafts when the k-gram has been seen at
  least this many times (filters noisy single-occurrence matches).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple


@dataclass
class NGramSpeculator:
    """N-gram speculator with FIFO continuation memory.

    Maintains a hashtable mapping k-grams seen in this request's history
    to the most recent token(s) that followed them. When the latest k
    tokens match an entry in the table, the stored continuation is offered
    as the draft.
    """

    ngram_size: int = 3
    num_draft_tokens: int = 4
    min_hits: int = 1
    # Cap the per-(k-gram) continuation buffer. We only need the latest
    # ``num_draft_tokens`` since proposing earlier-stored sequences is
    # equivalent to proposing the most recent continuation in expectation.
    max_continuations_per_kgram: int = 8

    _history: List[int] = field(default_factory=list)
    _kgram_to_continuations: Dict[Tuple[int, ...], Deque[List[int]]] = field(default_factory=lambda: defaultdict(deque))
    _kgram_hits: Dict[Tuple[int, ...], int] = field(default_factory=lambda: defaultdict(int))

    def reset(self) -> None:
        """Clear all history (call when a request finishes or a new one starts)."""
        self._history.clear()
        self._kgram_to_continuations.clear()
        self._kgram_hits.clear()

    def observe(self, tokens: Sequence[int]) -> None:
        """Append ``tokens`` to the history and update the n-gram table.

        Call this for every token that gets emitted (whether it was a
        speculated-and-accepted token or a model-emitted token from a
        forced single-token step).
        """
        for tok in tokens:
            self._history.append(tok)
            # Once the history is long enough, record the k-gram that ENDED
            # at position ``self._history[-(k+1)]`` paired with what followed.
            # Specifically: the k-gram ``history[i:i+k]`` was followed by
            # ``history[i+k]`` (and the next few tokens). Store the latest
            # ``num_draft_tokens`` that followed for each k-gram.
            k = self.ngram_size
            if len(self._history) > k:
                kgram = tuple(self._history[-(k + 1) : -1])
                continuation = self._history[-1:]  # just the next token for now
                self._kgram_to_continuations[kgram].append(continuation)
                if len(self._kgram_to_continuations[kgram]) > self.max_continuations_per_kgram:
                    self._kgram_to_continuations[kgram].popleft()
                self._kgram_hits[kgram] += 1

    def propose(self) -> Optional[List[int]]:
        """Return up to ``num_draft_tokens`` draft tokens, or None if no hit.

        Looks at the most recent ``ngram_size`` tokens and checks whether
        that k-gram has been seen at least ``min_hits`` times. If so,
        builds a draft by walking forward through the history starting
        from where the k-gram last occurred.

        Returns:
            A list of up to ``num_draft_tokens`` ints, or None if no
            speculation is appropriate this step.
        """
        k = self.ngram_size
        if len(self._history) < k:
            return None

        latest_kgram = tuple(self._history[-k:])
        if self._kgram_hits.get(latest_kgram, 0) < self.min_hits:
            return None

        continuations = self._kgram_to_continuations.get(latest_kgram)
        if not continuations:
            return None

        # Take the most recent continuation, then extend by walking forward
        # in the history starting from that token. (This implicitly assumes
        # the same k-gram tends to be followed by the same continuation.)
        last_seen = continuations[-1][0]
        # Find where this continuation occurred in history.
        # We know it followed latest_kgram somewhere; search the latest
        # occurrence (linear in worst case; OK for POC).
        draft: List[int] = []
        seen_at = -1
        # Walk history from the end backward to find the most recent
        # occurrence of latest_kgram so we can extend its continuation.
        for i in range(len(self._history) - k - 1, -1, -1):
            if tuple(self._history[i : i + k]) == latest_kgram:
                seen_at = i
                break
        if seen_at < 0:
            return None
        # Walk forward num_draft_tokens from the position AFTER the k-gram.
        start = seen_at + k
        end = min(start + self.num_draft_tokens, len(self._history))
        draft = list(self._history[start:end])
        if not draft:
            return None
        return draft

    def stats(self) -> dict:
        """Snapshot of internal state for diagnostics."""
        return {
            "history_len": len(self._history),
            "unique_kgrams": len(self._kgram_to_continuations),
            "max_kgram_hits": max(self._kgram_hits.values(), default=0),
        }
