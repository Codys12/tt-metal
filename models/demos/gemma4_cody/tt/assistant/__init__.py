# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 MTP assistant (drafter) on Tenstorrent.

A 4-layer companion model that consumes the target model's last-layer
hidden state + per-layer-type KV cache and proposes T draft tokens for
speculative decoding.

See README.md in this directory for the architecture map + porting plan.
"""
