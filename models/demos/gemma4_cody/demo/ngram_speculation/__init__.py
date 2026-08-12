# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from .speculator import NGramSpeculator

# MTPDrafter is imported lazily — it pulls in transformers, which is heavy.
# Use `from ...mtp_drafter import MTPDrafter` at the call site when you need it.

__all__ = ["NGramSpeculator"]
