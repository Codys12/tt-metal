# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RoPE helpers shared by the native DFlash server and parity tests."""

import torch


def build_rope_cache(positions: torch.Tensor, head_dim: int, theta: float, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    embedding = torch.cat([freqs, freqs], dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)
