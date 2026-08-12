# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter — block-diffusion speculative-decoding drafter for Gemma-4.

Pairs with the same target model the MTP assistant uses (gemma-4-31B-it,
hidden=5376) but uses a different drafting strategy: 5 Llama-style layers
emit ``block_size=8`` draft tokens in a single non-causal forward pass,
conditioning on 5 target hidden-state taps fused through ``fc`` and reused
inside every layer's K/V (the same ``k_proj`` / ``v_proj`` apply to both the
mask-token "noise" and the projected target context).

Public surface
--------------

  * :class:`DFlashConfig` — HF-config-driven dataclass.
  * :class:`DFlashDrafter` — subclass of :class:`tt.drafter_base.Drafter`.
  * :func:`load_dflash_weights` — converted-cache → TT mesh tensors.
"""

from .config import DFlashConfig
from .model import DFlashDrafter
from .weights import DFlashWeights, load_dflash_weights

__all__ = ["DFlashConfig", "DFlashDrafter", "DFlashWeights", "load_dflash_weights"]
