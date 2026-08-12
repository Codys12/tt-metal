# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation: gelu(gate) * up (different from GPT-OSS SwiGLU).
"""

import ttnn

# LoFi math for expert matmuls (bf4 weights: 4 mantissa bits are read in
# full by a single fidelity phase; ~2x compute ceiling vs HiFi2).
_LOFI_KCFG = None


def expert_kernel_config(mesh_device):
    global _LOFI_KCFG
    if _LOFI_KCFG is None:
        _LOFI_KCFG = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    return _LOFI_KCFG


def apply_geglu(gate, up):
    """GeGLU activation: gelu(gate) * up. Gemma4 uses gelu_pytorch_tanh."""
    activated = ttnn.gelu(gate, fast_and_approximate_mode=True)
    result = ttnn.mul(activated, up)
    return result
