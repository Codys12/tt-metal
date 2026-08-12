# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate DiffusionGemma TTNN weight cache (tensor_cache_diff_bf16) without hardware.

TT hosts have ~3 GB RAM, while the first cache build needs to materialize the
1.5 GB embed/lm_head torch tensors. Run this on a big-RAM host with a mock
cluster matching the inference TP:

    export TT_METAL_MOCK_CLUSTER_DESC_PATH=/mnt/nas/1x4.yaml
    export HF_MODEL=/mnt/nas/gemma-diff TT_CACHE_PATH=/mnt/nas/gemma_cache
    python models/demos/gemma4_cody/diffusion/generate_weight_cache.py --mesh-shape 1x4
"""

import argparse
import os
import time

from loguru import logger

import ttnn
from models.demos.gemma4_cody.demo.generate_weight_cache import _parse_mesh_shape, _patch_for_mock_mode_if_needed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("HF_MODEL", "/mnt/nas/gemma-diff"))
    parser.add_argument("--mesh-shape", type=_parse_mesh_shape, default=(1, 4))
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    _patch_for_mock_mode_if_needed()
    if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        # CCL persistent semaphores/buffers are inference scratch, not weights.
        import models.demos.gemma4_cody.diffusion.tt_model as ttm

        class _StubCCL:
            def __init__(self, *a, **k):
                pass

        ttm.CCLManager = _StubCCL

    rows, cols = args.mesh_shape
    fabric = None
    if not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        fabric = ttnn.FabricConfig.FABRIC_1D
        ttnn.set_fabric_config(fabric)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    try:
        t0 = time.time()
        from models.demos.gemma4_cody.diffusion.tt_model import TTDiffusionGemma

        TTDiffusionGemma(
            mesh_device, args.model_path, num_layers=args.num_layers, max_seq_len=args.max_seq_len, cache_gen_only=True
        )
        logger.info(f"Diffusion weight cache generated in {time.time() - t0:.1f}s")
    finally:
        ttnn.close_mesh_device(mesh_device)
        if fabric is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
