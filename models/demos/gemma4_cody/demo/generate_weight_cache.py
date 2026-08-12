# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate Gemma4 TTNN weight cache files for the text demo.

This intentionally uses the same create_tt_model() path as text_demo.py so the
emitted tensorbin names and layouts match what the demo later tries to load.
"""

import argparse
import os
import time

from loguru import logger

import ttnn
from models.demos.gemma4_cody.tt.common import create_tt_model
from models.tt_transformers.tt.common import PagedAttentionConfig


def _parse_mesh_shape(value):
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mesh shape must be ROWSxCOLS, for example 1x8")
    rows, cols = (int(part) for part in parts)
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("mesh shape dimensions must be positive")
    return rows, cols


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "/mnt/nas/gemma"),
        help="Path to the HuggingFace Gemma4 checkpoint.",
    )
    parser.add_argument(
        "--cache-path",
        default=os.getenv("TT_CACHE_PATH", "/mnt/nas/gemma_cache"),
        help="Root cache directory. The demo will create tensor_cache_bf16 under this path.",
    )
    parser.add_argument("--mesh-shape", type=_parse_mesh_shape, default=(1, 8), help="Mesh shape, for example 1x8.")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="KV cache sequence length to match the demo.")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Batch size to match the demo.")
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Optional layer count for partial cache generation."
    )
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        help="Do not enable FABRIC_1D before opening the mesh device.",
    )
    return parser.parse_args()


def _patch_for_mock_mode_if_needed():
    """When running against a mock cluster descriptor, skip device-side
    scratch-buffer allocations that would fail without a real command queue.

    The only such allocation during ``create_tt_model(create_kv_cache=False,
    create_rope_cache=False)`` is the fused matmul-reduce-scatter persistent
    buffer pair created inside ``Gemma4Attention.__init__`` and
    ``SharedMLP.__init__``. They're scratch space for the inference-time
    fused CCL kernel, not weight tensors — cache generation produces no
    tensorbins for them, so returning ``(None, None)`` is a clean no-op
    for the cache-gen path and harmless on real hardware (because we only
    patch when mock mode is detected).
    """
    if not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        return
    import models.demos.gemma4_cody.tt.attention as _attn_mod
    from models.demos.gemma4_cody.tt import ccl as _ccl_mod
    from models.demos.gemma4_cody.tt import model as _model_mod
    from models.demos.gemma4_cody.tt import shared_mlp as _mlp_mod

    def _stub_buffers(*_args, **_kwargs):
        return None, None

    _ccl_mod.make_reduce_scatter_persistent_buffers = _stub_buffers
    _attn_mod.make_reduce_scatter_persistent_buffers = _stub_buffers
    _mlp_mod.make_reduce_scatter_persistent_buffers = _stub_buffers

    # SamplingGenerator allocates per-user k/p/temp/seed device tensors at
    # __init__ via uncached ttnn.from_torch. Cache generation never invokes
    # the sampling op, so replace the type with a marker that satisfies the
    # ``self.sampling is not None`` checks downstream — we don't need any
    # methods because nothing calls them in cache-gen.
    class _StubSampling:
        def __init__(self, *_args, **_kwargs):
            pass

    _model_mod.SamplingGenerator = _StubSampling

    logger.info(
        "Mock cluster descriptor detected (TT_METAL_MOCK_CLUSTER_DESC_PATH); "
        "stubbing make_reduce_scatter_persistent_buffers and SamplingGenerator "
        "for cache generation."
    )


def main():
    args = parse_args()

    os.environ["HF_MODEL"] = args.model_path
    os.environ["GEMMA4_MODEL_PATH"] = args.model_path
    os.environ["TT_CACHE_PATH"] = args.cache_path

    _patch_for_mock_mode_if_needed()

    rows, cols = args.mesh_shape
    num_devices = rows * cols
    available_devices = ttnn.get_num_devices()
    if num_devices > available_devices:
        raise RuntimeError(f"Requested {num_devices} devices but only {available_devices} are available")

    fabric_config = None
    if not args.disable_fabric:
        fabric_config = ttnn.FabricConfig.FABRIC_1D
        ttnn.set_fabric_config(fabric_config)

    mesh_device = None
    start = time.time()
    try:
        logger.info(
            "Generating Gemma4 weight cache: model_path={}, cache_path={}, mesh_shape={}x{}, max_seq_len={}, layers={}",
            args.model_path,
            args.cache_path,
            rows,
            cols,
            args.max_seq_len,
            args.num_layers or "all",
        )
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
        page_params = {"page_block_size": 64, "page_max_num_blocks": args.max_seq_len // 64}
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        model_args, _model, _tt_kv_cache, _state_dict = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            model_path=args.model_path,
            create_kv_cache=False,
            create_rope_cache=False,
            paged_attention_config=paged_attention_config,
        )
        logger.info(
            "Generated Gemma4 cache for {} layers at {} in {:.1f}s",
            args.num_layers or model_args.num_hidden_layers,
            model_args.weight_cache_path(args.model_path, ttnn.bfloat16),
            time.time() - start,
        )
    finally:
        if mesh_device is not None:
            for submesh in mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
