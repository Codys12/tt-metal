# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Convert Gemma4 MTP assistant safetensors to TT cache layout.

Mirrors cody's existing target-model cache pattern at
``/mnt/nas/gemma_cache/tensor_cache_bf16/`` so the assistant cache lives
at ``$TT_CACHE_PATH/tensor_cache_assistant_bf16/``.

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_CACHE_PATH=/mnt/nas/gemma_cache
    python -m models.demos.gemma4_cody.tt.assistant.convert_weights \\
        --src /mnt/nas/gemma-assistant \\
        --dst $TT_CACHE_PATH/tensor_cache_assistant_bf16

This is a host-side conversion: it loads the safetensors, casts each tensor
to bfloat16, and saves to a torch ``.tensorbin``-style layout that cody's
``cached_tensor_placeholder`` can pick up. NO sharding / TP layout is
applied here — that's done at model-load time by the per-tensor loader.

Layout produced
---------------

    $dst/
        embed_tokens.weight.tensorbin
        norm.weight.tensorbin
        pre_projection.weight.tensorbin
        post_projection.weight.tensorbin
        layer_0/
            input_layernorm.weight.tensorbin
            post_attention_layernorm.weight.tensorbin
            pre_feedforward_layernorm.weight.tensorbin
            post_feedforward_layernorm.weight.tensorbin
            layer_scalar.tensorbin
            self_attn/
                q_proj.weight.tensorbin
                q_norm.weight.tensorbin
                o_proj.weight.tensorbin
            mlp/
                gate_proj.weight.tensorbin
                up_proj.weight.tensorbin
                down_proj.weight.tensorbin
        layer_1/ ... layer_3/

The drafter has NO k_proj/v_proj/k_norm/v_norm — those come from the
target via shared_kv_states at runtime, NOT from the drafter weights.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


# HF parameter name → (cache subdirectory, cache filename) under the cache root.
# None means "save at root" with the given filename.
def _hf_to_cache_path(hf_name: str) -> tuple[str | None, str]:
    """Map an HF state-dict key to (subdir, filename) under the cache root."""
    if hf_name == "model.embed_tokens.weight":
        return None, "embed_tokens.weight.tensorbin"
    if hf_name == "model.norm.weight":
        return None, "norm.weight.tensorbin"
    if hf_name == "pre_projection.weight":
        return None, "pre_projection.weight.tensorbin"
    if hf_name == "post_projection.weight":
        return None, "post_projection.weight.tensorbin"

    # model.layers.{i}.{rest}
    parts = hf_name.split(".")
    assert parts[0] == "model" and parts[1] == "layers", f"unexpected name: {hf_name}"
    layer_idx = int(parts[2])
    rest = parts[3:]  # e.g. ["self_attn", "q_proj", "weight"]
    layer_dir = f"layer_{layer_idx}"

    if rest[0] in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        return layer_dir, ".".join(rest) + ".tensorbin"
    if rest[0] == "layer_scalar":
        return layer_dir, "layer_scalar.tensorbin"
    if rest[0] == "self_attn":
        # self_attn.q_proj.weight, self_attn.q_norm.weight, self_attn.o_proj.weight
        return f"{layer_dir}/self_attn", ".".join(rest[1:]) + ".tensorbin"
    if rest[0] == "mlp":
        # mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
        return f"{layer_dir}/mlp", ".".join(rest[1:]) + ".tensorbin"

    raise ValueError(f"Unrecognized HF parameter name: {hf_name}")


def convert(src: Path, dst: Path, dtype=torch.bfloat16) -> dict:
    """Load safetensors from ``src``, cast to ``dtype``, write to ``dst`` layout.

    Returns a stats dict.
    """
    safetensors_path = src / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Expected {safetensors_path}; got {list(src.iterdir())}")

    print(f"Loading {safetensors_path} ...")
    weights = load_file(str(safetensors_path))
    print(
        f"  loaded {len(weights)} tensors, total {sum(t.numel() * t.element_size() for t in weights.values()) / 1e9:.2f} GB"
    )

    dst.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for hf_name, tensor in weights.items():
        subdir, filename = _hf_to_cache_path(hf_name)
        out_dir = dst if subdir is None else (dst / subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        # Cast and save.
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        # Use torch.save for the .tensorbin to match cody's existing pattern
        # (`cached_tensor_placeholder` reads via `torch.load`).
        torch.save(tensor, str(out_path))
        n_written += 1

    print(f"Wrote {n_written} tensors to {dst}")
    print(f"  layout:")
    for path in sorted(dst.rglob("*.tensorbin")):
        size_mb = path.stat().st_size / 1e6
        rel = path.relative_to(dst)
        print(f"    {rel}  ({size_mb:.1f} MB)")
    return {"num_tensors": n_written, "dst": str(dst)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default="/mnt/nas/gemma-assistant",
        help="Path to the downloaded HF assistant checkpoint (must contain model.safetensors + config.json).",
    )
    p.add_argument(
        "--dst",
        default=None,
        help="Output cache directory. Default: $TT_CACHE_PATH/tensor_cache_assistant_bf16",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Output dtype.",
    )
    args = p.parse_args()

    src = Path(args.src).resolve()
    if args.dst is None:
        cache_root = os.environ.get("TT_CACHE_PATH")
        if not cache_root:
            raise SystemExit("Must set $TT_CACHE_PATH or pass --dst")
        dst = Path(cache_root) / "tensor_cache_assistant_bf16"
    else:
        dst = Path(args.dst).resolve()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    convert(src, dst, dtype=dtype)


if __name__ == "__main__":
    main()
