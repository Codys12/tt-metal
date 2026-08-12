# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Convert a DFlash speculator checkpoint to the TT tensorbin cache layout.

The HF safetensors flatten DFlash as:

    fc.weight                                    [out=hidden, in=K*target_hidden]
    hidden_norm.weight                           [hidden]
    norm.weight                                  [hidden]
    layers.{i}.input_layernorm.weight            [hidden]
    layers.{i}.post_attention_layernorm.weight   [hidden]
    layers.{i}.self_attn.q_proj.weight           [Q_dim, hidden]
    layers.{i}.self_attn.k_proj.weight           [KV_dim, hidden]
    layers.{i}.self_attn.v_proj.weight           [KV_dim, hidden]
    layers.{i}.self_attn.o_proj.weight           [hidden, Q_dim]
    layers.{i}.mlp.gate_proj.weight              [intermediate, hidden]
    layers.{i}.mlp.up_proj.weight                [intermediate, hidden]
    layers.{i}.mlp.down_proj.weight              [hidden, intermediate]

For the Gemma variant the checkpoint also contains:

    embed_tokens.weight                          [target_vocab, hidden]
    lm_head.weight                               [draft_vocab, hidden]
    d2t (draft_id_to_target_id)                  [draft_vocab] int64
        — offset encoding: target_id = draft_id + d2t[draft_id]

Qwen3-style variants additionally carry per-layer ``q_norm`` / ``k_norm``
(head_dim-sized RMSNorms). We preserve any such keys we find — the loader
ignores them on Llama-style configs.

Run:
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_CACHE_PATH=/mnt/nas/gemma_cache
    python -m models.demos.gemma4_cody.tt.dflash.convert_weights \\
        --src /mnt/nas/dflash-gemma \\
        --dst $TT_CACHE_PATH/tensor_cache_dflash_bf16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


def _hf_to_cache_path(hf_name: str) -> tuple[str | None, str]:
    """Map an HF state-dict key to ``(subdir, filename)`` under the cache root."""
    # Top-level
    if hf_name == "fc.weight":
        return None, "fc.weight.tensorbin"
    if hf_name == "hidden_norm.weight":
        return None, "hidden_norm.weight.tensorbin"
    if hf_name == "norm.weight":
        return None, "norm.weight.tensorbin"
    if hf_name == "embed_tokens.weight":
        return None, "embed_tokens.weight.tensorbin"
    if hf_name == "lm_head.weight":
        return None, "lm_head.weight.tensorbin"
    # Vocab remap tables — names in the published checkpoint are `d2t` / `t2d`.
    # We accept the longer aliases too in case future versions rename them.
    if hf_name in ("d2t", "draft_id_to_target_id"):
        return None, "d2t.tensorbin"
    if hf_name in ("t2d", "target_id_to_draft_mask"):
        return None, "t2d.tensorbin"

    # layers.{i}.{rest}
    parts = hf_name.split(".")
    if parts[0] != "layers":
        raise ValueError(f"Unrecognized DFlash key: {hf_name}")
    layer_idx = int(parts[1])
    rest = parts[2:]
    layer_dir = f"layer_{layer_idx}"

    if rest[0] in ("input_layernorm", "post_attention_layernorm"):
        return layer_dir, ".".join(rest) + ".tensorbin"
    if rest[0] == "self_attn":
        return f"{layer_dir}/self_attn", ".".join(rest[1:]) + ".tensorbin"
    if rest[0] == "mlp":
        return f"{layer_dir}/mlp", ".".join(rest[1:]) + ".tensorbin"

    raise ValueError(f"Unrecognized DFlash key: {hf_name}")


def convert(src: Path, dst: Path, dtype=torch.bfloat16) -> dict:
    safetensors_path = src / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Expected {safetensors_path}; got {sorted(p.name for p in src.iterdir())}")

    print(f"Loading {safetensors_path} ...")
    weights = load_file(str(safetensors_path))
    total_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    print(f"  loaded {len(weights)} tensors, {total_bytes / 1e9:.2f} GB")

    dst.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for hf_name, tensor in weights.items():
        subdir, filename = _hf_to_cache_path(hf_name)
        out_dir = dst if subdir is None else (dst / subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Index/bool tables keep their semantic dtypes; floats cast to `dtype`.
        if hf_name in ("d2t", "draft_id_to_target_id"):
            t = tensor.to(torch.int32)
        elif hf_name in ("t2d", "target_id_to_draft_mask"):
            t = tensor.to(torch.bool)
        elif tensor.dtype != dtype and tensor.is_floating_point():
            t = tensor.to(dtype)
        else:
            t = tensor
        torch.save(t, str(out_dir / filename))
        n_written += 1

    print(f"Wrote {n_written} tensors to {dst}")
    return {"num_tensors": n_written, "dst": str(dst)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        required=True,
        help="Path to the downloaded HF DFlash checkpoint (must contain model.safetensors + config.json).",
    )
    p.add_argument(
        "--dst",
        default=None,
        help="Output cache directory. Default: $TT_CACHE_PATH/tensor_cache_dflash_bf16",
    )
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = p.parse_args()

    src = Path(args.src).resolve()
    if args.dst is None:
        cache_root = os.environ.get("TT_CACHE_PATH")
        if not cache_root:
            raise SystemExit("Must set $TT_CACHE_PATH or pass --dst")
        dst = Path(cache_root) / "tensor_cache_dflash_bf16"
    else:
        dst = Path(args.dst).resolve()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    convert(src, dst, dtype=dtype)


if __name__ == "__main__":
    main()
