# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter weight loader (TT mesh tensors from converted cache).

Mirrors the layout of :mod:`models.demos.gemma4_cody.tt.assistant.weights`
with three additions that DFlash needs:

  1. ``fc`` — projects the concatenated K aux target hiddens
     ``[B, S, K*target_hidden]`` down to ``[B, S, hidden]``.
  2. ``hidden_norm`` — RMSNorm applied to ``fc(target_hidden)``.
  3. Full Q/K/V per layer instead of Q-only (DFlash owns its own KV).

Plus the narrow LM head + remap table introduced by the Gemma variant:

  * ``lm_head`` — ``[draft_vocab, hidden]``, applied after the final norm.
  * ``draft_id_to_target_id`` — int32 ``[draft_vocab]`` lookup that maps
    32k drafter argmaxes back into the target's 262k vocab.

TP sharding mirrors the MTP assistant: column-parallel q/k/v (sharded on
the head dim) and replicated o/MLP, with ``ccl_allgather`` between SDPA and
``o_proj``.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

import ttnn

from ...utils.lazy_state_dict import LazyStateDict


@dataclass(frozen=True)
class DFlashLayerWeights:
    input_layernorm: ttnn.Tensor
    post_attention_layernorm: ttnn.Tensor

    q_proj: ttnn.Tensor  # [hidden, Q_dim]   — column-parallel on Q heads
    k_proj: ttnn.Tensor  # [hidden, KV_dim]  — column-parallel on KV heads
    v_proj: ttnn.Tensor  # [hidden, KV_dim]  — column-parallel on KV heads
    o_proj: ttnn.Tensor  # [Q_dim, hidden]   — replicated (after all-gather)

    # Per-head norms — present in the published Gemma DFlash despite its
    # ``model_type: llama`` config. They're applied to Q (post q_proj+reshape)
    # and to K (post the cat of context-K and noise-K, before RoPE).
    q_norm: ttnn.Tensor  # [head_dim]
    k_norm: ttnn.Tensor  # [head_dim]

    mlp_gate: ttnn.Tensor  # [hidden, intermediate]
    mlp_up: ttnn.Tensor  # [hidden, intermediate]
    mlp_down: ttnn.Tensor  # [intermediate, hidden]


@dataclass(frozen=True)
class DFlashWeights:
    fc: ttnn.Tensor  # [K*target_hidden, hidden]
    hidden_norm: ttnn.Tensor  # [hidden]
    final_norm: ttnn.Tensor  # [hidden]
    layers: List[DFlashLayerWeights]

    # Optional pieces — Gemma DFlash ships its own; Qwen variant reuses target's.
    embed_tokens: Optional[ttnn.Tensor]  # [target_vocab, hidden] — for mask + noise embedding
    lm_head: Optional[ttnn.Tensor]  # [hidden, draft_vocab] — narrow draft head
    draft_id_to_target_id: Optional[ttnn.Tensor]  # int32 [draft_vocab]


def _load_cached(
    cache_dir: Path,
    rel_path: str,
    mesh_device,
    dtype,
    layout,
    mesh_mapper,
    *,
    transpose_for_matmul: bool = False,
    reshape_for_norm: bool = False,
) -> ttnn.Tensor:
    abs_path = cache_dir / rel_path
    if not abs_path.is_file():
        raise FileNotFoundError(
            f"Missing weight cache file {abs_path}. "
            "Run `python -m models.demos.gemma4_cody.tt.dflash.convert_weights` first."
        )
    tensor = torch.load(str(abs_path), weights_only=True)
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if transpose_for_matmul:
        tensor = tensor.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
    elif reshape_for_norm:
        if tensor.dim() == 1 and tensor.shape[0] >= ttnn.TILE_SIZE and tensor.shape[0] % ttnn.TILE_SIZE == 0:
            tensor = tensor.reshape(1, 1, -1, ttnn.TILE_SIZE).contiguous()
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        mesh_mapper=mesh_mapper,
    )


def _from_hf_lazy(
    lsd: LazyStateDict,
    key: str,
    mesh_device,
    dtype,
    layout,
    mesh_mapper,
    *,
    transpose_for_matmul: bool = False,
    reshape_for_norm: bool = False,
    on_raw=None,
) -> ttnn.Tensor:
    """Stream a single tensor from safetensors via :class:`LazyStateDict`.

    Reads ``key`` from the lazy state dict, applies the matmul-transpose /
    norm-reshape on the host, transfers to the mesh, and drops every
    intermediate host buffer before returning. Peak host RAM is bounded by
    one tensor's transposed-contiguous size (≈ 2 × bf16 size of the tensor).

    ``on_raw(key, tensor)`` (optional) is invoked with the host tensor AFTER the
    bf16 cast but BEFORE the transpose/reshape — i.e. exactly the raw layout
    :func:`convert_weights.convert` persists — so a streaming load can also
    write the tensorbin cache for next time without re-reading the safetensors.
    """
    tensor = lsd[key]
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if on_raw is not None:
        on_raw(key, tensor)
    if transpose_for_matmul:
        tensor_t = tensor.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
        del tensor
        tensor = tensor_t
    elif reshape_for_norm:
        if tensor.dim() == 1 and tensor.shape[0] >= ttnn.TILE_SIZE and tensor.shape[0] % ttnn.TILE_SIZE == 0:
            tensor = tensor.reshape(1, 1, -1, ttnn.TILE_SIZE).contiguous()
    tt_tensor = ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        mesh_mapper=mesh_mapper,
    )
    del tensor
    gc.collect()
    return tt_tensor


def load_dflash_weights(
    mesh_device,
    config,  # DFlashConfig
    cache_dir: str | Path | None = None,
    weight_dtype=ttnn.bfloat16,
    *,
    safetensors_dir: str | Path | None = None,
    load_embed_tokens: bool = True,
    write_cache_dir: str | Path | None = None,
) -> DFlashWeights:
    """Load DFlash weights for the TT mesh.

    Two source modes (``cache_dir`` and ``safetensors_dir`` are mutually
    exclusive — pass exactly one):

      * ``cache_dir`` — converted tensorbin layout produced by
        :mod:`tt.dflash.convert_weights`. Per-tensor ``torch.load`` →
        ``ttnn.from_torch`` flow; same eager pattern the MTP assistant uses.
      * ``safetensors_dir`` — directory containing the source
        ``model.safetensors``. Streams tensors lazily via
        :class:`LazyStateDict` — one tensor in RAM at a time, freed before
        the next is read. Needed on RAM-constrained hosts where the eager
        path's peak (≈ 2× the largest weight, ≈ 5.4 GB for ``embed_tokens``)
        would OOM.

    ``load_embed_tokens=False`` skips the 2.7 GB ``embed_tokens.weight``
    altogether — the drafter's ``forward()`` doesn't use it directly
    (caller supplies pre-embedded ``noise_embeddings``); the server uses it
    only if it needs to embed mask tokens on-device.
    """
    if safetensors_dir is not None and cache_dir is not None:
        raise ValueError("Pass exactly one of cache_dir or safetensors_dir")
    if safetensors_dir is not None:
        return _load_dflash_weights_lazy(
            mesh_device=mesh_device,
            config=config,
            safetensors_dir=Path(safetensors_dir),
            weight_dtype=weight_dtype,
            load_embed_tokens=load_embed_tokens,
            write_cache_dir=Path(write_cache_dir) if write_cache_dir else None,
        )
    if cache_dir is None:
        cache_root = os.environ.get("TT_CACHE_PATH")
        if not cache_root:
            raise RuntimeError("Pass cache_dir or set $TT_CACHE_PATH")
        cache_dir = Path(cache_root) / "tensor_cache_dflash_bf16"
    cache_dir = Path(cache_dir)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tp = mesh_device.shape[1] if is_mesh else 1
    col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3) if tp > 1 else replicate

    def load_matmul(rel, layout=ttnn.TILE_LAYOUT, mesh_mapper=None, dtype=None):
        return _load_cached(
            cache_dir,
            rel,
            mesh_device,
            dtype or weight_dtype,
            layout,
            mesh_mapper if mesh_mapper is not None else replicate,
            transpose_for_matmul=True,
        )

    def load_norm(rel):
        return _load_cached(
            cache_dir,
            rel,
            mesh_device,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            replicate,
            reshape_for_norm=True,
        )

    def load_lookup(rel, dtype):
        # int32 lookup table — keep as ROW_MAJOR int32, replicated. Optional.
        path = cache_dir / rel
        if not path.is_file():
            return None
        tensor = torch.load(str(path), weights_only=True).to(dtype)
        return ttnn.from_torch(
            tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # [1,1,1,N]
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32 if dtype == torch.int32 else ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def load_optional_matmul(rel, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
        if not (cache_dir / rel).is_file():
            return None
        return load_matmul(rel, layout=layout, mesh_mapper=mesh_mapper)

    # NOTE: bf8 fc/MLP was tried (halves the heaviest weight streams) and
    # dropped acceptance from ~6 to ~2.4 tokens/step — drafter quality is
    # precision-bound. Keep bf16 everywhere.
    fc = load_matmul("fc.weight.tensorbin")
    hidden_norm = load_norm("hidden_norm.weight.tensorbin")
    final_norm = load_norm("norm.weight.tensorbin")

    # Gemma variant: own narrow lm_head + remap; column-parallel on draft_vocab dim.
    # embed_tokens is 2.7 GB bf16 (262144 × 5376) — peak host RAM during the
    # contiguous() inside `_load_cached` is ~5.4 GB, so skip it when not needed.
    embed_tokens = (
        load_optional_matmul("embed_tokens.weight.tensorbin", mesh_mapper=col_mapper) if load_embed_tokens else None
    )
    lm_head = load_optional_matmul("lm_head.weight.tensorbin", mesh_mapper=col_mapper)
    # Vocab remap — the published checkpoint stores this as `d2t`.
    draft_id_to_target_id = load_lookup("d2t.tensorbin", torch.int32)

    layers: List[DFlashLayerWeights] = []
    for i in range(config.num_hidden_layers):
        layer_dir = f"layer_{i}"
        layers.append(
            DFlashLayerWeights(
                input_layernorm=load_norm(f"{layer_dir}/input_layernorm.weight.tensorbin"),
                post_attention_layernorm=load_norm(f"{layer_dir}/post_attention_layernorm.weight.tensorbin"),
                q_proj=load_matmul(f"{layer_dir}/self_attn/q_proj.weight.tensorbin", mesh_mapper=col_mapper),
                k_proj=load_matmul(f"{layer_dir}/self_attn/k_proj.weight.tensorbin", mesh_mapper=col_mapper),
                v_proj=load_matmul(f"{layer_dir}/self_attn/v_proj.weight.tensorbin", mesh_mapper=col_mapper),
                o_proj=load_matmul(f"{layer_dir}/self_attn/o_proj.weight.tensorbin"),
                q_norm=load_norm(f"{layer_dir}/self_attn/q_norm.weight.tensorbin"),
                k_norm=load_norm(f"{layer_dir}/self_attn/k_norm.weight.tensorbin"),
                mlp_gate=load_matmul(f"{layer_dir}/mlp/gate_proj.weight.tensorbin"),
                mlp_up=load_matmul(f"{layer_dir}/mlp/up_proj.weight.tensorbin"),
                mlp_down=load_matmul(f"{layer_dir}/mlp/down_proj.weight.tensorbin"),
            )
        )

    return DFlashWeights(
        fc=fc,
        hidden_norm=hidden_norm,
        final_norm=final_norm,
        layers=layers,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        draft_id_to_target_id=draft_id_to_target_id,
    )


def _load_dflash_weights_lazy(
    mesh_device,
    config,
    safetensors_dir: Path,
    weight_dtype,
    load_embed_tokens: bool,
    write_cache_dir: Path | None = None,
) -> DFlashWeights:
    """Streaming TT weight loader — reads each tensor from
    ``model.safetensors`` via :class:`LazyStateDict`, transfers to the mesh,
    frees the host buffer, then moves on. Bounded peak host RAM.

    Same TP layout as the eager path: column-parallel q/k/v + lm_head,
    replicated everything else.

    When ``write_cache_dir`` is set, each streamed tensor is ALSO persisted (in
    the exact raw layout :mod:`convert_weights` produces) so the next server run
    can use the fast eager path. Tensors are written to a temp sibling dir and
    atomically renamed into place only after a fully successful load, so a crash
    mid-load never leaves a half-written cache that the eager path would later
    fail to read. Note the cache reflects ``load_embed_tokens`` — a load with
    ``load_embed_tokens=False`` (the server default) writes no ``embed_tokens``
    entry, which is fine for every server consumer (noise uses the target embed)
    but means the cache is not a drop-in for an embed-needing eager load.
    """
    import shutil
    import tempfile

    from .convert_weights import _hf_to_cache_path

    lsd = LazyStateDict(safetensors_dir)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tp = mesh_device.shape[1] if is_mesh else 1
    col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3) if tp > 1 else replicate

    cache_tmp = None
    if write_cache_dir is not None:
        write_cache_dir = Path(write_cache_dir)
        write_cache_dir.parent.mkdir(parents=True, exist_ok=True)
        cache_tmp = Path(tempfile.mkdtemp(prefix=write_cache_dir.name + ".partial-", dir=str(write_cache_dir.parent)))

    def _save_raw(key, tensor):
        # Persist the raw (pre-transpose/reshape) host tensor under the same
        # path convert_weights uses, so the eager loader reads it identically.
        if cache_tmp is None:
            return
        subdir, filename = _hf_to_cache_path(key)
        out_dir = cache_tmp if subdir is None else (cache_tmp / subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, str(out_dir / filename))

    def matmul(key, mesh_mapper=None, dtype=None):
        return _from_hf_lazy(
            lsd,
            key,
            mesh_device,
            dtype or weight_dtype,
            ttnn.TILE_LAYOUT,
            mesh_mapper if mesh_mapper is not None else replicate,
            transpose_for_matmul=True,
            on_raw=_save_raw,
        )

    def norm(key):
        return _from_hf_lazy(
            lsd,
            key,
            mesh_device,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            replicate,
            reshape_for_norm=True,
            on_raw=_save_raw,
        )

    def optional_matmul(key, mesh_mapper=None):
        return matmul(key, mesh_mapper=mesh_mapper) if key in lsd else None

    def optional_lookup(key, torch_dtype, tt_dtype):
        if key not in lsd:
            return None
        t = lsd[key].to(torch_dtype)
        _save_raw(key, t)  # convert_weights stores d2t as int32 — same dtype here.
        tt = ttnn.from_torch(
            t.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=tt_dtype,
            mesh_mapper=replicate,
        )
        del t
        gc.collect()
        return tt

    fc = matmul("fc.weight")
    hidden_norm_w = norm("hidden_norm.weight")
    final_norm_w = norm("norm.weight")

    embed_tokens = optional_matmul("embed_tokens.weight", mesh_mapper=col_mapper) if load_embed_tokens else None
    lm_head = optional_matmul("lm_head.weight", mesh_mapper=col_mapper)
    draft_id_to_target_id = optional_lookup("d2t", torch.int32, ttnn.uint32)

    layers: List[DFlashLayerWeights] = []
    for i in range(config.num_hidden_layers):
        prefix = f"layers.{i}"
        layers.append(
            DFlashLayerWeights(
                input_layernorm=norm(f"{prefix}.input_layernorm.weight"),
                post_attention_layernorm=norm(f"{prefix}.post_attention_layernorm.weight"),
                q_proj=matmul(f"{prefix}.self_attn.q_proj.weight", mesh_mapper=col_mapper),
                k_proj=matmul(f"{prefix}.self_attn.k_proj.weight", mesh_mapper=col_mapper),
                v_proj=matmul(f"{prefix}.self_attn.v_proj.weight", mesh_mapper=col_mapper),
                o_proj=matmul(f"{prefix}.self_attn.o_proj.weight"),
                q_norm=norm(f"{prefix}.self_attn.q_norm.weight"),
                k_norm=norm(f"{prefix}.self_attn.k_norm.weight"),
                mlp_gate=matmul(f"{prefix}.mlp.gate_proj.weight"),
                mlp_up=matmul(f"{prefix}.mlp.up_proj.weight"),
                mlp_down=matmul(f"{prefix}.mlp.down_proj.weight"),
            )
        )

    lsd.close()

    if cache_tmp is not None:
        # Reveal the cache atomically: rename the fully-written temp dir into the
        # final name. os.replace is atomic on the same filesystem; if the target
        # appeared meanwhile (a racing process won), drop our temp copy.
        try:
            os.replace(str(cache_tmp), str(write_cache_dir))
            print(f"[dflash] wrote tensorbin weight cache → {write_cache_dir} (next run loads from cache)", flush=True)
        except OSError as e:
            shutil.rmtree(str(cache_tmp), ignore_errors=True)
            print(f"[dflash] skipped writing weight cache to {write_cache_dir}: {e}", flush=True)

    return DFlashWeights(
        fc=fc,
        hidden_norm=hidden_norm_w,
        final_norm=final_norm_w,
        layers=layers,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        draft_id_to_target_id=draft_id_to_target_id,
    )
