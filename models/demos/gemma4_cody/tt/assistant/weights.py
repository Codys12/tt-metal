# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading for the Gemma4 MTP assistant (drafter) on Tenstorrent.

Differs from cody's target-model weight loader in three places:

  1. **No fused QKV.** The drafter has only ``q_proj`` (no k_proj / v_proj).
     K/V at attention time come from ``shared_kv_states`` provided by the
     target model at runtime.
  2. **No k_norm.** Drafter has only ``q_norm`` per layer. K/V come pre-
     normed from the target.
  3. **Two top-level linear projections** that bridge the drafter's
     residual stream to the target's:
       - ``pre_projection``: input shape ``[B, T, 2 * backbone_hidden]``
         (target's last-layer hidden state, concatenated with something
         else — likely token embedding — by the target's integration
         code) → drafter's ``hidden_size``.
       - ``post_projection``: drafter's ``hidden_size`` → target's
         ``backbone_hidden`` (so the drafter's per-token "draft logits"
         can be projected back to a vocab-ready embedding).

Sharding / TP
-------------

The drafter mirrors cody's target sharding so each per-layer stage runs at
the same per-device cost as a target layer:

  - ``q_proj``: column-parallel (heads sharded over TP). Per-device output
    is ``[1, 1, B, num_heads/tp * head_dim]``, which feeds the local SDPA
    directly against the target's per-device KV-cache shard.
  - ``o_proj``: row-parallel (input dim sharded over TP). The local SDPA
    output is already sharded on heads, so we skip the all-gather and feed
    o_proj its native sharded input; the trailing all-reduce sums partial
    outputs back to the replicated residual stream.
  - ``mlp_gate`` / ``mlp_up``: column-parallel (intermediate dim sharded).
    Output is ``[1, 1, B, intermediate/tp]`` — each device only does
    ``1/tp`` of the MLP intermediate work.
  - ``mlp_down``: row-parallel (input dim sharded) — partial-sum output,
    summed with all-reduce.

Norm weights are loaded as torch tensors and wrapped in the shared
``models.demos.gemma4_cody.tt.rms_norm.RMSNorm`` adapter at module init
time, which opts into the width-sharded multi-core decode kernel (≈32
cores at hidden=1024 vs the default interleaved kernel's ≈4 cores).
``q_norm`` stays as a raw TT tensor — it is consumed by
``apply_per_head_norm`` directly and the kernel does not benefit from the
sharded decode path at head_dim=256/512.

Cache layout
------------

Reads from ``$TT_CACHE_PATH/tensor_cache_assistant_bf16/`` (produced by
``convert_weights.py``). One subdir per layer plus four top-level files
(embed_tokens, norm, pre_projection, post_projection).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch

import ttnn
from models.demos.gemma4_cody.tt.rms_norm import RMSNorm


@dataclass(frozen=True)
class DrafterLayerWeights:
    """Per-layer weights for the drafter."""

    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    pre_feedforward_layernorm: RMSNorm
    post_feedforward_layernorm: RMSNorm
    # layer_scalar is a single learned float (shape [1] in HF). Stored as a
    # Python scalar so ttnn.mul broadcasts unambiguously, matching cody's
    # pattern (`self.layer_scalar = layer_state["layer_scalar"].item()`).
    layer_scalar: float

    q_proj: ttnn.Tensor  # [layer_q_dim, hidden_size] (column-parallel)
    q_norm: ttnn.Tensor  # [head_dim] (layer-type specific: 256 or 512)
    o_proj: ttnn.Tensor  # [hidden_size, layer_q_dim] (row-parallel)

    mlp_gate: ttnn.Tensor  # [intermediate_size, hidden_size] (column-parallel)
    mlp_up: ttnn.Tensor  # [intermediate_size, hidden_size] (column-parallel)
    mlp_down: ttnn.Tensor  # [hidden_size, intermediate_size] (row-parallel)


@dataclass(frozen=True)
class DrafterWeights:
    """All weights for the drafter."""

    embed_tokens: ttnn.Tensor  # [vocab_size, hidden_size]
    final_norm: RMSNorm
    pre_projection: ttnn.Tensor  # [hidden_size, 2 * backbone_hidden]
    post_projection: ttnn.Tensor  # [backbone_hidden, hidden_size]
    layers: List[DrafterLayerWeights]


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
    """Load a single .tensorbin into a TT tensor (replicated across the mesh).

    ``transpose_for_matmul=True``: cody's matmul weight convention —
        ``.transpose(-2, -1).unsqueeze(0).unsqueeze(0)``. HF stores
        nn.Linear weights as [out, in]; ``ttnn.linear(x, w)`` computes
        ``x @ w`` directly, so weight needs to be [in, out].

    ``reshape_for_norm=True``: cody's norm weight convention — reshape from
        ``[H]`` to ``[1, 1, H/TILE_SIZE, TILE_SIZE]``. ``ttnn.rms_norm``
        requires gamma's last padded dim == tile width (32).
    """
    abs_path = cache_dir / rel_path
    if not abs_path.is_file():
        raise FileNotFoundError(
            f"Missing weight cache file {abs_path}. "
            "Run `python -m models.demos.gemma4_cody.tt.assistant.convert_weights` first."
        )
    tensor = torch.load(str(abs_path), weights_only=True)
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if transpose_for_matmul:
        tensor = tensor.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
    elif reshape_for_norm:
        # [H] -> [1, 1, H/32, 32]. Norms with dims smaller than 32 (like
        # layer_scalar [1]) skip this; they're broadcast across the row.
        if tensor.dim() == 1 and tensor.shape[0] >= ttnn.TILE_SIZE and tensor.shape[0] % ttnn.TILE_SIZE == 0:
            tensor = tensor.reshape(1, 1, -1, ttnn.TILE_SIZE).contiguous()
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        mesh_mapper=mesh_mapper,
    )


def _load_torch_norm(cache_dir: Path, rel_path: str) -> torch.Tensor:
    """Load a raw torch norm weight tensor (1-D [hidden]) for adapter consumption."""
    abs_path = cache_dir / rel_path
    if not abs_path.is_file():
        raise FileNotFoundError(
            f"Missing weight cache file {abs_path}. "
            "Run `python -m models.demos.gemma4_cody.tt.assistant.convert_weights` first."
        )
    tensor = torch.load(str(abs_path), weights_only=True)
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    return tensor


class _DrafterNormConfig:
    """Adapter shim — exposes the fields ``RMSNorm`` reads from ``hf_config``.

    The shared ``RMSNorm`` adapter expects ``hf_config.rms_norm_eps`` and
    ``hf_config.hidden_size``. ``Gemma4AssistantConfig`` already carries
    both fields, but it is a frozen dataclass with extra members the
    adapter does not touch — wrapping keeps the surface explicit and
    decouples the assistant config schema from the adapter's API.
    """

    def __init__(self, hidden_size: int, rms_norm_eps: float):
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps


def _build_norm(
    mesh_device,
    norm_cfg: _DrafterNormConfig,
    weight: torch.Tensor,
    mesh_config,
) -> RMSNorm:
    """Wrap a torch norm weight in the shared sharded-decode RMSNorm adapter."""
    return RMSNorm(
        mesh_device=mesh_device,
        hf_config=norm_cfg,
        state_dict={"weight": weight},
        tensor_cache_path=None,
        mesh_config=mesh_config,
        with_scale=True,
        enable_sharded_decode=True,
    )


def load_drafter_weights(
    mesh_device,
    config,  # Gemma4AssistantConfig
    cache_dir: str | Path | None = None,
    weight_dtype=ttnn.bfloat16,
    mesh_config: Any = None,
) -> DrafterWeights:
    """Load the converted drafter weights from disk into TT tensors.

    Matmul weights are TP-sharded so the per-layer stages match the target's
    per-device work (see module docstring). Norm weights are loaded as torch
    tensors and wrapped in the shared ``RMSNorm`` adapter, which routes to
    the width-sharded multi-core decode kernel at hidden=1024.

    Sharding gating: the row-parallel MLP / o_proj path is only used when
    ``mesh_config`` is provided (production server + parity test). Without
    it those four weights load as replicated and ``model._all_reduce``
    becomes a no-op, so each device's matmul produces a complete replicated
    output — preserving the original loader's behaviour for the smoke test
    path which only sets ``mesh_device``. ``q_proj`` and the tied lm head
    stay column-parallel either way, since their sharding is load-bearing
    for the per-device SDPA layout.
    """
    if cache_dir is None:
        cache_root = os.environ.get("TT_CACHE_PATH")
        if not cache_root:
            raise RuntimeError("Pass cache_dir or set $TT_CACHE_PATH")
        cache_dir = Path(cache_root) / "tensor_cache_assistant_bf16"
    cache_dir = Path(cache_dir)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    tp = mesh_device.shape[1] if is_mesh else 1
    # The new row-parallel MLP/o_proj path requires an all-reduce to sum
    # partial outputs back to the replicated residual stream. That needs
    # both ``mesh_config`` (for the TP axis + semaphore plumbing) and
    # ``ccl_manager`` (for the actual op) — see ``model._all_reduce``. When
    # either is missing (smoke tests like ``test_assistant_forward.py``) we
    # fall back to replicated MLP/o_proj weights so each device's matmul is
    # self-contained; the model then takes the legacy ``ccl_allgather``
    # path before o_proj (also a no-op without CCL) to keep behaviour at
    # parity with the original loader.
    sharded = tp > 1 and mesh_config is not None
    if tp > 1:
        # q_proj / embed are column-parallel unconditionally — sharding the
        # query heads is load-bearing for the per-device SDPA layout
        # (``num_heads_local = num_heads // tp`` heads on each device).
        if mesh_config is not None:
            col_mapper_q = mesh_config.column_parallel(mesh_device)
        else:
            col_mapper_q = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    else:
        col_mapper_q = replicate
    if sharded:
        # Full row/column-parallel MLP + row-parallel o_proj (production
        # path — matches target's per-device cost; all-reduce in model).
        col_mapper = mesh_config.column_parallel(mesh_device)
        row_mapper = mesh_config.row_parallel(mesh_device)
    else:
        col_mapper = replicate
        row_mapper = replicate

    def load_matmul(rel, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
        """Load a 2D matmul weight, transpose to [in, out] + unsqueeze to 4D."""
        return _load_cached(
            cache_dir,
            rel,
            mesh_device,
            weight_dtype,
            layout,
            mesh_mapper if mesh_mapper is not None else replicate,
            transpose_for_matmul=True,
        )

    def load_per_head_norm(rel):
        # Per-head norm weights (q_norm) are consumed by `apply_per_head_norm`,
        # which calls `ttnn.rms_norm(weight=...)` directly — keep the TT tensor
        # in [1, 1, H/32, 32] layout for that path.
        return _load_cached(
            cache_dir,
            rel,
            mesh_device,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            replicate,
            reshape_for_norm=True,
        )

    norm_cfg = _DrafterNormConfig(hidden_size=config.hidden_size, rms_norm_eps=config.rms_norm_eps)

    def build_layer_norm(rel):
        weight = _load_torch_norm(cache_dir, rel)
        return _build_norm(mesh_device, norm_cfg, weight, mesh_config)

    # embed_tokens is consumed by `ttnn.linear(h, embed_tokens)` for tied-lm-head logits.
    # That treats it as a [vocab, hidden] matmul weight — we need it as [hidden, vocab]
    # to make x @ w produce [B, vocab]. So we transpose at load.
    # Column-parallel (shard the vocab/output dim across TP), matching the
    # target's lm-head (tt/model.py:244): each device computes h @ w_shard →
    # [B, vocab/tp] sharded logits. Without this every device ran the full
    # [B,hidden]·[hidden,262144] matmul + 537 MB weight read — ~8× the needed
    # per-device work, a large chunk of the drafter forward (see
    # .claude-spec-decode-scoping/ITEM1_PACKED_VERIFY_RESULTS.md).
    embed_tokens_lm = load_matmul("embed_tokens.weight.tensorbin", layout=ttnn.TILE_LAYOUT, mesh_mapper=col_mapper_q)
    final_norm_weight = _load_torch_norm(cache_dir, "norm.weight.tensorbin")
    final_norm = _build_norm(mesh_device, norm_cfg, final_norm_weight, mesh_config)
    pre_projection = load_matmul("pre_projection.weight.tensorbin")
    post_projection = load_matmul("post_projection.weight.tensorbin")

    layers = []
    for i in range(config.num_hidden_layers):
        layer_dir = f"layer_{i}"
        # Load layer_scalar as a Python float (NOT a TT tensor) so ttnn.mul
        # broadcasts unambiguously. The HF model stores this as a 1-element
        # buffer; using it as a scalar matches cody's existing pattern and
        # avoids tile/row-major layout mismatch with the activation tensor.
        scalar_tensor = torch.load(str(cache_dir / f"{layer_dir}/layer_scalar.tensorbin"), weights_only=True)
        layer_scalar = float(scalar_tensor.float().item())

        layers.append(
            DrafterLayerWeights(
                input_layernorm=build_layer_norm(f"{layer_dir}/input_layernorm.weight.tensorbin"),
                post_attention_layernorm=build_layer_norm(f"{layer_dir}/post_attention_layernorm.weight.tensorbin"),
                pre_feedforward_layernorm=build_layer_norm(f"{layer_dir}/pre_feedforward_layernorm.weight.tensorbin"),
                post_feedforward_layernorm=build_layer_norm(f"{layer_dir}/post_feedforward_layernorm.weight.tensorbin"),
                layer_scalar=layer_scalar,
                q_proj=load_matmul(f"{layer_dir}/self_attn/q_proj.weight.tensorbin", mesh_mapper=col_mapper_q),
                q_norm=load_per_head_norm(f"{layer_dir}/self_attn/q_norm.weight.tensorbin"),
                o_proj=load_matmul(f"{layer_dir}/self_attn/o_proj.weight.tensorbin", mesh_mapper=row_mapper),
                mlp_gate=load_matmul(f"{layer_dir}/mlp/gate_proj.weight.tensorbin", mesh_mapper=col_mapper),
                mlp_up=load_matmul(f"{layer_dir}/mlp/up_proj.weight.tensorbin", mesh_mapper=col_mapper),
                mlp_down=load_matmul(f"{layer_dir}/mlp/down_proj.weight.tensorbin", mesh_mapper=row_mapper),
            )
        )

    return DrafterWeights(
        embed_tokens=embed_tokens_lm,
        final_norm=final_norm,
        pre_projection=pre_projection,
        post_projection=post_projection,
        layers=layers,
    )
