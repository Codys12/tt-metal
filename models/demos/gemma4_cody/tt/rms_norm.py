# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 RMSNorm — adapter delegating to the shared tt_transformers RMSNorm.

We wrap `models.common.rmsnorm.RMSNorm` so Gemma4 inherits the optimizations
that ship with the shared class (HiFi2 compute kernel, sharded multi-core
program-config path, future fusion landings) without keeping a Gemma4-specific
fork of the kernel call.

The adapter preserves Gemma4's existing constructor / `.forward(x)` interface
so the call sites in `layer.py`, `model.py`, and `router.py` are unchanged.

Sharded decode opt-in:
  Pass `enable_sharded_decode=True` to enable a width-sharded multi-core RMS
  path for decode-like activations. This covers both single-token decode
  (height=32) and packed multi-token decode (height=B*P, e.g. 128 at B=32,
  P=4) — the config is built lazily per activation height and cached. The
  adapter handles the reshard internally — input is moved to L1 width-sharded,
  the sharded kernel runs, and the output is resharded back to whatever layout
  the caller passed in.

  Motivation: the default interleaved rms_norm parallelizes a short/wide
  activation over only its height tiles (≈4 cores at B*P=128) and reduces the
  full hidden row per core. The sharded path spreads the hidden reduction
  across the grid (e.g. 56 cores at hidden=5376). At small hidden / few cores
  the in/out reshard can cancel the gain; once surrounding ops consume the
  sharded layout the reshards collapse (a layer.py restructure tracked
  separately). Tall prefill activations (> `_sharded_max_h_tiles`) stay on
  the interleaved kernel.
"""

from pathlib import Path

from torch import nn

import ttnn
from models.common.rmsnorm import RMSNorm as TTRMSNorm
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.tt_transformers.tt.common import Mode

_TILE = 32


def _build_decode_sharded_config(hidden_size, decode_height_tiles=1):
    """Build a sharded width-multi-core RMS config for decode at `hidden_size`.

    Returns a dict with `input_memcfg`, `output_memcfg`, `program_config` — or
    None if `hidden_size` can't be cleanly width-sharded onto a Blackhole-friendly
    grid (8 columns × ≤8 rows, integer tiles per core).
    """
    if hidden_size % _TILE != 0:
        return None
    hidden_tiles = hidden_size // _TILE

    # Width-shard the hidden dim across as many Blackhole worker cores as divide
    # it evenly. Pick the largest core count on an (≤8 × ≤8) grid whose product
    # divides hidden_tiles, so every core gets an integer tile count and the
    # cross-core reduction stays balanced. The old fixed candidate list only
    # tried {64,32,16,8,4} cores; for hidden=5376 (168 tiles) none of those
    # divide 168 except 8, so it fell back to an 8×1=8-core grid. The search
    # below finds 8×7=56 cores (block_w=3) instead — the interleaved kernel,
    # by contrast, parallelizes a short/wide decode activation over only its
    # ~4 height tiles.
    grid = None
    best_cores = 0
    for gy in range(1, 9):
        for gx in range(1, 9):
            n = gx * gy
            if hidden_tiles % n == 0 and n > best_cores:
                best_cores = n
                grid = (gx, gy)
    if grid is None:
        return None
    grid_x, grid_y = grid
    num_cores = grid_x * grid_y
    block_w = hidden_tiles // num_cores

    # Decode shape: [1, 1, 32, hidden_size] — TILE pads height to 32 (one tile).
    height = decode_height_tiles * _TILE

    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(1, 1, height, hidden_size),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_memcfg = input_memcfg
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid,
        subblock_w=1,
        block_h=decode_height_tiles,
        block_w=block_w,
        inplace=False,
    )
    return {
        "input_memcfg": input_memcfg,
        "output_memcfg": output_memcfg,
        "program_config": program_config,
    }


class RMSNorm(nn.Module):
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        tensor_cache_path=None,
        mesh_config=None,
        with_scale=True,
        enable_sharded_decode=False,
    ):
        super().__init__()
        self.with_scale = with_scale
        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))

        # Bridge Gemma4's per-norm tensor_cache_path stem to the shared class's
        # weight_cache_path / weight_name convention. Using the parent dir + leaf
        # as weight_key lands the cache alongside the existing path layout.
        if tensor_cache_path:
            stem = Path(tensor_cache_path)
            weight_cache_path = stem.parent
            weight_key = stem.name
        else:
            weight_cache_path = None
            weight_key = "weight"

        if with_scale and state_dict and "weight" in state_dict:
            wrapped_state = {f"{weight_key}.weight": state_dict["weight"]}
            self._impl = TTRMSNorm(
                device=mesh_device,
                dim=hf_config.hidden_size,
                state_dict=wrapped_state,
                weight_key=weight_key,
                eps=self.eps,
                weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=False,
                # On Gemma4's interleaved prefill path each core walks 88 tiles
                # wide for hidden=2816 — fp32 dest doubles the dest CB and blows
                # past Blackhole's 1.5 MB L1.
                fp32_dest_acc_en=False,
            )
        else:
            self._impl = None

        # Width-sharded multi-core RMSNorm for decode-like activations, built
        # lazily per activation height (in tiles) and cached. Single-token
        # decode is height=32 (1 tile); packed multi-token decode is
        # height=B*P (e.g. 128 = 4 tiles at B=32, P=4). The default interleaved
        # rms_norm parallelizes a short/wide activation over only its height
        # tiles (≈4 cores at B*P=128, hidden=5376) and reduces the full hidden
        # row per core — the sharded path spreads the hidden reduction across
        # the grid instead. Real prefill (tall activations) is left on the
        # interleaved kernel: width-sharding its height into L1 would blow the
        # per-core budget, so it is gated out by ``_sharded_max_h_tiles``.
        self._sharded_enabled = enable_sharded_decode and self._impl is not None
        self._hidden_size = hf_config.hidden_size
        self._sharded_cfg_cache: dict = {}
        self._sharded_max_h_tiles = 8

    def _sharded_cfg_for(self, h_tiles):
        """Width-sharded config for an activation `h_tiles` tall, or None.

        Cached per height. Returns None when sharding is disabled, the height
        exceeds the decode budget, or the hidden dim is not cleanly shardable.
        """
        if not self._sharded_enabled or h_tiles > self._sharded_max_h_tiles:
            return None
        if h_tiles not in self._sharded_cfg_cache:
            self._sharded_cfg_cache[h_tiles] = _build_decode_sharded_config(
                self._hidden_size, decode_height_tiles=h_tiles
            )
        return self._sharded_cfg_cache[h_tiles]

    def forward(self, x, mode=None):
        if self._impl is None:
            return ttnn.rms_norm(x, epsilon=self.eps)

        if mode is None:
            mode = Mode.DECODE if x.shape[2] <= _TILE else Mode.PREFILL

        # Width-sharded multi-core path for decode-like heights — covers both
        # single-token decode (height<=32) and packed multi-token decode
        # (height=B*P). The in/out reshard is cheap next to the kernel win once
        # the hidden reduction is spread across the grid (≈4 -> 56 cores at
        # hidden=5376), which the default interleaved kernel does not do for a
        # short, wide activation. Tall (prefill) activations fall through.
        h_tiles = (int(x.shape[2]) + _TILE - 1) // _TILE
        cfg = self._sharded_cfg_for(h_tiles)
        if cfg is not None:
            orig_mem = x.memory_config()
            x_s = ttnn.to_memory_config(x, cfg["input_memcfg"])
            y = self._impl(
                x_s,
                mode,
                in_sharded=True,
                out_sharded=True,
                norm_config={
                    "sharded_program_config": cfg["program_config"],
                    "sharded_output_config": cfg["output_memcfg"],
                },
            )
            return ttnn.to_memory_config(y, orig_mem)

        return self._impl(x, mode)
