# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
KV cache initialization for Gemma4 attention with TP support.

Per-device cache uses local KV head count (num_kv_heads // tp).
Follows gpt-oss kv_cache.py pattern.
"""

import ttnn

# Hot cache blocks held in staging per decode slot for the loop-free packed KV
# write: a P-token speculative tail (P <= block_size) straddles at most 2 pages,
# so each slot reserves 2 staging blocks (cur_pos block + spill). Must match the
# server's ``self._pv_blk``.
PV_HOT_BLOCKS = 2


def init_kv_cache(
    mesh_device,
    config,
    max_batch_size=1,
    max_seq_len=131072,
    paged_attention_config=None,
    cache_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
):
    """
    Initialize KV cache for a single attention layer.

    For TP > 1, each device gets num_kv_heads // tp heads (column-parallel sharding).
    The cache tensor is replicated to each device with the local head count.

    Args:
        mesh_device: TT device or mesh device
        config: Gemma4AttentionConfig for this layer
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        paged_attention_config: Optional paged attention config
        cache_dtype: Cache tensor dtype (bfloat16, bfloat8_b, bfloat4_b, ...)
        tensor_cache_path: Unused — KV caches are zero-initialized on device.

    Returns:
        [k_cache, v_cache] list of TT tensors
    """
    del tensor_cache_path  # zero-init caches buy nothing from a host cache file.

    # Determine TP from mesh shape (column axis)
    is_mesh = hasattr(mesh_device, "shape")
    tp = mesh_device.shape[1] if is_mesh else 1

    # When KV heads < TP, each device gets 1 KV head (GQA-assigned, not all heads)
    num_local_kv_heads = 1 if config.num_key_value_heads < tp else config.num_key_value_heads // tp
    head_dim = config.head_dim

    if paged_attention_config:
        cache_shape = [
            paged_attention_config.max_num_blocks,
            num_local_kv_heads,
            paged_attention_config.block_size,
            head_dim,
        ]
    else:
        cache_shape = [
            max_batch_size,
            num_local_kv_heads,
            max_seq_len,
            head_dim,
        ]

    # Allocate fully on device for every cache dtype.
    #
    # Block-float dtypes (bfp8_b / bfp4_b) cannot go through ttnn.as_tensor for
    # large shapes: the host-side packer (pack_as_bfp_tiles) uses int32 element
    # indexing internally and segfaults once a single tensor exceeds ~2.15 G
    # elements. KV caches at large max_seq_len blow past that easily.
    #
    # `allocate_tensor_on_device` reserves a buffer with no host data, and
    # `ttnn.fill` writes zeros via a kernel. Neither path enqueues a host→device
    # transfer, so this is safe to run right before trace capture (a stray host
    # write that hasn't drained by `begin_trace_capture` is fatal: see
    # fd_mesh_command_queue.cpp:587).
    def _build_zero_cache():
        cache = ttnn.allocate_tensor_on_device(
            ttnn.Shape(cache_shape),
            cache_dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.fill(cache, 0.0, output_tensor=cache)
        return cache

    return [_build_zero_cache(), _build_zero_cache()]


def init_kv_staging(
    mesh_device,
    config,
    max_batch_size,
    block_size,
    blk,
    cache_dtype=ttnn.bfloat16,
):
    """Per-layer staging buffers for the loop-free packed KV write.

    Holds, per decode slot, the ``blk`` "hot" cache blocks it is currently
    appending into (the partial block at cur_pos + one spill block). The
    packed-verify write merges this resident copy with the step's new K/V and
    re-fills the committed cache from it — so the committed cache is never read
    on the hot path (see ``decode.py::_packed_fill_kv_loopfree``).

    Shape ``[1, num_local_kv_heads, max_batch_size*blk*block_size, head_dim]``
    in TILE/bf16/DRAM — same seq layout as the ``paged_fill_cache`` input. Slot
    ``s`` owns seq positions ``[s*blk*block_size, (s+1)*blk*block_size)``;
    block-slot 0 is the cur_pos block, block-slot 1 the spill block.

    Returns ``[k_staging, v_staging]``.
    """
    is_mesh = hasattr(mesh_device, "shape")
    tp = mesh_device.shape[1] if is_mesh else 1
    num_local_kv_heads = 1 if config.num_key_value_heads < tp else config.num_key_value_heads // tp
    head_dim = config.head_dim
    stage_shape = [1, num_local_kv_heads, max_batch_size * blk * block_size, head_dim]

    def _build_zero_staging():
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(stage_shape),
            cache_dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.fill(t, 0.0, output_tensor=t)
        return t

    return [_build_zero_staging(), _build_zero_staging()]
