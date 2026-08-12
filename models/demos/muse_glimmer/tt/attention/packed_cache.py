# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Direct batch-1 BF16 writes for a packed block in the paged KV cache."""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor

_KERNEL_PATH = "models/demos/muse_glimmer/tt/attention/kernels/packed_kv_update.cpp"
_TILE = ttnn.Tile((32, 32))
_SCRATCH_CB = 0


def packed_kv_update(k_cache, v_cache, k_new, v_new, position_idx, *, count):
    """Write consecutive K/V rows without cloning and rebuilding cache pages.

    Muse Glimmer's batch-1 server owns an identity page table. Eight data-
    movement cores copy the two KV heads and four head-dimension tiles in
    parallel; each core serializes its row writes, so several packed rows can
    safely update the same physical cache tile.
    """
    if k_cache.dtype != ttnn.bfloat16 or k_new.dtype != ttnn.bfloat16:
        raise ValueError("packed_kv_update supports BF16 caches and inputs only")
    num_heads = int(k_cache.shape[1])
    page_size = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])
    if num_heads != 2 or page_size != 64 or head_dim % 32 or head_dim > 128:
        raise ValueError(f"Unexpected Muse Glimmer cache shape {k_cache.shape}")
    batch_major = int(k_new.shape[2]) == num_heads
    if not batch_major and int(k_new.shape[1]) != num_heads:
        raise ValueError(f"Unexpected Muse Glimmer packed K/V shape {k_new.shape}")
    if int(k_new.shape[3]) != head_dim:
        raise ValueError(f"Unexpected Muse Glimmer packed K/V shape {k_new.shape}")
    if position_idx.dtype != ttnn.uint32 or position_idx.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("packed_kv_update requires a row-major uint32 position tensor")
    if not 1 <= count <= 32:
        raise ValueError(f"Invalid packed cache count={count}")

    width_tiles = head_dim // 32
    last_core_x = num_heads * width_tiles - 1
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(last_core_x, 0))])
    cache_accessor = ttnn.TensorAccessorArgs(k_cache)
    input_accessor = ttnn.TensorAccessorArgs(k_new)
    position_accessor = ttnn.TensorAccessorArgs(position_idx)
    compile_time_args = (
        cache_accessor.get_compile_time_args()
        + input_accessor.get_compile_time_args()
        + position_accessor.get_compile_time_args()
    )
    runtime_args = [
        k_cache.buffer_address(),
        v_cache.buffer_address(),
        k_new.buffer_address(),
        v_new.buffer_address(),
        position_idx.buffer_address(),
        int(count),
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_PATH,
        core_ranges=core_grid,
        ncrisc_compile_time_args=compile_time_args,
        ncrisc_named_compile_time_args=[
            ("scratch_cb", _SCRATCH_CB),
            ("grid_start_x", 0),
            ("grid_start_y", 0),
            ("grid_end_x", last_core_x),
            ("grid_end_y", 0),
            ("num_heads", num_heads),
            ("width_tiles", width_tiles),
            ("input_batch_major", int(batch_major)),
        ],
        ncrisc_common_runtime_args=runtime_args,
    )
    page_size = _TILE.get_tile_size(ttnn.bfloat16)
    scratch = ttnn.CBDescriptor(
        total_size=page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_SCRATCH_CB,
                data_format=ttnn.bfloat16,
                page_size=page_size,
                tile=ttnn.TileDescriptor(_TILE),
            )
        ],
    )
    program = ttnn.ProgramDescriptor(
        kernels=kernel.get_kernel_descriptors().kernels,
        cbs=[scratch],
    )
    return ttnn.generic_op([k_cache, v_cache, k_new, v_new, position_idx, k_cache], program)
