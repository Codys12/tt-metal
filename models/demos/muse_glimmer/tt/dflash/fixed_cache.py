# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-safe packed writes into the fixed DFlash K/V caches."""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor

_KERNEL_PATH = "models/demos/muse_glimmer/tt/dflash/kernels/fixed_kv_update.cpp"
_TILE = ttnn.Tile((32, 32))
_SCRATCH_CB = 0


def fixed_kv_update(k_cache, v_cache, k_new, v_new, control):
    """Write ``control[1]`` consecutive rows starting at ``control[0]``."""
    if k_cache.dtype != ttnn.bfloat16 or k_new.dtype != ttnn.bfloat16:
        raise ValueError("fixed_kv_update supports BF16 caches and inputs only")
    num_heads = int(k_cache.shape[1])
    capacity = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])
    if int(k_new.shape[1]) != num_heads or int(k_new.shape[3]) != head_dim:
        raise ValueError(f"Unexpected DFlash packed K/V shape {k_new.shape}")
    if int(k_new.shape[2]) > 32 or capacity % 32 or head_dim % 32:
        raise ValueError(f"Unexpected DFlash fixed-cache shape {k_cache.shape}")
    if control.dtype != ttnn.uint32 or control.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("fixed_kv_update requires row-major uint32 control")

    width_tiles = head_dim // 32
    num_cores = num_heads * width_tiles
    grid_width = min(num_cores, 8)
    if num_cores % grid_width:
        raise ValueError(f"Cannot map {num_cores} DFlash cache workers to a rectangular grid")
    grid_height = num_cores // grid_width
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_width - 1, grid_height - 1))]
    )
    cache_accessor = ttnn.TensorAccessorArgs(k_cache)
    input_accessor = ttnn.TensorAccessorArgs(k_new)
    control_accessor = ttnn.TensorAccessorArgs(control)
    compile_time_args = (
        cache_accessor.get_compile_time_args()
        + input_accessor.get_compile_time_args()
        + control_accessor.get_compile_time_args()
    )
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_PATH,
        core_ranges=core_grid,
        ncrisc_compile_time_args=compile_time_args,
        ncrisc_named_compile_time_args=[
            ("scratch_cb", _SCRATCH_CB),
            ("grid_start_x", 0),
            ("grid_start_y", 0),
            ("grid_end_x", grid_width - 1),
            ("grid_end_y", grid_height - 1),
            ("num_heads", num_heads),
            ("width_tiles", width_tiles),
            ("cache_height_tiles", capacity // 32),
        ],
        ncrisc_common_runtime_args=[
            k_cache.buffer_address(),
            v_cache.buffer_address(),
            k_new.buffer_address(),
            v_new.buffer_address(),
            control.buffer_address(),
        ],
    )
    tile_bytes = _TILE.get_tile_size(ttnn.bfloat16)
    scratch = ttnn.CBDescriptor(
        total_size=tile_bytes,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_SCRATCH_CB,
                data_format=ttnn.bfloat16,
                page_size=tile_bytes,
                tile=ttnn.TileDescriptor(_TILE),
            )
        ],
    )
    program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=[scratch])
    return ttnn.generic_op([k_cache, v_cache, k_new, v_new, control, k_cache], program)
