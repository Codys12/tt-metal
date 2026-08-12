// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Direct BF16 packed K/V writer for Muse Glimmer batch-1 decode.

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
    constexpr uint32_t grid_start_x = get_named_compile_time_arg_val("grid_start_x");
    constexpr uint32_t grid_start_y = get_named_compile_time_arg_val("grid_start_y");
    constexpr uint32_t grid_end_x = get_named_compile_time_arg_val("grid_end_x");
    constexpr uint32_t grid_end_y = get_named_compile_time_arg_val("grid_end_y");
    constexpr uint32_t num_heads = get_named_compile_time_arg_val("num_heads");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr bool input_batch_major = get_named_compile_time_arg_val("input_batch_major") != 0;
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_height = 16;
    constexpr uint32_t bytes_per_element = 2;
    constexpr uint32_t face_bytes = face_width * face_height * bytes_per_element;
    constexpr uint32_t face_line_bytes = face_width * bytes_per_element;
    constexpr uint32_t cache_page_rows = 64;
    constexpr uint32_t cache_height_tiles = cache_page_rows / tile_height;

    const uint32_t core_id =
        unified_kernels::linear_id_in_grid<true>(grid_start_x, grid_start_y, grid_end_x, grid_end_y);
    const uint32_t head = core_id / width_tiles;
    const uint32_t width_tile = core_id % width_tiles;

    const uint32_t k_cache_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t v_cache_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t k_new_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t v_new_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t position_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t count = get_common_arg_val<uint32_t>(5);

    constexpr auto cache_args = TensorAccessorArgs<0>();
    constexpr auto input_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto position_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto k_cache = TensorAccessor(cache_args, k_cache_addr);
    const auto v_cache = TensorAccessor(cache_args, v_cache_addr);
    const auto k_new = TensorAccessor(input_args, k_new_addr);
    const auto v_new = TensorAccessor(input_args, v_new_addr);
    const auto positions = TensorAccessor(position_args, position_addr);

    // Position is a trace input. Read it on device so one captured verifier
    // can update any page/offset through 128K without recapture.
    cb_reserve_back(scratch_cb, 1);
    const uint32_t position_scratch = get_write_ptr(scratch_cb);
    noc_async_read_page(0, positions, position_scratch);
    noc_async_read_barrier();
    const uint32_t first_position = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(position_scratch);
    const uint32_t first_page = first_position / cache_page_rows;
    const uint32_t page_offset = first_position % cache_page_rows;
    cb_push_back(scratch_cb, 1);
    cb_pop_front(scratch_cb, 1);

    auto copy_rows = [&](const auto& source, const auto& cache) {
        cb_reserve_back(scratch_cb, 1);
        const uint32_t scratch = get_write_ptr(scratch_cb);

        for (uint32_t source_row = 0; source_row < count; ++source_row) {
            const uint32_t source_tile =
                input_batch_major ? source_row * width_tiles + width_tile : head * width_tiles + width_tile;
            noc_async_read_page(source_tile, source, scratch);
            noc_async_read_barrier();
            const uint32_t absolute_row = first_page * cache_page_rows + page_offset + source_row;
            const uint32_t physical_page = absolute_row / cache_page_rows;
            const uint32_t row_in_page = absolute_row % cache_page_rows;
            const uint32_t cache_tile =
                ((physical_page * num_heads + head) * cache_height_tiles + row_in_page / tile_height) * width_tiles +
                width_tile;

            const uint32_t source_tile_row = input_batch_major ? head : source_row;
            const uint32_t source_face_y = source_tile_row / face_height;
            const uint32_t source_line = source_tile_row % face_height;
            const uint32_t dest_tile_row = row_in_page % tile_height;
            const uint32_t dest_face_y = dest_tile_row / face_height;
            const uint32_t dest_line = dest_tile_row % face_height;
            for (uint32_t face_x = 0; face_x < 2; ++face_x) {
                const uint32_t source_offset =
                    (source_face_y * 2 + face_x) * face_bytes + source_line * face_line_bytes;
                const uint32_t dest_offset = (dest_face_y * 2 + face_x) * face_bytes + dest_line * face_line_bytes;
                noc_async_write(scratch + source_offset, cache.get_noc_addr(cache_tile, dest_offset), face_line_bytes);
            }
        }
        noc_async_write_barrier();
        cb_push_back(scratch_cb, 1);
        cb_pop_front(scratch_cb, 1);
    };

    copy_rows(k_new, k_cache);
    copy_rows(v_new, v_cache);
#endif
}
