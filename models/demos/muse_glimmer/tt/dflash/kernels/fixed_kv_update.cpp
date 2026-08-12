// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

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
    constexpr uint32_t cache_height_tiles = get_named_compile_time_arg_val("cache_height_tiles");
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_height = 16;
    constexpr uint32_t bytes_per_element = 2;
    constexpr uint32_t face_bytes = face_width * face_height * bytes_per_element;
    constexpr uint32_t face_line_bytes = face_width * bytes_per_element;

    const uint32_t core_id =
        unified_kernels::linear_id_in_grid<true>(grid_start_x, grid_start_y, grid_end_x, grid_end_y);
    const uint32_t head = core_id / width_tiles;
    const uint32_t width_tile = core_id % width_tiles;

    const uint32_t k_cache_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t v_cache_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t k_new_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t v_new_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t control_addr = get_common_arg_val<uint32_t>(4);

    constexpr auto cache_args = TensorAccessorArgs<0>();
    constexpr auto input_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto control_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto k_cache = TensorAccessor(cache_args, k_cache_addr);
    const auto v_cache = TensorAccessor(cache_args, v_cache_addr);
    const auto k_new = TensorAccessor(input_args, k_new_addr);
    const auto v_new = TensorAccessor(input_args, v_new_addr);
    const auto control = TensorAccessor(control_args, control_addr);

    cb_reserve_back(scratch_cb, 1);
    const uint32_t control_scratch = get_write_ptr(scratch_cb);
    noc_async_read_page(0, control, control_scratch);
    noc_async_read_barrier();
    const auto control_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control_scratch);
    const uint32_t write_base = control_words[0];
    const uint32_t valid_count = control_words[1];
    cb_push_back(scratch_cb, 1);
    cb_pop_front(scratch_cb, 1);

    const uint32_t source_tile = head * width_tiles + width_tile;
    auto copy_rows = [&](const auto& source, const auto& cache) {
        cb_reserve_back(scratch_cb, 1);
        const uint32_t scratch = get_write_ptr(scratch_cb);
        noc_async_read_page(source_tile, source, scratch);
        noc_async_read_barrier();
        for (uint32_t source_row = 0; source_row < valid_count; ++source_row) {
            const uint32_t destination_row = write_base + source_row;
            const uint32_t cache_tile =
                (head * cache_height_tiles + destination_row / tile_height) * width_tiles + width_tile;
            const uint32_t source_face_y = source_row / face_height;
            const uint32_t source_line = source_row % face_height;
            const uint32_t destination_tile_row = destination_row % tile_height;
            const uint32_t destination_face_y = destination_tile_row / face_height;
            const uint32_t destination_line = destination_tile_row % face_height;
            for (uint32_t face_x = 0; face_x < 2; ++face_x) {
                const uint32_t source_offset =
                    (source_face_y * 2 + face_x) * face_bytes + source_line * face_line_bytes;
                const uint32_t destination_offset =
                    (destination_face_y * 2 + face_x) * face_bytes + destination_line * face_line_bytes;
                noc_async_write(
                    scratch + source_offset, cache.get_noc_addr(cache_tile, destination_offset), face_line_bytes);
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
