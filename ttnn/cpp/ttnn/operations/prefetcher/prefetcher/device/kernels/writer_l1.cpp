// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

uint32_t increment_arg_idx(uint32_t& arg_idx, uint32_t num_args = 1) {
    uint32_t old_arg_idx = arg_idx;
    arg_idx += num_args;
    return old_arg_idx;
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);
    constexpr uint32_t max_block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t local_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(7);
    constexpr bool posted_payload = get_compile_time_arg_val(8);

    // Runtime args
    // Note: Coalesced sizes -> wrt to receiver cores, sizes -> wrt to dram reader cores
    uint32_t rt_args_idx = 0;
    const uint32_t* coalesced_page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* coalesced_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_height_in_tiles =
        (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));  // Kt / num_blocks = in_block_h;

    uint32_t noc = noc_index;
    uint32_t total_aligned_pages_sent = experimental::remote_cb_local_pages_sent(remote_cb_id);
    {
        DeviceZoneScopedN("PREFETCH-L1-FANOUT");
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            for (uint32_t t = 0; t < num_tensors; t++) {
                uint32_t curr_coalesced_page_size = coalesced_page_sizes[t];
                uint32_t curr_coalesced_num_pages = coalesced_num_pages[t];
                uint32_t curr_block_height_in_tiles = block_height_in_tiles[t];

                constexpr uint32_t blocks_per_credit = 4;
                for (uint32_t block = 0; block < num_blocks; block += blocks_per_credit) {
                    uint32_t remaining_blocks = num_blocks - block;
                    uint32_t blocks_this_credit =
                        remaining_blocks < blocks_per_credit ? remaining_blocks : blocks_per_credit;
                    experimental::remote_cb_reserve_back_fixed_pages_tracked(
                        remote_cb_id, blocks_this_credit, total_aligned_pages_sent);
                    for (uint32_t block_in_credit = 0; block_in_credit < blocks_this_credit; ++block_in_credit) {
                        cb_wait_front(local_cb_id, max_block_num_tiles);
                        uint32_t local_cb_addr = get_read_ptr(local_cb_id);
                        if constexpr (posted_payload) {
                            if (block_in_credit + 1 == blocks_this_credit) {
                                experimental::remote_cb_write_pages<true, true>(
                                    remote_cb_id,
                                    local_cb_addr,
                                    1,
                                    curr_block_height_in_tiles,
                                    curr_coalesced_num_pages,
                                    curr_coalesced_page_size,
                                    noc);
                            } else {
                                experimental::remote_cb_write_pages<true, false>(
                                    remote_cb_id,
                                    local_cb_addr,
                                    1,
                                    curr_block_height_in_tiles,
                                    curr_coalesced_num_pages,
                                    curr_coalesced_page_size,
                                    noc);
                            }
                            // The local reader slot can be recycled as soon as all
                            // posted payload requests have left this core.
                            noc_async_posted_writes_flushed(noc);
                            if (block_in_credit + 1 == blocks_this_credit) {
                                // The last block's one-word receiver fences make
                                // the entire chunk visible before credit publish.
                                noc_async_write_barrier(noc);
                            }
                        } else {
                            experimental::remote_cb_write_pages<false>(
                                remote_cb_id,
                                local_cb_addr,
                                1,
                                curr_block_height_in_tiles,
                                curr_coalesced_num_pages,
                                curr_coalesced_page_size,
                                noc);
                            noc_async_writes_flushed(noc);
                        }
                        cb_pop_front(local_cb_id, max_block_num_tiles);
                    }
                    if constexpr (!posted_payload) {
                        noc_async_write_barrier(noc);
                    }
                    total_aligned_pages_sent += blocks_this_credit;
                    if (block + blocks_this_credit == num_blocks) {
                        experimental::remote_cb_publish_fixed_pages<true>(remote_cb_id, total_aligned_pages_sent, noc);
                        noc_async_write_barrier(noc);
                    } else {
                        experimental::remote_cb_publish_fixed_pages<false>(remote_cb_id, total_aligned_pages_sent, noc);
                        noc_async_posted_writes_flushed(noc);
                    }
                }
            }
        }
    }

    {
        DeviceZoneScopedN("PREFETCH-FINAL-DRAIN");
        // Every payload and credit must be visible before the producer exits.
        // The consumer phase boundary owns the final drain.
        noc_async_posted_writes_flushed(noc);
        experimental::remote_cb_sender_barrier(remote_cb_id);
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
    // reset noc counters here because we didn't properly update ptrs for better perf.
    if (noc_mode == DM_DEDICATED_NOC) {
        ncrisc_noc_counters_init();
    } else {
        dynamic_noc_local_state_init();
    }
    // signal reader can exit, since reader cannot exit early due to the ongoing traffic on the same noc.
    cb_reserve_back(sync_cb_id, 1);
    cb_push_back(sync_cb_id, 1);
}
