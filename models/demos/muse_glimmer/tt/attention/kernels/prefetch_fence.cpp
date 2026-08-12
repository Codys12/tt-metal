// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#if defined(COMPILE_FOR_NCRISC)
#include "api/remote_circular_buffer.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t remote_cb_id = get_named_compile_time_arg_val("cb_remote");
    constexpr uint32_t num_aligned_pages = get_named_compile_time_arg_val("num_aligned_pages");

    auto& remote_cb = get_remote_receiver_cb_interface(remote_cb_id);
    volatile tt_l1_ptr uint32_t* pages_acked =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_acked_ptr);
    volatile tt_l1_ptr uint32_t* pages_sent =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_acked_ptr - L1_ALIGNMENT);
    do {
        invalidate_l1_cache();
    } while (*pages_sent - *pages_acked < num_aligned_pages);
#endif
}
