// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstring>

#include <umd/device/driver_atomics.hpp>

namespace tt::tt_metal {

template <bool debug_sync = false>
__attribute((nonnull(1, 2))) inline void memcpy_to_device(
    void* __restrict dst, const void* __restrict src, std::size_t n) {
    std::memcpy(dst, src, n);
    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}

}  // namespace tt::tt_metal
