// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "noc_nonblocking_api.h"
#include "internal/ethernet/tunneling.h"
#include "risc_common.h"
#include "lf_dev_mem_map.hpp"

namespace lite_fabric {

// Interface to the connected RISC processor via ethernet
struct ConnectedRiscInterface {
    // ETH_TXQ_CMD_START_REG (remote register write) is only supported on TXQ0
    static constexpr uint32_t k_Txq = 0;
    static constexpr uint32_t k_SoftResetAddr = 0xFFB121B0;

    // Put the connected RISC into reset.
    // Must include bit 11 (ERISC0/BRISC) to keep ERISC0 in reset.  The previous
    // value 0x47000 omitted bit 11, which turned a direct register write into an
    // accidental ERISC0 deassert when the core was in POR state (0x47800).
    // ERISC0 would then boot and its base firmware init could overwrite the lite
    // fabric config/binary being sent over ethernet.
    inline static void assert_connected_dm1_reset() {
        constexpr uint32_t k_ResetValue = 0x47800;
        internal_::eth_write_remote_reg(k_Txq, k_SoftResetAddr, k_ResetValue);
        while (internal_::eth_txq_is_busy(k_Txq)) {
        }
    }

    // Take ERISC1 out of reset while keeping ERISC0 in reset.
    // ERISC0 will be deasserted later by Metal's initialize_and_launch_firmware.
    inline static void deassert_connected_dm1_reset() {
        constexpr uint32_t k_ResetValue = 0x46800;
        internal_::eth_write_remote_reg(k_Txq, k_SoftResetAddr, k_ResetValue);
        while (internal_::eth_txq_is_busy(k_Txq)) {
        }
    }

    inline static void set_pc(uint32_t pc) {
        constexpr uint32_t k_ResetPcAddr = LITE_FABRIC_RESET_PC;
        internal_::eth_write_remote_reg(k_Txq, k_ResetPcAddr, pc);
        while (internal_::eth_txq_is_busy(k_Txq)) {
        }
    }
};

}  // namespace lite_fabric
