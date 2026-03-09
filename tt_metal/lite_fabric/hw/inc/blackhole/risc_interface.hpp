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

// Bounded wait for local TXQ to become idle, with recovery if stuck.
// Used during init (before channels.hpp helpers are available).
inline void txq_wait_or_recover_init(uint32_t txq_id) {
    const uint32_t txq_base = ETH_TXQ0_REGS_START + txq_id * ETH_TXQ_REGS_SIZE;
    constexpr uint32_t k_MaxIters = 5000000;

    for (uint32_t i = 0; i < k_MaxIters; i++) {
        if (!internal_::eth_txq_is_busy(txq_id)) {
            return;
        }
    }

    // TXQ stuck — attempt recovery
    *reinterpret_cast<volatile uint32_t*>(txq_base + ETH_TXQ_CTRL) = 0;
    for (volatile uint32_t i = 0; i < 10000; i++) {
    }
    *reinterpret_cast<volatile uint32_t*>(txq_base + ETH_TXQ_CMD) = 0x8;  // MAC queue flush
    for (volatile uint32_t i = 0; i < 10000; i++) {
    }
    *reinterpret_cast<volatile uint32_t*>(txq_base + ETH_TXQ_CTRL) = ETH_TXQ_CTRL_KEEPALIVE;
    for (volatile uint32_t i = 0; i < 10000; i++) {
    }
}

// Interface to the connected RISC processor via ethernet
struct ConnectedRiscInterface {
    // ETH_TXQ_CMD_START_REG (remote register write) is only supported on TXQ0
    static constexpr uint32_t k_Txq = 0;
    static constexpr uint32_t k_SoftResetAddr = 0xFFB121B0;

    // Put the connected RISC into reset.
    //
    // On fresh chip boot (after FLR), ERISC0's bootrom actively uses TXQ0 for
    // ETH link training.  Hard-resetting ERISC0 while TXQ0 has an in-flight
    // operation can halt the DMA engine mid-transfer, leaving CMD_ONGOING
    // permanently asserted.  Remote ERISC1 would then hang at
    // eth_txq_is_busy(0) and never send its handshake packet.
    //
    // Two-step sequence:
    // Step 1: Assert only ERISC1 reset, keeping ERISC0 running so it
    //         can finish any in-flight TXQ0 operation naturally.
    // Step 2: Brief delay for the TXQ0 DMA to drain.
    // Step 3: Assert both ERISC0 + ERISC1 reset.  TXQ0 is now idle.
    // Step 4: MAC queue flush on remote TXQ0.
    //
    // NOTE: Do NOT disable KEEPALIVE on remote TXQ0 via WRITE_REG here.
    // KEEPALIVE must be set on both sides for reliable packet delivery.
    // Disabling it breaks ACK delivery for all subsequent WRITE_REGs,
    // including the re-enable itself.
    inline static void assert_connected_dm1_reset() {
        // Step 0: Ensure remote TXQ0 KEEPALIVE is enabled.  A prior run may
        // have left it disabled (e.g. a debug WRITE_REG that cleared CTRL).
        // WRITE_REG delivery via START_REG is ACK'd at the MAC level and
        // does not depend on the remote's TXQ KEEPALIVE setting.
        constexpr uint32_t k_RemoteTxqCtrlAddr = 0xFFB90000;  // ETH_TXQ0 CTRL
        internal_::eth_write_remote_reg(k_Txq, k_RemoteTxqCtrlAddr, 0x1);
        txq_wait_or_recover_init(k_Txq);

        // Step 1: ERISC1 in reset, ERISC0 stays running.
        // 0x47000 is safe here because the chip has already booted (ETH link
        // is up), so ERISC0 is already deasserted.
        constexpr uint32_t k_ResetErisc1Only = 0x47000;
        internal_::eth_write_remote_reg(k_Txq, k_SoftResetAddr, k_ResetErisc1Only);
        txq_wait_or_recover_init(k_Txq);

        // Step 2: Let ERISC0 drain any pending TXQ0 operation (~50 µs).
        for (volatile uint32_t i = 0; i < 50000; i++) {
        }

        // Step 3: Now assert ERISC0 reset too.  TXQ0 is now idle.
        constexpr uint32_t k_ResetAll = 0x47800;
        internal_::eth_write_remote_reg(k_Txq, k_SoftResetAddr, k_ResetAll);
        txq_wait_or_recover_init(k_Txq);

        // Step 4: Issue MAC queue flush on remote TXQ0 to clear any residual
        // state, then wait for the WRITE_REG to complete.
        constexpr uint32_t k_RemoteTxqCmdAddr = 0xFFB90004;  // ETH_TXQ0 CMD
        constexpr uint32_t k_FlushCmd = 0x8;                 // ETH_TXQ_CMD_FLUSH
        internal_::eth_write_remote_reg(k_Txq, k_RemoteTxqCmdAddr, k_FlushCmd);
        txq_wait_or_recover_init(k_Txq);
    }

    // Put only the connected ERISC1 into reset while leaving ERISC0 running.
    // Used during lite-fabric shutdown after the remote fabric router has been launched.
    inline static void assert_connected_erisc1_reset_only() {
        constexpr uint32_t k_ResetErisc1Only = 0x47000;
        internal_::eth_write_remote_reg(k_Txq, k_SoftResetAddr, k_ResetErisc1Only);
        txq_wait_or_recover_init(k_Txq);
    }

    // Take ERISC1 out of reset while keeping ERISC0 in reset.
    // ERISC0 will be deasserted later by Metal's initialize_and_launch_firmware.
    inline static void deassert_connected_dm1_reset() {
        constexpr uint32_t k_ResetValue = 0x46800;
        internal_::eth_write_remote_reg(k_Txq, k_SoftResetAddr, k_ResetValue);
        txq_wait_or_recover_init(k_Txq);
    }

    inline static void set_pc(uint32_t pc) {
        constexpr uint32_t k_ResetPcAddr = LITE_FABRIC_RESET_PC;
        internal_::eth_write_remote_reg(k_Txq, k_ResetPcAddr, pc);
        txq_wait_or_recover_init(k_Txq);
    }
};

}  // namespace lite_fabric
