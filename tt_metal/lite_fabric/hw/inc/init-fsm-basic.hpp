// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Basic handshake (only supports tunnel depth of 1)
// Expects the host to write the lite fabric kernel to the MMIO device
// 1. Each MMIO kernel Mi copies itself to the neighbour (over ethernet) Ni
// 2. Ni will wait for go signal from Mi
// 3. Mi sends first handshake to Ni
// 4. Ni returns ack and is ready. Now Mi is also ready.

#pragma once

#include <cstdint>
#include <stddef.h>
#include "eth_chan_noc_mapping.h"
#include "dataflow_api_addrgen.h"
#include "ethernet/dataflow_api.h"
#include "dataflow_api.h"
#include "ethernet/tunneling.h"
#include "lf_dev_mem_map.hpp"
#include "risc_common.h"
#include "host_interface.hpp"
#include "risc_interface.hpp"

namespace lite_fabric {

static_assert(sizeof(uint32_t) == sizeof(uintptr_t));

// TX queue for init-time data transfers over ethernet.
// During init (Phase 2), ERISC0 is in reset on both MMIO and remote sides,
// so TXQ0 is exclusively available to ERISC1.  Using the same TXQ for both
// data (eth_send_packet) and register writes (eth_write_remote_reg) ensures
// natural serialization — no cross-TXQ barriers needed.
// Steady-state uses active_txq (switched to TXQ2 by host before fabric
// router launches on ERISC0) for ERISC0/ERISC1 coexistence.
static constexpr uint32_t k_DataTxq = 0;

inline void wait_val(uint32_t addr, uint32_t val) {
    do {
        invalidate_l1_cache();
    } while (reinterpret_cast<volatile uint32_t*>(addr)[0] != val);
}

inline void routing_init(volatile lite_fabric::FabricLiteConfig* config_struct) {
    invalidate_l1_cache();

    // Ensure TXQ0 packet resend mode is active for init-time data transfers.
    // TXQ0 is normally configured by the syseng base firmware at POR, but a
    // prior assert_connected_dm1_reset MAC flush may have cleared it.
    // Redundant if already set — harmless.
    eth_txq_reg_write(k_DataTxq, ETH_TXQ_CTRL, ETH_TXQ_CTRL_KEEPALIVE);

    // This value should not be used. It comes from metal.
    // auto my_y = get_absolute_logical_y();
    int number_of_other_eth_chs = __builtin_popcount(config_struct->eth_chans_mask) - 1;
    ASSERT(number_of_other_eth_chs > 0);

    // Send the binary over ethernet to the connected core
    const auto eth_send_binary = [=]() {
        internal_::eth_send_packet<false>(
            k_DataTxq, LITE_FABRIC_DATA_START >> 4, LITE_FABRIC_DATA_START >> 4, LITE_FABRIC_DATA_SIZE >> 4);
        internal_::eth_send_packet<false>(
            k_DataTxq,
            config_struct->binary_addr >> 4,
            config_struct->binary_addr >> 4,
            config_struct->binary_size >> 4);
    };

    const auto eth_send_config = [=]() {
        internal_::eth_send_packet<false>(
            k_DataTxq,
            (uintptr_t)config_struct >> 4,
            (uintptr_t)config_struct >> 4,
            sizeof(lite_fabric::FabricLiteConfig) >> 4);
    };

    auto original_init_state = config_struct->initial_state;
    bool is_mmio = config_struct->is_mmio;
    bool is_primary = config_struct->is_primary;
    while (config_struct->current_state != lite_fabric::InitState::READY) {
        invalidate_l1_cache();

        switch (config_struct->current_state) {
            case lite_fabric::InitState::UNKNOWN: {
                break;
            }
            case lite_fabric::InitState::ETH_INIT_FROM_HOST: {
                break;
            }
            case lite_fabric::InitState::ETH_INIT_LOCAL: {
                break;
            }
            case lite_fabric::InitState::ETH_HANDSHAKE_NEIGHBOUR: {
                auto handshake_addr = (uintptr_t)&config_struct->neighbour_handshake;
                auto local_handshake_addr = (uintptr_t)&config_struct->primary_local_handshake;

                if (is_mmio) {
                    wait_val(handshake_addr, 1);
                    // Safe to modify config_struct now
                    config_struct->primary_local_handshake = 2;
                    // Use <false> to skip risc_context_switch / ncrisc_noc_full_sync.
                    // ERISC0 may have used NOC0 before being reset, leaving HW counters
                    // out of sync with ERISC1's freshly-initialized SW counters.
                    internal_::eth_send_packet<false>(k_DataTxq, local_handshake_addr >> 4, handshake_addr >> 4, 1);

                    // Wait for ack
                    wait_val(handshake_addr, 3);
                } else {
                    // Send first signal to mmio to indicate we have started
                    config_struct->primary_local_handshake = 1;
                    // Use <false>: on the remote side, ERISC0 (syseng FW) is still running
                    // and sharing NOC0.  risc_context_switch() -> ncrisc_noc_full_sync()
                    // would hang because HW NOC counters (incremented by ERISC0) don't
                    // match ERISC1's SW counters.
                    internal_::eth_send_packet<false>(k_DataTxq, local_handshake_addr >> 4, handshake_addr >> 4, 1);

                    // wait for signal from mmio
                    wait_val(handshake_addr, 2);

                    // send ack to mmio
                    config_struct->primary_local_handshake = 3;
                    internal_::eth_send_packet<false>(k_DataTxq, local_handshake_addr >> 4, handshake_addr >> 4, 1);
                }
                config_struct->current_state = lite_fabric::InitState::READY;
                // Restore is_mmio after handshake clobber.  eth_send_packet sends
                // 16 bytes from primary_local_handshake [offset 16..31] to
                // neighbour_handshake [offset 32..47].  is_mmio sits at offset 44,
                // so the handshake zeroes it (padding1[2] = 0).  Without this
                // restore, object_init's noc_self_read_word sees is_mmio=0 in L1
                // and sets on_mmio_chip=false on the MMIO side, breaking completion
                // gating and d2h receiver updates.
                config_struct->is_mmio = is_mmio;
                break;
            }
            case lite_fabric::InitState::ETH_INIT_NEIGHBOUR: {
                ASSERT(is_primary);
                ASSERT(is_mmio);
                // Breadcrumb 0x10: entering ETH_INIT_NEIGHBOUR
                config_struct->primary_local_handshake = 0x10;
                config_struct->is_primary = false;
                config_struct->is_mmio = false;
                config_struct->routing_enabled = lite_fabric::RoutingEnabledState::ENABLED;
                config_struct->current_state = lite_fabric::InitState::ETH_HANDSHAKE_NEIGHBOUR;
                config_struct->initial_state = lite_fabric::InitState::ETH_HANDSHAKE_NEIGHBOUR;
                // Breadcrumb 0x11: about to assert remote reset
                config_struct->primary_local_handshake = 0x11;
                ConnectedRiscInterface::assert_connected_dm1_reset();
                // Breadcrumb 0x12: about to set remote PC
                config_struct->primary_local_handshake = 0x12;
                ConnectedRiscInterface::set_pc(LITE_FABRIC_TEXT_START);
                // Clear forwarding.is_reverse_relay before sending config to
                // neighbor.  The downstream sender's ForwardingConfig has
                // is_reverse_relay=1, but the neighbor (final-destination chip)
                // must NOT use the reverse-relay NOC_READ path — it needs the
                // normal !on_mmio_chip handler to execute reads locally.
                uint8_t saved_reverse_relay = config_struct->forwarding.is_reverse_relay;
                config_struct->forwarding.is_reverse_relay = 0;
                // Breadcrumb 0x13: about to send config
                config_struct->primary_local_handshake = 0x13;
                eth_send_config();
                config_struct->forwarding.is_reverse_relay = saved_reverse_relay;
                // Breadcrumb 0x14: about to send binary (DATA section)
                config_struct->primary_local_handshake = 0x14;
                eth_send_binary();
                // Wait for TXQ0 data transfers to complete before deasserting.
                // With k_DataTxq=0, both data and WRITE_REG share TXQ0 so this
                // is naturally serialized, but the explicit barrier is a safety
                // net in case the TXQ assignment changes in the future.
                while (internal_::eth_txq_is_busy(k_DataTxq)) {
                }
                // Breadcrumb 0x15: about to deassert remote reset
                config_struct->primary_local_handshake = 0x15;
                ConnectedRiscInterface::deassert_connected_dm1_reset();
                // Breadcrumb 0x16: ETH_INIT_NEIGHBOUR complete, entering handshake
                config_struct->primary_local_handshake = 0x16;
                // Restore is_mmio for this core.  We set it to false above so the
                // connected core receives an accurate config (it's not on the MMIO
                // chip).  But this core IS on the MMIO side and object_init reads
                // config->is_mmio to set the on_mmio_chip global, which gates
                // return-forwarding and receiver completion updates.
                config_struct->is_mmio = is_mmio;
                break;
            }
            case lite_fabric::InitState::ETH_HANDSHAKE_LOCAL: {
                break;
            }
            default: {
                ASSERT(false);
                while (true) {
                };
            }
        }
    }
}

}  // namespace lite_fabric
