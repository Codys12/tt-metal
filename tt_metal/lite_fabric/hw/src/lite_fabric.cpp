// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include <utility>

#include "tt_metal/api/tt-metalium/hal_types.hpp"
// Forward-declare invalidate_l1_cache so noc_nonblocking_api.h templates compile
// (full definition comes later via risc_common.h from init-fsm-basic.hpp)
inline __attribute__((always_inline)) void invalidate_l1_cache();
#include "noc_nonblocking_api.h"
#include "dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include "firmware_common.h"
#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"
#include "tt_metal/lite_fabric/hw/inc/init-fsm-basic.hpp"
#include "tt_metal/lite_fabric/hw/inc/constants.hpp"
#include "tt_metal/lite_fabric/hw/inc/channels.hpp"
#include "tt_metal/lite_fabric/hw/inc/channel_util.hpp"
#include "tt_metal/lite_fabric/hw/inc/header.hpp"
#include "tt_metal/lite_fabric/hw/inc/types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "internal/ethernet/tunneling.h"
#include "risc_interface.hpp"

#if !defined(tt_l1_ptr)
#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#endif

/////////////////////
// Metal globals -- Mainly to make it compile
/////////////////////
uint8_t noc_index __attribute__((used));

extern uint32_t __ldm_bss_start[];
extern uint32_t __ldm_bss_end[];
extern uint32_t __ldm_data_start[];
extern uint32_t __ldm_data_end[];

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[tt::tt_metal::NumHalProgrammableCoreTypes] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

// Not initialized anywhere and not used
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
// Not initialized anywhere and not used. Used for address generator apis
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

/////////////////////
// Lite Fabric globals
/////////////////////
namespace lite_fabric {

// Global variable definitions matching extern declarations in lite_fabric_channels.hpp
RemoteReceiverChannelsType remote_receiver_channels __attribute__((used));

LocalSenderChannelsType local_sender_channels __attribute__((used));

// These are used by the other files
bool on_mmio_chip __attribute__((used));

volatile HostInterface* host_interface __attribute__((used));

WriteTridTracker receiver_channel_0_trid_tracker __attribute__((used));

volatile lite_fabric::FabricLiteConfig::ForwardingConfig* forwarding_config __attribute__((used));
uint8_t forwarding_downstream_wr_idx __attribute__((used));

OutboundReceiverChannelPointersTupleImpl outbound_to_receiver_channel_pointers_tuple __attribute__((used));

ReceiverChannelPointersTupleImpl receiver_channel_pointers_tuple __attribute__((used));

uint32_t diag_loop_counter __attribute__((used));

// object_init and routing_init are expected to be called before this
__attribute__((noinline)) void service_lite_fabric() {
    invalidate_l1_cache();
    // Compiler memory barrier: invalidate_l1_cache() is asm("fence") which provides
    // hardware ordering but does NOT clobber "memory", so the compiler may keep C/C++
    // variables (like num_free_slots) in registers across iterations.  Force a full
    // reload so that values written in main() or by previous iterations are visible.
    asm volatile("" ::: "memory");
    auto* mem_map = reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);
    switch (mem_map->config.routing_enabled) {
        case lite_fabric::RoutingEnabledState::ENABLED: break;
        case lite_fabric::RoutingEnabledState::STOPPED: return;
        case lite_fabric::RoutingEnabledState::STOP:
            mem_map->config.routing_enabled = lite_fabric::RoutingEnabledState::STOPPED;
            ConnectedRiscInterface::assert_connected_dm1_reset();
            constexpr uint32_t routing_enabled_address =
                LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteConfig, routing_enabled);
            internal_::eth_send_packet<false>(
                lite_fabric::k_DataTxq, routing_enabled_address >> 4, routing_enabled_address >> 4, 1);
            return;
    }
    // Self-healing: if num_free_slots is 0 but there are no pending packets
    // (h2d == d2h), force-init to RECEIVER_NUM_BUFFERS.  This catches:
    //   - template-based init or explicit reinit elided/corrupted by compiler/LTO
    //   - binding reset: host wrote h2d = d2h to device L1 after a channel switch
    //     or ETH handshake corruption, leaving num_free_slots stale at 0
    // Use volatile to prevent the compiler from optimizing away this safety net.
    {
        volatile uint32_t* nfs_ptr = &outbound_to_receiver_channel_pointers_tuple.template get<0>().num_free_slots;
        bool no_pending_packets =
            host_interface->h2d.sender_host_write_index == host_interface->d2h.fabric_sender_channel_index;
        if (*nfs_ptr == 0 && no_pending_packets) {
            *nfs_ptr = RECEIVER_NUM_BUFFERS_ARRAY[0];
        }
    }

    // Defensive: sanitize h2d.sender_host_write_index.  Valid values are
    // [0, SENDER_NUM_BUFFERS_ARRAY[0]).  Out-of-range values indicate L1
    // corruption (e.g. stale data from a previous iteration, or a NOC write
    // landing at the h2d address from an unexpected source).  Reset to d2h
    // to suppress phantom sends that would pollute the downstream receiver.
    {
        uint8_t h2d_s = host_interface->h2d.sender_host_write_index;
        if (h2d_s >= SENDER_NUM_BUFFERS_ARRAY[0]) {
            host_interface->h2d.sender_host_write_index = host_interface->d2h.fabric_sender_channel_index;
        }
    }

    // Lazy-init forwarding_downstream_wr_idx when forwarding is first enabled.
    // The host writes initial_wr_idx and enabled=1 AFTER all reads through the
    // upstream sender complete, so the value matches the current upstream d2h.
    if (forwarding_config->enabled && forwarding_downstream_wr_idx == 0xFF) {
        invalidate_l1_cache();
        forwarding_downstream_wr_idx = forwarding_config->initial_wr_idx;
        // Reset initial_wr_idx to match h2d.sender so the mailbox polling below
        // doesn't fire prematurely.  On the downstream core, initial_wr_idx may
        // be non-zero (upstream_sender_d2h) for return forwarding init, but
        // h2d.sender is 0 from init().  Without this reset, the polling would
        // set h2d.sender to upstream_sender_d2h, causing a phantom sender send.
        forwarding_config->initial_wr_idx = host_interface->h2d.sender_host_write_index;
    }

    // Mailbox polling: forwarding writes new_wr_idx to our initial_wr_idx
    // via a 16B-aligned NOC write (avoiding the d2h-clobbering 16B write to
    // h2d.sender).  Copy the mailbox value to h2d.sender locally using a
    // RISC-V store (no alignment restrictions).
    if (forwarding_config->enabled && forwarding_downstream_wr_idx != 0xFF) {
        invalidate_l1_cache();
        uint8_t mailbox = forwarding_config->initial_wr_idx;
        if (mailbox != host_interface->h2d.sender_host_write_index) {
            host_interface->h2d.sender_host_write_index = mailbox;
        }
    }

    lite_fabric::run_sender_channel_step<0>();
    lite_fabric::run_receiver_channel_step<0>();

    // Diagnostic: write sender flow-control state so the host can read it
    // primary_local_handshake layout:
    //   bits 31-24: num_free_slots (capped at 0xFF)
    //   bits 23-16: raw completion stream register value (capped at 0xFF)
    //   bit 8:      has_unsent_packet
    //   bit 0:      can_send
    {
        auto& optr = outbound_to_receiver_channel_pointers_tuple.template get<0>();
        bool has_unsent =
            host_interface->h2d.sender_host_write_index != host_interface->d2h.fabric_sender_channel_index;
        int32_t completion_reg = get_ptr_val(to_sender_pkts_completed_ids[0]);
        mem_map->config.primary_local_handshake = (static_cast<uint32_t>(optr.num_free_slots & 0xFF) << 24) |
                                                  (static_cast<uint32_t>(completion_reg & 0xFF) << 16) |
                                                  (static_cast<uint32_t>(has_unsent) << 8) |
                                                  static_cast<uint32_t>(optr.num_free_slots > 0 && has_unsent);
    }
    // Diagnostic: write receiver flow-control state to padding1[0]
    // padding1[0] layout:
    //   bits 31-24: receiver wr_sent_counter (low byte)
    //   bits 23-16: receiver completion_counter (low byte)
    //   bits 15-8:  d2h.fabric_receiver_channel_index
    //   bits 7-0:   h2d.receiver_host_read_index
    {
        auto& rptr = receiver_channel_pointers_tuple.template get<0>();
        mem_map->config.padding1[0] = (static_cast<uint32_t>(rptr.wr_sent_counter.counter & 0xFF) << 24) |
                                      (static_cast<uint32_t>(rptr.completion_counter.counter & 0xFF) << 16) |
                                      (static_cast<uint32_t>(host_interface->d2h.fabric_receiver_channel_index) << 8) |
                                      static_cast<uint32_t>(host_interface->h2d.receiver_host_read_index);
    }
    // Diagnostic: forwarding_downstream_wr_idx so host can see relay state
    // bits 7-0: forwarding_downstream_wr_idx
    // bits 15-8: forwarding_config->enabled
    // bit 16: on_mmio_chip
    mem_map->config.padding2[0] = static_cast<uint32_t>(forwarding_downstream_wr_idx) |
                                  (static_cast<uint32_t>(forwarding_config->enabled) << 8) |
                                  (static_cast<uint32_t>(on_mmio_chip) << 16);
    // Loop counter so the host can verify firmware is alive
    mem_map->config.neighbour_handshake = ++diag_loop_counter;
}

inline void object_init(volatile lite_fabric::FabricLiteMemoryMap* mem_map) {
    local_sender_channels =
        tt::tt_fabric::StaticSizedSenderEthChannelBuffers<lite_fabric::FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY>::
            make(std::make_index_sequence<NUM_SENDER_CHANNELS>{});
    remote_receiver_channels =
        tt::tt_fabric::StaticSizedEthChannelBuffers<lite_fabric::FabricLiteHeader, RECEIVER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});
    outbound_to_receiver_channel_pointers_tuple = OutboundReceiverChannelPointersTuple::make();
    receiver_channel_pointers_tuple = ReceiverChannelPointersTuple::make();

    const uint32_t lf_local_sender_0_channel_address = (uintptr_t)&mem_map->sender_channel_buffer;
    const uint32_t lf_local_sender_channel_0_connection_info_addr = (uintptr_t)&mem_map->sender_location_info;
    const uint32_t lf_remote_receiver_0_channel_buffer_address = (uintptr_t)&mem_map->receiver_channel_buffer;
    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {
        lf_local_sender_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        lf_remote_receiver_0_channel_buffer_address};

    const uint32_t lf_local_sender_channel_0_connection_semaphore_addr =
        (uintptr_t)&mem_map->sender_connection_live_semaphore;
    auto lf_sender0_worker_semaphore_ptr =
        reinterpret_cast<volatile uint32_t*>((uintptr_t)&mem_map->sender_flow_control_semaphore);

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        lf_local_sender_channel_0_connection_info_addr};

    // Note: Do not use stream 17
    init_ptr_val<to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<to_sender_0_pkts_acked_id>(0);
    init_ptr_val<to_sender_0_pkts_completed_id>(0);

    lite_fabric::remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::FabricLiteHeader),
        RECEIVER_CHANNEL_BASE_ID);
    lite_fabric::init_receiver_headers(lite_fabric::remote_receiver_channels);

    lite_fabric::local_sender_channels.init(
        local_sender_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::FabricLiteHeader),
        SENDER_CHANNEL_BASE_ID);

    (lite_fabric::receiver_channel_pointers_tuple.template get<0>()).reset();
    lite_fabric::on_mmio_chip = mem_map->config.is_mmio;
    lite_fabric::host_interface = &mem_map->host_interface;
    lite_fabric::forwarding_config = &mem_map->config.forwarding;
    lite_fabric::forwarding_downstream_wr_idx = 0xFF;  // sentinel: lazy-init when forwarding enabled
    mem_map->service_lite_fabric_addr = reinterpret_cast<uint32_t>(&service_lite_fabric);
    lite_fabric::host_interface->init();
}

inline void data_init() { wzerorange(__ldm_bss_start, __ldm_bss_end); }

inline void teardown(volatile lite_fabric::FabricLiteMemoryMap* mem_map) {
    lite_fabric::receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    lite_fabric::ConnectedRiscInterface::assert_connected_dm1_reset();

    mem_map->config.current_state = lite_fabric::InitState::READY;
}

}  // namespace lite_fabric

int main() {
    invalidate_l1_cache();
    configure_csr();
    noc_index = NOC_INDEX;
    lite_fabric::data_init();
    risc_init();
    noc_init(MEM_LITE_FABRIC_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    // Drain stale per-TRID HW counters left from a previous ERISC1 incarnation.
    // noc_local_state_init only clears global NOC counters — it does NOT clear
    // NIU_MST_WRITE_REQS_OUTGOING_ID(trid) per-TRID registers, which persist
    // across RISC soft resets.  If a TRID counter is stuck non-zero,
    // transaction_flushed() returns false, blocking all receiver completions.
    // Poll each lite fabric TRID with a bounded timeout.  In-flight NOC writes
    // from the previous incarnation should complete within a few hundred cycles
    // (all destinations are local on-chip L1).
    {
        constexpr uint32_t noc = lite_fabric::edm_to_local_chip_noc;
        constexpr uint32_t k_MaxDrainIters = 100000;
        for (uint8_t i = 0; i < lite_fabric::NUM_TRANSACTION_IDS; i++) {
            uint32_t trid = lite_fabric::TRID_OFFSET + i;
            for (uint32_t iter = 0; iter < k_MaxDrainIters; iter++) {
                if (ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, trid)) {
                    break;
                }
            }
        }
    }

    auto structs = reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);

    // Self-healing TXQ0 check for remote (non-MMIO) ERISC1.
    // After assert_connected_dm1_reset hard-resets remote ERISC0, TXQ0 may have
    // CMD_ONGOING stuck from an interrupted bootrom DMA.  Check and fix locally.
    if (!structs->config.is_mmio) {
        constexpr uint32_t TXQ0_CTRL = 0xFFB90000;
        constexpr uint32_t TXQ0_CMD = 0xFFB90004;
        constexpr uint32_t TXQ0_STATUS = 0xFFB90008;

        // BH-55 workaround: dummy read of CMD before reading STATUS
        (void)*reinterpret_cast<volatile uint32_t*>(TXQ0_CMD);
        uint32_t status = *reinterpret_cast<volatile uint32_t*>(TXQ0_STATUS);
        uint32_t ctrl = *reinterpret_cast<volatile uint32_t*>(TXQ0_CTRL);
        bool cmd_ongoing = (status >> 16) & 1;

        if (cmd_ongoing) {
            // TXQ0 stuck! Disable KEEPALIVE to abort resend loop.
            *reinterpret_cast<volatile uint32_t*>(TXQ0_CTRL) = 0;
            for (volatile uint32_t i = 0; i < 10000; i++) {
            }
            // Flush MAC queue
            *reinterpret_cast<volatile uint32_t*>(TXQ0_CMD) = 0x8;
            for (volatile uint32_t i = 0; i < 10000; i++) {
            }
            // Re-enable KEEPALIVE
            *reinterpret_cast<volatile uint32_t*>(TXQ0_CTRL) = 0x1;
            for (volatile uint32_t i = 0; i < 10000; i++) {
            }
            // Re-check
            (void)*reinterpret_cast<volatile uint32_t*>(TXQ0_CMD);
            status = *reinterpret_cast<volatile uint32_t*>(TXQ0_STATUS);
            cmd_ongoing = (status >> 16) & 1;
        }

        // Send diagnostic breadcrumb to MMIO side via eth_send_packet.
        // Writes to remote's primary_local_handshake → MMIO's neighbour_handshake.
        // padding1[0] carries raw TXQ0 status, padding1[1] carries CTRL value.
        if (!cmd_ongoing) {
            auto* cfg = &structs->config;
            auto src_addr = (uintptr_t)&cfg->primary_local_handshake;
            auto dst_addr = (uintptr_t)&cfg->neighbour_handshake;
            cfg->primary_local_handshake = 0xA0;  // "Remote alive, TXQ0 OK"
            cfg->padding1[0] = status;
            cfg->padding1[1] = ctrl;
            internal_::eth_send_packet<false>(0, src_addr >> 4, dst_addr >> 4, 1);
        }
        // If cmd_ongoing is still set, skip breadcrumb (eth_send_packet would hang).
        // The handshake will fail, but at least we won't deadlock here.
    }

    lite_fabric::object_init(structs);
    lite_fabric::routing_init(&structs->config);

    // Explicitly reinitialize sender flow control after routing handshake.
    // The template-based OutboundReceiverChannelPointersTuple::make() init may
    // silently fail on the embedded RISC-V target (std::apply + fold expressions).
    lite_fabric::outbound_to_receiver_channel_pointers_tuple.template get<0>().num_free_slots =
        lite_fabric::RECEIVER_NUM_BUFFERS_ARRAY[0];
    lite_fabric::outbound_to_receiver_channel_pointers_tuple.template get<0>().remote_receiver_buffer_index =
        tt::tt_fabric::BufferIndex{0};
    // Re-init stream registers to ensure clean state after handshake ethernet traffic
    init_ptr_val<lite_fabric::to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<lite_fabric::to_sender_0_pkts_acked_id>(0);
    init_ptr_val<lite_fabric::to_sender_0_pkts_completed_id>(0);

    // Re-zero h2d after routing_init to prevent phantom packets from stale
    // ETH handshake traffic that may have corrupted h2d values.
    lite_fabric::host_interface->init();

    invalidate_l1_cache();
    while (true) {
        lite_fabric::service_lite_fabric();
    }

    lite_fabric::teardown(structs);

    return 0;
}
