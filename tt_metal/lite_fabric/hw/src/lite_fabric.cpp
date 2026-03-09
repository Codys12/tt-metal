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

// Global variable definitions matching extern declarations in channels.hpp
RemoteReceiverChannelsType remote_receiver_channels __attribute__((used));

LocalSenderChannelsType local_sender_channels __attribute__((used));

// These are used by the other files
bool on_mmio_chip __attribute__((used));

volatile HostInterface* host_interface __attribute__((used));
volatile HostInterface1* host_interface_ch1 __attribute__((used));

WriteTridTracker receiver_channel_0_trid_tracker __attribute__((used));
WriteTridTracker1 receiver_channel_1_trid_tracker __attribute__((used));

volatile lite_fabric::FabricLiteConfig::ForwardingConfig* forwarding_config __attribute__((used));
uint8_t forwarding_downstream_wr_idx __attribute__((used));
bool cached_is_reverse_relay __attribute__((used));

OutboundReceiverChannelPointersTupleImpl outbound_to_receiver_channel_pointers_tuple __attribute__((used));

ReceiverChannelPointersTupleImpl receiver_channel_pointers_tuple __attribute__((used));

uint32_t diag_loop_counter __attribute__((used));

uint32_t txq_recovery_count __attribute__((used));

// Runtime TXQ: starts at 0 (TXQ0) for init and early steady-state.
// Switched to 2 (TXQ2) when host writes active_txq_request before launching
// fabric router on ERISC0 (which uses TXQ0/TXQ1).
uint32_t active_txq __attribute__((used)) = 0;

// L1-based notification counters — replaces stream register COMMAND frames.
// ETH_TXQ_CMD_START_REG is TXQ0-only on BH, so notifications are sent as
// DATA frames to these L1 counters instead of COMMAND frame register writes.
// Must be in L1 SRAM (BSS), NOT on the stack (DMEM), for NOC DMA access.
volatile uint32_t pkts_sent_notify[2][4] __attribute__((aligned(16), used));
volatile uint32_t pkts_completed_notify[2][4] __attribute__((aligned(16), used));
uint32_t pkts_sent_writer[2] __attribute__((used));
uint32_t pkts_completed_writer[2] __attribute__((used));
uint32_t pkts_sent_reader[2] __attribute__((used));
uint32_t pkts_completed_reader[2] __attribute__((used));

// Ch1 reverse forwarding mailbox and write index.
// The downstream sender writes responses to the upstream_rx's ch1 sender buffer
// and signals via ch1_fwd_mailbox.  The upstream_rx polls it via noc_self_read.
volatile uint32_t ch1_fwd_mailbox[4] __attribute__((aligned(16), used));
uint8_t ch1_reverse_wr_idx __attribute__((used));

// object_init and routing_init are expected to be called before this
__attribute__((noinline)) void service_lite_fabric() {
    invalidate_l1_cache();
    // Compiler memory barrier: invalidate_l1_cache() is asm("fence") which provides
    // hardware ordering but does NOT clobber "memory", so the compiler may keep C/C++
    // variables (like num_free_slots) in registers across iterations.  Force a full
    // reload so that values written in main() or by previous iterations are visible.
    asm volatile("" ::: "memory");
    auto* mem_map = reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);

    // Check for TXQ switch request from host (one-shot: 0 -> requested value).
    // Must use NOC self-read to bypass the BH ERISC D-cache: host writes this
    // field via PCIe after boot, and stale zero lines can survive soft reset.
    if (active_txq == 0) {
        uint32_t req_l1 = reinterpret_cast<uint32_t>(&mem_map->config.forwarding.active_txq_request);
        uint32_t req_word = noc_self_read_word(req_l1 & ~0xFu, (req_l1 & 0xFu) / 4);
        uint8_t req = static_cast<uint8_t>((req_word >> ((req_l1 & 0x3u) * 8)) & 0xFF);
        if (req != 0) {
            active_txq = req;
        }
    }

    // Static flag: once the host requests STOP and we process it, don't
    // self-heal STOPPED back to ENABLED.  Reset to false on next FW
    // incarnation via BSS zeroing (data_init).
    static bool terminate_processed = false;

    switch (mem_map->config.routing_enabled) {
        case lite_fabric::RoutingEnabledState::ENABLED: break;
        case lite_fabric::RoutingEnabledState::STOPPED:
            // Self-healing: STOPPED is the zero-init default (value 0).  A stray
            // zero-write to routing_enabled's L1 address silently disables the FW.
            if (diag_loop_counter > 100) {
                mem_map->config.routing_enabled = lite_fabric::RoutingEnabledState::ENABLED;
                break;
            }
            return;
        case lite_fabric::RoutingEnabledState::STOP:
            if (terminate_processed) {
                mem_map->config.routing_enabled = lite_fabric::RoutingEnabledState::ENABLED;
                break;
            }
            terminate_processed = true;
            mem_map->config.routing_enabled = lite_fabric::RoutingEnabledState::STOPPED;
            // Preserve any remote fabric router already running on ERISC0.
            // Shutdown only the lite-fabric ERISC1 on the connected core.
            ConnectedRiscInterface::assert_connected_erisc1_reset_only();
            {
                constexpr uint32_t routing_enabled_address =
                    LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteConfig, routing_enabled);
                // Use active_txq for steady-state sends (TXQ0 initially, TXQ2 after host signal).
                internal_::eth_send_packet<false>(
                    lite_fabric::active_txq, routing_enabled_address >> 4, routing_enabled_address >> 4, 1);
            }
            return;
    }

    // Self-healing: if num_free_slots is 0 but there are no pending packets
    // (h2d == d2h), force-init to RECEIVER_NUM_BUFFERS.
    // Read h2d via NOC self-read to bypass D-cache (host writes via PCIe).
    // Channel 0
    {
        volatile uint32_t* nfs_ptr = &outbound_to_receiver_channel_pointers_tuple.template get<0>().num_free_slots;
        uint8_t h2d_s = read_h2d_sender_via_noc(host_interface);
        bool no_pending_packets = h2d_s == host_interface->d2h.fabric_sender_channel_index;
        if (*nfs_ptr == 0 && no_pending_packets) {
            *nfs_ptr = RECEIVER_NUM_BUFFERS_ARRAY[0];
        }
        // Defensive: sanitize h2d.sender_host_write_index
        if (h2d_s >= SENDER_NUM_BUFFERS_ARRAY[0]) {
            host_interface->h2d.sender_host_write_index = host_interface->d2h.fabric_sender_channel_index;
        }
    }
    // Channel 1: h2d written by local FW only (NOC_READ handler), so direct
    // volatile load is correct — NOC self-read would miss write-back cache data.
    {
        volatile uint32_t* nfs_ptr = &outbound_to_receiver_channel_pointers_tuple.template get<1>().num_free_slots;
        uint8_t h2d_s = host_interface_ch1->h2d.sender_host_write_index;
        bool no_pending_packets = h2d_s == host_interface_ch1->d2h.fabric_sender_channel_index;
        if (*nfs_ptr == 0 && no_pending_packets) {
            *nfs_ptr = RECEIVER_NUM_BUFFERS_ARRAY[1];
        }
        // Defensive: sanitize h2d.sender_host_write_index
        if (h2d_s >= SENDER_NUM_BUFFERS_ARRAY[1]) {
            host_interface_ch1->h2d.sender_host_write_index = host_interface_ch1->d2h.fabric_sender_channel_index;
        }
    }

    // Lazy-init forwarding_downstream_wr_idx when forwarding is activated.
    // Only applies to channel 0 (forwarding is for outbound commands).
    if (forwarding_downstream_wr_idx == 0xFF) {
        uint32_t init_wr_l1 = reinterpret_cast<uint32_t>(&forwarding_config->initial_wr_idx);
        uint32_t word = noc_self_read_word(init_wr_l1 & ~0xFu, (init_wr_l1 & 0xFu) / 4);
        uint8_t init_wr = static_cast<uint8_t>(word & 0xFF);
        if (init_wr != 0xFF) {
            forwarding_downstream_wr_idx = init_wr;
            cached_is_reverse_relay = static_cast<bool>((word >> 8) & 0xFF);
            // Refresh D-cache for ForwardingConfig block 0
            {
                constexpr uint32_t SENTINEL = 0xCAFEDEAD;
                static volatile uint32_t fwd_buf[4] __attribute__((aligned(16)));
                uint32_t fwd_l1 = reinterpret_cast<uint32_t>(forwarding_config);
                fwd_buf[0] = SENTINEL;
                uint64_t fwd_noc = get_noc_addr(my_x[0], my_y[0], fwd_l1);
                lite_fabric::lf_noc_async_read(
                    fwd_noc, reinterpret_cast<uint32_t>(&fwd_buf[0]), 16, lite_fabric::edm_to_local_chip_noc);
                while (fwd_buf[0] == SENTINEL) {
                    invalidate_l1_cache();
                }
                auto* fwd_raw = reinterpret_cast<volatile uint32_t*>(forwarding_config);
                fwd_raw[0] = fwd_buf[0];
                fwd_raw[1] = fwd_buf[1];
                fwd_raw[2] = fwd_buf[2];
                fwd_raw[3] = fwd_buf[3];
            }
            forwarding_config->initial_wr_idx = read_h2d_sender_via_noc(host_interface);
        }
    }

    // Mailbox polling: forwarding writes new_wr_idx to our initial_wr_idx.
    // Only for channel 0 forwarding (downstream senders and remote upstream receivers).
    if (forwarding_downstream_wr_idx != 0xFF && (!on_mmio_chip || cached_is_reverse_relay)) {
        uint32_t init_wr_l1 = reinterpret_cast<uint32_t>(&forwarding_config->initial_wr_idx);
        uint32_t word = noc_self_read_word(init_wr_l1 & ~0xFu, (init_wr_l1 & 0xFu) / 4);
        uint8_t mailbox = static_cast<uint8_t>(word & 0xFF);
        if (mailbox != host_interface->h2d.sender_host_write_index) {
            host_interface->h2d.sender_host_write_index = mailbox;
        }
    }

    // Ch1 mailbox polling: when this core is an upstream_rx (outbound forwarder,
    // NOT reverse-relay), the downstream sender writes ch1 read responses to our
    // ch1 sender buffer and signals via ch1_fwd_mailbox.  Update h2d_ch1.sender
    // so run_sender_channel_step<1> picks up the response and sends it via ETH.
    if (forwarding_downstream_wr_idx != 0xFF && !cached_is_reverse_relay) {
        uint32_t mb_l1 = reinterpret_cast<uint32_t>(&ch1_fwd_mailbox[0]);
        uint32_t word = noc_self_read_word(mb_l1 & ~0xFu, (mb_l1 & 0xFu) / 4);
        uint8_t mb_val = static_cast<uint8_t>(word & 0xFF);
        if (mb_val != host_interface_ch1->h2d.sender_host_write_index) {
            host_interface_ch1->h2d.sender_host_write_index = mb_val;
        }
    }

    // Run both sender channels before receiver channels so that any pending
    // responses (sender ch1) are flushed before processing new commands
    // (receiver ch0) that may fill more response slots.
    lite_fabric::run_sender_channel_step<0>();
    lite_fabric::run_sender_channel_step<1>();
    lite_fabric::run_receiver_channel_step<0>();
    lite_fabric::run_receiver_channel_step<1>();

    // Diagnostic: write sender ch0 flow-control state
    {
        auto& optr = outbound_to_receiver_channel_pointers_tuple.template get<0>();
        bool has_unsent =
            host_interface->h2d.sender_host_write_index != host_interface->d2h.fabric_sender_channel_index;
        int32_t completion_reg = static_cast<int32_t>(pkts_completed_notify[0][0] - pkts_completed_reader[0]);
        mem_map->config.primary_local_handshake = (static_cast<uint32_t>(optr.num_free_slots & 0xFF) << 24) |
                                                  (static_cast<uint32_t>(completion_reg & 0xFF) << 16) |
                                                  (static_cast<uint32_t>(has_unsent) << 8) |
                                                  static_cast<uint32_t>(optr.num_free_slots > 0 && has_unsent);
    }
    // Diagnostic: write receiver ch0 flow-control state to padding1[0]
    {
        auto& rptr = receiver_channel_pointers_tuple.template get<0>();
        mem_map->config.padding1[0] = (static_cast<uint32_t>(rptr.wr_sent_counter.counter & 0xFF) << 24) |
                                      (static_cast<uint32_t>(rptr.completion_counter.counter & 0xFF) << 16) |
                                      (static_cast<uint32_t>(host_interface->d2h.fabric_receiver_channel_index) << 8) |
                                      static_cast<uint32_t>(host_interface->h2d.receiver_host_read_index);
    }
    // Diagnostic: forwarding state
    mem_map->config.padding2[0] =
        static_cast<uint32_t>(forwarding_downstream_wr_idx) | (static_cast<uint32_t>(forwarding_config->enabled) << 8) |
        (static_cast<uint32_t>(on_mmio_chip) << 16) | (static_cast<uint32_t>(mem_map->config.routing_enabled) << 24);
    // Diagnostic: TXQ recovery count and is_reverse_relay state
    mem_map->config.padding1[1] =
        (txq_recovery_count & 0xFFFF) | (static_cast<uint32_t>(cached_is_reverse_relay) << 16);
    // Loop counter so the host can verify firmware is alive
    mem_map->config.neighbour_handshake = ++diag_loop_counter;

    // Software keepalive safety net: BH MAC may time out (~300ms) if ERISC0
    // stops generating traffic (e.g., during Phase 3 reset/relaunch).  Send a
    // harmless DATA frame every ~65K iterations as extra insurance.
    // Use primary_local_handshake region (offsets 16-31: handshake + padding1)
    // instead of neighbour_handshake — the latter's 16B span covers is_mmio
    // and initial_state at offsets 44-47, clobbering important config fields.
    if ((diag_loop_counter & 0xFFFF) == 0) {
        auto* cfg = &mem_map->config;
        auto addr = reinterpret_cast<uintptr_t>(&cfg->primary_local_handshake);
        internal_::eth_send_packet<false>(lite_fabric::active_txq, addr >> 4, addr >> 4, 1);
    }
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

    // Sender channel buffer addresses (dual channels)
    const uint32_t lf_local_sender_0_channel_address = (uintptr_t)&mem_map->sender_ch0_buffer;
    const uint32_t lf_local_sender_1_channel_address = (uintptr_t)&mem_map->sender_ch1_buffer;
    const uint32_t lf_local_sender_channel_0_connection_info_addr = (uintptr_t)&mem_map->sender_ch0_location_info;
    const uint32_t lf_local_sender_channel_1_connection_info_addr = (uintptr_t)&mem_map->sender_ch1_location_info;
    const uint32_t lf_remote_receiver_0_channel_buffer_address = (uintptr_t)&mem_map->receiver_ch0_buffer;
    const uint32_t lf_remote_receiver_1_channel_buffer_address = (uintptr_t)&mem_map->receiver_ch1_buffer;
    const std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {
        lf_local_sender_0_channel_address, lf_local_sender_1_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        lf_remote_receiver_0_channel_buffer_address, lf_remote_receiver_1_channel_buffer_address};

    const uint32_t lf_local_sender_channel_0_connection_semaphore_addr =
        (uintptr_t)&mem_map->sender_connection_live_semaphore;
    auto lf_sender0_worker_semaphore_ptr =
        reinterpret_cast<volatile uint32_t*>((uintptr_t)&mem_map->sender_flow_control_semaphore);

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        lf_local_sender_channel_0_connection_info_addr, lf_local_sender_channel_1_connection_info_addr};

    // Initialize stream registers for both channels
    init_ptr_val<to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<to_sender_0_pkts_acked_id>(0);
    init_ptr_val<to_sender_0_pkts_completed_id>(0);
    init_ptr_val<to_receiver_1_pkts_sent_id>(0);
    init_ptr_val<to_sender_1_pkts_acked_id>(0);
    init_ptr_val<to_sender_1_pkts_completed_id>(0);

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
    (lite_fabric::receiver_channel_pointers_tuple.template get<1>()).reset();

    // NOC self-read for is_mmio: D-cache may retain stale value from a
    // previous FW incarnation (cache survives soft resets).
    {
        uint32_t is_mmio_l1 = reinterpret_cast<uint32_t>(&mem_map->config.is_mmio);
        uint32_t aligned = is_mmio_l1 & ~0xFu;
        uint32_t woff = (is_mmio_l1 & 0xFu) / 4;
        uint32_t word = noc_self_read_word(aligned, woff);
        lite_fabric::on_mmio_chip = static_cast<bool>(word & 0xFFFF);
    }
    lite_fabric::host_interface = &mem_map->host_interface;
    lite_fabric::host_interface_ch1 = &mem_map->host_interface_ch1;
    lite_fabric::forwarding_config = &mem_map->config.forwarding;
    lite_fabric::forwarding_downstream_wr_idx = 0xFF;  // sentinel: lazy-init when forwarding activated
    mem_map->config.forwarding.initial_wr_idx = 0xFF;
    mem_map->service_lite_fabric_addr = reinterpret_cast<uint32_t>(&service_lite_fabric);
    lite_fabric::host_interface->init();
    lite_fabric::host_interface_ch1->init();
}

inline void data_init() { wzerorange(__ldm_bss_start, __ldm_bss_end); }

inline void teardown(volatile lite_fabric::FabricLiteMemoryMap* mem_map) {
    lite_fabric::receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();
    lite_fabric::receiver_channel_1_trid_tracker.all_buffer_slot_transactions_acked();

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

    // Disable L1 data cache allocations BEFORE data_init and ERISC0 kill.
    // data_init() zeroes BSS via RISC-V stores.  With the D-cache in normal
    // allocating mode, each store creates a D-cache line.  Later, ETH DMA
    // reads from L1 SRAM (NOT D-cache), so RISC-V writes to BSS variables
    // used as ETH DMA sources (sent_notify_scratch, comp_notify_scratch)
    // would never reach L1 — the ETH DMA reads stale zeros, and remote
    // receivers never detect incoming packets.
    //
    // Setting CSR 0x7c0 bit 3 BEFORE data_init prevents D-cache line
    // allocation.  All BSS writes (by data_init and the FW) go directly
    // to L1 SRAM (write miss with no-allocate → write-through to L1).
    // ETH DMA reads then see the correct values.
#if defined(ARCH_BLACKHOLE)
    asm volatile("li t1, 0x8\n\tcsrs 0x7c0, t1" ::: "t1", "memory");
#endif

    // Kill ERISC0 to prevent TXQ0 contention during init-fsm.
    // ERISC0 syseng FW shares TXQ0 with ERISC1.  With ERISC0 dead, TXQ0 is
    // exclusively ERISC1's for init (k_DataTxq=0) and early steady-state
    // (active_txq=0 → TXQ0).  Before fabric router launches on ERISC0
    // (TXQ0/TXQ1), the host signals active_txq_request=2 and ERISC1 switches
    // to TXQ2.  NOC cmd buffer contention is prevented by ERISC1 using
    // cmd buffers 2/3 (vs ERISC0's 0/1).
    {
        constexpr uint32_t kSoftResetAddr = 0xFFB121B0;
        *reinterpret_cast<volatile uint32_t*>(kSoftResetAddr) = 0x46800;  // ERISC0 in reset
        *reinterpret_cast<volatile uint32_t*>(0xFFB90000) = 0x1;          // TXQ0 enable
        for (volatile uint32_t i = 0; i < 50000; i++) {
        }  // Wait for reset
    }

    noc_index = NOC_INDEX;
    lite_fabric::data_init();

    risc_init();
    noc_init(MEM_LITE_FABRIC_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    // Initialize NOC cmd buffers 2 (write) and 3 (read) for ERISC1.
    // ERISC0 uses cmd buffers 0/1, ERISC1 uses 2/3 to avoid contention.
    // Mirror noc_init() setup for cmd buffers 0/1 onto 2/3.
    {
        for (uint32_t n = 0; n < NUM_NOCS; n++) {
            uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_CFG(NOC_ID_LOGICAL));
            uint32_t mx = noc_id_reg & NOC_NODE_ID_MASK;
            uint32_t my_noc = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
            uint64_t xy_local = NOC_XY_ADDR(mx, my_noc, 0);

            // Cmd buf 3 (read): same as cmd buf 1 in noc_init()
            uint32_t noc_rd_cmd_field =
                NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
            NOC_CMD_BUF_WRITE_REG(n, lite_fabric::LF_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
            NOC_CMD_BUF_WRITE_REG(n, lite_fabric::LF_RD_CMD_BUF, NOC_RET_ADDR_MID, 0x0);
            NOC_CMD_BUF_WRITE_REG(
                n,
                lite_fabric::LF_RD_CMD_BUF,
                NOC_RET_ADDR_COORDINATE,
                (uint32_t)(xy_local >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

            // Cmd buf 2 (write): same as cmd buf 0 in noc_init()
            NOC_CMD_BUF_WRITE_REG(n, lite_fabric::LF_WR_CMD_BUF, NOC_TARG_ADDR_MID, 0x0);
            NOC_CMD_BUF_WRITE_REG(
                n,
                lite_fabric::LF_WR_CMD_BUF,
                NOC_TARG_ADDR_COORDINATE,
                (uint32_t)(xy_local >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        }
    }

    // Drain stale per-TRID HW counters left from a previous ERISC1 incarnation.
    // Drain both channel 0 and channel 1 TRID ranges.
    {
        constexpr uint32_t noc = lite_fabric::edm_to_local_chip_noc;
        constexpr uint32_t k_MaxDrainIters = 100000;
        // Channel 0 TRIDs
        for (uint8_t i = 0; i < lite_fabric::NUM_TRANSACTION_IDS; i++) {
            uint32_t trid = lite_fabric::TRID_OFFSET + i;
            for (uint32_t iter = 0; iter < k_MaxDrainIters; iter++) {
                if (ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, trid)) {
                    break;
                }
            }
        }
        // Channel 1 TRIDs
        for (uint8_t i = 0; i < lite_fabric::NUM_TRANSACTION_IDS; i++) {
            uint32_t trid = lite_fabric::TRID_OFFSET_CH1 + i;
            for (uint32_t iter = 0; iter < k_MaxDrainIters; iter++) {
                if (ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, trid)) {
                    break;
                }
            }
        }
    }

    // Sweep FabricLiteMemoryMap to invalidate stale D-cache lines from previous
    // FW incarnation.  BH ERISC D-cache is write-back and survives soft resets
    // and FLR.  CSR 0x7c0 bit 3 prevents NEW allocations but stale dirty lines
    // from the prior incarnation still accept RISC-V stores (write-back, not
    // reaching L1).  ETH DMA reads from L1 → sends stale data.  NOC self-read
    // completions snoop-invalidate D-cache lines.  After this sweep, all RISC-V
    // stores to FabricLiteMemoryMap go directly to L1 (write miss, no allocate).
    {
        constexpr uint32_t sweep_start = LITE_FABRIC_CONFIG_START;
        constexpr uint32_t sweep_size = LITE_FABRIC_CONFIG_SIZE;
        constexpr uint32_t chunk = 256;
        constexpr uint8_t noc = lite_fabric::edm_to_local_chip_noc;
        for (uint32_t off = 0; off < sweep_size; off += chunk) {
            uint32_t addr = sweep_start + off;
            uint32_t sz = ((sweep_size - off) < chunk) ? (sweep_size - off) : chunk;
            uint64_t noc_addr = get_noc_addr(my_x[0], my_y[0], addr);
            lite_fabric::lf_noc_async_read(noc_addr, addr, sz, noc);
            while (!noc_cmd_buf_ready(noc, lite_fabric::LF_RD_CMD_BUF)) {
            }
        }
    }

    auto structs = reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);

    // Self-healing TXQ2 check for ALL ERISC1 instances.
    {
        constexpr uint32_t TXQ2_BASE = ETH_TXQ0_REGS_START + 2 * ETH_TXQ_REGS_SIZE;
        constexpr uint32_t TXQ2_CTRL = TXQ2_BASE + ETH_TXQ_CTRL;
        constexpr uint32_t TXQ2_CMD = TXQ2_BASE + ETH_TXQ_CMD;
        constexpr uint32_t TXQ2_STATUS = TXQ2_BASE + ETH_TXQ_STATUS;

        // BH-55 workaround: dummy read of CMD before reading STATUS
        (void)*reinterpret_cast<volatile uint32_t*>(TXQ2_CMD);
        uint32_t status = *reinterpret_cast<volatile uint32_t*>(TXQ2_STATUS);
        uint32_t ctrl = *reinterpret_cast<volatile uint32_t*>(TXQ2_CTRL);
        bool cmd_ongoing = (status >> 16) & 1;

        if (cmd_ongoing) {
            *reinterpret_cast<volatile uint32_t*>(TXQ2_CTRL) = 0;
            for (volatile uint32_t i = 0; i < 10000; i++) {
            }
            *reinterpret_cast<volatile uint32_t*>(TXQ2_CMD) = 0x8;  // MAC queue flush
            for (volatile uint32_t i = 0; i < 10000; i++) {
            }
            *reinterpret_cast<volatile uint32_t*>(TXQ2_CTRL) = ETH_TXQ_CTRL_KEEPALIVE;
            for (volatile uint32_t i = 0; i < 10000; i++) {
            }
            (void)*reinterpret_cast<volatile uint32_t*>(TXQ2_CMD);
            status = *reinterpret_cast<volatile uint32_t*>(TXQ2_STATUS);
            cmd_ongoing = (status >> 16) & 1;
        }
    }

    // Enable TXQ2 packet resend mode.  TXQ2 is used after the host signals
    // active_txq_request=2 (before fabric router launches on ERISC0).
    *reinterpret_cast<volatile uint32_t*>(ETH_TXQ0_REGS_START + 2 * ETH_TXQ_REGS_SIZE + ETH_TXQ_CTRL) =
        ETH_TXQ_CTRL_KEEPALIVE;

    // Clone all configuration registers from TXQ0 to TXQ2.
    // The POR syseng firmware configures TXQ0 with the correct MAC DA, timeouts,
    // packet sizes, and pipelining depth.  TXQ2 starts with hardware defaults
    // (often 0) which can cause silent frame drops.
    {
        static constexpr uint32_t regs_to_copy[] = {
            0x0C,  // MAX_PKT_SIZE_BYTES
            0x10,  // BURST_LEN
            0x48,  // REMOTE_SEQ_TIMEOUT
            0x4C,  // LOCAL_SEQ_UPDATE_TIMEOUT
            0x50,  // DEST_MAC_ADDR_HI
            0x54,  // DEST_MAC_ADDR_LO
            0x58,  // SRC_MAC_ADDR_HI
            0x5C,  // SRC_MAC_ADDR_LO
            0x60,  // ETH_TYPE
            0x64,  // MIN_PACKET_SIZE_WORDS
            0x6C,  // DATA_PACKET_ACCEPT_AHEAD
            0x80,  // TXPKT_CFG_SEL_SW
            0x84,  // TXPKT_CFG_SEL_HW
        };
        for (uint32_t reg : regs_to_copy) {
            eth_txq_reg_write(2, reg, eth_txq_reg_read(0, reg));
        }
    }

    // Diagnostic breadcrumb: send on TXQ2 (now configured) for non-MMIO cores.
    if (!structs->config.is_mmio) {
        auto* cfg = &structs->config;
        auto src_addr = (uintptr_t)&cfg->primary_local_handshake;
        auto dst_addr = (uintptr_t)&cfg->neighbour_handshake;
        cfg->primary_local_handshake = 0xA0;  // "Remote alive, TXQ2 OK"
        internal_::eth_send_packet<false>(lite_fabric::active_txq, src_addr >> 4, dst_addr >> 4, 1);
    }

    lite_fabric::object_init(structs);
    lite_fabric::routing_init(&structs->config);

    // Explicitly reinitialize sender flow control for both channels after routing handshake.
    lite_fabric::outbound_to_receiver_channel_pointers_tuple.template get<0>().num_free_slots =
        lite_fabric::RECEIVER_NUM_BUFFERS_ARRAY[0];
    lite_fabric::outbound_to_receiver_channel_pointers_tuple.template get<0>().remote_receiver_buffer_index =
        tt::tt_fabric::BufferIndex{0};
    lite_fabric::outbound_to_receiver_channel_pointers_tuple.template get<1>().num_free_slots =
        lite_fabric::RECEIVER_NUM_BUFFERS_ARRAY[1];
    lite_fabric::outbound_to_receiver_channel_pointers_tuple.template get<1>().remote_receiver_buffer_index =
        tt::tt_fabric::BufferIndex{0};

    // Re-init stream registers to ensure clean state after handshake ethernet traffic
    init_ptr_val<lite_fabric::to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<lite_fabric::to_sender_0_pkts_acked_id>(0);
    init_ptr_val<lite_fabric::to_sender_0_pkts_completed_id>(0);
    init_ptr_val<lite_fabric::to_receiver_1_pkts_sent_id>(0);
    init_ptr_val<lite_fabric::to_sender_1_pkts_acked_id>(0);
    init_ptr_val<lite_fabric::to_sender_1_pkts_completed_id>(0);

    // Re-zero h2d after routing_init to prevent phantom packets from stale
    // ETH handshake traffic that may have corrupted h2d values.
    lite_fabric::host_interface->init();
    lite_fabric::host_interface_ch1->init();

    invalidate_l1_cache();
    while (true) {
        lite_fabric::service_lite_fabric();
    }

    lite_fabric::teardown(structs);

    return 0;
}
