// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "tt_metal/hw/inc/api/alignment.h"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/lite_fabric/hw/inc/constants.hpp"
#include "tt_metal/lite_fabric/hw/inc/header.hpp"

#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

#include <fmt/ranges.h>
#include <umd/device/types/xy_pair.hpp>
#include <tt-logger/tt-logger.hpp>

#endif

namespace lite_fabric {

template <size_t LIMIT = 0, typename T>
auto wrap_increment(T val) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + 1) & (static_cast<T>(LIMIT - 1));
    } else {
        return (val == static_cast<T>(LIMIT - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
    }
}

enum class InitState : uint16_t {
    // Unknown initial state
    UNKNOWN = 0,
    // Indicates that this is written directly from host
    ETH_INIT_FROM_HOST,
    // Write kernel to local ethernet cores and wait for ack
    ETH_INIT_LOCAL,
    // Wait for ack from connected ethernet core
    ETH_HANDSHAKE_NEIGHBOUR,
    // Write primary kernel to connected ethernet core and wait for ack
    ETH_INIT_NEIGHBOUR,
    // Wait for ack from local ethernet cores
    ETH_HANDSHAKE_LOCAL,
    // Ready for traffic
    READY,
    // Terminated
    TERMINATED,
};

enum class RoutingEnabledState : uint16_t {
    // Write to disable routing. This will stop all routing activity and propagate routing enabled state to the
    // connected core
    STOPPED = 0,
    // Enabled. Call functions to service channels.
    ENABLED = 1,
    // Stopped
    STOP = 2,
};

struct FabricLiteConfig {
    // Starting address of the Lite Fabric binary to be copied locally and to the neighbour.
    volatile uint32_t binary_addr = 0;

    // Size of the Lite Fabric binary.
    volatile uint32_t binary_size = 0;

    // Bit N is 1 if channel N is an active ethernet core. Relies on eth_chan_to_noc_xy to
    // get the ethernet core coordinate.
    volatile uint32_t eth_chans_mask = 0;

    uint32_t padding0{};

    // Subordinate cores on the same chip increment this value when they are ready. The primary core
    // will stall until this value shows all eth cores are ready.
    volatile uint32_t primary_local_handshake = 0;

    uint32_t padding1[3]{};

    // Becomes 1 when the neighbour is ready
    volatile uint32_t neighbour_handshake = 0;

    uint32_t padding2[1]{};

    // This is the local primary core
    volatile uint16_t is_primary = false;

    volatile uint8_t primary_eth_core_x = 0;

    volatile uint8_t primary_eth_core_y = 0;

    // This is on the MMIO
    volatile uint16_t is_mmio = false;

    volatile InitState initial_state = InitState::UNKNOWN;

    volatile InitState current_state = InitState::UNKNOWN;

    unsigned char padding3[14]{};

    volatile RoutingEnabledState routing_enabled = RoutingEnabledState::STOPPED;

    unsigned char padding4[14]{};

    // Multi-hop forwarding configuration.  Written by the host after lite fabric
    // is running to turn an endpoint into a relay.  The receiver inspects this
    // when processing packets with FORWARD_ONLY or WRITE_AND_FORWARD routing.
    struct ForwardingConfig {
        volatile uint8_t enabled = 0;                      // 0 = endpoint only, 1 = relay mode
        uint8_t downstream_noc_x = 0;                      // NOC X of downstream ETH core on same chip
        uint8_t downstream_noc_y = 0;                      // NOC Y of downstream ETH core on same chip
        uint8_t downstream_num_buffers = 0;                // Sender buffer count on downstream core
        volatile uint32_t downstream_sender_buf_addr = 0;  // Sender buffer base L1 addr on downstream core
        volatile uint32_t downstream_h2d_addr = 0;         // h2d L1 addr on downstream core
        uint32_t downstream_buffer_size = 0;               // Buffer slot size (= CHANNEL_BUFFER_SIZE)
        // Initial value for forwarding_downstream_wr_idx.  Must match the
        // target sender's current d2h.sender so that wrap_increment(initial)
        // produces a value != d2h, triggering the sender to pick up the packet.
        uint8_t initial_wr_idx = 0;
        // 1 = this core is a downstream sender that reverse-forwards read
        // responses upstream.  Used to gate mailbox polling, reverse-forwarding
        // in the NOC_READ handler, and receiver completion gating.  Upstream
        // receivers (including the real MMIO receiver) leave this at 0.
        uint8_t is_reverse_relay = 0;
        uint8_t _forwarding_pad[14]{};  // Pad ForwardingConfig to maintain 16-byte struct alignment
    } __attribute__((packed)) forwarding;
} __attribute__((packed));

static_assert(sizeof(FabricLiteConfig) % 16 == 0);
static_assert(offsetof(FabricLiteConfig, primary_local_handshake) % 16 == 0);
static_assert(offsetof(FabricLiteConfig, neighbour_handshake) % 16 == 0);

class HostToFabricLiteReadEvent {
private:
    inline static std::atomic<uint64_t> event{0};

public:
    static uint64_t get() { return event.load(); }

    static void increment() { event.fetch_add(1); }
};

// Interface for Host to MMIO Lite Fabric (per-channel).
// d2h and h2d are on-device (L1); the rest are host-only.
template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
struct HostToFabricLiteInterface {
    static constexpr uint32_t k_ConnectedDeviceId = 1;

    // This values are updated by the device and read to the host
    struct DeviceToHost {
        volatile uint8_t fabric_sender_channel_index = 0;
        volatile uint8_t fabric_receiver_channel_index = 0;
    } __attribute((packed)) d2h;

    // Padding to ensure d2h and h2d occupy separate 4-byte words.
    // Without this, firmware byte writes to d2h (RISC-V SB -> word-level
    // RMW on L1) can clobber concurrent host writes to h2d in the same word.
    uint8_t _d2h_h2d_pad[2]{};

    // These values are updated by the host and written to the device
    struct HostToDevice {
        volatile uint8_t sender_host_write_index = 0;
        volatile uint8_t receiver_host_read_index = 0;
    } __attribute((packed)) h2d;

    // Host only fields
    uint32_t host_interface_on_device_addr = 0;
    uint32_t sender_channel_base = 0;
    uint32_t receiver_channel_base = 0;
    uint32_t eth_barrier_addr = 0;
    uint32_t tensix_barrier_addr = 0;
    uint32_t l1_alignment_bytes = 0;  // Assumed to be 16B
    // The core to process requests
    uint32_t mmio_device_id = 0;
    uint32_t mmio_eth_core_x = 0;
    uint32_t mmio_eth_core_y = 0;

    explicit HostToFabricLiteInterface() = default;

    void init() volatile {
        h2d.sender_host_write_index = 0;
        h2d.receiver_host_read_index = 0;
        d2h.fabric_sender_channel_index = 0;
        d2h.fabric_receiver_channel_index = 0;
    }

    constexpr uint32_t get_max_payload_data_size_bytes() const {
        // Additional 16B to be used only for unaligned reads/writes
        return CHANNEL_BUFFER_SIZE - sizeof(FabricLiteHeader) - 16;
    }
} __attribute__((packed));

// Helper to compute total sender buffers across all channels
constexpr size_t total_sender_buffers() {
    size_t total = 0;
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        total += SENDER_NUM_BUFFERS_ARRAY[i];
    }
    return total;
}

constexpr size_t total_receiver_buffers() {
    size_t total = 0;
    for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
        total += RECEIVER_NUM_BUFFERS_ARRAY[i];
    }
    return total;
}

struct FabricLiteMemoryMap {
    uint32_t sender_flow_control_semaphore{};
    uint32_t padding0[3]{};
    uint32_t sender_connection_live_semaphore{};
    uint32_t padding1[3]{};
    uint32_t worker_semaphore{};
    uint32_t padding2[7]{};

    // Channel 0 sender buffers (outbound commands: writes + read commands)
    unsigned char sender_ch0_buffer[lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0] * lite_fabric::CHANNEL_BUFFER_SIZE]{};
    // Channel 1 sender buffers (read responses going back to host)
    unsigned char sender_ch1_buffer[lite_fabric::SENDER_NUM_BUFFERS_ARRAY[1] * lite_fabric::CHANNEL_BUFFER_SIZE]{};

    unsigned char padding3[64]{};

    // Channel 0 receiver buffers (incoming commands on remote side)
    unsigned char receiver_ch0_buffer[lite_fabric::RECEIVER_NUM_BUFFERS_ARRAY[0] * lite_fabric::CHANNEL_BUFFER_SIZE]{};
    // Channel 1 receiver buffers (incoming read responses on MMIO side)
    unsigned char receiver_ch1_buffer[lite_fabric::RECEIVER_NUM_BUFFERS_ARRAY[1] * lite_fabric::CHANNEL_BUFFER_SIZE]{};

    // L1 address of the service_lite_fabric function
    uint32_t service_lite_fabric_addr{};
    unsigned char padding4[12]{};

    lite_fabric::FabricLiteConfig config;
    tt::tt_fabric::EDMChannelWorkerLocationInfo sender_ch0_location_info;
    tt::tt_fabric::EDMChannelWorkerLocationInfo sender_ch1_location_info;

    // Channel 0 host interface (outbound commands)
    HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>
        host_interface;
    // Channel 1 host interface (read responses)
    HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[1], lite_fabric::CHANNEL_BUFFER_SIZE>
        host_interface_ch1;
};

static_assert(offsetof(FabricLiteMemoryMap, sender_flow_control_semaphore) % 16 == 0);
static_assert(offsetof(FabricLiteMemoryMap, sender_connection_live_semaphore) % 16 == 0);
static_assert(offsetof(FabricLiteMemoryMap, worker_semaphore) % 16 == 0);
static_assert(offsetof(FabricLiteMemoryMap, sender_ch0_buffer) % GLOBAL_ALIGNMENT == 0);
static_assert(offsetof(FabricLiteMemoryMap, receiver_ch0_buffer) % GLOBAL_ALIGNMENT == 0);
static_assert(offsetof(FabricLiteMemoryMap, config) % 16 == 0);
static_assert(offsetof(FabricLiteMemoryMap, host_interface) % 16 == 0);

}  // namespace lite_fabric
