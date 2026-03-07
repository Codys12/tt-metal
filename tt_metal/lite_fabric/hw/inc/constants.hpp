// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include "tt_metal/lite_fabric/hw/inc/header.hpp"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"
#include "noc_nonblocking_api.h"
#endif

namespace lite_fabric {

// STREAM REGISTER ASSIGNMENT
// Consult tt_metal/fabric/erisc_datamover_builder.hpp StreamRegAssignments to ensure no conflicts.
// Fabric router (ERISC0) uses IDs 0-22 and 29-31.  Lite fabric (ERISC1) uses 23-28.
// Channel 0: outbound commands (host → remote writes + read commands)
constexpr uint32_t to_receiver_0_pkts_sent_id = 23;
constexpr uint32_t to_sender_0_pkts_acked_id = 24;
constexpr uint32_t to_sender_0_pkts_completed_id = 25;
// Channel 1: inbound read responses (remote → host)
constexpr uint32_t to_receiver_1_pkts_sent_id = 26;
constexpr uint32_t to_sender_1_pkts_acked_id = 27;
constexpr uint32_t to_sender_1_pkts_completed_id = 28;

// Dual channels: ch0 for outbound commands, ch1 for read responses
constexpr size_t MAX_NUM_RECEIVER_CHANNELS = 2;
constexpr size_t MAX_NUM_SENDER_CHANNELS = 2;

constexpr std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> to_receiver_pkts_sent_ids = {
    to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id};
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_pkts_acked_ids = {
    to_sender_0_pkts_acked_id, to_sender_1_pkts_acked_id};
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_pkts_completed_ids = {
    to_sender_0_pkts_completed_id, to_sender_1_pkts_completed_id};

constexpr uint32_t NUM_RECEIVER_CHANNELS = 2;
constexpr uint32_t NUM_USED_RECEIVER_CHANNELS = 2;
constexpr uint32_t NUM_SENDER_CHANNELS = 2;

constexpr uint8_t NUM_TRANSACTION_IDS = 4;
// Offset into the NOC TRID space so lite fabric (ERISC1) doesn't collide with
// the fabric router (ERISC0) which shares the same per-tile NOC0 TRID counters.
// Fabric router uses TRIDs starting at 0; we start at 8 to stay clear.
constexpr uint8_t TRID_OFFSET = 8;
// Channel 1 TRIDs must not collide with channel 0's range [TRID_OFFSET, TRID_OFFSET+NUM_TRANSACTION_IDS).
constexpr uint8_t TRID_OFFSET_CH1 = TRID_OFFSET + NUM_TRANSACTION_IDS;  // 12

// 4 buffer slots per channel for pipelining.  With dual channels, writes
// can pipeline up to 3 slots ahead while a read blocks the receiver.
constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY = {4, 4};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY = {4, 4};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_NUM_BUFFERS_ARRAY = RECEIVER_NUM_BUFFERS_ARRAY;

static_assert(NUM_SENDER_CHANNELS == 2);

// Alignment for read and write to work on all core types
constexpr uint32_t GLOBAL_ALIGNMENT = 64;
// Additional space reserved for data alignment
constexpr uint32_t ALIGNMENT_BUFFER_SIZE = GLOBAL_ALIGNMENT;
constexpr uint32_t CHANNEL_BUFFER_SIZE = 2048 + ALIGNMENT_BUFFER_SIZE + sizeof(lite_fabric::FabricLiteHeader);

constexpr size_t RECEIVER_CHANNEL_BASE_ID = NUM_SENDER_CHANNELS;
constexpr size_t SENDER_CHANNEL_BASE_ID = 0;

// Lite fabric uses TXQ0 (ERISC0 is killed at boot, so no contention).
// TODO: Move to TXQ2 for coexistence with fabric router (ERISC0) once
// TXQ2 DATA frame delivery issues are resolved.
// ETH_TXQ_CMD_START_REG (remote register writes) is TXQ0-only, so
// WRITE_REG is handled by the receiver doing a local RISC-V store
// instead of the sender using eth_write_remote_reg.
constexpr uint32_t DEFAULT_ETH_TXQ = 0;
constexpr bool multi_txq_enabled = false;
constexpr uint32_t sender_txq_id = DEFAULT_ETH_TXQ;
constexpr uint32_t receiver_txq_id = DEFAULT_ETH_TXQ;
constexpr bool enable_first_level_ack = false;
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> local_receiver_completion_counter_ptrs = {0, 0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> local_receiver_ack_counter_ptrs = {0, 0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> to_sender_remote_completion_counter_addrs = {0, 0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> to_sender_remote_ack_counter_addrs = {0, 0};

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
// ERISC1 uses NOC cmd buffer 2 for writes (DYNAMIC_NOC_NCRISC_WR_CMD_BUF)
// to avoid contention with ERISC0's cmd buffer 0 (BRISC_WR_CMD_BUF).
constexpr uint8_t local_chip_data_cmd_buf = DYNAMIC_NOC_NCRISC_WR_CMD_BUF;
#endif

// Default NoC to use for Reads/Writes
// Must match the noc_index set in the packet header by UMD (NOC0, since UMD uses
// TRANSLATED coordinates which are NOC0 coordinates on Blackhole).
constexpr uint8_t edm_to_local_chip_noc = 0;
constexpr uint8_t forward_and_local_write_noc_vc = 2;  // FabricEriscDatamoverConfig::DEFAULT_NOC_VC
constexpr uint8_t edm_to_downstream_noc = 0;

}  // namespace lite_fabric
