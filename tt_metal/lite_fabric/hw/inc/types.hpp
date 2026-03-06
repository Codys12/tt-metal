// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstdint>
#include "tt_metal/lite_fabric/hw/inc/header.hpp"
#include "tt_metal/lite_fabric/hw/inc/constants.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp"

namespace lite_fabric {

template <template <uint8_t> class ChannelType, auto& BufferSizes, typename Seq>
using ChannelPointersTupleImpl = tt::tt_fabric::ChannelPointersTupleImpl<ChannelType, BufferSizes, Seq>;

template <template <uint8_t> class ChannelType, auto& BufferSizes>
using ChannelPointersTuple = tt::tt_fabric::ChannelPointersTuple<ChannelType, BufferSizes>;

template <uint8_t RECEIVER_NUM_BUFFERS>
using OutboundReceiverChannelPointers = tt::tt_fabric::OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS>;
using OutboundReceiverChannelPointersTuple =
    lite_fabric::ChannelPointersTuple<OutboundReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>;
using OutboundReceiverChannelPointersTupleImpl =
    decltype(lite_fabric::ChannelPointersTuple<OutboundReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make());

template <uint8_t RECEIVER_NUM_BUFFERS>
using ReceiverChannelPointers = tt::tt_fabric::ReceiverChannelPointers<RECEIVER_NUM_BUFFERS>;
using ReceiverChannelPointersTuple =
    lite_fabric::ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>;
using ReceiverChannelPointersTupleImpl =
    decltype(lite_fabric::ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make());

// Per-channel type aliases.  Channel 0 and 1 have the same buffer count (4)
// so they share the same concrete type.
using SenderEthChannelBuffer = tt::tt_fabric::SenderEthChannel<FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY[0]>;
using ReceiverEthChannelBuffer = tt::tt_fabric::EthChannelBuffer<FabricLiteHeader, RECEIVER_NUM_BUFFERS_ARRAY[0]>;

// Channel 1 aliases (same type when buffer counts match)
using SenderEthChannelBuffer1 = tt::tt_fabric::SenderEthChannel<FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY[1]>;
using ReceiverEthChannelBuffer1 = tt::tt_fabric::EthChannelBuffer<FabricLiteHeader, RECEIVER_NUM_BUFFERS_ARRAY[1]>;

using HostInterface = HostToFabricLiteInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE>;
using HostInterface1 = HostToFabricLiteInterface<SENDER_NUM_BUFFERS_ARRAY[1], CHANNEL_BUFFER_SIZE>;

using WriteTridTracker = WriteTransactionIdTracker<
    RECEIVER_NUM_BUFFERS_ARRAY[0],
    NUM_TRANSACTION_IDS,
    TRID_OFFSET,
    lite_fabric::edm_to_local_chip_noc,
    lite_fabric::edm_to_downstream_noc>;

using WriteTridTracker1 = WriteTransactionIdTracker<
    RECEIVER_NUM_BUFFERS_ARRAY[1],
    NUM_TRANSACTION_IDS,
    TRID_OFFSET_CH1,
    lite_fabric::edm_to_local_chip_noc,
    lite_fabric::edm_to_downstream_noc>;

using RemoteReceiverChannelsType =
    decltype(tt::tt_fabric::StaticSizedEthChannelBuffers<FabricLiteHeader, RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_RECEIVER_CHANNELS>{}));

using LocalSenderChannelsType =
    decltype(tt::tt_fabric::StaticSizedSenderEthChannelBuffers<FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_SENDER_CHANNELS>{}));

}  // namespace lite_fabric
