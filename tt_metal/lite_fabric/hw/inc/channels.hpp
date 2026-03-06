// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstdint>
#include "tt_metal/lite_fabric/hw/inc/constants.hpp"
#include "tt_metal/lite_fabric/hw/inc/header.hpp"
#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"
#include "tt_metal/lite_fabric/hw/inc/types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

namespace lite_fabric {

// Linked from main
extern bool on_mmio_chip;
extern volatile HostInterface* host_interface;
extern volatile HostInterface1* host_interface_ch1;
extern RemoteReceiverChannelsType remote_receiver_channels;
extern LocalSenderChannelsType local_sender_channels;
extern WriteTridTracker receiver_channel_0_trid_tracker;
extern WriteTridTracker1 receiver_channel_1_trid_tracker;

extern OutboundReceiverChannelPointersTupleImpl outbound_to_receiver_channel_pointers_tuple;
extern ReceiverChannelPointersTupleImpl receiver_channel_pointers_tuple;

// Forwarding state for multi-hop relay
extern volatile FabricLiteConfig::ForwardingConfig* forwarding_config;
extern uint8_t forwarding_downstream_wr_idx;
extern bool cached_is_reverse_relay;

// Per-channel host interface accessor
template <uint32_t CHANNEL_INDEX>
FORCE_INLINE auto& get_host_interface_ref() {
    if constexpr (CHANNEL_INDEX == 0) {
        return *host_interface;
    } else {
        return *host_interface_ch1;
    }
}

// Per-channel trid tracker accessor
template <uint32_t CHANNEL_INDEX>
FORCE_INLINE auto& get_trid_tracker() {
    if constexpr (CHANNEL_INDEX == 0) {
        return receiver_channel_0_trid_tracker;
    } else {
        return receiver_channel_1_trid_tracker;
    }
}

// Count of TXQ0 recoveries during the main loop (diagnostic)
extern uint32_t txq_recovery_count;

// Bounded wait for TXQ to be ready, with recovery if stuck.
// TXQ can get stuck from interrupted DMA operations or transient ETH link
// issues.  Without recovery, the FW hangs forever in the spin loop.
FORCE_INLINE bool eth_txq_wait_or_recover(uint32_t txq_id) {
    const uint32_t txq_base = ETH_TXQ0_REGS_START + txq_id * ETH_TXQ_REGS_SIZE;
    constexpr uint32_t k_MaxIters = 5000000;

    for (uint32_t i = 0; i < k_MaxIters; i++) {
        if (!internal_::eth_txq_is_busy(txq_id)) {
            return true;
        }
    }

    // TXQ stuck — attempt recovery
    txq_recovery_count++;
    *reinterpret_cast<volatile uint32_t*>(txq_base + ETH_TXQ_CTRL) = 0;
    for (volatile uint32_t i = 0; i < 10000; i++) {
    }
    *reinterpret_cast<volatile uint32_t*>(txq_base + ETH_TXQ_CMD) = 0x8;  // MAC queue flush
    for (volatile uint32_t i = 0; i < 10000; i++) {
    }
    *reinterpret_cast<volatile uint32_t*>(txq_base + ETH_TXQ_CTRL) = ETH_TXQ_CTRL_KEEPALIVE;
    for (volatile uint32_t i = 0; i < 10000; i++) {
    }

    return !internal_::eth_txq_is_busy(txq_id);
}

// Read a 32-bit word from a 16B-aligned L1 address via NOC self-read,
// bypassing the BH ERISC D-cache.  The D-cache survives soft resets and
// the CSR 0x7c0 disable only prevents new allocations — stale lines from
// a previous FW incarnation still serve hits.  External NOC writes (from
// other tiles or the host via lite fabric) update L1 SRAM but do NOT
// snoop-invalidate the D-cache.  A local NOC DMA read-completion DOES
// snoop-invalidate, so reading from our own L1 via NOC guarantees fresh
// data.  word_index selects which 32-bit word (0-3) within the 16B block.
FORCE_INLINE uint32_t noc_self_read_word(uint32_t aligned_l1_addr, uint32_t word_index) {
    constexpr uint32_t SENTINEL = 0xCAFEDEAD;
    // CRITICAL: buf must be in L1 SRAM, NOT on the stack.  The lite fabric
    // stack is in ERISC DMEM (0xFFB00xxx) which is NOT reachable by NOC DMA.
    // NOC DMA read completions write to tile-local L1 SRAM only.  A stack-
    // allocated buffer would never receive the DMA data, causing the sentinel
    // spin loop to hang forever.  Using a static variable places it in BSS
    // (LITE_FABRIC_DATA section, L1 SRAM at 0x6Dxxx) where NOC DMA works.
    static volatile uint32_t buf[4] __attribute__((aligned(16)));
    buf[word_index] = SENTINEL;
    uint64_t noc_addr = get_noc_addr(my_x[0], my_y[0], aligned_l1_addr);
    noc_async_read(noc_addr, reinterpret_cast<uint32_t>(&buf[0]), 16, lite_fabric::edm_to_local_chip_noc);
    while (buf[word_index] == SENTINEL) {
        invalidate_l1_cache();
    }
    return buf[word_index];
}

/////////////////////
// Sender Channel
/////////////////////
template <uint32_t CHANNEL_INDEX>
FORCE_INLINE void send_next_data(
    SenderEthChannelBuffer& sender_buffer_channel,
    lite_fabric::OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>&
        outbound_to_receiver_channel_pointers,
    ReceiverEthChannelBuffer& receiver_buffer_channel) {
    auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    constexpr uint32_t sender_txq_id = lite_fabric::sender_txq_id;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile lite_fabric::FabricLiteHeader*>(src_addr);
    pkt_header->debug = 0xd05e0000;

    // WRITE_REG: no sender-side handling needed.  ETH_TXQ_CMD_START_REG is
    // TXQ0-only, and lite fabric uses TXQ2.  The receiver handles WRITE_REG
    // by doing a local RISC-V store (see service_fabric_request).  The
    // packet is sent as regular data below to keep pointers synced.

    size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
    // Actual payload may be offset by an unaligned offset. Ensure we include this in the payload size
    // Buffer slots have 16B padding at the end which is unused.
    payload_size_bytes += pkt_header->unaligned_offset;
    payload_size_bytes = (payload_size_bytes + 15) & ~15;
    uint32_t dest_addr = receiver_buffer_channel.get_cached_next_buffer_slot_addr();
    pkt_header->src_ch_id = CHANNEL_INDEX;

    if (!eth_txq_wait_or_recover(sender_txq_id)) {
        return;
    }
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    // Wait for data send to complete, then notify receiver BEFORE advancing
    // local pointers.  If TXQ is stuck, bail out — pointers haven't advanced,
    // so the same packet will be retried on the next iteration.
    static constexpr uint32_t packets_to_forward = 1;
    if (!eth_txq_wait_or_recover(sender_txq_id)) {
        return;
    }
    remote_update_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX], sender_txq_id>(packets_to_forward);

    auto& send_hi = get_host_interface_ref<CHANNEL_INDEX>();
    send_hi.d2h.fabric_sender_channel_index =
        tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(send_hi.d2h.fabric_sender_channel_index);

    remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
        tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(remote_receiver_buffer_index.get())};
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    sender_buffer_channel.advance_to_next_cached_buffer_slot_addr();
    remote_receiver_num_free_slots--;
}

template <uint32_t CHANNEL_INDEX>
FORCE_INLINE void run_sender_channel_step() {
    auto& outbound_to_receiver_channel_pointers =
        outbound_to_receiver_channel_pointers_tuple.template get<CHANNEL_INDEX>();
    auto& local_sender_channel = lite_fabric::local_sender_channels.template get<CHANNEL_INDEX>();
    auto& remote_receiver_channel = remote_receiver_channels.template get<CHANNEL_INDEX>();
    auto& sender_hi = get_host_interface_ref<CHANNEL_INDEX>();
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    bool has_unsent_packet = sender_hi.h2d.sender_host_write_index != sender_hi.d2h.fabric_sender_channel_index;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if (can_send) {
        send_next_data<CHANNEL_INDEX>(
            local_sender_channel, outbound_to_receiver_channel_pointers, remote_receiver_channel);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check = get_ptr_val(to_sender_pkts_completed_ids[CHANNEL_INDEX]);
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        increment_local_update_ptr_val(to_sender_pkts_completed_ids[CHANNEL_INDEX], -completions_since_last_check);
    }
}

/////////////////////
// Receiver Channel
/////////////////////
template <uint32_t CHANNEL_INDEX>
__attribute__((optimize("jump-tables"))) FORCE_INLINE void service_fabric_request(
    tt_l1_ptr lite_fabric::FabricLiteHeader* const packet_start,
    uint16_t payload_size_bytes,
    uint32_t transaction_id,
    tt::tt_fabric::SenderEthChannel<lite_fabric::FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>&
        sender_buffer_channel) {
    invalidate_l1_cache();
    const auto& header = *packet_start;

    // Multi-hop forwarding only applies to channel 0 (outbound commands).
    // Channel 1 carries read responses which are not forwarded through this path.
    if constexpr (CHANNEL_INDEX == 0) {
        // Inspect routing_fields to decide whether to forward this packet to a
        // downstream ETH core on the same chip via NOC write.
        // Use forwarding_downstream_wr_idx != 0xFF (BSS variable, always accurate)
        // instead of forwarding_config->enabled (L1 variable, potentially stale
        // due to BH ERISC L1 read caching) as the activation check.
        uint32_t current_hop_action = header.routing_fields.value & lite_fabric::LiteFabricRoutingFields::FIELD_MASK;
        if (forwarding_downstream_wr_idx != 0xFF &&
            (current_hop_action == lite_fabric::LiteFabricRoutingFields::FORWARD_ONLY ||
             current_hop_action == lite_fabric::LiteFabricRoutingFields::WRITE_AND_FORWARD)) {
            // Shift routing fields right to consume this hop
            const_cast<lite_fabric::FabricLiteHeader*>(packet_start)->routing_fields.value >>=
                lite_fabric::LiteFabricRoutingFields::FIELD_WIDTH;

            // Calculate destination in the downstream core's sender buffer
            uint32_t downstream_buf = forwarding_config->downstream_sender_buf_addr +
                                      forwarding_downstream_wr_idx * forwarding_config->downstream_buffer_size;
            uint64_t downstream_noc_addr =
                get_noc_addr(forwarding_config->downstream_noc_x, forwarding_config->downstream_noc_y, downstream_buf);

            // Copy the full packet (header + unaligned_offset + payload) to downstream sender buffer
            uint32_t total_size = sizeof(lite_fabric::FabricLiteHeader) + header.unaligned_offset + payload_size_bytes;
            total_size = (total_size + 15) & ~15;  // 16-byte aligned
            // Use per-TRID writes and barriers instead of noc_async_write + noc_async_write_barrier().
            // noc_async_write_barrier() checks NIU_MST_WR_ACK_RECEIVED == noc_nonposted_writes_acked[noc],
            // but the HW register is shared between ERISC0 and ERISC1 on the same tile.  ERISC0's
            // service_eth_msg() (called from risc_context_switch) can do NOC0 non-posted writes that
            // increment the shared HW counter without updating ERISC1's SW counter, causing the ==
            // check to never pass and ERISC1 to hang forever.  Per-TRID barriers check
            // NIU_MST_WRITE_REQS_OUTGOING_ID(trid) which is specific to our TRID and immune to ERISC0.
            noc_async_write_one_packet_with_trid<true, false>(
                reinterpret_cast<uint32_t>(packet_start),
                downstream_noc_addr,
                total_size,
                transaction_id,
                lite_fabric::local_chip_data_cmd_buf,
                lite_fabric::edm_to_local_chip_noc,
                lite_fabric::forward_and_local_write_noc_vc);
            while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(
                lite_fabric::edm_to_local_chip_noc, transaction_id)) {
                invalidate_l1_cache();
            }

            // Signal the downstream sender to pick up the forwarded packet.
            // Instead of writing h2d.sender directly (not 16B-aligned, would clobber
            // d2h via the 16B-aligned write workaround), write new_wr_idx to the
            // target core's ForwardingConfig.initial_wr_idx which IS 16B-aligned.
            // The target FW polls initial_wr_idx and copies to h2d.sender locally.
            uint8_t new_wr_idx = lite_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(forwarding_downstream_wr_idx);
            // Build 16B scratch block with new_wr_idx at byte 0 (initial_wr_idx is
            // at byte 0 of its 16B-aligned block).  Byte 1 is is_reverse_relay —
            // must be preserved (always 1 for downstream senders targeted by
            // outbound forwarding).  Without this, the 16B write clobbers
            // is_reverse_relay to 0, disabling the downstream's mailbox polling.
            volatile uint32_t* h2d_scratch =
                reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint32_t>(packet_start) + total_size);
            h2d_scratch[0] = static_cast<uint32_t>(new_wr_idx) | (1u << 8);  // initial_wr_idx + is_reverse_relay=1
            h2d_scratch[1] = 0;
            h2d_scratch[2] = 0;
            h2d_scratch[3] = 0;
            asm volatile("fence w,w" ::: "memory");
            // initial_wr_idx L1 address is the same on all cores (identical memory map)
            uint32_t mailbox_l1_addr = reinterpret_cast<uint32_t>(&forwarding_config->initial_wr_idx);
            uint64_t mailbox_noc_addr =
                get_noc_addr(forwarding_config->downstream_noc_x, forwarding_config->downstream_noc_y, mailbox_l1_addr);
            noc_async_write_one_packet_with_trid<true, false>(
                reinterpret_cast<uint32_t>(h2d_scratch),
                mailbox_noc_addr,
                16,
                transaction_id,
                lite_fabric::local_chip_data_cmd_buf,
                lite_fabric::edm_to_local_chip_noc,
                lite_fabric::forward_and_local_write_noc_vc);
            while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(
                lite_fabric::edm_to_local_chip_noc, transaction_id)) {
                invalidate_l1_cache();
            }

            forwarding_downstream_wr_idx = new_wr_idx;

            // Diagnostic: record outbound forwarding target in padding0 so the host
            // can verify the NOC write destination.
            auto* diag_map = reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);
            diag_map->config.padding0 = (static_cast<uint32_t>(forwarding_config->downstream_noc_x) << 24) |
                                        (static_cast<uint32_t>(forwarding_config->downstream_noc_y) << 16) |
                                        (static_cast<uint32_t>(new_wr_idx) << 8) |
                                        (forwarding_config->downstream_h2d_addr & 0xFF);
        }

        // For FORWARD_ONLY, skip local processing entirely
        if (current_hop_action == lite_fabric::LiteFabricRoutingFields::FORWARD_ONLY) {
            return;
        }
    }  // if constexpr (CHANNEL_INDEX == 0)

    lite_fabric::NocSendTypeEnum noc_send_type = header.get_base_send_type();
    uint8_t noc_index = header.get_noc_index();
    if (static_cast<int>(noc_send_type) > static_cast<int>(lite_fabric::NocSendTypeEnum::NOC_SEND_TYPE_LAST)) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case lite_fabric::NocSendTypeEnum::NOC_UNICAST_WRITE: {
            const uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) +
                                                   sizeof(lite_fabric::FabricLiteHeader) + header.unaligned_offset;

            const auto dest_address = header.command_fields.noc_unicast.noc_address;

            // Non-posted writes (with ACK) are required here because
            // transaction_flushed() checks NIU_MST_WRITE_REQS_OUTGOING_ID(trid),
            // which only tracks non-posted writes.  Posted writes would cause the
            // counter to be untracked, leading to premature buffer reuse (the
            // sender refills the receiver buffer before the NOC finishes reading
            // the previous payload) or a deadlocked completion path.
            noc_async_write_one_packet_with_trid<true, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                lite_fabric::local_chip_data_cmd_buf,
                noc_index,
                lite_fabric::forward_and_local_write_noc_vc);
        } break;

        case lite_fabric::NocSendTypeEnum::NOC_READ: {
            // NOC_READ handling is channel-0 only (commands arrive on ch0).
            // Channel 1 receiver on MMIO side receives read responses — they are
            // left in the receiver buffer for the host to consume directly.
            if constexpr (CHANNEL_INDEX == 0) {
                if (forwarding_downstream_wr_idx != 0xFF && cached_is_reverse_relay) {
                    // Relay node (downstream receiver): forward read response upstream.
                    // The downstream tunnel has is_mmio=true and forwarding configured
                    // to point to the upstream core's sender buffer.  The response
                    // arrived from the downstream chip; relay it upstream so it
                    // eventually reaches the real MMIO receiver for the host to read.
                    //
                    // Resync forwarding_downstream_wr_idx with the upstream sender's
                    // actual h2d.sender before writing.
                    {
                        constexpr uint32_t RESYNC_SENTINEL = 0xDEADCAFE;
                        static volatile uint32_t resync_buf[4] __attribute__((aligned(16)));
                        uint32_t raw_addr = forwarding_config->downstream_h2d_addr;
                        uint32_t aligned_addr = raw_addr & ~0xFu;
                        uint32_t word_offset = (raw_addr & 0xFu) / 4;
                        resync_buf[word_offset] = RESYNC_SENTINEL;
                        uint64_t upstream_h2d_noc = get_noc_addr(
                            forwarding_config->downstream_noc_x, forwarding_config->downstream_noc_y, aligned_addr);
                        noc_async_read(
                            upstream_h2d_noc,
                            reinterpret_cast<uint32_t>(&resync_buf[0]),
                            16,
                            lite_fabric::edm_to_local_chip_noc);
                        while (resync_buf[word_offset] == RESYNC_SENTINEL) {
                            invalidate_l1_cache();
                        }
                        forwarding_downstream_wr_idx = static_cast<uint8_t>(resync_buf[word_offset] & 0xFF);
                    }

                    uint32_t upstream_buf = forwarding_config->downstream_sender_buf_addr +
                                            forwarding_downstream_wr_idx * forwarding_config->downstream_buffer_size;
                    uint64_t upstream_noc_addr = get_noc_addr(
                        forwarding_config->downstream_noc_x, forwarding_config->downstream_noc_y, upstream_buf);

                    uint32_t total_size =
                        sizeof(lite_fabric::FabricLiteHeader) + header.unaligned_offset + payload_size_bytes;
                    total_size = (total_size + 15) & ~15;
                    noc_async_write_one_packet_with_trid<true, false>(
                        reinterpret_cast<uint32_t>(packet_start),
                        upstream_noc_addr,
                        total_size,
                        transaction_id,
                        lite_fabric::local_chip_data_cmd_buf,
                        lite_fabric::edm_to_local_chip_noc,
                        lite_fabric::forward_and_local_write_noc_vc);
                    while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(
                        lite_fabric::edm_to_local_chip_noc, transaction_id)) {
                        invalidate_l1_cache();
                    }

                    uint8_t new_wr_idx =
                        lite_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(forwarding_downstream_wr_idx);
                    volatile uint32_t* h2d_scratch =
                        reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint32_t>(packet_start) + total_size);
                    h2d_scratch[0] = static_cast<uint32_t>(new_wr_idx);
                    h2d_scratch[1] = 0;
                    h2d_scratch[2] = 0;
                    h2d_scratch[3] = 0;
                    asm volatile("fence w,w" ::: "memory");
                    uint32_t mailbox_l1_addr = reinterpret_cast<uint32_t>(&forwarding_config->initial_wr_idx);
                    uint64_t mailbox_noc_addr = get_noc_addr(
                        forwarding_config->downstream_noc_x, forwarding_config->downstream_noc_y, mailbox_l1_addr);
                    noc_async_write_one_packet_with_trid<true, false>(
                        reinterpret_cast<uint32_t>(h2d_scratch),
                        mailbox_noc_addr,
                        16,
                        transaction_id,
                        lite_fabric::local_chip_data_cmd_buf,
                        lite_fabric::edm_to_local_chip_noc,
                        lite_fabric::forward_and_local_write_noc_vc);
                    while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(
                        lite_fabric::edm_to_local_chip_noc, transaction_id)) {
                        invalidate_l1_cache();
                    }

                    forwarding_downstream_wr_idx = new_wr_idx;
                } else if (!on_mmio_chip) {
                    // Direct 1-hop read: execute the NOC read locally and place the
                    // response in sender channel 1 (dedicated read-response channel).
                    // Sender ch1 sends it back via ETH to the MMIO receiver ch1.
                    auto& sender_ch1 = local_sender_channels.template get<1>();
                    const uint64_t src_address = header.command_fields.noc_read.noc_address;
                    uint32_t dst_header_address = sender_ch1.get_cached_next_buffer_slot_addr();
                    // Create packet header for writing back
                    tt_l1_ptr lite_fabric::FabricLiteHeader* packet_header_in_sender_ch =
                        reinterpret_cast<lite_fabric::FabricLiteHeader*>(dst_header_address);
                    *packet_header_in_sender_ch = header;

                    // Calculate natural payload location (immediately after header)
                    uint32_t natural_payload_address = dst_header_address + sizeof(lite_fabric::FabricLiteHeader);

                    // Get the lower 6 bits that we need to match from source address
                    uint32_t src_alignment = src_address & (GLOBAL_ALIGNMENT - 1);
                    uint32_t natural_alignment = natural_payload_address & (GLOBAL_ALIGNMENT - 1);

                    // Calculate offset needed to align to source's lower 6 bits
                    uint32_t alignment_offset;
                    if (src_alignment >= natural_alignment) {
                        alignment_offset = src_alignment - natural_alignment;
                    } else {
                        alignment_offset = GLOBAL_ALIGNMENT + src_alignment - natural_alignment;
                    }

                    // Final aligned payload address
                    uint32_t payload_dst_address = natural_payload_address + alignment_offset;

                    // Store the offset for the host to know where the actual data starts
                    packet_header_in_sender_ch->unaligned_offset = alignment_offset;

                    // Diagnostic: record the NOC read target address for debugging
                    auto* diag_map =
                        reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);
                    diag_map->config.padding1[1] = static_cast<uint32_t>(src_address);
                    diag_map->config.padding1[2] = static_cast<uint32_t>(src_address >> 32);

                    // Sentinel-based read barrier: ERISC0 and ERISC1 share NOC0 HW
                    // read counters, so we use sentinel polling instead of
                    // ncrisc_noc_reads_flushed().
                    volatile uint32_t* sentinel_ptr = reinterpret_cast<volatile uint32_t*>(payload_dst_address);
                    constexpr uint32_t SENTINEL = 0xFACECA5E;
                    *sentinel_ptr = SENTINEL;

                    // Resync SW counter for bookkeeping (not used for barrier)
                    noc_reads_num_issued[noc_index] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);

                    noc_async_read(src_address, payload_dst_address, payload_size_bytes, noc_index);

                    bool read_completed = false;
                    {
                        constexpr uint32_t k_MaxBarrierIters = 5000000;
                        for (uint32_t i = 0; i < k_MaxBarrierIters; i++) {
                            if (*sentinel_ptr != SENTINEL) {
                                read_completed = true;
                                break;
                            }
                        }
                        invalidate_l1_cache();
                    }

                    // Resync SW counter to prevent drift for future reads
                    noc_reads_num_issued[noc_index] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);

                    if (read_completed) {
                        // Signal sender ch1 that there is a read response to send back.
                        // run_sender_channel_step<1>() will pick this up and send via ETH.
                        host_interface_ch1->h2d.sender_host_write_index =
                            tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[1]>(
                                host_interface_ch1->h2d.sender_host_write_index);
                    }
                    // else: read timed out, don't signal — host will time out too
                }
                // else: real MMIO chip without forwarding — host reads from receiver buffer directly
            }  // if constexpr (CHANNEL_INDEX == 0)
        } break;

        case lite_fabric::NocSendTypeEnum::WRITE_REG: {
            // Apply the register write locally via RISC-V store.  The sender
            // no longer uses eth_write_remote_reg (TXQ0-only) since lite fabric
            // runs on TXQ2.  The receiver is on the target chip, so a local
            // store to the register address achieves the same effect.
            const uint32_t reg_address = header.command_fields.write_reg.reg_address;
            const uint32_t reg_value = header.command_fields.write_reg.reg_value;
            *reinterpret_cast<volatile uint32_t*>(reg_address) = reg_value;
        } break;

        default: {
            ASSERT(false);
        } break;
    };
}

template <uint32_t CHANNEL_INDEX>
FORCE_INLINE void run_receiver_channel_step() {
    auto& receiver_channel_pointers = receiver_channel_pointers_tuple.template get<CHANNEL_INDEX>();
    auto& local_sender_channel = lite_fabric::local_sender_channels.template get<CHANNEL_INDEX>();
    auto& remote_receiver_channel = remote_receiver_channels.template get<CHANNEL_INDEX>();
    auto& trid_tracker = get_trid_tracker<CHANNEL_INDEX>();
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX]>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = pkts_received_since_last_check != 0;

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr lite_fabric::FabricLiteHeader* packet_header = const_cast<lite_fabric::FabricLiteHeader*>(
            remote_receiver_channel.template get_packet_header<lite_fabric::FabricLiteHeader>(receiver_buffer_index));

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);

        uint8_t trid = trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(receiver_buffer_index);
        service_fabric_request<CHANNEL_INDEX>(
            packet_header, packet_header->payload_size_bytes, trid, local_sender_channel);

        wr_sent_counter.increment();
        increment_local_update_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX]>(-1);
    }

    // flush and completion are fused, so we only need to update one of the counters
    auto& completion_counter = receiver_channel_pointers.completion_counter;
    bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
    auto receiver_buffer_index = completion_counter.get_buffer_index();
    bool next_trid_flushed = trid_tracker.transaction_flushed(receiver_buffer_index);
    bool can_send_completion = unflushed_writes && next_trid_flushed;

    auto& recv_hi = get_host_interface_ref<CHANNEL_INDEX>();
    if (on_mmio_chip && !cached_is_reverse_relay) {
        // On the real MMIO receiver, gate completion on the host having consumed
        // previous responses.  Skip for reverse-relay downstream senders.
        can_send_completion = can_send_completion &&
                              (((recv_hi.d2h.fabric_receiver_channel_index + 1) %
                                RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]) != recv_hi.h2d.receiver_host_read_index);
    }

    if (can_send_completion) {
        if (!eth_txq_wait_or_recover(DEFAULT_ETH_TXQ)) {
            return;
        }
        remote_update_ptr_val<DEFAULT_ETH_TXQ>(to_sender_pkts_completed_ids[CHANNEL_INDEX], 1);

        trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
        if (on_mmio_chip) {
            recv_hi.d2h.fabric_receiver_channel_index =
                tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(
                    recv_hi.d2h.fabric_receiver_channel_index);
        }
    }
}

}  // namespace lite_fabric
