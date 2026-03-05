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
extern RemoteReceiverChannelsType remote_receiver_channels;
extern LocalSenderChannelsType local_sender_channels;
extern WriteTridTracker receiver_channel_0_trid_tracker;

extern OutboundReceiverChannelPointersTupleImpl outbound_to_receiver_channel_pointers_tuple;
extern ReceiverChannelPointersTupleImpl receiver_channel_pointers_tuple;

// Forwarding state for multi-hop relay
extern volatile FabricLiteConfig::ForwardingConfig* forwarding_config;
extern uint8_t forwarding_downstream_wr_idx;

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
    constexpr uint32_t sender_txq_id = 0;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile lite_fabric::FabricLiteHeader*>(src_addr);
    pkt_header->debug = 0xd05e0000;

    if (pkt_header->get_base_send_type() == lite_fabric::NocSendTypeEnum::WRITE_REG) {
        // Only apply WRITE_REG locally if this is the last hop before the destination.
        // For intermediate FORWARD_ONLY hops, skip eth_write_remote_reg — the downstream
        // sender at the final hop will apply it via its own ETH link.
        uint32_t current_hop = pkt_header->routing_fields.value & lite_fabric::LiteFabricRoutingFields::FIELD_MASK;
        if (current_hop == lite_fabric::LiteFabricRoutingFields::WRITE_ONLY ||
            current_hop == lite_fabric::LiteFabricRoutingFields::WRITE_AND_FORWARD) {
            const uint32_t reg_address = pkt_header->command_fields.write_reg.reg_address;
            const uint32_t reg_value = pkt_header->command_fields.write_reg.reg_value;
            while (internal_::eth_txq_is_busy(sender_txq_id));
            internal_::eth_write_remote_reg(sender_txq_id, reg_address, reg_value);
        }
        // Continue to forward the packet to ensure pointers are synced
    }

    size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
    // Actual payload may be offset by an unaligned offset. Ensure we include this in the payload size
    // Buffer slots have 16B padding at the end which is unused.
    payload_size_bytes += pkt_header->unaligned_offset;
    payload_size_bytes = (payload_size_bytes + 15) & ~15;
    uint32_t dest_addr = receiver_buffer_channel.get_cached_next_buffer_slot_addr();
    pkt_header->src_ch_id = 0;

    while (internal_::eth_txq_is_busy(sender_txq_id));
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    host_interface->d2h.fabric_sender_channel_index =
        tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(
            host_interface->d2h.fabric_sender_channel_index);

    remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
        tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(remote_receiver_buffer_index.get())};
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    sender_buffer_channel.advance_to_next_cached_buffer_slot_addr();
    remote_receiver_num_free_slots--;
    // update the remote reg
    static constexpr uint32_t packets_to_forward = 1;
    while (internal_::eth_txq_is_busy(sender_txq_id));
    remote_update_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX], sender_txq_id>(packets_to_forward);
}

template <uint32_t CHANNEL_INDEX>
FORCE_INLINE void run_sender_channel_step() {
    auto& outbound_to_receiver_channel_pointers =
        outbound_to_receiver_channel_pointers_tuple.template get<CHANNEL_INDEX>();
    auto& local_sender_channel = lite_fabric::local_sender_channels.template get<CHANNEL_INDEX>();
    auto& remote_receiver_channel = remote_receiver_channels.template get<CHANNEL_INDEX>();
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    bool has_unsent_packet =
        host_interface->h2d.sender_host_write_index != host_interface->d2h.fabric_sender_channel_index;
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

    // Multi-hop forwarding: inspect routing_fields to decide whether to forward
    // this packet to a downstream ETH core on the same chip via NOC write.
    uint32_t current_hop_action = header.routing_fields.value & lite_fabric::LiteFabricRoutingFields::FIELD_MASK;
    if (forwarding_config->enabled && (current_hop_action == lite_fabric::LiteFabricRoutingFields::FORWARD_ONLY ||
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
        while (
            !ncrisc_noc_nonposted_write_with_transaction_id_sent(lite_fabric::edm_to_local_chip_noc, transaction_id)) {
            invalidate_l1_cache();
        }

        // Signal the downstream sender to pick up the forwarded packet.
        // Instead of writing h2d.sender directly (not 16B-aligned, would clobber
        // d2h via the 16B-aligned write workaround), write new_wr_idx to the
        // target core's ForwardingConfig.initial_wr_idx which IS 16B-aligned.
        // The target FW polls initial_wr_idx and copies to h2d.sender locally.
        uint8_t new_wr_idx =
            lite_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(forwarding_downstream_wr_idx);
        // Build 16B scratch block with new_wr_idx at byte 0 (initial_wr_idx is
        // at byte 0 of its 16B-aligned block), rest is padding.
        volatile uint32_t* h2d_scratch =
            reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint32_t>(packet_start) + total_size);
        h2d_scratch[0] = static_cast<uint32_t>(new_wr_idx);  // initial_wr_idx in LSB
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
        while (
            !ncrisc_noc_nonposted_write_with_transaction_id_sent(lite_fabric::edm_to_local_chip_noc, transaction_id)) {
            invalidate_l1_cache();
        }

        forwarding_downstream_wr_idx = new_wr_idx;

        // Diagnostic: record outbound forwarding target in padding0 so the host
        // can verify the NOC write destination.
        // bits 31-24: downstream_noc_x
        // bits 23-16: downstream_noc_y
        // bits 15-8:  new_wr_idx (value written to downstream h2d.sender)
        // bits 7-0:   downstream_h2d_addr low byte
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
            if (on_mmio_chip && forwarding_config->enabled) {
                // Relay node (downstream receiver): forward read response upstream.
                // The downstream tunnel has is_mmio=true and forwarding configured
                // to point to the upstream core's sender buffer.  The response
                // arrived from the downstream chip; relay it upstream so it
                // eventually reaches the real MMIO receiver for the host to read.
                uint32_t upstream_buf = forwarding_config->downstream_sender_buf_addr +
                                        forwarding_downstream_wr_idx * forwarding_config->downstream_buffer_size;
                uint64_t upstream_noc_addr = get_noc_addr(
                    forwarding_config->downstream_noc_x, forwarding_config->downstream_noc_y, upstream_buf);

                uint32_t total_size =
                    sizeof(lite_fabric::FabricLiteHeader) + header.unaligned_offset + payload_size_bytes;
                total_size = (total_size + 15) & ~15;
                // Per-TRID writes + barriers (see outbound forwarding comment above for rationale)
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

                // Signal the upstream sender to pick up the response.
                // Write new_wr_idx to the target's ForwardingConfig.initial_wr_idx
                // (16B-aligned mailbox) instead of h2d.sender (not 16B-aligned).
                // This avoids clobbering the upstream sender's d2h, which may be
                // non-zero from processing previous forwarding config writes.
                uint8_t new_wr_idx =
                    lite_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(forwarding_downstream_wr_idx);
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
                const uint64_t src_address = header.command_fields.noc_read.noc_address;
                // This assumes nobody else is using the sender channel on device 1 because
                // the tunnel depth is only 1 at the moment
                uint32_t dst_header_address = sender_buffer_channel.get_cached_next_buffer_slot_addr();
                // Create packet header for writing back
                tt_l1_ptr lite_fabric::FabricLiteHeader* packet_header_in_sender_ch =
                    reinterpret_cast<lite_fabric::FabricLiteHeader*>(dst_header_address);
                *packet_header_in_sender_ch = header;
                // Read the data into the buffer
                // This is safe only if the data at the sender buffer slot has been flushed out
                // We rely on the host to not do a read until the received data has been read out
                // When doing reads, ensure that the lower bits of the src_address and payload_dst_address are the same
                // we will let the host know of the data offset by setting header.unaligned_offset

                // Calculate natural payload location (immediately after header)
                uint32_t natural_payload_address = dst_header_address + sizeof(lite_fabric::FabricLiteHeader);

                // Get the lower 6 bits that we need to match from source address
                uint32_t src_alignment = src_address & (GLOBAL_ALIGNMENT - 1);  // Lower 6 bits of source
                uint32_t natural_alignment =
                    natural_payload_address & (GLOBAL_ALIGNMENT - 1);  // Lower 6 bits of natural location

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
                auto* diag_map = reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);
                diag_map->config.padding1[1] = static_cast<uint32_t>(src_address);
                diag_map->config.padding1[2] = static_cast<uint32_t>(src_address >> 32);

                // Sentinel-based read barrier: ERISC0 and ERISC1 share NOC0 HW
                // read counters.  ERISC0's service_eth_msg() can issue NOC reads
                // that increment NIU_MST_RD_RESP_RECEIVED, causing the equality
                // check in ncrisc_noc_reads_flushed() to permanently fail (HW
                // overshoots SW).  Instead of using the shared counter, we write
                // a sentinel to the read destination and poll for the NOC DMA to
                // overwrite it.  On BH ERISC, L1 is SRAM (no D-cache), so
                // volatile reads see NOC DMA writes immediately.
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
                    // Tell ourselves there is data to send
                    // NOTE: sender_buffer_channel index will be incremented in send_next_data
                    host_interface->h2d.sender_host_write_index =
                        tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(
                            host_interface->h2d.sender_host_write_index);
                    // Keep the forwarding mailbox in sync: service_lite_fabric()
                    // polls initial_wr_idx and copies to h2d.sender every iteration.
                    // If we don't update initial_wr_idx here, the polling clobbers
                    // h2d.sender back to the stale mailbox value, preventing the
                    // sender from ever picking up this 1-hop read response.
                    if (forwarding_config->enabled) {
                        forwarding_config->initial_wr_idx = host_interface->h2d.sender_host_write_index;
                    }
                } else {
                    // Don't update sender_host_write_index — no response is sent back.
                    // The host-side wait_for_read_event will time out and report the error.
                }
            }
            // else: real MMIO chip without forwarding — host reads from receiver buffer directly
        } break;

        case lite_fabric::NocSendTypeEnum::WRITE_REG: {
            // Do nothing. Sender directly wrote to us with eth_write_remote_reg
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
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX]>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = pkts_received_since_last_check != 0;

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr lite_fabric::FabricLiteHeader* packet_header = const_cast<lite_fabric::FabricLiteHeader*>(
            remote_receiver_channel.template get_packet_header<lite_fabric::FabricLiteHeader>(receiver_buffer_index));

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);

        uint8_t trid = receiver_channel_0_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
            receiver_buffer_index);
        // lite fabric tunnel depth is 1 so any fabric cmds being sent here will be writes to/reads from this chip
        service_fabric_request<CHANNEL_INDEX>(
            packet_header, packet_header->payload_size_bytes, trid, local_sender_channel);

        wr_sent_counter.increment();
        // decrement the to_receiver_0_pkts_sent_id stream register by 1 since current packet has been processed.
        increment_local_update_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX]>(-1);
    }

    // flush and completion are fused, so we only need to update one of the counters
    // update completion since other parts of the code check against completion
    auto& completion_counter = receiver_channel_pointers.completion_counter;
    // Currently unclear if it's better to loop here or not...
    bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
    auto receiver_buffer_index = completion_counter.get_buffer_index();
    bool next_trid_flushed = receiver_channel_0_trid_tracker.transaction_flushed(receiver_buffer_index);
    bool can_send_completion = unflushed_writes && next_trid_flushed;
    if (on_mmio_chip && !forwarding_config->enabled) {
        // On the real MMIO receiver (no forwarding), gate completion on the host
        // having consumed previous responses.  Skip this for relay receivers
        // (on_mmio_chip=true + forwarding enabled) because no host drains their
        // receiver buffer — the relay forwards responses upstream directly.
        can_send_completion =
            can_send_completion &&
            (((host_interface->d2h.fabric_receiver_channel_index + 1) % RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]) !=
             host_interface->h2d.receiver_host_read_index);
    }

    if (can_send_completion) {
        // Completion pointer is to the host
        while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ));
        remote_update_ptr_val<DEFAULT_ETH_TXQ>(to_sender_pkts_completed_ids[CHANNEL_INDEX], 1);

        receiver_channel_0_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
        if (on_mmio_chip) {
            host_interface->d2h.fabric_receiver_channel_index =
                tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(
                    host_interface->d2h.fabric_receiver_channel_index);
        }
    }
}

}  // namespace lite_fabric
