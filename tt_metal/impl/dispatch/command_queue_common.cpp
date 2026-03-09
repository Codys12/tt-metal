// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"

#include "impl/context/metal_context.hpp"

#include <algorithm>

#include <tt_stl/assert.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal {

namespace {

constexpr uint32_t kLogicalHostChannelStride = 4;

uint32_t get_num_host_channel_share_levels(ChipId chip_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    ChipId mmio_device_id = cluster.get_associated_mmio_device(chip_id);
    uint32_t max_share_index = 0;
    for (ChipId controlled_device_id : cluster.get_devices_controlled_by_mmio_device(mmio_device_id)) {
        uint16_t controlled_channel = cluster.get_assigned_channel_for_device(controlled_device_id);
        max_share_index =
            std::max(max_share_index, static_cast<uint32_t>(controlled_channel / kLogicalHostChannelStride));
    }
    return max_share_index + 1;
}

}  // namespace

uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) { return cq_id * cq_size; }

uint16_t get_umd_channel(uint16_t channel) { return channel & 0x3; }

uint32_t get_per_device_host_channel_size(ChipId chip_id, uint16_t channel) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    ChipId mmio_device_id = cluster.get_associated_mmio_device(chip_id);
    uint32_t host_channel_size = cluster.get_host_channel_size(mmio_device_id, channel);
    uint32_t num_share_levels = get_num_host_channel_share_levels(chip_id);
    if (num_share_levels == 1) {
        return host_channel_size;
    }

    uint32_t logical_channel_size =
        std::min(host_channel_size / num_share_levels, DispatchSettings::MAX_DEV_CHANNEL_SIZE);
    uint32_t host_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    logical_channel_size -= logical_channel_size % host_alignment;
    TT_FATAL(
        logical_channel_size >= host_alignment,
        "Logical host channel size {} too small for device {} on channel {}",
        logical_channel_size,
        chip_id,
        channel);
    return logical_channel_size;
}

uint32_t get_per_device_host_channel_offset(ChipId chip_id, uint16_t channel) {
    return static_cast<uint32_t>(channel / kLogicalHostChannelStride) *
           get_per_device_host_channel_size(chip_id, channel);
}

uint32_t get_absolute_cq_offset(ChipId chip_id, uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint32_t host_channel_stride = static_cast<uint32_t>(
        tt::tt_metal::MetalContext::instance().get_cluster().get_host_channel_stride(mmio_device_id, channel));
    uint32_t channel_offset =
        host_channel_stride * get_umd_channel(channel) + get_per_device_host_channel_offset(chip_id, channel);
    return channel_offset + get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
uint32_t get_cq_issue_rd_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = get_per_device_host_channel_offset(chip_id, channel);
    uint32_t issue_q_rd_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_RD);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv,
        sizeof(uint32_t),
        issue_q_rd_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_rd_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_rd_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_issue_wr_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = get_per_device_host_channel_offset(chip_id, channel);
    uint32_t issue_q_wr_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv,
        sizeof(uint32_t),
        issue_q_wr_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_wr_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_wr_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_wr_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = get_per_device_host_channel_offset(chip_id, channel);
    uint32_t completion_q_wr_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_WR);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv,
        sizeof(uint32_t),
        completion_q_wr_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_wr_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_wr_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
inline uint32_t get_cq_completion_rd_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = get_per_device_host_channel_offset(chip_id, channel);
    uint32_t completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_RD);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv,
        sizeof(uint32_t),
        completion_q_rd_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_rd_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_rd_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

uint32_t calculate_expected_workers_to_finish(const tt::tt_metal::IDevice* device, const SubDeviceId& sub_device_id, tt::tt_metal::HalProgrammableCoreType core_type) {
    // Sub Device manager state must be correct (from device init)
    // If core type is active ethernet, it does not include fabric routers which were created using slow dispatch
    // Not managed by fast dispatch
    const auto num_workers = device->num_worker_cores(core_type, sub_device_id);
    return num_workers;
}

}  // namespace tt::tt_metal
