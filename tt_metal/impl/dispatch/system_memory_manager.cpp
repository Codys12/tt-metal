// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/context/metal_context.hpp"
#include "system_memory_manager.hpp"
#include "device/device_manager.hpp"
#include "fabric/fabric_builder_context.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include <tt-metalium/tt_align.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <thread>
#include <string>
#include <tuple>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "hal_types.hpp"
#include "memcpy.hpp"
#include "command_queue_common.hpp"
#include "system_memory_cq_interface.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/tt_io.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <tracy/Tracy.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <impl/debug/inspector/inspector.hpp>
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal {

void on_dispatch_timeout_detected();

namespace {

constexpr size_t kMaxRecentFetchTraceEntries = 8192;
constexpr size_t kMaxFetchSequenceCommandsToLog = 6;
constexpr uint32_t kMaxNextFetchBytesToLog = 0x400;

const char* prefetch_cmd_name(uint8_t cmd_id) {
    switch (static_cast<CQPrefetchCmdId>(cmd_id)) {
        case CQ_PREFETCH_CMD_RELAY_LINEAR: return "RELAY_LINEAR";
        case CQ_PREFETCH_CMD_RELAY_LINEAR_H: return "RELAY_LINEAR_H";
        case CQ_PREFETCH_CMD_RELAY_PAGED: return "RELAY_PAGED";
        case CQ_PREFETCH_CMD_RELAY_PAGED_PACKED: return "RELAY_PAGED_PACKED";
        case CQ_PREFETCH_CMD_RELAY_INLINE: return "RELAY_INLINE";
        case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH: return "RELAY_INLINE_NOFLUSH";
        case CQ_PREFETCH_CMD_EXEC_BUF: return "EXEC_BUF";
        case CQ_PREFETCH_CMD_EXEC_BUF_END: return "EXEC_BUF_END";
        case CQ_PREFETCH_CMD_STALL: return "STALL";
        case CQ_PREFETCH_CMD_DEBUG: return "DEBUG";
        case CQ_PREFETCH_CMD_TERMINATE: return "TERMINATE";
        case CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER: return "PAGED_TO_RINGBUFFER";
        case CQ_PREFETCH_CMD_SET_RINGBUFFER_OFFSET: return "SET_RINGBUFFER_OFFSET";
        case CQ_PREFETCH_CMD_RELAY_RINGBUFFER: return "RELAY_RINGBUFFER";
        default: return "UNKNOWN";
    }
}

const char* dispatch_cmd_name(uint8_t cmd_id) {
    switch (static_cast<CQDispatchCmdId>(cmd_id)) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR: return "WRITE_LINEAR";
        case CQ_DISPATCH_CMD_WRITE_LINEAR_H: return "WRITE_LINEAR_H";
        case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST: return "WRITE_LINEAR_H_HOST";
        case CQ_DISPATCH_CMD_WRITE_PAGED: return "WRITE_PAGED";
        case CQ_DISPATCH_CMD_WRITE_PACKED: return "WRITE_PACKED";
        case CQ_DISPATCH_CMD_WRITE_PACKED_LARGE: return "WRITE_PACKED_LARGE";
        case CQ_DISPATCH_CMD_WAIT: return "WAIT";
        case CQ_DISPATCH_CMD_SINK: return "SINK";
        case CQ_DISPATCH_CMD_DEBUG: return "DEBUG";
        case CQ_DISPATCH_CMD_DELAY: return "DELAY";
        case CQ_DISPATCH_CMD_EXEC_BUF_END: return "EXEC_BUF_END";
        case CQ_DISPATCH_CMD_SET_WRITE_OFFSET: return "SET_WRITE_OFFSET";
        case CQ_DISPATCH_CMD_TERMINATE: return "TERMINATE";
        case CQ_DISPATCH_CMD_SEND_GO_SIGNAL: return "SEND_GO_SIGNAL";
        case CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL: return "NOTIFY_SUBORDINATE_GO_SIGNAL";
        case CQ_DISPATCH_SET_NUM_WORKER_SEMS: return "SET_NUM_WORKER_SEMS";
        case CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA: return "SET_GO_SIGNAL_NOC_DATA";
        default: return "UNKNOWN";
    }
}

std::string summarize_first_dispatch_cmd(const uint8_t* payload, uint32_t payload_size) {
    if (payload_size < sizeof(CQDispatchCmd)) {
        return fmt::format("<short:{}B>", payload_size);
    }

    const auto* cmd = reinterpret_cast<const CQDispatchCmd*>(payload);
    const auto cmd_id = static_cast<uint8_t>(cmd->base.cmd_id);
    std::string summary = dispatch_cmd_name(cmd_id);

    switch (static_cast<CQDispatchCmdId>(cmd_id)) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR:
        case CQ_DISPATCH_CMD_WRITE_LINEAR_H: {
            if (payload_size < sizeof(CQDispatchCmdLarge)) {
                summary += " <short-large>";
                break;
            }
            const auto* large = reinterpret_cast<const CQDispatchCmdLarge*>(payload);
            summary += fmt::format(
                " noc=0x{:x} addr=0x{:x} len=0x{:x} mcast={}",
                large->write_linear.noc_xy_addr,
                large->write_linear.addr,
                large->write_linear.length,
                large->write_linear.num_mcast_dests);
            break;
        }
        case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
            summary +=
                fmt::format(" len=0x{:x} is_event={}", cmd->write_linear_host.length, cmd->write_linear_host.is_event);
            break;
        case CQ_DISPATCH_CMD_WRITE_PAGED:
            summary += fmt::format(
                " base=0x{:x} page=0x{:x} pages={} dram={}",
                cmd->write_paged.base_addr,
                cmd->write_paged.page_size,
                cmd->write_paged.pages,
                cmd->write_paged.is_dram);
            break;
        case CQ_DISPATCH_CMD_WRITE_PACKED:
            summary += fmt::format(
                " count={} size=0x{:x} flags=0x{:x} addr=0x{:x}",
                cmd->write_packed.count,
                cmd->write_packed.size,
                cmd->write_packed.flags,
                cmd->write_packed.addr);
            break;
        case CQ_DISPATCH_CMD_WRITE_PACKED_LARGE:
            summary += fmt::format(
                " count={} align={} type={} woff={}",
                cmd->write_packed_large.count,
                cmd->write_packed_large.alignment,
                cmd->write_packed_large.type,
                cmd->write_packed_large.write_offset_index);
            break;
        case CQ_DISPATCH_CMD_WAIT:
            summary += fmt::format(
                " flags=0x{:x} count={} addr=0x{:x} stream={}",
                cmd->wait.flags,
                cmd->wait.count,
                cmd->wait.addr,
                cmd->wait.stream);
            break;
        case CQ_DISPATCH_CMD_DEBUG:
            summary += fmt::format(" key={} size={} stride=0x{:x}", cmd->debug.key, cmd->debug.size, cmd->debug.stride);
            break;
        case CQ_DISPATCH_CMD_DELAY: summary += fmt::format(" delay={}", cmd->delay.delay); break;
        case CQ_DISPATCH_SET_NUM_WORKER_SEMS:
            summary += fmt::format(" num_worker_sems={}", cmd->set_num_worker_sems.num_worker_sems);
            break;
        case CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA:
            summary += fmt::format(" words={}", cmd->set_go_signal_noc_data.num_words);
            break;
        default: break;
    }

    return summary;
}

std::vector<std::string> summarize_fetch_sequence(
    const char* cq_sysmem_start, uint32_t channel_offset, const FetchTraceEntry& entry) {
    std::vector<std::string> lines;
    if (entry.size_bytes == 0 || entry.start_addr < channel_offset) {
        return lines;
    }

    const auto host_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    const auto* sequence =
        reinterpret_cast<const uint8_t*>(cq_sysmem_start + static_cast<ptrdiff_t>(entry.start_addr - channel_offset));

    uint32_t offset = 0;
    for (size_t cmd_index = 0; cmd_index < kMaxFetchSequenceCommandsToLog && offset < entry.size_bytes; ++cmd_index) {
        if (entry.size_bytes - offset < sizeof(CQPrefetchCmd)) {
            lines.push_back(fmt::format(
                "prefetch{} @0x{:x} <short:{}B>", cmd_index, entry.start_addr + offset, entry.size_bytes - offset));
            break;
        }

        const auto* cmd = reinterpret_cast<const CQPrefetchCmd*>(sequence + offset);
        const auto cmd_id = static_cast<uint8_t>(cmd->base.cmd_id);
        uint32_t stride = host_alignment;
        std::string line =
            fmt::format("prefetch{} @0x{:x} {}", cmd_index, entry.start_addr + offset, prefetch_cmd_name(cmd_id));

        switch (static_cast<CQPrefetchCmdId>(cmd_id)) {
            case CQ_PREFETCH_CMD_RELAY_LINEAR:
            case CQ_PREFETCH_CMD_RELAY_LINEAR_H: {
                if (entry.size_bytes - offset < sizeof(CQPrefetchCmdLarge)) {
                    line += " <short-large>";
                    stride = entry.size_bytes - offset;
                    break;
                }
                const auto* large = reinterpret_cast<const CQPrefetchCmdLarge*>(sequence + offset);
                const auto len =
                    cmd_id == CQ_PREFETCH_CMD_RELAY_LINEAR ? large->relay_linear.length : large->relay_linear_h.length;
                line += fmt::format(
                    " noc=0x{:x} addr=0x{:x} len=0x{:x}",
                    large->relay_linear.noc_xy_addr,
                    large->relay_linear.addr,
                    len);
                break;
            }
            case CQ_PREFETCH_CMD_RELAY_PAGED:
                line += fmt::format(
                    " base=0x{:x} page=0x{:x} pages={} start_page={} flags=0x{:x}",
                    cmd->relay_paged.base_addr,
                    cmd->relay_paged.page_size,
                    cmd->relay_paged.pages,
                    cmd->relay_paged.start_page,
                    cmd->relay_paged.is_dram_and_length_adjust);
                break;
            case CQ_PREFETCH_CMD_RELAY_PAGED_PACKED:
                stride = cmd->relay_paged_packed.stride;
                line += fmt::format(
                    " count={} total=0x{:x} stride=0x{:x}",
                    cmd->relay_paged_packed.count,
                    cmd->relay_paged_packed.total_length,
                    stride);
                break;
            case CQ_PREFETCH_CMD_RELAY_INLINE:
            case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            case CQ_PREFETCH_CMD_EXEC_BUF_END: {
                stride = cmd->relay_inline.stride;
                line += fmt::format(
                    " len=0x{:x} stride=0x{:x} disp={}",
                    cmd->relay_inline.length,
                    stride,
                    cmd->relay_inline.dispatcher_type);
                const auto payload_offset = offset + sizeof(CQPrefetchCmd);
                if (payload_offset <= entry.size_bytes) {
                    const uint32_t payload_remaining = entry.size_bytes - payload_offset;
                    const uint32_t payload_size = std::min<uint32_t>(cmd->relay_inline.length, payload_remaining);
                    if (payload_size != 0) {
                        line += fmt::format(
                            " dispatch0=[{}]", summarize_first_dispatch_cmd(sequence + payload_offset, payload_size));
                    }
                }
                break;
            }
            case CQ_PREFETCH_CMD_EXEC_BUF:
                line += fmt::format(
                    " base=0x{:x} log_page={} pages={}",
                    cmd->exec_buf.base_addr,
                    cmd->exec_buf.log_page_size,
                    cmd->exec_buf.pages);
                break;
            case CQ_PREFETCH_CMD_DEBUG:
                stride = cmd->debug.stride;
                line += fmt::format(" key={} size={} stride=0x{:x}", cmd->debug.key, cmd->debug.size, stride);
                break;
            case CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER:
                line += fmt::format(
                    " base=0x{:x} len=0x{:x} wp_update=0x{:x} flags=0x{:x}",
                    cmd->paged_to_ringbuffer.base_addr,
                    cmd->paged_to_ringbuffer.length,
                    cmd->paged_to_ringbuffer.wp_offset_update,
                    cmd->paged_to_ringbuffer.flags);
                break;
            case CQ_PREFETCH_CMD_SET_RINGBUFFER_OFFSET:
                line += fmt::format(
                    " offset=0x{:x} update_wp={}",
                    cmd->set_ringbuffer_offset.offset,
                    cmd->set_ringbuffer_offset.update_wp);
                break;
            case CQ_PREFETCH_CMD_RELAY_RINGBUFFER:
                stride = cmd->relay_ringbuffer.stride;
                line += fmt::format(" count={} stride=0x{:x}", cmd->relay_ringbuffer.count, stride);
                break;
            case CQ_PREFETCH_CMD_STALL:
            case CQ_PREFETCH_CMD_TERMINATE:
            case CQ_PREFETCH_CMD_ILLEGAL:
            default: break;
        }

        if (stride == 0 || stride > entry.size_bytes - offset) {
            line += fmt::format(" invalid_stride=0x{:x} remaining=0x{:x}", stride, entry.size_bytes - offset);
            lines.push_back(std::move(line));
            break;
        }

        lines.push_back(std::move(line));
        offset += stride;
    }

    if (offset < entry.size_bytes) {
        lines.push_back(fmt::format("remaining=0x{:x}", entry.size_bytes - offset));
    }

    return lines;
}

bool wrap_ge(uint32_t a, uint32_t b) {
    // SIgned Diff uses 2's Complement to handle wrap
    // Works as long as a and b are 2^31 apart
    int32_t diff = a - b;
    return diff >= 0;
}

// Cancellable timeout wrapper: invokes on_timeout() before throwing and waits for task to exit
// Please note that the FuncBody is going to loop until the FuncWait returns false.
template <typename FuncBody, typename FuncWait, typename OnTimeout>
void loop_and_wait_with_timeout(
    const FuncBody& func_body,
    const FuncWait& wait_condition,
    const OnTimeout& on_timeout,
    std::chrono::duration<float> timeout_duration) {
    if (timeout_duration.count() > 0.0f) {
        auto start_time = std::chrono::high_resolution_clock::now();

        do {
            func_body();
            if (wait_condition()) {
                // If somehow finished up the operation, we don't need to yield
                std::this_thread::yield();
            }

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(current_time - start_time).count();

            if (elapsed >= timeout_duration.count()) {
                on_timeout();
                break;
            }
        } while (wait_condition());
    } else {
        do {
            func_body();
        } while (wait_condition());
    }
}

std::optional<uint32_t> read_l1_u32(
    IDevice* device, const tt_cxy_pair& logical_core, uint32_t addr, CoreType core_type) {
    if (device == nullptr) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> value(1, 0);
    tt_metal::detail::ReadFromDeviceL1(
        device, CoreCoord(logical_core.x, logical_core.y), addr, sizeof(std::uint32_t), value, core_type);
    return value[0];
}
}  // namespace

SystemMemoryManager::SystemMemoryManager(ChipId device_id, uint8_t num_hw_cqs) : device_id(device_id) {
    this->completion_byte_addrs.resize(num_hw_cqs);
    this->prefetcher_cores.resize(num_hw_cqs);
    this->prefetch_q_writers.reserve(num_hw_cqs);
    this->completion_q_writers.reserve(num_hw_cqs);
    this->prefetch_q_dev_ptrs.resize(num_hw_cqs);
    this->prefetch_q_dev_fences.resize(num_hw_cqs);
    this->last_issue_push_start_addrs.resize(num_hw_cqs, 0);
    this->last_issue_push_sizes.resize(num_hw_cqs, 0);
    this->recent_fetch_traces.resize(num_hw_cqs);
    this->total_fetch_trace_counts.resize(num_hw_cqs, 0);

    // Split hugepage into however many pieces as there are CQs
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    ChipId mmio_device_id = cluster.get_associated_mmio_device(device_id);
    uint16_t channel = cluster.get_assigned_channel_for_device(device_id);
    char* hugepage_start = (char*)cluster.host_dma_address(0, mmio_device_id, channel);
    const uint32_t channel_share_offset = get_per_device_host_channel_offset(device_id, channel);
    hugepage_start += channel_share_offset;
    this->cq_sysmem_start = hugepage_start;

    // TODO(abhullar): Remove env var and expose sizing at the API level
    char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
    if (cq_size_override_env != nullptr) {
        uint32_t cq_size_override = std::stoi(std::string(cq_size_override_env));
        this->cq_size = cq_size_override;
    } else {
        uint32_t logical_channel_size = get_per_device_host_channel_size(device_id, channel);
        this->cq_size = logical_channel_size / num_hw_cqs;
        this->cq_size -= this->cq_size % MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    }
    uint32_t host_channel_stride = static_cast<uint32_t>(cluster.get_host_channel_stride(mmio_device_id, channel));
    TT_ASSERT(host_channel_stride != 0, "Host channel stride must be non-zero.");
    this->channel_offset = host_channel_stride * get_umd_channel(channel) + channel_share_offset;

    CoreType core_type = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    uint32_t completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    uint32_t prefetch_q_base = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::UNRESERVED);
    uint32_t cq_start =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair prefetcher_core =
            tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().prefetcher_core(
                device_id, channel, cq_id);
        auto prefetcher_virtual =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                prefetcher_core.chip, CoreCoord(prefetcher_core.x, prefetcher_core.y), core_type);
        this->prefetcher_cores[cq_id] = tt_cxy_pair(prefetcher_core.chip, prefetcher_virtual.x, prefetcher_virtual.y);
        this->prefetch_q_writers.emplace_back(
            tt::tt_metal::MetalContext::instance().get_cluster().get_static_tlb_writer(this->prefetcher_cores[cq_id]));

        tt_cxy_pair completion_queue_writer_core =
            tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().completion_queue_writer_core(
                device_id, channel, cq_id);
        auto completion_queue_writer_virtual =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                completion_queue_writer_core.chip,
                CoreCoord(completion_queue_writer_core.x, completion_queue_writer_core.y),
                core_type);

        const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = tt::tt_metal::MetalContext::instance()
                                                                                 .get_cluster()
                                                                                 .get_tlb_data(tt_cxy_pair(
                                                                                     completion_queue_writer_core.chip,
                                                                                     completion_queue_writer_virtual.x,
                                                                                     completion_queue_writer_virtual.y))
                                                                                 .value();
        auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;

        this->completion_byte_addrs[cq_id] = completion_q_rd_ptr % completion_tlb_size;
        this->completion_q_writers.emplace_back(
            tt::tt_metal::MetalContext::instance().get_cluster().get_static_tlb_writer(tt_cxy_pair(
                completion_queue_writer_core.chip,
                completion_queue_writer_virtual.x,
                completion_queue_writer_virtual.y)));

        this->cq_interfaces.push_back(SystemMemoryCQInterface(device_id, channel, cq_id, this->cq_size, cq_start));
        const auto& cq_interface = this->cq_interfaces.back();
        // Prefetch queue acts as the sync mechanism to ensure that issue queue has space to write, so issue queue
        // must be as large as the max amount of space the prefetch queue can specify Plus 1 to handle wrapping Plus
        // 1 to allow us to start writing to issue queue before we reserve space in the prefetch queue
        TT_FATAL(
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() *
                    (MetalContext::instance().dispatch_mem_map().prefetch_q_entries() + 2) <=
                this->get_issue_queue_size(cq_id),
            "Issue queue for cq_id {} has size of {} which is too small",
            cq_id,
            this->get_issue_queue_size(cq_id));
        log_info(
            tt::LogMetal,
            "DEBUG CQ-SETUP: device {} cq {} mmio={} channel={} umd={} share_offset=0x{:x} channel_offset=0x{:x} "
            "cq_size=0x{:x} issue_base=0x{:x} issue_limit=0x{:x} completion_base=0x{:x} completion_limit=0x{:x} "
            "prefetch_logical={} prefetch_virtual={} cq_writer_logical={} cq_writer_virtual={} "
            "completion_l1_off=0x{:x}",
            device_id,
            cq_id,
            mmio_device_id,
            channel,
            get_umd_channel(channel),
            channel_share_offset,
            this->channel_offset,
            this->cq_size,
            cq_start + cq_interface.offset,
            cq_interface.issue_fifo_limit << 4,
            cq_interface.issue_fifo_limit << 4,
            cq_interface.completion_fifo_limit << 4,
            prefetcher_core.str(),
            tt_cxy_pair(prefetcher_core.chip, prefetcher_virtual.x, prefetcher_virtual.y).str(),
            completion_queue_writer_core.str(),
            tt_cxy_pair(
                completion_queue_writer_core.chip, completion_queue_writer_virtual.x, completion_queue_writer_virtual.y)
                .str(),
            this->completion_byte_addrs[cq_id]);
        this->cq_to_event.push_back(0);
        this->cq_to_last_completed_event.push_back(0);
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        this->prefetch_q_dev_fences[cq_id] =
            prefetch_q_base + MetalContext::instance().dispatch_mem_map().prefetch_q_entries() *
                                  sizeof(DispatchSettings::prefetch_q_entry_type);
    }
    std::vector<std::mutex> temp_mutexes(num_hw_cqs);
    cq_to_event_locks.swap(temp_mutexes);
    std::vector<std::mutex> temp_fetch_trace_mutexes(num_hw_cqs);
    recent_fetch_trace_locks.swap(temp_fetch_trace_mutexes);
}

uint32_t SystemMemoryManager::get_next_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t next_event = ++this->cq_to_event[cq_id];  // Event ids start at 1

    cq_to_event_locks[cq_id].unlock();
    return next_event;
}

// Get last issued event to Command Queue
uint32_t SystemMemoryManager::get_last_event(const uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(cq_to_event_locks[cq_id]);
    return this->cq_to_event[cq_id];
}

void SystemMemoryManager::set_current_and_last_completed_event(
    const uint8_t cq_id, const uint32_t current_event_id, const uint32_t last_completed_event_id) {
    cq_to_event_locks[cq_id].lock();

    this->cq_to_event[cq_id] = current_event_id;
    this->cq_to_last_completed_event[cq_id] = last_completed_event_id;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::reset_event_id(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] = 0;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::increment_event_id(const uint8_t cq_id, const uint32_t val) {
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] += val;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
    TT_ASSERT(
        wrap_ge(event_id, this->cq_to_last_completed_event[cq_id]),
        "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
        "completed event is {}, manager {}",
        event_id,
        this->cq_to_last_completed_event[cq_id],
        fmt::ptr(this));
    cq_to_event_locks[cq_id].lock();

    this->cq_to_last_completed_event[cq_id] = event_id;
    cq_to_event_locks[cq_id].unlock();
}

uint32_t SystemMemoryManager::get_current_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t current_event = this->cq_to_event[cq_id];
    cq_to_event_locks[cq_id].unlock();
    return current_event;
}

uint32_t SystemMemoryManager::get_last_completed_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t last_completed_event = this->cq_to_last_completed_event[cq_id];
    cq_to_event_locks[cq_id].unlock();
    return last_completed_event;
}

void SystemMemoryManager::reset(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
    cq_interface.issue_fifo_wr_toggle = false;
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = false;
}

void SystemMemoryManager::set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_size = (issue_queue_size >> 4);
    cq_interface.issue_fifo_limit = (cq_interface.cq_start + cq_interface.offset + issue_queue_size) >> 4;
}

void SystemMemoryManager::set_bypass_mode(const bool enable, const bool clear) {
    this->bypass_enable = enable;
    if (clear) {
        this->bypass_buffer.clear();
        this->bypass_buffer_write_offset = 0;
    }
}

bool SystemMemoryManager::get_bypass_mode() const { return this->bypass_enable; }

std::vector<uint32_t>& SystemMemoryManager::get_bypass_data() { return this->bypass_buffer; }

uint32_t SystemMemoryManager::get_issue_queue_size(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_limit(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_size(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_limit(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_write_ptr(const uint8_t cq_id) const {
    if (this->bypass_enable) {
        return this->bypass_buffer_write_offset;
    } else {
        return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
    }
}

uint32_t SystemMemoryManager::get_completion_queue_read_ptr(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
}

void* SystemMemoryManager::get_completion_queue_ptr(uint8_t cq_id) const {
    // The completion queue follows issue queue in contiguous memory
    // get_issue_queue_limit() returns absolute device address where the issue queue ends.
    // We subtract channel_offset (absolute device channel base) to get relative offset,
    // then add it to cq_sysmem_start (host channel base) to get host virtual address
    return (void*)(this->cq_sysmem_start + (this->get_issue_queue_limit(cq_id) - this->channel_offset));
}

uint32_t SystemMemoryManager::get_completion_queue_read_toggle(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
}

uint32_t SystemMemoryManager::get_cq_size() const { return this->cq_size; }

ChipId SystemMemoryManager::get_device_id() const { return this->device_id; }

std::vector<SystemMemoryCQInterface>& SystemMemoryManager::get_cq_interfaces() { return this->cq_interfaces; }

void* SystemMemoryManager::issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
    TT_ASSERT(cmd_size_B > 0, "Command size must be greater than 0");
    if (this->bypass_enable) {
        uint32_t curr_size = this->bypass_buffer.size();
        uint32_t new_size = curr_size + (cmd_size_B / sizeof(uint32_t));
        this->bypass_buffer.resize(new_size);
        return (void*)((char*)this->bypass_buffer.data() + this->bypass_buffer_write_offset);
    }

    uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

    const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
    if (issue_q_write_ptr +
            align(
                cmd_size_B,
                tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST)) >
        command_issue_limit) {
        this->wrap_issue_queue_wr_ptr(cq_id);
        issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
    }

    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
    // queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on
    // https://github.com/tenstorrent/tt-metal/issues/4757
    void* issue_q_region = this->cq_sysmem_start + (issue_q_write_ptr - this->channel_offset);

    return issue_q_region;
}

void SystemMemoryManager::cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
    // queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on
    // https://github.com/tenstorrent/tt-metal/issues/4757
    void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

    if (this->bypass_enable) {
        std::copy((uint8_t*)data, (uint8_t*)data + size_in_bytes, (uint8_t*)this->bypass_buffer.data() + write_ptr);
    } else {
        memcpy_to_device(user_scratchspace, data, size_in_bytes);
    }
}

// TODO: RENAME issue_queue_stride ?
void SystemMemoryManager::issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
    TT_ASSERT(push_size_B > 0, "Push size must be greater than 0");
    if (this->bypass_enable) {
        this->bypass_buffer_write_offset += push_size_B;
        return;
    }

    // All data needs to be PCIE_ALIGNMENT aligned
    uint32_t push_size_16B =
        align(
            push_size_B, tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST)) >>
        4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    uint32_t issue_q_wr_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
    const uint32_t issue_push_start_addr = cq_interface.issue_fifo_wr_ptr << 4;
    this->last_issue_push_start_addrs[cq_id] = issue_push_start_addr;
    this->last_issue_push_sizes[cq_id] = push_size_B;

    if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;            // Flip the toggle
    } else {
        cq_interface.issue_fifo_wr_ptr += push_size_16B;
    }

    // Also store this data in hugepages, so if a hang happens we can see what was written by host.
    ChipId mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->device_id);
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(this->device_id);
    tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
        &cq_interface.issue_fifo_wr_ptr,
        sizeof(uint32_t),
        issue_q_wr_ptr + get_absolute_cq_offset(this->device_id, channel, cq_id, this->cq_size),
        mmio_device_id,
        channel);
}

void SystemMemoryManager::send_completion_queue_read_ptr(const uint8_t cq_id) const {
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    uint32_t read_ptr_and_toggle = cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
    this->completion_q_writers[cq_id].write(this->completion_byte_addrs[cq_id], read_ptr_and_toggle);

    // Also store this data in hugepages in case we hang and can't get it from the device.
    ChipId mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->device_id);
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(this->device_id);
    uint32_t completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_RD);
    tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
        &read_ptr_and_toggle,
        sizeof(uint32_t),
        completion_q_rd_ptr + get_absolute_cq_offset(this->device_id, channel, cq_id, this->cq_size),
        mmio_device_id,
        channel);
}

void SystemMemoryManager::fetch_queue_reserve_back(const uint8_t cq_id) {
    if (this->bypass_enable) {
        return;
    }

    const uint32_t prefetch_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::PREFETCH_Q_RD);

    // Helper to wait for fetch queue space, if needed
    uint32_t fence;
    auto wait_for_fetch_q_space = [&]() {
        if (this->prefetch_q_dev_ptrs[cq_id] != this->prefetch_q_dev_fences[cq_id]) {
            return;
        }
        ZoneScopedN("wait_for_fetch_q_space");

        // Body of the operation
        auto fetch_operation_body = [&]() {
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                &fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], prefetch_q_rd_ptr);
            this->prefetch_q_dev_fences[cq_id] = fence;
        };

        // Condition to check if should continue waiting
        auto fetch_wait_condition = [&]() -> bool {
            return this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id];
        };

        // Handler for timeout
        auto fetch_on_timeout = []() {
            MetalContext::instance().on_dispatch_timeout_detected();
            TT_THROW("TIMEOUT: device timeout in fetch queue wait, potential hang detected");
        };

        auto timeout_duration =
            tt::tt_metal::MetalContext::instance().rtoptions().get_timeout_duration_for_operations();

        loop_and_wait_with_timeout(fetch_operation_body, fetch_wait_condition, fetch_on_timeout, timeout_duration);
    };

    wait_for_fetch_q_space();
    // Wrap FetchQ if possible
    uint32_t prefetch_q_base = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::UNRESERVED);
    uint32_t prefetch_q_limit = prefetch_q_base + (MetalContext::instance().dispatch_mem_map().prefetch_q_entries() *
                                                   sizeof(DispatchSettings::prefetch_q_entry_type));
    if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        wait_for_fetch_q_space();
    }
}

uint32_t SystemMemoryManager::completion_queue_wait_front(
    const uint8_t cq_id, std::atomic<bool>& exit_condition) const {
    uint32_t write_ptr_and_toggle;
    uint32_t write_ptr;
    uint32_t write_toggle;
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    // Body of the operation to be timed out
    uint64_t cq_wait_iter_count = 0;
    size_t cq_wait_timeout_log_index = 0;
    auto cq_wait_start = std::chrono::steady_clock::now();
    auto wait_operation_body = [this,
                                cq_id,
                                &exit_condition,
                                &write_ptr_and_toggle,
                                &write_ptr,
                                &write_toggle,
                                &cq_wait_iter_count,
                                &cq_wait_timeout_log_index,
                                &cq_wait_start,
                                &cq_interface]() -> uint32_t {
        write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
        write_ptr = write_ptr_and_toggle & 0x7fffffff;
        write_toggle = write_ptr_and_toggle >> 31;

        if (exit_condition.load()) {
            return write_ptr_and_toggle;
        }

        // Escalating diagnostic: emit only a few timeout milestones while waiting.
        cq_wait_iter_count++;
        if ((cq_wait_iter_count & 0xFFFFF) == 0) {  // Check wall clock every ~1M iterations
            static constexpr uint64_t kCqWaitLogThresholds[] = {10, 30, 60, 300};
            static constexpr size_t kNumCqWaitLogThresholds =
                sizeof(kCqWaitLogThresholds) / sizeof(kCqWaitLogThresholds[0]);
            auto now = std::chrono::steady_clock::now();
            uint64_t elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - cq_wait_start).count();
            if (cq_wait_timeout_log_index < kNumCqWaitLogThresholds &&
                elapsed_s >= kCqWaitLogThresholds[cq_wait_timeout_log_index]) {
                log_info(
                    tt::LogMetal,
                    "DEBUG: CQ wait: device {} cq {} stuck for {}s: wr_ptr=0x{:x} wr_toggle={} "
                    "rd_ptr=0x{:x} rd_toggle={} (iters={})",
                    this->device_id,
                    cq_id,
                    elapsed_s,
                    write_ptr,
                    write_toggle,
                    cq_interface.completion_fifo_rd_ptr,
                    cq_interface.completion_fifo_rd_toggle,
                    cq_wait_iter_count);
                cq_wait_timeout_log_index++;

                // One-time extended diagnostic on first detection
                static bool cq_diag_dumped = false;
                if (!cq_diag_dumped && elapsed_s >= 10) {
                    cq_diag_dumped = true;
                    try {
                        const auto& cluster = MetalContext::instance().get_cluster();
                        ChipId mmio_id = cluster.get_associated_mmio_device(this->device_id);
                        uint16_t my_channel = cluster.get_assigned_channel_for_device(this->device_id);
                        log_info(
                            tt::LogMetal,
                            "DEBUG: CQ diag: stuck device {} uses mmio={} channel={}",
                            this->device_id,
                            mmio_id,
                            my_channel);

                        // Dump ALL devices' CQ wr_ptr to see if this is a global or local issue
                        for (const auto& controlled_id : cluster.get_devices_controlled_by_mmio_device(mmio_id)) {
                            try {
                                auto other_issue_rd = get_cq_issue_rd_ptr<true>(controlled_id, cq_id, this->cq_size);
                                auto other_wr = get_cq_completion_wr_ptr<true>(controlled_id, cq_id, this->cq_size);
                                uint32_t other_wr_ptr = other_wr & 0x7fffffff;
                                uint32_t other_toggle = other_wr >> 31;
                                auto other_issue_wr = get_cq_issue_wr_ptr<true>(controlled_id, cq_id, this->cq_size);
                                auto other_completion_rd =
                                    get_cq_completion_rd_ptr<true>(controlled_id, cq_id, this->cq_size);
                                log_info(
                                    tt::LogMetal,
                                    "DEBUG: CQ diag: device {} cq {} issue_rd=0x{:x} issue_wr=0x{:x} "
                                    "completion_rd=0x{:x} completion_wr=0x{:x} toggle={}",
                                    controlled_id,
                                    cq_id,
                                    other_issue_rd & 0x7fffffff,
                                    other_issue_wr & 0x7fffffff,
                                    other_completion_rd & 0x7fffffff,
                                    other_wr_ptr,
                                    other_toggle);
                            } catch (...) {
                            }
                        }

                        // Dump device-side queue state for the stuck device to localize whether the stall is on
                        // host issue consumption, MMIO-side service, or remote D-kernel execution.
                        try {
                            auto& dcm = MetalContext::instance().get_dispatch_core_manager();
                            const auto dispatch_core_type = dcm.get_dispatch_core_type();
                            const auto& dispatch_mem_map = MetalContext::instance().dispatch_mem_map();
                            auto* device_manager = MetalContext::instance().device_manager().get();

                            auto log_core_regs = [&](const char* label,
                                                     const tt_cxy_pair& logical_core,
                                                     CoreType core_type,
                                                     std::initializer_list<std::pair<const char*, uint32_t>> regs) {
                                auto* owner_dev = device_manager->get_active_device(logical_core.chip);
                                if (owner_dev == nullptr) {
                                    return;
                                }
                                std::string reg_dump;
                                for (const auto& [name, addr] : regs) {
                                    if (!reg_dump.empty()) {
                                        reg_dump += " ";
                                    }
                                    try {
                                        auto value = read_l1_u32(owner_dev, logical_core, addr, core_type);
                                        reg_dump += fmt::format("{}=0x{:x}", name, value.value_or(0));
                                    } catch (const std::exception& e) {
                                        reg_dump += fmt::format("{}=<err:{}>", name, e.what());
                                    }
                                }
                                const auto virtual_core =
                                    cluster.get_virtual_coordinate_from_logical_coordinates(logical_core, core_type);
                                log_info(
                                    tt::LogMetal,
                                    "DEBUG CQ-STALL: {} chip={} logical={} virtual={} regs:[{}]",
                                    label,
                                    logical_core.chip,
                                    logical_core.str(),
                                    virtual_core.str(),
                                    reg_dump);
                            };

                            log_info(
                                tt::LogMetal,
                                "DEBUG CQ-STALL: device {} cq {} host_window issue_base=0x{:x} issue_limit=0x{:x} "
                                "completion_base=0x{:x} completion_limit=0x{:x} channel_offset=0x{:x} cq_size=0x{:x}",
                                this->device_id,
                                cq_id,
                                cq_interface.cq_start + cq_interface.offset,
                                cq_interface.issue_fifo_limit << 4,
                                cq_interface.issue_fifo_limit << 4,
                                cq_interface.completion_fifo_limit << 4,
                                this->channel_offset,
                                this->cq_size);

                            const auto prefetch_q_rd_addr = dispatch_mem_map.get_device_command_queue_addr(
                                CommandQueueDeviceAddrType::PREFETCH_Q_RD);
                            const auto prefetch_q_pcie_rd_addr = dispatch_mem_map.get_device_command_queue_addr(
                                CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD);
                            const auto mmio_prefetch_core = dcm.prefetcher_core(this->device_id, my_channel, cq_id);
                            auto* mmio_prefetch_owner = device_manager->get_active_device(mmio_prefetch_core.chip);
                            const auto mmio_prefetch_q_rd = read_l1_u32(
                                mmio_prefetch_owner, mmio_prefetch_core, prefetch_q_rd_addr, dispatch_core_type);
                            const auto mmio_prefetch_q_pcie_rd = read_l1_u32(
                                mmio_prefetch_owner, mmio_prefetch_core, prefetch_q_pcie_rd_addr, dispatch_core_type);

                            log_core_regs(
                                "mmio_prefetch_h",
                                mmio_prefetch_core,
                                dispatch_core_type,
                                {
                                    {"prefetch_q_rd", prefetch_q_rd_addr},
                                    {"prefetch_q_pcie_rd", prefetch_q_pcie_rd_addr},
                                });
                            log_core_regs(
                                "mmio_completion_writer",
                                dcm.completion_queue_writer_core(this->device_id, my_channel, cq_id),
                                dispatch_core_type,
                                {
                                    {"completion_q_wr",
                                     dispatch_mem_map.get_device_command_queue_addr(
                                         CommandQueueDeviceAddrType::COMPLETION_Q_WR)},
                                    {"completion_q_rd",
                                     dispatch_mem_map.get_device_command_queue_addr(
                                         CommandQueueDeviceAddrType::COMPLETION_Q_RD)},
                                    {"last_event",
                                     dispatch_mem_map.get_device_command_queue_addr(
                                         cq_id == 0 ? CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT
                                                    : CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT)},
                                });

                            for (const auto& controlled_id : cluster.get_devices_controlled_by_mmio_device(mmio_id)) {
                                auto* controlled_dev = device_manager->get_active_device(controlled_id);
                                if (controlled_dev == nullptr) {
                                    continue;
                                }

                                uint16_t controlled_channel = cluster.get_assigned_channel_for_device(controlled_id);
                                auto service_prefetch_core =
                                    dcm.prefetcher_core(controlled_id, controlled_channel, cq_id);
                                auto service_completion_core =
                                    dcm.completion_queue_writer_core(controlled_id, controlled_channel, cq_id);
                                auto service_prefetch_owner =
                                    device_manager->get_active_device(service_prefetch_core.chip);
                                auto service_pcie_rd = read_l1_u32(
                                    service_prefetch_owner,
                                    service_prefetch_core,
                                    prefetch_q_pcie_rd_addr,
                                    dispatch_core_type);
                                auto service_prefetch_rd = read_l1_u32(
                                    service_prefetch_owner,
                                    service_prefetch_core,
                                    prefetch_q_rd_addr,
                                    dispatch_core_type);
                                auto service_completion_owner =
                                    device_manager->get_active_device(service_completion_core.chip);
                                auto service_completion_wr = read_l1_u32(
                                    service_completion_owner,
                                    service_completion_core,
                                    dispatch_mem_map.get_device_command_queue_addr(
                                        CommandQueueDeviceAddrType::COMPLETION_Q_WR),
                                    dispatch_core_type);
                                auto service_completion_rd = read_l1_u32(
                                    service_completion_owner,
                                    service_completion_core,
                                    dispatch_mem_map.get_device_command_queue_addr(
                                        CommandQueueDeviceAddrType::COMPLETION_Q_RD),
                                    dispatch_core_type);

                                auto& controlled_sysmem = controlled_dev->sysmem_manager();
                                const auto& controlled_cq = controlled_sysmem.get_cq_interfaces().at(cq_id);
                                const uint32_t issue_base_bytes = controlled_cq.cq_start + controlled_cq.offset;
                                const uint32_t issue_wr_bytes = controlled_sysmem.get_issue_queue_write_ptr(cq_id);
                                const uint32_t completion_base_bytes = controlled_sysmem.get_issue_queue_limit(cq_id);
                                log_info(
                                    tt::LogMetal,
                                    "DEBUG CQ-STALL: service_device {} channel={} issue_base=0x{:x} issue_wr=0x{:x} "
                                    "pcie_rd=0x{:x} prefetch_q_rd=0x{:x} completion_base=0x{:x} "
                                    "completion_wr=0x{:x} completion_rd=0x{:x}",
                                    controlled_id,
                                    controlled_channel,
                                    issue_base_bytes,
                                    issue_wr_bytes,
                                    service_pcie_rd.value_or(0),
                                    service_prefetch_rd.value_or(0),
                                    completion_base_bytes,
                                    service_completion_wr.value_or(0),
                                    service_completion_rd.value_or(0));
                            }

                            if (mmio_prefetch_q_pcie_rd.has_value()) {
                                std::vector<FetchTraceEntry> traces;
                                uint64_t trace_base_index = 0;
                                {
                                    std::lock_guard<std::mutex> lock(this->recent_fetch_trace_locks[cq_id]);
                                    const auto& stored_traces = this->recent_fetch_traces[cq_id];
                                    traces.assign(stored_traces.begin(), stored_traces.end());
                                    trace_base_index = this->total_fetch_trace_counts[cq_id] - traces.size();
                                }

                                const uint32_t pcie_rd = mmio_prefetch_q_pcie_rd.value();
                                const uint32_t host_issue_wr = this->get_issue_queue_write_ptr(cq_id);
                                log_info(
                                    tt::LogMetal,
                                    "DEBUG CQ-STALL: host_fetch_progress pcie_rd=0x{:x} issue_wr=0x{:x} "
                                    "prefetch_q_rd=0x{:x} traced_fetches={}",
                                    pcie_rd,
                                    host_issue_wr,
                                    mmio_prefetch_q_rd.value_or(0),
                                    traces.size());

                                if (!traces.empty()) {
                                    const uint32_t oldest_trace_end =
                                        traces.front().start_addr + traces.front().size_bytes;
                                    const uint32_t newest_trace_end =
                                        traces.back().start_addr + traces.back().size_bytes;
                                    log_info(
                                        tt::LogMetal,
                                        "DEBUG CQ-STALL: retained_fetches global_range=[{}, {}) "
                                        "oldest=[0x{:x},0x{:x}) newest=[0x{:x},0x{:x})",
                                        trace_base_index,
                                        trace_base_index + traces.size(),
                                        traces.front().start_addr,
                                        oldest_trace_end,
                                        traces.back().start_addr,
                                        newest_trace_end);

                                    std::optional<size_t> exact_match_idx;
                                    std::optional<size_t> nearest_completed_idx;
                                    for (size_t idx = 0; idx < traces.size(); ++idx) {
                                        const uint32_t trace_end = traces[idx].start_addr + traces[idx].size_bytes;
                                        if (trace_end == pcie_rd) {
                                            exact_match_idx = idx;
                                            break;
                                        }
                                        if (trace_end < pcie_rd) {
                                            nearest_completed_idx = idx;
                                        }
                                    }

                                    const auto focus_idx =
                                        exact_match_idx.value_or(nearest_completed_idx.value_or(traces.size() - 1));
                                    const size_t begin_idx = (focus_idx > 2) ? focus_idx - 2 : 0;
                                    const size_t end_idx = std::min(traces.size(), focus_idx + 3);
                                    log_info(
                                        tt::LogMetal,
                                        "DEBUG CQ-STALL: retained_fetch_window focus={} exact_match={} "
                                        "range=[{}, {}) pcie_rd_in_range={}",
                                        trace_base_index + focus_idx,
                                        exact_match_idx.has_value(),
                                        trace_base_index + begin_idx,
                                        trace_base_index + end_idx,
                                        pcie_rd >= oldest_trace_end && pcie_rd <= newest_trace_end);
                                    if (!exact_match_idx.has_value()) {
                                        log_info(
                                            tt::LogMetal,
                                            "DEBUG CQ-STALL: no retained fetch ends exactly at pcie_rd=0x{:x}; "
                                            "nearest_completed_end=0x{:x}",
                                            pcie_rd,
                                            traces[focus_idx].start_addr + traces[focus_idx].size_bytes);
                                    }
                                    for (size_t idx = begin_idx; idx < end_idx; ++idx) {
                                        const auto& trace = traces[idx];
                                        log_info(
                                            tt::LogMetal,
                                            "DEBUG CQ-STALL: fetch[{}] start=0x{:x} end=0x{:x} size=0x{:x} stall={}",
                                            trace_base_index + idx,
                                            trace.start_addr,
                                            trace.start_addr + trace.size_bytes,
                                            trace.size_bytes,
                                            trace.stall_prefetcher);
                                        for (const auto& line : summarize_fetch_sequence(
                                                 this->cq_sysmem_start, this->channel_offset, trace)) {
                                            log_info(
                                                tt::LogMetal,
                                                "DEBUG CQ-STALL: fetch[{}] {}",
                                                trace_base_index + idx,
                                                line);
                                        }
                                    }
                                }

                                if (host_issue_wr > pcie_rd) {
                                    const FetchTraceEntry next_fetch_window{
                                        pcie_rd,
                                        std::min<uint32_t>(host_issue_wr - pcie_rd, kMaxNextFetchBytesToLog),
                                        false};
                                    log_info(
                                        tt::LogMetal,
                                        "DEBUG CQ-STALL: next_from_pcie_rd start=0x{:x} size=0x{:x}",
                                        next_fetch_window.start_addr,
                                        next_fetch_window.size_bytes);
                                    for (const auto& line : summarize_fetch_sequence(
                                             this->cq_sysmem_start, this->channel_offset, next_fetch_window)) {
                                        log_info(tt::LogMetal, "DEBUG CQ-STALL: next_from_pcie_rd {}", line);
                                    }
                                }
                            }
                        } catch (const std::exception& e) {
                            log_info(tt::LogMetal, "DEBUG CQ-STALL: device-side queue diag failed: {}", e.what());
                        }

                        // Read MMIO fabric router connection semaphores at stall time
                        try {
                            const auto& cp = MetalContext::instance().get_control_plane();
                            const auto& fc = cp.get_fabric_context();
                            const auto& bc = fc.get_builder_context();
                            const auto& router_config = bc.get_fabric_router_config();
                            for (const auto& dev :
                                 MetalContext::instance().device_manager()->get_all_active_devices()) {
                                if (!cluster.mmio_chip_ids().count(dev->id())) {
                                    continue;
                                }
                                const auto& soc_desc = cluster.get_soc_desc(dev->id());
                                const auto fn = cp.get_fabric_node_id_from_physical_chip_id(dev->id());
                                for (const auto& [chan, dir] : cp.get_active_fabric_eth_channels(fn)) {
                                    auto lc = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
                                    std::string sem_str;
                                    for (uint32_t s = 0; s < router_config.num_used_sender_channels; s++) {
                                        auto addr = router_config.sender_channels_connection_semaphore_address[s];
                                        if (addr == 0) {
                                            continue;
                                        }
                                        std::vector<std::uint32_t> sem(1, 0);
                                        tt_metal::detail::ReadFromDeviceL1(dev, lc, addr, 4, sem, CoreType::ETH);
                                        sem_str += fmt::format(" s{}={}", s, sem[0]);
                                    }
                                    // Also read flow control (credits available)
                                    std::string fc_str;
                                    for (uint32_t s = 0; s < router_config.num_used_sender_channels; s++) {
                                        auto addr =
                                            router_config.sender_channels_local_flow_control_semaphore_address[s];
                                        if (addr == 0) {
                                            continue;
                                        }
                                        std::vector<std::uint32_t> sem(1, 0);
                                        tt_metal::detail::ReadFromDeviceL1(dev, lc, addr, 4, sem, CoreType::ETH);
                                        fc_str += fmt::format(" s{}={}", s, sem[0]);
                                    }
                                    // Blackhole remote dispatch links use credit counters instead of direct reg-writes.
                                    // Dump both sender-side received credits and receiver-side emitted credits so we
                                    // can tell whether completion traffic ever reaches the MMIO endpoint.
                                    std::string tx_ack_str;
                                    std::string tx_comp_str;
                                    std::string rx_ack_str;
                                    std::string rx_comp_str;
                                    for (uint32_t s = 0; s < router_config.num_used_sender_channels; s++) {
                                        if (router_config.to_sender_channel_remote_ack_counters_base_addr != 0) {
                                            std::vector<std::uint32_t> counter(1, 0);
                                            tt_metal::detail::ReadFromDeviceL1(
                                                dev,
                                                lc,
                                                router_config.to_sender_channel_remote_ack_counters_base_addr +
                                                    (s * sizeof(std::uint32_t)),
                                                sizeof(std::uint32_t),
                                                counter,
                                                CoreType::ETH);
                                            tx_ack_str += fmt::format(" s{}={}", s, counter[0]);
                                        }
                                        if (router_config.to_sender_channel_remote_completion_counters_base_addr != 0) {
                                            std::vector<std::uint32_t> counter(1, 0);
                                            tt_metal::detail::ReadFromDeviceL1(
                                                dev,
                                                lc,
                                                router_config.to_sender_channel_remote_completion_counters_base_addr +
                                                    (s * sizeof(std::uint32_t)),
                                                sizeof(std::uint32_t),
                                                counter,
                                                CoreType::ETH);
                                            tx_comp_str += fmt::format(" s{}={}", s, counter[0]);
                                        }
                                        if (router_config.receiver_channel_remote_ack_counters_base_addr != 0) {
                                            std::vector<std::uint32_t> counter(1, 0);
                                            tt_metal::detail::ReadFromDeviceL1(
                                                dev,
                                                lc,
                                                router_config.receiver_channel_remote_ack_counters_base_addr +
                                                    (s * sizeof(std::uint32_t)),
                                                sizeof(std::uint32_t),
                                                counter,
                                                CoreType::ETH);
                                            rx_ack_str += fmt::format(" s{}={}", s, counter[0]);
                                        }
                                        if (router_config.receiver_channel_remote_completion_counters_base_addr != 0) {
                                            std::vector<std::uint32_t> counter(1, 0);
                                            tt_metal::detail::ReadFromDeviceL1(
                                                dev,
                                                lc,
                                                router_config.receiver_channel_remote_completion_counters_base_addr +
                                                    (s * sizeof(std::uint32_t)),
                                                sizeof(std::uint32_t),
                                                counter,
                                                CoreType::ETH);
                                            rx_comp_str += fmt::format(" s{}={}", s, counter[0]);
                                        }
                                    }
                                    std::vector<std::uint32_t> edm(1, 0);
                                    tt_metal::detail::ReadFromDeviceL1(
                                        dev, lc, router_config.edm_status_address, 4, edm, CoreType::ETH);
                                    log_info(
                                        tt::LogMetal,
                                        "DEBUG CQ-STALL: MMIO dev {} ch={} dir={} EDM=0x{:08x} conn:[{}] fc:[{}] "
                                        "tx_ack:[{}] tx_comp:[{}] rx_ack:[{}] rx_comp:[{}]",
                                        dev->id(),
                                        chan,
                                        (int)dir,
                                        edm[0],
                                        sem_str,
                                        fc_str,
                                        tx_ack_str,
                                        tx_comp_str,
                                        rx_ack_str,
                                        rx_comp_str);
                                }
                            }
                        } catch (...) {
                            log_info(tt::LogMetal, "DEBUG CQ-STALL: router diag failed");
                        }

                        // Dump hop counts for all remote devices
                        try {
                            const auto& cp = MetalContext::instance().get_control_plane();
                            const auto mmio_fn = cp.get_fabric_node_id_from_physical_chip_id(mmio_id);
                            for (const auto& controlled_id : cluster.get_devices_controlled_by_mmio_device(mmio_id)) {
                                if (controlled_id == mmio_id) {
                                    continue;
                                }
                                int hops = MetalContext::instance().get_lite_fabric_hop_count(controlled_id);
                                auto fn = cp.get_fabric_node_id_from_physical_chip_id(controlled_id);
                                auto fwd_dir = cp.get_forwarding_direction(mmio_fn, fn);
                                std::string return_route = "<none>";
                                int return_first_hop = -1;
                                auto return_channels = cp.get_forwarding_eth_chans_to_chip(fn, mmio_fn);
                                if (!return_channels.empty()) {
                                    const auto route = cp.get_fabric_route(fn, mmio_fn, return_channels.front());
                                    if (!route.empty()) {
                                        return_route.clear();
                                        for (const auto& [next_fn, chan_id] : route) {
                                            if (!return_route.empty()) {
                                                return_route += " -> ";
                                            }
                                            return_route += fmt::format(
                                                "M{}D{}/ch{}", next_fn.mesh_id.get(), next_fn.chip_id, chan_id);
                                            if (return_first_hop < 0 && next_fn != fn) {
                                                return_first_hop = cp.get_physical_chip_id_from_fabric_node_id(next_fn);
                                            }
                                        }
                                    }
                                }
                                log_info(
                                    tt::LogMetal,
                                    "DEBUG CQ-STALL: device {} (M{}D{}) lite_fabric_hops={} fwd_dir={} "
                                    "return_first_hop={} return_route=[{}]",
                                    controlled_id,
                                    fn.mesh_id.get(),
                                    fn.chip_id,
                                    hops,
                                    fwd_dir.has_value() ? static_cast<int>(*fwd_dir) : -1,
                                    return_first_hop,
                                    return_route);
                            }
                        } catch (...) {
                            log_info(tt::LogMetal, "DEBUG CQ-STALL: hop count diag failed");
                        }
                    } catch (const std::exception& e) {
                        log_info(tt::LogMetal, "DEBUG: CQ diag: diagnostic failed: {}", e.what());
                    }
                }
            }
        }

        return write_ptr_and_toggle;
    };

    // Condition to check if the operation should continue
    auto wait_condition = [&cq_interface, &write_ptr, &write_toggle]() -> bool {
        return cq_interface.completion_fifo_rd_ptr == write_ptr and
               cq_interface.completion_fifo_rd_toggle == write_toggle;
    };

    // Handler for the timeout
    auto on_timeout = [&exit_condition]() {
        exit_condition.store(true);

        MetalContext::instance().on_dispatch_timeout_detected();

        TT_THROW("TIMEOUT: device timeout, potential hang detected, the device is unrecoverable");
    };

    loop_and_wait_with_timeout(
        wait_operation_body,
        wait_condition,
        on_timeout,
        tt::tt_metal::MetalContext::instance().rtoptions().get_timeout_duration_for_operations());

    return write_ptr_and_toggle;
}

void SystemMemoryManager::wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
    if (this->bypass_enable) {
        return;
    }
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;
    cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
}

void SystemMemoryManager::wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
}

void SystemMemoryManager::completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
    uint32_t data_read_B = num_pages_read * DispatchSettings::TRANSFER_PAGE_SIZE;
    uint32_t data_read_16B = data_read_B >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr += data_read_16B;
    if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    // Notify dispatch core
    this->send_completion_queue_read_ptr(cq_id);
}

void SystemMemoryManager::fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id, bool stall_prefetcher) {
    uint32_t max_command_size_B = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    TT_ASSERT(
        command_size_B <= max_command_size_B,
        "Generated prefetcher command of size {} B exceeds max command size {} B",
        command_size_B,
        max_command_size_B);
    TT_ASSERT(
        (command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");
    TT_ASSERT(command_size_B > 0, "Command size must be greater than 0");
    if (this->bypass_enable) {
        return;
    }
    tt_driver_atomics::sfence();
    DispatchSettings::prefetch_q_entry_type command_size_16B =
        command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE;

    // stall_prefetcher is used for enqueuing traces, as replaying a trace will hijack the cmd_data_q
    // so prefetcher fetches multiple cmds that include the trace cmd, they will be corrupted by trace pulling data
    // from DRAM stall flag prevents pulling prefetch q entries that occur after the stall entry Stall flag for
    // prefetcher is MSB of FetchQ entry.
    if (stall_prefetcher) {
        command_size_16B |= (1 << ((sizeof(DispatchSettings::prefetch_q_entry_type) * 8) - 1));
    }
    this->prefetch_q_writers[cq_id].write(this->prefetch_q_dev_ptrs[cq_id], command_size_16B);
    this->prefetch_q_dev_ptrs[cq_id] += sizeof(DispatchSettings::prefetch_q_entry_type);

    if (cq_id < this->recent_fetch_traces.size() && this->last_issue_push_sizes[cq_id] != 0) {
        std::lock_guard<std::mutex> lock(this->recent_fetch_trace_locks[cq_id]);
        auto& traces = this->recent_fetch_traces[cq_id];
        traces.push_back(FetchTraceEntry{this->last_issue_push_start_addrs[cq_id], command_size_B, stall_prefetcher});
        this->total_fetch_trace_counts[cq_id]++;
        while (traces.size() > kMaxRecentFetchTraceEntries) {
            traces.pop_front();
        }
        this->last_issue_push_sizes[cq_id] = 0;
    }
}

}  // namespace tt::tt_metal
