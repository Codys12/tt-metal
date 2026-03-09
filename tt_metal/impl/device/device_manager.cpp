// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_manager.hpp"

#include <numa.h>
#include <pthread.h>
#include <tracy/Tracy.hpp>
#include <unistd.h>  // Warning Linux Only, needed for _SC_NPROCESSORS_ONLN

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_metal.hpp>
#include <umd/device/warm_reset.hpp>
#include "common/executor.hpp"
#include "context/metal_context.hpp"

#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include <experimental/fabric/fabric.hpp>
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_builder_context.hpp"

#include "dispatch/dispatch_settings.hpp"
#include "dispatch/topology.hpp"
#include "dispatch/system_memory_manager.hpp"

#include <tt_metal_profiler.hpp>
#include "profiler/profiler_state.hpp"
#include "profiler/profiler_state_manager.hpp"

#include <device.hpp>
#include "device_impl.hpp"

using namespace tt::tt_metal;

namespace tt {

namespace llrt::internal_ {
void wait_until_cores_done(
    ChipId device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms);
}  // namespace llrt::internal_

namespace device_cpu_allocator {
std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node(std::unordered_set<uint32_t>& free_cores) {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    if (numa_available() != -1) {
        // Host has NUMA enabled. Group CPU IDs by the NUMA nodes they belong to.
        for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            int node = numa_node_of_cpu(cpu);
            if (!cpu_cores_per_numa_node.contains(node)) {
                cpu_cores_per_numa_node.insert({node, {}});
            }
            free_cores.insert(cpu);
            cpu_cores_per_numa_node.at(node).push_back(cpu);
        }
    } else {
        // Host does not have NUMA. Place all CPU Ids under a single node (0).
        log_warning(tt::LogMetal, "Host does not use NUMA. May see reduced performance.");
        for (int cpu = 0; cpu < sysconf(_SC_NPROCESSORS_ONLN); ++cpu) {
            free_cores.insert(cpu);
        }
    }
    return cpu_cores_per_numa_node;
}

std::pair<int, int> get_cpu_cores_for_dispatch_threads(
    int mmio_controlled_device_id,
    const std::unordered_map<int, std::vector<uint32_t>>& cpu_cores_per_numa_node,
    std::unordered_set<uint32_t>& free_cores,
    uint32_t num_devices,
    bool use_separate_procs) {
    int core_assigned_to_device_worker_thread = 0;
    int core_assigned_to_device_completion_queue_reader = 0;
    uint32_t num_online_processors = sysconf(_SC_NPROCESSORS_ONLN);
    // Get NUMA node that the current device is mapped to through UMD
    int numa_node_for_device =
        tt::tt_metal::MetalContext::instance().get_cluster().get_numa_node_for_device(mmio_controlled_device_id);

    if (numa_available() != -1 and cpu_cores_per_numa_node.contains(numa_node_for_device)) {
        // NUMA node reported by UMD exists on host. Choose a core on this numa-node using round robin policy
        const auto& cpu_core_for_numa_node = cpu_cores_per_numa_node.at(numa_node_for_device);
        int num_cores_in_numa_node = cpu_core_for_numa_node.size();
        core_assigned_to_device_worker_thread =
            cpu_core_for_numa_node.at(mmio_controlled_device_id % num_cores_in_numa_node);
        if (use_separate_procs) {
            core_assigned_to_device_completion_queue_reader =
                cpu_core_for_numa_node.at((mmio_controlled_device_id + num_devices) % num_cores_in_numa_node);
        } else {
            core_assigned_to_device_completion_queue_reader = core_assigned_to_device_worker_thread;
        }
    } else {
        // NUMA node reported by UMD does not exist on host. Use round-robin binding policy for this worker thread.
        log_warning(
            tt::LogMetal,
            "NUMA node {} for device {} does not exist on host or NUMA is not available.",
            numa_node_for_device,
            mmio_controlled_device_id);
        core_assigned_to_device_worker_thread = mmio_controlled_device_id % num_online_processors;
        if (use_separate_procs) {
            core_assigned_to_device_completion_queue_reader =
                (mmio_controlled_device_id + num_devices) % num_online_processors;
        } else {
            core_assigned_to_device_completion_queue_reader = core_assigned_to_device_worker_thread;
        }
    }

    free_cores.erase(core_assigned_to_device_worker_thread);
    if (use_separate_procs) {
        free_cores.erase(core_assigned_to_device_completion_queue_reader);
    }
    return std::make_pair(core_assigned_to_device_worker_thread, core_assigned_to_device_completion_queue_reader);
}

void bind_current_thread_to_free_cores(const std::unordered_set<uint32_t>& free_cores) {
    cpu_set_t cpuset;
    pthread_t current_thread = pthread_self();
    CPU_ZERO(&cpuset);

    for (const auto& free_core : free_cores) {
        CPU_SET(free_core, &cpuset);
    }
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind main thread to free CPU cores. May see performance degradation. Error Code: {}",
            rc);
    }
}

std::unordered_map<uint32_t, uint32_t> get_device_id_to_core_map(
    const uint8_t num_hw_cqs, std::unordered_map<uint32_t, uint32_t>& completion_queue_reader_to_cpu_core_map) {
    std::vector<ChipId> device_ids;
    for (ChipId device_id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
        device_ids.emplace_back(device_id);
    }
    bool use_numa_node_based_thread_binding =
        tt::tt_metal::MetalContext::instance().rtoptions().get_numa_based_affinity();
    std::unordered_set<uint32_t> free_cores = {};
    uint32_t num_online_processors = sysconf(_SC_NPROCESSORS_ONLN);
    constexpr uint32_t max_num_procs_per_device = 2;
    // When using multiple command queues, assign separate CPU cores to worker and completion queue reader threads,
    // if enough processors exist on host. Atleast one core is given to the main thread.
    bool separate_procs_for_worker_and_reader =
        (num_hw_cqs > 1) && (max_num_procs_per_device * device_ids.size() <= num_online_processors - 1);
    std::unordered_map<uint32_t, uint32_t> worker_thread_to_cpu_core_map = {};
    if (use_numa_node_based_thread_binding) {
        auto cpu_cores_per_numa_node = device_cpu_allocator::get_cpu_cores_per_numa_node(free_cores);
        for (const auto& device_id : device_ids) {
            auto [worker_thread_core, completion_queue_reader_core] =
                device_cpu_allocator::get_cpu_cores_for_dispatch_threads(
                    device_id,
                    cpu_cores_per_numa_node,
                    free_cores,
                    device_ids.size(),
                    separate_procs_for_worker_and_reader);
            worker_thread_to_cpu_core_map.insert({device_id, worker_thread_core});
            completion_queue_reader_to_cpu_core_map.insert({device_id, completion_queue_reader_core});
        }
    } else {
        // Round Robin CPU assignment for worker and completion queue reader threads
        for (const auto& device_id : device_ids) {
            uint32_t worker_thread_proc = device_id % num_online_processors;
            worker_thread_to_cpu_core_map.insert({device_id, worker_thread_proc});
            if (separate_procs_for_worker_and_reader) {
                uint32_t completion_queue_reader_proc = (device_id + device_ids.size()) % num_online_processors;
                completion_queue_reader_to_cpu_core_map.insert({device_id, completion_queue_reader_proc});
            } else {
                completion_queue_reader_to_cpu_core_map.insert({device_id, worker_thread_proc});
            }
        }
    }

    if (use_numa_node_based_thread_binding) {
        // Bind main thread to cores not being used by workers
        bind_current_thread_to_free_cores(free_cores);
    }

    return worker_thread_to_cpu_core_map;
}
}  // namespace device_cpu_allocator

namespace tt_metal {

void DeviceManager::init_profiler() const {
#if defined(TRACY_ENABLE)
    if (!getDeviceProfilerState()) {
        return;
    }
    for (const auto& dev : this->get_all_active_devices()) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        detail::InitDeviceProfiler(dev);
        log_info(tt::LogMetal, "Profiler started on device {}", mmio_device_id);
        if (not this->skip_remote_devices_) {
            for (const auto& tunnel : tunnels_from_mmio) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    auto* mmio_device = get_device(mmio_controlled_device_id);
                    detail::InitDeviceProfiler(mmio_device);
                    log_info(tt::LogMetal, "Profiler started on remote device {}", mmio_device->id());
                }
            }
        }
    }
    detail::ProfilerSync(ProfilerSyncState::INIT);
#endif
}

void DeviceManager::initialize(
    const std::vector<ChipId>& device_ids,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size,
    bool init_profiler,
    bool initialize_fabric_and_dispatch_fw) {
    ZoneScoped;
    log_debug(tt::LogMetal, "DeviceManager initialize");

    num_hw_cqs_ = num_hw_cqs;
    l1_small_size_ = l1_small_size;
    trace_region_size_ = trace_region_size;
    worker_l1_size_ = worker_l1_size;
    using_fast_dispatch_ = MetalContext::instance().rtoptions().get_fast_dispatch();
    init_profiler_ = init_profiler;
    initialize_fabric_and_dispatch_fw_ = initialize_fabric_and_dispatch_fw;

    worker_thread_to_cpu_core_map_ =
        device_cpu_allocator::get_device_id_to_core_map(num_hw_cqs_, completion_queue_reader_to_cpu_core_map_);

    l1_bank_remap_.assign(l1_bank_remap.begin(), l1_bank_remap.end());

    initialize_devices(device_ids);
    is_initialized_ = true;
}

void DeviceManager::initialize_devices(const std::vector<ChipId>& device_ids) {
    // Validate requested device IDs exist and are reachable before proceeding.
    for (auto dev_id : device_ids) {
        TT_FATAL(
            tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().contains(dev_id),
            "Device {} does not exist. There are {} devices available (IDs 0 through {}).",
            dev_id,
            tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(),
            tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices() - 1);
        TT_FATAL(
            !tt::tt_metal::MetalContext::instance().is_chip_unreachable(dev_id),
            "Device {} was discovered via BFS but is not reachable via lite fabric (N-hop chip without tunnel). "
            "Cannot open this device.",
            dev_id);
    }

    std::vector<ChipId> device_ids_to_open = device_ids;
    dispatch_device_ids_.clear();  // Reset for this open session
    // Never skip for TG Cluster
    bool is_galaxy = tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster();
    bool skip = !is_galaxy;
    bool any_remote_devices = false;

    // Fabric requires all devices to be open even though dispatch
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
    if (using_fast_dispatch_) {
        // Check if fabric needs to be enabled (any remote devices).
        // Note, all devices must be open to use fabric. This check will happen in add_devices_to_pool.
        for (auto dev_id : device_ids_to_open) {
            any_remote_devices =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev_id) != dev_id;
            if (any_remote_devices) {
                break;
            }
        }
        // Must launch for TG
        any_remote_devices |= is_galaxy;

        // For Galaxy clusters, must open all devices.
        // For non-Galaxy (e.g. lite fabric multi-hop), only open requested devices + MMIO.
        // Opening ALL reachable devices would exhaust dispatch cores on the MMIO device
        // (each remote device needs 2 dispatch cores on the MMIO, but only 10 are available).
        if (any_remote_devices) {
            if (is_galaxy) {
                device_ids_to_open.clear();
                for (int id = 0; id < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); ++id) {
                    if (!tt::tt_metal::MetalContext::instance().is_chip_unreachable(id)) {
                        device_ids_to_open.push_back(id);
                    }
                }
            } else {
                // For non-Galaxy with remote devices (BH lite fabric), activate ALL
                // reachable devices.  The dispatch relay mux between MMIO and a remote
                // device requires fabric routers on every intermediate chip in the
                // fabric mesh path.  Only the user-requested devices + their MMIO get
                // dispatch kernels; other devices are activated for fabric routing only.
                device_ids_to_open.clear();
                for (int id = 0; id < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); ++id) {
                    if (!tt::tt_metal::MetalContext::instance().is_chip_unreachable(id)) {
                        device_ids_to_open.push_back(id);
                    }
                }

                for (auto dev_id : device_ids) {
                    dispatch_device_ids_.insert(dev_id);
                    dispatch_device_ids_.insert(
                        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev_id));
                }
            }
        }
    }

    // Default: all opened devices get dispatch (Galaxy and MMIO-only cases).
    if (dispatch_device_ids_.empty()) {
        dispatch_device_ids_.insert(device_ids_to_open.begin(), device_ids_to_open.end());
    }

    std::vector<ChipId> target_mmio_ids;
    for (const auto& device_id : device_ids_to_open) {
        TT_FATAL(
            tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().contains(device_id),
            "Device index {} out of range. There are {} devices available.",
            device_id,
            tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices());
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
        if (std::find(target_mmio_ids.begin(), target_mmio_ids.end(), mmio_device_id) == target_mmio_ids.end()) {
            target_mmio_ids.push_back(mmio_device_id);
        }
        skip &= (device_id == mmio_device_id);
    }
    if (target_mmio_ids.size() != tt::tt_metal::MetalContext::instance().get_cluster().number_of_pci_devices()) {
        log_warning(
            tt::LogMetal,
            "Opening subset of mmio devices slows down UMD read/write to remote chips. If opening more devices, "
            "consider using CreateDevices API.");
    }

    // Need to reserve eth cores for fabric before we initialize individual devices to maintain consistent state
    // while initializing default sub device state.
    // This call will be a no-op if fabric is disabled.
    // May be called again below
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    if (any_remote_devices) {
        auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
        if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
            fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D;
            tt::tt_fabric::SetFabricConfig(
                fabric_config, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
            // Call initialize again because previously it was a no-op
            tt::tt_metal::MetalContext::instance().initialize_fabric_config();
            log_info(
                tt::LogMetal,
                "Enabling {} only for dispatch. If your workload requires fabric, please set the fabric config "
                "accordingly.",
                fabric_config);
        } else {
            // Use the same mode
            tt::tt_fabric::SetFabricConfig(
                fabric_config, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        }
        log_info(tt::LogMetal, "Dispatch on {} with {} Command Queues\n", fabric_config, num_hw_cqs_);

        // For non-Galaxy clusters with partial device open, release fabric router
        // reservations for links to inactive devices.  This prevents deploying
        // routers on links where the peer won't have a router, avoiding handshake
        // timeouts during wait_for_fabric_router_sync.
        if (!is_galaxy) {
            std::set<ChipId> active_set(device_ids_to_open.begin(), device_ids_to_open.end());
            tt::tt_metal::MetalContext::instance().get_cluster().release_fabric_routers_for_inactive_links(active_set);
            // Rebuild routing tables and fabric context to reflect the reduced set of
            // fabric router links (only links between active devices remain).
            auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
            cp.clear_fabric_context();
            cp.initialize_fabric_context(fabric_config);
            cp.configure_routing_tables_for_fabric_ethernet_channels(
                fabric_config, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        }
    }

    skip_remote_devices_ = skip;
    log_info(tt::LogMetal, "DEBUG: initialize_devices: adding devices to pool");
    add_devices_to_pool(device_ids_to_open);

    // Initialize fabric tensix datamover config after devices are added to the pool
    log_info(tt::LogMetal, "DEBUG: initialize_devices: initializing fabric tensix datamover config");
    tt::tt_metal::MetalContext::instance().initialize_fabric_tensix_datamover_config();

    log_info(tt::LogMetal, "DEBUG: initialize_devices: init firmware on active devices");
    init_firmware_on_active_devices();
    log_info(tt::LogMetal, "DEBUG: initialize_devices: done");
}

void DeviceManager::initialize_fabric_and_dispatch_fw() {
    if (using_fast_dispatch_ && tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        // Due to galaxy taking potentially taking a 2-3 minutes to compile all the firmware kernels
        log_info(
            tt::LogMetal, "Initializing Fabric and Dispatch Firmware for Galaxy cluster (this may take a few minutes)");
    }
    this->initialize_active_devices();

    log_info(tt::LogMetal, "DEBUG: initialize_fabric_and_dispatch_fw: active devices done, checking INIT_FABRIC flag");
    if (has_flag(
            tt::tt_metal::MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        log_info(
            tt::LogMetal,
            "DEBUG: initialize_fabric_and_dispatch_fw: calling wait_for_fabric_router_sync (timeout={}ms)",
            this->get_fabric_router_sync_timeout_ms());
        this->wait_for_fabric_router_sync(this->get_fabric_router_sync_timeout_ms());
        log_info(tt::LogMetal, "DEBUG: initialize_fabric_and_dispatch_fw: wait_for_fabric_router_sync returned");

        // Diagnostic: read EDM status from ALL fabric router ETH cores on MMIO device(s)
        try {
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
            const auto& fc = cp.get_fabric_context();
            const auto& bc = fc.get_builder_context();
            auto [sync_addr, _expected] = bc.get_fabric_router_sync_address_and_status();
            for (const auto& dev : this->get_all_active_devices()) {
                if (!cluster.mmio_chip_ids().count(dev->id())) {
                    continue;
                }
                auto num_routers = bc.get_num_fabric_initialized_routers(dev->id());
                if (num_routers == 0) {
                    continue;
                }
                // Read EDM status from all ETH channels that might have fabric routers
                const auto& soc_desc = cluster.get_soc_desc(dev->id());
                for (uint32_t chan = 0; chan < 14; chan++) {
                    try {
                        auto lc = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
                        std::vector<std::uint32_t> status(1, 0);
                        tt_metal::detail::ReadFromDeviceL1(dev, lc, sync_addr, 4, status, CoreType::ETH);
                        // Only log if it looks like a valid EDM status value
                        if ((status[0] & 0xF0F0F0F0) == 0xa0b0c0d0 || status[0] == 0) {
                            log_info(
                                tt::LogMetal,
                                "DEBUG: MMIO device {} ETH chan={} logical={} EDM_status=0x{:08x}",
                                dev->id(),
                                chan,
                                lc.str(),
                                status[0]);
                        }
                    } catch (...) {
                    }
                }
            }
        } catch (...) {
            log_info(tt::LogMetal, "DEBUG: Failed to read post-sync EDM status diagnostics");
        }
    } else {
        log_info(tt::LogMetal, "DEBUG: initialize_fabric_and_dispatch_fw: INIT_FABRIC flag not set, skipping sync");
    }

    if (using_fast_dispatch_) {
        tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
        const bool check_chip_mapped = tt_fabric::is_tt_fabric_config(fabric_config);
        const auto* control_plane_ptr =
            check_chip_mapped ? &tt::tt_metal::MetalContext::instance().get_control_plane() : nullptr;
        auto is_dispatch_device = [&](ChipId id) {
            if (!dispatch_device_ids_.contains(id)) {
                return false;
            }
            if (tt::tt_metal::MetalContext::instance().is_chip_unreachable(id)) {
                return false;
            }
            return !check_chip_mapped || control_plane_ptr->is_chip_mapped(id);
        };

        log_info(tt::LogMetal, "DEBUG: initialize_fabric_and_dispatch_fw: initializing CQ runtime state");
        for (auto* dev : this->get_all_active_devices()) {
            if (!is_dispatch_device(dev->id())) {
                continue;
            }
            dev->initialize_command_queue_runtime_state();
        }
        log_info(tt::LogMetal, "DEBUG: initialize_fabric_and_dispatch_fw: CQ runtime state initialized");

        // Diagnostic: Enumerate fabric MUX cores on MMIO device
        try {
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            auto& dcm = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager();
            for (const auto& dev : this->get_all_active_devices()) {
                if (!cluster.mmio_chip_ids().count(dev->id())) {
                    continue;
                }
                uint16_t channel = cluster.get_assigned_channel_for_device(dev->id());
                for (int tunnel = 0; tunnel < 4; tunnel++) {
                    if (dcm.is_fabric_mux_core_allocated(dev->id(), channel, 0, tunnel)) {
                        const auto& mux_core = dcm.fabric_mux_core(dev->id(), channel, 0, tunnel);
                        log_info(
                            tt::LogMetal,
                            "DEBUG POST-INIT: MMIO dev {} fabric_mux tunnel={} core={}",
                            dev->id(),
                            tunnel,
                            mux_core.str());
                    }
                }
            }
        } catch (const std::exception& e) {
            log_info(tt::LogMetal, "DEBUG POST-INIT: Failed to enumerate fabric mux cores: {}", e.what());
        }

        // Diagnostic: Log control plane routing info for each device from MMIO
        if (tt_fabric::is_tt_fabric_config(fabric_config)) {
            try {
                const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
                const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
                for (const auto& dev : this->get_all_active_devices()) {
                    if (!cluster.mmio_chip_ids().count(dev->id())) {
                        continue;
                    }
                    auto src_node = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(dev->id());
                    for (const auto& target_dev : this->get_all_active_devices()) {
                        if (target_dev->id() == dev->id()) {
                            continue;
                        }
                        try {
                            auto dst_node = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_dev->id());
                            auto fwd_dir = cp.get_forwarding_direction(src_node, dst_node);
                            auto links = tt::tt_fabric::get_forwarding_link_indices(src_node, dst_node);
                            std::string link_str;
                            for (auto l : links) {
                                link_str += std::to_string(l) + " ";
                            }
                            log_info(
                                tt::LogMetal,
                                "DEBUG ROUTE: MMIO phys{} (M{}D{}) -> phys{} (M{}D{}): dir={} links=[{}]",
                                dev->id(),
                                src_node.mesh_id.get(),
                                src_node.chip_id,
                                target_dev->id(),
                                dst_node.mesh_id.get(),
                                dst_node.chip_id,
                                fwd_dir.has_value() ? static_cast<int>(*fwd_dir) : -1,
                                link_str);
                        } catch (...) {
                            log_info(
                                tt::LogMetal,
                                "DEBUG ROUTE: MMIO phys{} -> phys{}: FAILED",
                                dev->id(),
                                target_dev->id());
                        }
                    }
                }
            } catch (const std::exception& e) {
                log_info(tt::LogMetal, "DEBUG POST-INIT: Failed to log routing info: {}", e.what());
            }
        }

        // Diagnostic: Dump active fabric router channels per device
        if (tt_fabric::is_tt_fabric_config(fabric_config)) {
            try {
                const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
                const auto& fc = cp.get_fabric_context();
                const auto& bc = fc.get_builder_context();
                for (const auto& dev : this->get_all_active_devices()) {
                    try {
                        auto fabric_node = cp.get_fabric_node_id_from_physical_chip_id(dev->id());
                        auto num_routers = bc.get_num_fabric_initialized_routers(dev->id());
                        auto channels = cp.get_active_fabric_eth_channels(fabric_node);
                        std::string chan_str;
                        for (const auto& [chan, dir] : channels) {
                            chan_str += fmt::format("ch{}(dir={}) ", chan, (int)dir);
                        }
                        // Also check each direction for routing planes
                        std::string dir_str;
                        for (int d = 0; d < 4; d++) {
                            auto rd = static_cast<tt::tt_fabric::RoutingDirection>(d);
                            auto planes = cp.get_active_fabric_eth_routing_planes_in_direction(fabric_node, rd);
                            if (!planes.empty()) {
                                dir_str += fmt::format("dir{}={}_planes ", d, planes.size());
                            }
                        }
                        log_info(
                            tt::LogMetal,
                            "DEBUG POST-INIT: dev {} (phys{} M{}D{}) routers={} active_chans=[{}] routing=[{}]",
                            dev->id(),
                            dev->id(),
                            fabric_node.mesh_id.get(),
                            fabric_node.chip_id,
                            num_routers,
                            chan_str,
                            dir_str);
                    } catch (...) {
                    }
                }
            } catch (const std::exception& e) {
                log_info(tt::LogMetal, "DEBUG POST-INIT: Failed to enumerate per-device fabric state: {}", e.what());
            }
        }

        // Diagnostic: Read fabric router sender channel connection state on MMIO device.
        // After CQ runtime init, the FABRIC_MUX should have opened connections to the
        // fabric router.  Reading the sender channel connection semaphores tells us which
        // channels are connected.
        if (tt_fabric::is_tt_fabric_config(fabric_config)) {
            try {
                const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
                const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
                const auto& fc = cp.get_fabric_context();
                const auto& bc = fc.get_builder_context();
                const auto& router_config = bc.get_fabric_router_config();

                for (const auto& dev : this->get_all_active_devices()) {
                    if (!cluster.mmio_chip_ids().count(dev->id())) {
                        continue;
                    }
                    auto num_routers = bc.get_num_fabric_initialized_routers(dev->id());
                    if (num_routers == 0) {
                        continue;
                    }

                    const auto& soc_desc = cluster.get_soc_desc(dev->id());
                    const auto fabric_node_id = cp.get_fabric_node_id_from_physical_chip_id(dev->id());
                    const auto router_chans_and_dir = cp.get_active_fabric_eth_channels(fabric_node_id);

                    for (const auto& [chan, dir] : router_chans_and_dir) {
                        auto lc = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
                        // Read EDM status
                        std::vector<std::uint32_t> edm_status(1, 0);
                        tt_metal::detail::ReadFromDeviceL1(
                            dev, lc, router_config.edm_status_address, 4, edm_status, CoreType::ETH);

                        // Read sender channel connection semaphores
                        std::string conn_sems;
                        for (uint32_t s = 0; s < router_config.num_used_sender_channels; s++) {
                            auto addr = router_config.sender_channels_connection_semaphore_address[s];
                            if (addr == 0) {
                                continue;
                            }
                            std::vector<std::uint32_t> sem(1, 0);
                            tt_metal::detail::ReadFromDeviceL1(dev, lc, addr, 4, sem, CoreType::ETH);
                            conn_sems += fmt::format(" s{}={}@0x{:x}", s, sem[0], addr);
                        }

                        // Read sender channel flow control semaphores
                        std::string fc_sems;
                        for (uint32_t s = 0; s < router_config.num_used_sender_channels; s++) {
                            auto addr = router_config.sender_channels_local_flow_control_semaphore_address[s];
                            if (addr == 0) {
                                continue;
                            }
                            std::vector<std::uint32_t> sem(1, 0);
                            tt_metal::detail::ReadFromDeviceL1(dev, lc, addr, 4, sem, CoreType::ETH);
                            fc_sems += fmt::format(" s{}={}@0x{:x}", s, sem[0], addr);
                        }

                        // Read routing table header (first 16 bytes = mesh_id, device_id, intra_mesh_direction bytes)
                        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
                        auto rt_addr = hal.get_dev_addr(
                            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH,
                            tt::tt_metal::HalL1MemAddrType::ROUTING_TABLE);
                        std::vector<std::uint32_t> rt_data(8, 0);  // 32 bytes
                        tt_metal::detail::ReadFromDeviceL1(dev, lc, rt_addr, 32, rt_data, CoreType::ETH);
                        // First word: mesh_id(16) | device_id(16)
                        uint16_t rt_mesh_id = rt_data[0] & 0xFFFF;
                        uint16_t rt_device_id = (rt_data[0] >> 16) & 0xFFFF;

                        log_info(
                            tt::LogMetal,
                            "DEBUG POST-INIT: MMIO dev {} chan={} dir={} logical={} EDM=0x{:08x} "
                            "conn_sems:[{}] fc_sems:[{}] num_sender_ch={} "
                            "RT: mesh={} dev={} raw=[0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} "
                            "0x{:08x}]",
                            dev->id(),
                            chan,
                            (int)dir,
                            lc.str(),
                            edm_status[0],
                            conn_sems,
                            fc_sems,
                            router_config.num_used_sender_channels,
                            rt_mesh_id,
                            rt_device_id,
                            rt_data[0],
                            rt_data[1],
                            rt_data[2],
                            rt_data[3],
                            rt_data[4],
                            rt_data[5],
                            rt_data[6],
                            rt_data[7]);
                    }
                }
            } catch (const std::exception& e) {
                log_info(tt::LogMetal, "DEBUG POST-INIT: Failed to read fabric connection diagnostics: {}", e.what());
            }
        }
    }

    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    // Diagnostic: Read MUX kernel status on MMIO device.
    // The MUX status is at the address in the MUX config, on the MUX Tensix core.
    if (using_fast_dispatch_ && tt_fabric::is_tt_fabric_config(fabric_config)) {
        try {
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            auto& dcm = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager();
            for (const auto& dev : this->get_all_active_devices()) {
                if (!cluster.mmio_chip_ids().count(dev->id())) {
                    continue;
                }
                uint16_t channel = cluster.get_assigned_channel_for_device(dev->id());
                for (int tunnel = 0; tunnel < 8; tunnel++) {
                    if (!dcm.is_fabric_mux_core_allocated(dev->id(), channel, 0, tunnel)) {
                        continue;
                    }
                    const auto& mux_core = dcm.fabric_mux_core(dev->id(), channel, 0, tunnel);
                    // Read first 32 bytes from MUX L1 at the buffer base to see status
                    CoreCoord logical(mux_core.x, mux_core.y);
                    // Try reading MUX status. The status address is at the beginning of the MUX buffer region.
                    auto l1_base = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
                        tt::tt_metal::HalProgrammableCoreType::TENSIX,
                        tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
                    std::vector<std::uint32_t> mux_data(4, 0);
                    try {
                        tt_metal::detail::ReadFromDeviceL1(dev, logical, l1_base, 16, mux_data, CoreType::WORKER);
                        log_info(
                            tt::LogMetal,
                            "DEBUG POST-INIT: MMIO dev {} MUX tunnel={} core={} status_region=[0x{:08x} 0x{:08x} "
                            "0x{:08x} 0x{:08x}]",
                            dev->id(),
                            tunnel,
                            mux_core.str(),
                            mux_data[0],
                            mux_data[1],
                            mux_data[2],
                            mux_data[3]);
                    } catch (...) {
                    }
                }
            }
        } catch (const std::exception& e) {
            log_info(tt::LogMetal, "DEBUG POST-INIT: MUX status read failed: {}", e.what());
        }
    }

    // Diagnostic: Delayed re-read of connection semaphores to check if MUX connects later
    if (using_fast_dispatch_ && tt_fabric::is_tt_fabric_config(fabric_config)) {
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
            const auto& fc = cp.get_fabric_context();
            const auto& bc = fc.get_builder_context();
            const auto& router_config = bc.get_fabric_router_config();

            for (const auto& dev : this->get_all_active_devices()) {
                if (!cluster.mmio_chip_ids().count(dev->id())) {
                    continue;
                }
                auto num_routers = bc.get_num_fabric_initialized_routers(dev->id());
                if (num_routers == 0) {
                    continue;
                }
                const auto& soc_desc = cluster.get_soc_desc(dev->id());
                const auto fabric_node_id = cp.get_fabric_node_id_from_physical_chip_id(dev->id());
                const auto router_chans_and_dir = cp.get_active_fabric_eth_channels(fabric_node_id);

                for (const auto& [chan, dir] : router_chans_and_dir) {
                    auto lc = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
                    // Read sender channel connection semaphores
                    std::string conn_sems;
                    for (uint32_t s = 0; s < router_config.num_used_sender_channels; s++) {
                        auto addr = router_config.sender_channels_connection_semaphore_address[s];
                        if (addr == 0) {
                            continue;
                        }
                        std::vector<std::uint32_t> sem(1, 0);
                        tt_metal::detail::ReadFromDeviceL1(dev, lc, addr, 4, sem, CoreType::ETH);
                        conn_sems += fmt::format(" s{}={}", s, sem[0]);
                    }

                    std::string rx_info;

                    log_info(
                        tt::LogMetal,
                        "DEBUG DELAYED-200ms: MMIO dev {} chan={} dir={} conn:[{}] {}",
                        dev->id(),
                        chan,
                        (int)dir,
                        conn_sems,
                        rx_info);
                }
            }
        } catch (const std::exception& e) {
            log_info(tt::LogMetal, "DEBUG DELAYED: connection re-read failed: {}", e.what());
        }
    }

    log_info(tt::LogMetal, "DEBUG: initialize_fabric_and_dispatch_fw: returning");
}

void DeviceManager::initialize_host(IDevice* dev) const {
    detail::ClearProfilerControlBuffer(dev);

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e.
    // hugepage). Need to do this before FW init so we know what dispatch cores to reset.
    if (using_fast_dispatch_) {
        detail::DispatchStateCheck(true);
        dev->init_command_queue_host();
    } else {
        detail::DispatchStateCheck(false);
        TT_ASSERT(dev->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }
}

void DeviceManager::init_fabric(const std::vector<tt_metal::IDevice*>& active_devices) const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    std::vector<std::shared_future<tt_metal::IDevice*>> events;
    events.reserve(active_devices.size());
    for (auto* dev : active_devices) {
        // Skip devices not in the fabric mesh (e.g. when the mesh graph maps fewer
        // chips than the full cluster because of odd chip counts).
        if (!control_plane.is_chip_mapped(dev->id())) {
            continue;
        }
        events.emplace_back(detail::async([dev]() {
            if (dev->compile_fabric()) {
                return dev;
            } else {
                // compile failure mostly come from Nebula (TG)
                log_trace(tt::LogMetal, "Did not build fabric on Device {}", dev->id());
                return (tt_metal::IDevice*)nullptr;
            }
        }));
    }

    if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        return;
    }

    // Collect compiled devices.
    std::vector<tt_metal::IDevice*> compiled_devices;
    for (const auto& event : events) {
        auto* dev = event.get();
        if (dev) {
            compiled_devices.push_back(dev);
        }
    }

    // Stage remote devices FIRST (deepest-hop first), then MMIO devices.
    //
    // Remote configure_fabric() in the lite-fabric path only stages firmware,
    // router binaries, and launch mailboxes.  ERISC0 launch is deferred until
    // after remote CQ programs are written, so all heavy relay traffic happens
    // while ERISC1 is the only firmware using the ETH tile.
    //
    // MMIO devices are staged last to keep the gateway links free while remote
    // binaries are still being pushed through the relay.
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& mmio_ids = cluster.mmio_chip_ids();

    // Partition remote devices: those WITHOUT MMIO-peering ETH cores first,
    // those WITH MMIO-peering ETH cores second.
    std::vector<tt_metal::IDevice*> remote_deep;    // no MMIO-peering cores
    std::vector<tt_metal::IDevice*> remote_onehop;  // has MMIO-peering core(s)
    for (auto* dev : compiled_devices) {
        if (mmio_ids.count(dev->id())) {
            continue;  // MMIO devices handled separately
        }
        bool has_mmio_peer = false;
        auto connected_chips = cluster.get_ethernet_cores_grouped_by_connected_chips(dev->id());
        for (const auto& [peer_chip, cores] : connected_chips) {
            if (mmio_ids.count(peer_chip)) {
                has_mmio_peer = true;
                break;
            }
        }
        if (has_mmio_peer) {
            remote_onehop.push_back(dev);
        } else {
            remote_deep.push_back(dev);
        }
    }

    // Deep-hop devices first, deepest-first within the group.
    // configure_fabric for a shallower device (e.g., 2-hop) generates heavy
    // traffic through intermediate chips.  Processing deeper devices first
    // ensures their longer forwarding chains are used while pristine.
    std::sort(remote_deep.begin(), remote_deep.end(), [](tt_metal::IDevice* a, tt_metal::IDevice* b) {
        int hops_a = tt::tt_metal::MetalContext::instance().get_lite_fabric_hop_count(a->id());
        int hops_b = tt::tt_metal::MetalContext::instance().get_lite_fabric_hop_count(b->id());
        return hops_a != hops_b ? hops_a > hops_b : a->id() > b->id();
    });
    auto resync_remote_lite_fabric = [](tt_metal::IDevice* dev) {
        MetalContext::instance()
            .get_cluster()
            .get_driver()
            ->get_remote_chip(dev->id())
            ->get_remote_communication()
            ->resync_remote_transfer_ethernet_cores();
    };
    for (auto* dev : remote_deep) {
        // Phase 3 reads on other remote devices can advance the shared MMIO-side
        // ch1 ring. Re-sync the current binding before this device starts doing
        // remote ETH readbacks during configure_fabric().
        resync_remote_lite_fabric(dev);
        dev->configure_fabric();
    }
    // 1-hop devices second.
    for (auto* dev : remote_onehop) {
        resync_remote_lite_fabric(dev);
        dev->configure_fabric();
    }
    // MMIO devices last.
    for (auto* dev : compiled_devices) {
        if (mmio_ids.count(dev->id())) {
            dev->configure_fabric();
        }
    }
}

void DeviceManager::initialize_active_devices() {
    const auto& active_devices = this->get_all_active_devices();
    auto finalize_lite_fabric_bootstrap = [&]() {
        auto& context = tt::tt_metal::MetalContext::instance();
        if (!context.is_lite_fabric_bootstrap_active()) {
            return;
        }

        std::vector<IDevice*> remote_devices;
        std::vector<IDevice*> mmio_devices;
        remote_devices.reserve(active_devices.size());
        mmio_devices.reserve(active_devices.size());
        for (auto* dev : active_devices) {
            if (!context.get_control_plane().is_chip_mapped(dev->id())) {
                continue;
            }
            if (context.get_cluster().mmio_chip_ids().count(dev->id())) {
                mmio_devices.push_back(dev);
            } else {
                remote_devices.push_back(dev);
            }
        }

        std::sort(remote_devices.begin(), remote_devices.end(), [](IDevice* a, IDevice* b) {
            int hops_a = tt::tt_metal::MetalContext::instance().get_lite_fabric_hop_count(a->id());
            int hops_b = tt::tt_metal::MetalContext::instance().get_lite_fabric_hop_count(b->id());
            return hops_a != hops_b ? hops_a > hops_b : a->id() > b->id();
        });

        for (auto* dev : remote_devices) {
            context.launch_remote_eth_cores_for_fabric(dev->id());
        }
        // Single wait for all remote ERISC0s to boot (they boot in parallel).
        // 500ms is conservative for: trampoline → main() → base FW init → go-signal → kernel entry.
        log_info(tt::LogMetal, "All remote ERISC0 fabric routers deasserted, waiting 500ms for boot");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        log_info(tt::LogMetal, "DEBUG: finalize: calling terminate_lite_fabric_bootstrap");
        context.terminate_lite_fabric_bootstrap();
        log_info(
            tt::LogMetal,
            "DEBUG: finalize: terminate returned, calling configure_fabric on {} MMIO devices",
            mmio_devices.size());
        for (auto* dev : mmio_devices) {
            log_info(tt::LogMetal, "DEBUG: finalize: configure_fabric for MMIO device {}", dev->id());
            dev->configure_fabric();
            log_info(tt::LogMetal, "DEBUG: finalize: configure_fabric for MMIO device {} done", dev->id());
        }
        context.update_lite_fabric_bindings_for_fabric_routers();
        log_info(tt::LogMetal, "Fabric Initialized with config {}", context.get_fabric_config());
    };

    // Activate fabric (must be before FD)
    tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        if (has_flag(
                tt::tt_metal::MetalContext::instance().get_fabric_manager(),
                tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            log_info(tt::LogMetal, "Initializing Fabric");
            tt::tt_metal::MetalContext::instance().get_control_plane().write_routing_tables_to_all_chips();

            // Initialize fabric on all devices.  Remote devices are configured first
            // (they need the lite fabric relay + UMD bindings to WRITE_REG and write L1),
            // then MMIO devices.
            init_fabric(active_devices);

            if (!tt::tt_metal::MetalContext::instance().is_lite_fabric_bootstrap_active()) {
                tt::tt_metal::MetalContext::instance().update_lite_fabric_bindings_for_fabric_routers();
                log_info(tt::LogMetal, "Fabric Initialized with config {}", fabric_config);
            }
        } else if (has_flag(
                       tt::tt_metal::MetalContext::instance().get_fabric_manager(),
                       tt_fabric::FabricManagerMode::TERMINATE_FABRIC)) {
            log_info(tt::LogMetal, "Compiling fabric to setup fabric context for fabric termination");
            for (auto* dev : active_devices) {
                if (tt::tt_metal::MetalContext::instance().get_control_plane().is_chip_mapped(dev->id())) {
                    dev->compile_fabric();
                }
            }
        } else {
            log_info(tt::LogMetal, "Fabric initialized through Fabric Manager");
        }
    }

    // Activate FD kernels
    // Remaining steps are for setting up FD
    if (!using_fast_dispatch_) {
        finalize_lite_fabric_bootstrap();
        return;
    }

    // When fabric is active, only configure dispatch for devices that are mapped in the control plane.
    // With lite fabric, the topology mapper may map only a subset of physical chips to the fabric mesh.
    const bool check_chip_mapped = tt_fabric::is_tt_fabric_config(fabric_config);
    const auto* control_plane_ptr =
        check_chip_mapped ? &tt::tt_metal::MetalContext::instance().get_control_plane() : nullptr;
    auto is_dispatch_device = [&](ChipId id) {
        if (!dispatch_device_ids_.contains(id)) {
            return false;
        }
        if (tt::tt_metal::MetalContext::instance().is_chip_unreachable(id)) {
            return false;
        }
        return !check_chip_mapped || control_plane_ptr->is_chip_mapped(id);
    };
    auto resync_remote_lite_fabric = [&](ChipId id) {
        auto& context = tt::tt_metal::MetalContext::instance();
        if (context.get_cluster().mmio_chip_ids().count(id)) {
            return;
        }
        auto* remote_chip = context.get_cluster().get_driver()->get_remote_chip(id);
        if (remote_chip) {
            remote_chip->get_remote_communication()->resync_remote_transfer_ethernet_cores();
        }
    };

    // Generate static args
    log_info(tt::LogMetal, "DEBUG: FD init: generating static args for {} devices", active_devices.size());
    for (auto* dev : active_devices) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        if (!is_dispatch_device(dev->id())) {
            continue;
        }

        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        populate_cq_static_args(dev);
        if (not this->skip_remote_devices_) {
            std::unordered_set<ChipId> handled_devices;
            handled_devices.insert(dev->id());
            for (const auto& tunnel : tunnels_from_mmio) {
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    handled_devices.insert(mmio_controlled_device_id);
                    if (!is_dispatch_device(mmio_controlled_device_id)) {
                        continue;
                    }
                    auto* device = get_device(mmio_controlled_device_id);
                    if (!device || !device->is_initialized()) {
                        continue;
                    }
                    populate_cq_static_args(device);
                }
            }
            for (const auto& controlled_device_id :
                 tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                     mmio_device_id)) {
                if (handled_devices.count(controlled_device_id) == 0 && is_dispatch_device(controlled_device_id)) {
                    auto* device = get_device(controlled_device_id);
                    if (device && device->is_initialized()) {
                        populate_cq_static_args(device);
                    }
                }
            }
        }
    }

    // Create command queue programs
    log_info(tt::LogMetal, "DEBUG: FD init: creating CQ programs");
    for (auto* dev : active_devices) {
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        if (!is_dispatch_device(dev->id())) {
            continue;
        }

        create_cq_program(dev);
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        if (not this->skip_remote_devices_) {
            std::unordered_set<ChipId> handled_devices;
            handled_devices.insert(dev->id());
            for (const auto& tunnel : tunnels_from_mmio) {
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    handled_devices.insert(mmio_controlled_device_id);
                    if (!is_dispatch_device(mmio_controlled_device_id)) {
                        continue;
                    }
                    auto* device = get_device(mmio_controlled_device_id);
                    if (!device || !device->is_initialized()) {
                        continue;
                    }
                    create_cq_program(device);
                }
            }
            for (const auto& controlled_device_id :
                 tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                     mmio_device_id)) {
                if (handled_devices.count(controlled_device_id) == 0 && is_dispatch_device(controlled_device_id)) {
                    auto* device = get_device(controlled_device_id);
                    if (device && device->is_initialized()) {
                        create_cq_program(device);
                    }
                }
            }
        }
    }

    // Compile programs
    log_info(tt::LogMetal, "DEBUG: FD init: compiling CQ programs");
    compile_cq_programs();

    // Init command queue
    log_info(tt::LogMetal, "DEBUG: FD init: initializing command queues");
    for (auto* dev : active_devices) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        if (!is_dispatch_device(dev->id())) {
            continue;
        }

        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        log_info(tt::LogMetal, "DEBUG: FD init: CQ device init for MMIO device {}", dev->id());
        dev->init_command_queue_device();
        log_info(tt::LogMetal, "DEBUG: FD init: CQ device {} done", dev->id());
        // Diagnostic barrier: only for active devices
        for (const auto& tunnel : tunnels_from_mmio) {
            for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                uint32_t rid = tunnel[ts];
                auto* rdev = get_device(rid);
                if (!rdev || !rdev->is_initialized()) {
                    continue;
                }
                resync_remote_lite_fabric(rid);
                log_info(tt::LogMetal, "DEBUG: FD init: post-MMIO-CQ barrier for remote device {}", rid);
                tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(rid);
                log_info(tt::LogMetal, "DEBUG: FD init: post-MMIO-CQ barrier for remote device {} passed", rid);
            }
        }
        if (not this->skip_remote_devices_) {
            std::unordered_set<ChipId> handled_devices;
            handled_devices.insert(dev->id());
            for (const auto& tunnel : tunnels_from_mmio) {
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    handled_devices.insert(mmio_controlled_device_id);
                    if (!is_dispatch_device(mmio_controlled_device_id)) {
                        continue;
                    }
                    auto* device = get_device(mmio_controlled_device_id);
                    if (!device || !device->is_initialized()) {
                        continue;
                    }
                    log_info(
                        tt::LogMetal, "DEBUG: FD init: CQ device init for tunnel device {}", mmio_controlled_device_id);
                    resync_remote_lite_fabric(mmio_controlled_device_id);
                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(mmio_controlled_device_id);
                    device->init_command_queue_device();
                    log_info(tt::LogMetal, "Command Queue initialized on Device {}", device->id());
                }
            }
            for (const auto& controlled_device_id :
                 tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                     mmio_device_id)) {
                if (handled_devices.count(controlled_device_id) == 0 && is_dispatch_device(controlled_device_id)) {
                    auto* device = get_device(controlled_device_id);
                    if (device && device->is_initialized()) {
                        log_info(
                            tt::LogMetal, "DEBUG: FD init: CQ device init for N-hop device {}", controlled_device_id);
                        resync_remote_lite_fabric(controlled_device_id);
                        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(controlled_device_id);
                        device->init_command_queue_device();
                        log_info(tt::LogMetal, "Command Queue initialized on N-hop Device {}", device->id());
                    }
                }
            }
        }
    }
    finalize_lite_fabric_bootstrap();
    log_info(tt::LogMetal, "DEBUG: FD init: complete");
    dispatch_firmware_active_ = true;
}

void DeviceManager::activate_device(ChipId id) {
    TT_FATAL(
        tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().contains(id),
        "Device index {} out of range. There are {} devices available.",
        id,
        tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices());
    const std::lock_guard<std::mutex> lock(lock_);
    if (this->devices_.size() < id + 1) {
        this->devices_.reserve(id + 1);
    }
    auto* device = get_device(id);
    if (!device) {
        log_debug(tt::LogMetal, "DeviceManager new device {}", id);
        int worker_core_thread_core = this->worker_thread_to_cpu_core_map_.at(id);
        int completion_queue_reader_core = this->completion_queue_reader_to_cpu_core_map_.at(id);
        device = new Device(
            id,
            this->num_hw_cqs_,
            this->l1_small_size_,
            this->trace_region_size_,
            this->l1_bank_remap_,
            false,
            worker_core_thread_core,
            completion_queue_reader_core,
            this->worker_l1_size_);
        devices_.emplace_back(std::unique_ptr<IDevice>(device));
    } else {
        log_debug(tt::LogMetal, "DeviceManager re-initialize device {}", id);
        if (not device->is_initialized()) {
            device->initialize(this->num_hw_cqs_, this->l1_small_size_, this->trace_region_size_, this->worker_l1_size_, this->l1_bank_remap_);
        } else {
            TT_THROW("Cannot re-initialize device {}, must first call close()", id);
        }
    }
}

bool DeviceManager::is_device_active(ChipId id) const {
    auto* device = this->get_device(id);
    if (!device) {
        return false;
    }

    return device->is_initialized();
}

IDevice* DeviceManager::get_device(ChipId id) const {
    auto it = std::find_if(devices_.begin(), devices_.end(), [&id](const auto& device) { return device->id() == id; });
    if (it == devices_.end()) {
        return nullptr;
    }

    return it->get();
}

std::size_t DeviceManager::get_max_num_eth_cores_across_all_devices() const {
    // This API is needed due to Issue #19729:
    // Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
    // Records the maximum number of active ethernet cores across all devices opened in the cluster.
    // TT-Mesh dispatch assumes that all physical devices in the Mesh have the maximum number of active
    // ethernet cores (uniformity assumption)
    // Dispatch firmware running on each physical device knows how many ethernet cores are actually
    // available and will dispatch to/wait on the correct number of cores (effectively ignoring the
    // value host dispatch provides, if its incorrect).
    std::size_t max_eth_core_count = 0;
    for (const auto& device : this->devices_) {
        max_eth_core_count = std::max(
            MetalContext::instance()
                .get_control_plane()
                .get_active_ethernet_cores(device->id(), /*skip_reserved_cores*/ true)
                .size(),
            max_eth_core_count);
    }
    return max_eth_core_count;
}

void DeviceManager::add_devices_to_pool(const std::vector<ChipId>& device_ids) {
    std::set<ChipId> devices_to_activate;

    if (this->skip_remote_devices_) {
        for (const auto& device_id : device_ids) {
            const auto& mmio_device_id =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
            TT_ASSERT(device_id == mmio_device_id, "Skipping remote devices is only available for mmio devices");
            devices_to_activate.insert(device_id);
        }
    } else {
        bool is_galaxy = tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster();
        for (const auto& device_id : device_ids) {
            const auto& mmio_device_id =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
            if (is_galaxy) {
                // Galaxy: expand to all devices controlled by the MMIO device.
                for (const auto& mmio_controlled_device_id :
                     tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                         mmio_device_id)) {
                    if (tt::tt_metal::MetalContext::instance().is_chip_unreachable(mmio_controlled_device_id)) {
                        continue;
                    }
                    devices_to_activate.insert(mmio_controlled_device_id);
                }
            } else {
                // Non-Galaxy: activate exactly the devices passed in (already includes
                // MMIO + tunnel path from initialize_devices).
                devices_to_activate.insert(device_id);
            }
        }
    }

    for (const auto& device_id : devices_to_activate) {
        if (not this->is_device_active(device_id)) {
            this->activate_device(device_id);
        }
    }

    // For Galaxy: Fabric requires all devices to be active.
    // For non-Galaxy (lite fabric): only a subset of devices may be active.
    tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config) &&
        tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        for (int i = 0; i < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); i++) {
            if (tt::tt_metal::MetalContext::instance().is_chip_unreachable(i)) {
                continue;
            }
            TT_FATAL(
                this->is_device_active(i),
                "Fabric is being used but Device {} is not active. "
                "This may indicate that the fabric was launched on a subset of the devices available in the system, "
                "which is currently not supported. "
                "To launch on a subset of devices, first create a MeshDevice of the full system size, then create "
                "submeshes accordingly.\n"
                "For example, on a 6u system (8x4), if you wanted to run a 2x4 workload you could do:\n"
                "ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)\n"
                "mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))\n"
                "submeshes = mesh_device.create_submeshes(ttnn.MeshShape(2,8))",
                i);
        }
    }

    if (this->using_fast_dispatch_ && !devices_to_activate.empty()) {
        // Only generate dispatch topology for devices that actually need dispatch.
        // Intermediate tunnel devices activated for fabric routing are excluded to
        // avoid exhausting dispatch cores on the MMIO device.
        std::set<ChipId> dispatch_set;
        for (const auto& id : devices_to_activate) {
            if (dispatch_device_ids_.contains(id)) {
                dispatch_set.insert(id);
            }
        }

        if (tt_fabric::is_tt_fabric_config(fabric_config)) {
            // Further filter to devices mapped in the control plane.
            const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
            std::set<ChipId> mapped_devices;
            for (const auto& id : dispatch_set) {
                if (control_plane.is_chip_mapped(id)) {
                    mapped_devices.insert(id);
                }
            }
            populate_fd_kernels(mapped_devices, this->num_hw_cqs_);
        } else {
            populate_fd_kernels(dispatch_set, this->num_hw_cqs_);
        }
    }
}

uint32_t DeviceManager::get_fabric_router_sync_timeout_ms() {
    // Return user-configured timeout or default value
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rtoptions.get_simulator_enabled()) {
        return 15000;  // Keep simulator timeout unchanged
    }

    auto timeout = rtoptions.get_fabric_router_sync_timeout_ms();

    // Return user override if set, otherwise use fabric default
    return timeout.value_or(10000);
}

void DeviceManager::wait_for_fabric_router_sync(uint32_t timeout_ms) const {
    tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (!tt::tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    // (deferred_lite_remote_devices removed — remote poll after lite fabric termination can never work)

    auto wait_for_handshake = [&](IDevice* dev) {
        if (!dev) {
            TT_THROW("Fabric router sync on null device. All devices must be opened for Fabric.");
        }
        auto did = dev->id();
        bool is_mmio = dev->is_mmio_capable();
        bool lite_term = tt::tt_metal::MetalContext::instance().was_lite_fabric_bootstrap_terminated();
        log_info(
            tt::LogMetal, "DEBUG: wait_for_handshake: device {} mmio={} lite_terminated={}", did, is_mmio, lite_term);
        auto num_routers = builder_context.get_num_fabric_initialized_routers(did);
        if (num_routers == 0) {
            log_info(tt::LogMetal, "DEBUG: wait_for_handshake: device {} early return (0 routers)", did);
            return;
        }
        bool launched = tt::tt_metal::MetalContext::instance().has_fabric_routers_launched(did);
        if (!launched) {
            log_info(tt::LogMetal, "DEBUG: wait_for_handshake: device {} early return (not launched)", did);
            return;
        }

        const auto master_router_chan = builder_context.get_fabric_master_router_chan(dev->id());
        const auto master_router_logical_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(dev->id()).get_eth_core_for_channel(
                master_router_chan, CoordSystem::LOGICAL);

        log_info(
            tt::LogMetal,
            "DEBUG: wait_for_handshake: device {} routers={} chan={}",
            did,
            num_routers,
            master_router_chan);
        if (lite_term && !is_mmio) {
            // Remote devices can't be polled after lite fabric termination — ReadFromDeviceL1
            // goes through UMD read_non_mmio which speaks lite fabric protocol, but nobody is
            // running lite fabric anymore.  Skip the deferred poll entirely; trust MMIO sync.
            log_info(
                tt::LogMetal,
                "DEBUG: wait_for_handshake: device {} skipping remote READY poll (lite fabric terminated, trusting "
                "MMIO sync)",
                did);
            return;
        }
        log_info(
            tt::LogMetal,
            "DEBUG: wait_for_handshake: device {} ENTERING POLLING LOOP (lite_term={} is_mmio={})",
            did,
            lite_term,
            is_mmio);

        const auto [router_sync_address, expected_status] = builder_context.get_fabric_router_sync_address_and_status();
        std::vector<std::uint32_t> master_router_status{0};
        auto start_time = std::chrono::steady_clock::now();
        while (master_router_status[0] != expected_status) {
            tt_metal::detail::ReadFromDeviceL1(
                dev, master_router_logical_core, router_sync_address, 4, master_router_status, CoreType::ETH);
            // If the read value matches expected status, then we can break out of the loop
            // No need to check for timeout in this case.
            if (master_router_status[0] == expected_status) {
                break;
            }
            // Check for timeout
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            if (elapsed_ms > timeout_ms) {
                // Diagnostic reads for debugging
                const auto& hal = tt::tt_metal::MetalContext::instance().hal();
                auto eth_core_type = HalProgrammableCoreType::ACTIVE_ETH;
                uint64_t go_addr = hal.get_dev_addr(eth_core_type, HalL1MemAddrType::GO_MSG);
                uint64_t go_idx_addr = hal.get_dev_addr(eth_core_type, HalL1MemAddrType::GO_MSG_INDEX);
                uint64_t launch_addr = hal.get_dev_addr(eth_core_type, HalL1MemAddrType::LAUNCH);
                uint64_t mailbox_addr = hal.get_dev_addr(eth_core_type, HalL1MemAddrType::MAILBOX);

                // Read go message (first 8 bytes)
                std::vector<std::uint32_t> go_data(2, 0);
                tt_metal::detail::ReadFromDeviceL1(dev, master_router_logical_core, go_addr, 8, go_data, CoreType::ETH);
                // Read go_message_index
                std::vector<std::uint32_t> go_idx_data(1, 0);
                tt_metal::detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, go_idx_addr, 4, go_idx_data, CoreType::ETH);
                // Read launch message first 80 bytes (enough for kernel_text_offset + enables)
                std::vector<std::uint32_t> launch_data(20, 0);
                tt_metal::detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, launch_addr, 80, launch_data, CoreType::ETH);
                // Read mailbox area (ncrisc_halt + go_message_index area, 16 bytes)
                std::vector<std::uint32_t> mailbox_data(4, 0);
                tt_metal::detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, mailbox_addr, 16, mailbox_data, CoreType::ETH);
                // Read first 16 bytes of L1 (trampoline area)
                std::vector<std::uint32_t> l1_start(4, 0);
                tt_metal::detail::ReadFromDeviceL1(dev, master_router_logical_core, 0, 16, l1_start, CoreType::ETH);

                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync: master chan={}, logical core={}, sync address=0x{:08x}",
                    master_router_chan,
                    master_router_logical_core.str(),
                    router_sync_address);
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: go_addr=0x{:x} go_data=[0x{:08x}, 0x{:08x}] "
                    "go_idx_addr=0x{:x} go_message_index={}",
                    go_addr,
                    go_data[0],
                    go_data[1],
                    go_idx_addr,
                    go_idx_data[0]);
                // Decode launch message fields (kernel_config_msg_t layout):
                // kernel_config_base[3] at words 0-2, kernel_text_offset[0] at word 11, enables at word 19
                uint32_t kconfig_tensix = launch_data[0];
                uint32_t kconfig_active_eth = launch_data[1];
                uint32_t kconfig_idle_eth = launch_data[2];
                uint32_t kernel_text_offset_0 = launch_data[11];  // offset 44 / 4
                uint32_t enables = launch_data[19];               // offset 76 / 4
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: launch_addr=0x{:x} kconfig=[0x{:x}, 0x{:x}, 0x{:x}] "
                    "kernel_text_offset[0]=0x{:x} enables=0x{:08x}",
                    launch_addr,
                    kconfig_tensix,
                    kconfig_active_eth,
                    kconfig_idle_eth,
                    kernel_text_offset_0,
                    enables);
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: mailbox_addr=0x{:x} mailbox_data=[0x{:08x}, 0x{:08x}, 0x{:08x}, "
                    "0x{:08x}]",
                    mailbox_addr,
                    mailbox_data[0],
                    mailbox_data[1],
                    mailbox_data[2],
                    mailbox_data[3]);
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: L1[0x0..0xF]=[0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                    l1_start[0],
                    l1_start[1],
                    l1_start[2],
                    l1_start[3]);

                // Read syseng API table (0x7CF00) and ret stub (0x7CF10)
                std::vector<std::uint32_t> api_table(5, 0);
                tt_metal::detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, 0x7CF00, 20, api_table, CoreType::ETH);
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: API_TABLE[0x7CF00]=[0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}] "
                    "RET_STUB[0x7CF10]=0x{:08x}",
                    api_table[0],
                    api_table[1],
                    api_table[2],
                    api_table[3],
                    api_table[4]);

                // Read kernel config buffer start (ACTIVE_ETH base)
                if (kconfig_active_eth != 0) {
                    std::vector<std::uint32_t> kcfg_data(4, 0);
                    tt_metal::detail::ReadFromDeviceL1(
                        dev, master_router_logical_core, kconfig_active_eth, 16, kcfg_data, CoreType::ETH);
                    log_info(
                        tt::LogMetal,
                        "Fabric Router Sync DIAG: L1[kconfig_base=0x{:x}]=[0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                        kconfig_active_eth,
                        kcfg_data[0],
                        kcfg_data[1],
                        kcfg_data[2],
                        kcfg_data[3]);
                }
                // Read actual kernel entry point (kconfig_base + kernel_text_offset[0])
                uint32_t kernel_entry = kconfig_active_eth + kernel_text_offset_0;
                if (kconfig_active_eth != 0) {
                    std::vector<std::uint32_t> kentry_data(4, 0);
                    tt_metal::detail::ReadFromDeviceL1(
                        dev, master_router_logical_core, kernel_entry, 16, kentry_data, CoreType::ETH);
                    log_info(
                        tt::LogMetal,
                        "Fabric Router Sync DIAG: L1[kernel_entry=0x{:x}]=[0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                        kernel_entry,
                        kentry_data[0],
                        kentry_data[1],
                        kentry_data[2],
                        kentry_data[3]);
                }

                // Read EDM status area: check for STARTED (0xA0B0C0D0)
                std::vector<std::uint32_t> status_area(8, 0);
                tt_metal::detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, router_sync_address - 16, 32, status_area, CoreType::ETH);
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: status_area[sync-16..sync+15]="
                    "[0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                    status_area[0],
                    status_area[1],
                    status_area[2],
                    status_area[3],
                    status_area[4],
                    status_area[5],
                    status_area[6],
                    status_area[7]);

                // Read all active router ETH cores on this device
                auto num_routers = builder_context.get_num_fabric_initialized_routers(dev->id());
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: Device {} has {} initialized routers",
                    dev->id(),
                    num_routers);

                // Try to read peer (MMIO) device's master router status for comparison
                auto mmio_dev_id =
                    tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
                if (mmio_dev_id != dev->id()) {
                    try {
                        auto* mmio_dev = this->get_device(mmio_dev_id);
                        if (mmio_dev && builder_context.get_num_fabric_initialized_routers(mmio_dev_id) > 0) {
                            const auto peer_router_chan = builder_context.get_fabric_master_router_chan(mmio_dev_id);
                            const auto peer_router_core =
                                tt::tt_metal::MetalContext::instance()
                                    .get_cluster()
                                    .get_soc_desc(mmio_dev_id)
                                    .get_eth_core_for_channel(peer_router_chan, CoordSystem::LOGICAL);
                            std::vector<std::uint32_t> peer_status(1, 0);
                            tt_metal::detail::ReadFromDeviceL1(
                                mmio_dev, peer_router_core, router_sync_address, 4, peer_status, CoreType::ETH);
                            log_info(
                                tt::LogMetal,
                                "Fabric Router Sync DIAG: Peer MMIO device {} master router (chan={}) status=0x{:08x}",
                                mmio_dev_id,
                                peer_router_chan,
                                peer_status[0]);
                        }
                    } catch (...) {
                        log_info(
                            tt::LogMetal,
                            "Fabric Router Sync DIAG: Failed to read peer MMIO device {} status",
                            mmio_dev_id);
                    }
                }

                // Read handshake area on the master router
                std::vector<std::uint32_t> hs_data(8, 0);
                // handshake_addr is typically right after erisc_l1_unreserved_base
                uint32_t hs_addr = router_sync_address + 32;  // approximate; read a range
                tt_metal::detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, hs_addr, 32, hs_data, CoreType::ETH);
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync DIAG: Device {} handshake area [sync+32..sync+63]="
                    "[0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                    dev->id(),
                    hs_data[0],
                    hs_data[1],
                    hs_data[2],
                    hs_data[3],
                    hs_data[4],
                    hs_data[5],
                    hs_data[6],
                    hs_data[7]);

                // Re-read the current status right before throwing
                std::vector<std::uint32_t> final_status(1, 0);
                tt_metal::detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, router_sync_address, 4, final_status, CoreType::ETH);

                log_warning(
                    tt::LogMetal,
                    "Fabric Router Sync: Timeout after {} ms. Device {}: Expected status 0x{:08x}, got 0x{:08x} (final "
                    "re-read: 0x{:08x}). Continuing without fabric routers on this device.",
                    timeout_ms,
                    dev->id(),
                    expected_status,
                    master_router_status[0],
                    final_status[0]);
                return;  // Skip this device's fabric routers
            }
        }

        auto ready_address_and_signal = builder_context.get_fabric_router_ready_address_and_signal();
        if (ready_address_and_signal) {
            std::vector<uint32_t> signal(1, ready_address_and_signal->second);
            tt_metal::detail::WriteToDeviceL1(
                dev, master_router_logical_core, ready_address_and_signal->first, signal, CoreType::ETH);
        }
    };

    for (const auto& dev : this->get_all_active_devices()) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }
        // Skip devices not in the fabric mesh
        if (!control_plane.is_chip_mapped(dev->id())) {
            continue;
        }

        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(dev->id());

        std::unordered_set<ChipId> handled_devices;
        handled_devices.insert(dev->id());

        for (const auto& tunnel : tunnels_from_mmio) {
            // Need to poll on devices from farthest to the closest.
            for (auto j = tunnel.size() - 1; j > 0; j--) {
                handled_devices.insert(tunnel[j]);
                if (!control_plane.is_chip_mapped(tunnel[j])) {
                    continue;
                }
                auto* tunnel_dev = get_device(tunnel[j]);
                if (!tunnel_dev || !tunnel_dev->is_initialized()) {
                    continue;
                }
                wait_for_handshake(tunnel_dev);
            }
        }

        // Handle N-hop devices not in UMD tunnels
        for (const auto& controlled_device_id :
             tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(dev->id())) {
            if (handled_devices.count(controlled_device_id) == 0 &&
                control_plane.is_chip_mapped(controlled_device_id)) {
                auto* device = get_device(controlled_device_id);
                if (device && device->is_initialized()) {
                    wait_for_handshake(device);
                }
            }
        }

        wait_for_handshake(dev);
    }

    // Note: deferred_lite_remote_devices is no longer populated — remote devices
    // are skipped entirely when lite fabric is terminated (can't read via UMD).
}

void DeviceManager::init_firmware_on_active_devices() {
    const auto& active_devices = this->get_all_active_devices();
    for (const auto& dev : active_devices) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        this->initialize_host(dev);
        if (not this->skip_remote_devices_) {
            // Track which devices have been initialized via tunnels
            std::unordered_set<ChipId> initialized_devices;
            initialized_devices.insert(dev->id());

            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    log_debug(tt::LogMetal, "Tunnel {} Device {} Tunnel Stop: {}", t, mmio_controlled_device_id, ts);
                    auto* device = get_device(mmio_controlled_device_id);
                    if (!device || !device->is_initialized()) {
                        continue;  // Device not active (partial open)
                    }
                    // Only initialize host (dispatch CQ) for dispatch-enabled devices.
                    // Fabric-only intermediate devices don't need dispatch.
                    if (!dispatch_device_ids_.contains(mmio_controlled_device_id)) {
                        log_debug(
                            tt::LogMetal,
                            "Skipping dispatch init for fabric-only device {}",
                            mmio_controlled_device_id);
                        initialized_devices.insert(mmio_controlled_device_id);
                        continue;
                    }
                    this->initialize_host(device);
                    initialized_devices.insert(mmio_controlled_device_id);
                }
            }

            // N-hop devices discovered via BFS may not be in UMD tunnels.
            // Initialize any remaining active devices controlled by this MMIO device.
            for (const auto& controlled_device_id :
                 tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                     mmio_device_id)) {
                if (initialized_devices.count(controlled_device_id) == 0) {
                    auto* device = get_device(controlled_device_id);
                    if (device && device->is_initialized() && dispatch_device_ids_.contains(controlled_device_id)) {
                        log_debug(
                            tt::LogMetal,
                            "Initializing host for N-hop device {} not in UMD tunnels",
                            controlled_device_id);
                        this->initialize_host(device);
                    }
                }
            }
        }
    }

    if (init_profiler_) {
        this->init_profiler();
    }
    if (initialize_fabric_and_dispatch_fw_) {
        this->initialize_fabric_and_dispatch_fw();
    }
}

DeviceManager::DeviceManager() {
    ZoneScoped;
    log_debug(tt::LogMetal, "DeviceManager constructor");
}

IDevice* DeviceManager::get_active_device(ChipId device_id) const {
    auto* device = get_device(device_id);
    TT_ASSERT(device != nullptr, "DeviceManager does not contain device {}", device_id);
    TT_ASSERT(device->is_initialized(), "Device {} is not initialized", device_id);
    return device;
}

std::vector<IDevice*> DeviceManager::get_all_active_devices() const {
    std::vector<IDevice*> user_devices;
    for (const auto& device : this->devices_) {
        if (device && device->is_initialized()) {
            user_devices.push_back(device.get());
        }
    }
    return user_devices;
}

// Get all active device ids
// This function needs to be thread-safe as its called in inspector::data on a different thread
std::vector<ChipId> DeviceManager::get_all_active_device_ids() const {
    std::vector<ChipId> device_ids;
    std::lock_guard<std::mutex> lock(this->lock_);
    device_ids.reserve(this->devices_.size());
    for (const auto& device : devices_) {
        if (device && device->is_initialized()) {
            device_ids.emplace_back(device->id());
        }
    }
    return device_ids;
}

// Get all command queue event infos for all active devices
// The key is the device id and the value is a vector of event ids for each command queue
// This function needs to be thread-safe as its called in inspector::data on a different thread
std::unordered_map<ChipId, std::vector<uint32_t>> DeviceManager::get_all_command_queue_event_infos() const {
    std::unordered_map<ChipId, std::vector<uint32_t>> cq_to_event_by_device;
    std::lock_guard<std::mutex> lock(lock_);
    cq_to_event_by_device.reserve(devices_.size());
    for (const auto& device : devices_) {
        if (device && device->is_initialized()) {
            auto& vec = cq_to_event_by_device[device->id()];
            const auto num_hw_cqs = device->num_hw_cqs();
            vec.resize(num_hw_cqs);
            for (size_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                const auto event_id = device->sysmem_manager().get_last_event(static_cast<uint8_t>(cq_id));
                vec[cq_id] = event_id;
            }
        }
    }
    return cq_to_event_by_device;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void DeviceManager::teardown_fd(const std::unordered_set<ChipId>& devices_to_close) {
    for (const auto& dev_id : devices_to_close) {
        // Device is still active at this point
        auto* dev = this->get_active_device(dev_id);
        if (!this->using_fast_dispatch_) {
            continue;
        }

        // Fabric-only devices were activated for routing but never had
        // dispatch (command queues) initialized.  Skip them.
        if (!dispatch_device_ids_.contains(dev_id)) {
            continue;
        }

        for (int cq_id = 0; cq_id < dev->num_hw_cqs(); cq_id++) {
            auto& cq = dev->command_queue(cq_id);
            if (cq.sysmem_manager().get_bypass_mode()) {
                cq.record_end();
            }
            cq.terminate();
        }
    }
}

bool DeviceManager::is_dispatch_firmware_active() const { return this->dispatch_firmware_active_; }

bool DeviceManager::close_device(ChipId device_id) {
    // Sync and close one device
    // Currently can only call this on mmio chips, once we split dispatch kernel shutdown
    // from device close, we can call this on remote devices too
    ZoneScoped;
    const auto& mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    std::vector<IDevice*> devices_to_close;
    for (const auto& mmio_controlled_device_id :
         tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id)) {
        auto* device = this->get_device(mmio_controlled_device_id);
        if (device && device->is_initialized()) {
            devices_to_close.push_back(device);
        }
    }
    return this->close_devices(devices_to_close);
}

bool DeviceManager::close_devices(const std::vector<IDevice*>& devices, bool /*skip_synchronize*/) {
    ZoneScoped;
    const bool lite_fabric_bootstrap_terminated =
        tt::tt_metal::MetalContext::instance().was_lite_fabric_bootstrap_terminated();

    // Ordered, because we need to shutdown tunnels from the farthest to the closest.
    std::vector<ChipId> devices_to_close;

    // Loop over all devices and add remote devices to devices_to_close
    // For Galaxy if an mmio device's tunnels are being closed, close the mmio device as well
    std::unordered_set<ChipId> mmio_devices_to_close;
    for (const auto& dev : devices) {
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_devices_to_close.contains(mmio_device_id)) {
            continue;
        }
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);

        // Collect devices that are in UMD tunnels
        std::unordered_set<ChipId> tunnel_devices;
        tunnel_devices.insert(mmio_device_id);
        for (const auto& t : tunnels_from_mmio) {
            for (uint32_t ts = 1; ts < t.size(); ts++) {
                tunnel_devices.insert(t[ts]);
            }
        }

        // Close N-hop devices first (farthest from MMIO, not in UMD tunnels)
        for (const auto& controlled_device_id :
             tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                 mmio_device_id)) {
            if (tunnel_devices.count(controlled_device_id) == 0 && this->is_device_active(controlled_device_id)) {
                devices_to_close.push_back(controlled_device_id);
            }
        }

        // iterate over all tunnels origination from this mmio device
        for (auto t : tunnels_from_mmio) {
            // iterate over all tunneled devices (tunnel stops) in this tunnel
            for (uint32_t ts = t.size() - 1; ts > 0; ts--) {
                if (this->is_device_active(t[ts])) {
                    devices_to_close.push_back(t[ts]);
                }
            }
        }
        devices_to_close.push_back(mmio_device_id);
        mmio_devices_to_close.insert(mmio_device_id);
    }

    const bool dispatch_firmware_was_active = this->using_fast_dispatch_ && this->dispatch_firmware_active_;
    if (dispatch_firmware_was_active) {
        // TODO(MO): Remove when legacy non-mesh device is removed
        for (const ChipId device_id : devices_to_close) {
            IDevice* device = get_active_device(device_id);
            if (lite_fabric_bootstrap_terminated && !device->is_mmio_capable()) {
                continue;
            }
            detail::ReadDeviceProfilerResults(device, ProfilerReadState::LAST_FD_READ);
        }
    }

    dispatch_firmware_active_ = false;
    if (dispatch_firmware_was_active) {
        // Once lite fabric is terminated, remote accesses are rebound to fabric routers.
        // Teardown still needs to stop dispatch cleanly before the final warm reset.
        teardown_fd(std::unordered_set<ChipId>(devices_to_close.begin(), devices_to_close.end()));
        // Terminate sent to each device. Wait for dispatch to finish. MMIO only to prevent clogging SD path.
        // Dispatch kernels internally have a sync at the end to ensure all credits are returned
        for (const auto& dev_id : devices_to_close) {
            auto* dev = get_active_device(dev_id);
            if (!dev->is_mmio_capable()) {
                continue;
            }

            auto dispatch_cores = tt::tt_metal::get_virtual_dispatch_cores(dev_id);
            tt::llrt::internal_::wait_until_cores_done(dev_id, dev_msgs::RUN_MSG_GO, dispatch_cores, 0);
        }

        // Process registered termination signals from topology
        for (const auto& dev_id : devices_to_close) {
            auto* dev = this->get_active_device(dev_id);
            // After lite-fabric teardown, remote L1 writes are routed through fabric
            // routers. Keep profiler readbacks disabled for remotes, but still send
            // write-only shutdown signals to remote relay muxes.
            const auto& info = tt::tt_metal::get_registered_termination_cores(dev_id);
            for (const auto& core_to_terminate : info) {
                std::vector<uint32_t> val{core_to_terminate.val};
                tt_metal::detail::WriteToDeviceL1(
                    dev, core_to_terminate.logical_core, core_to_terminate.address, val, core_to_terminate.core_type);
            }
            tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(dev_id);
        }
    }

    // Terminate fabric routers if not using fabric manager
    if (has_flag(
            tt::tt_metal::MetalContext::instance().get_fabric_manager(),
            tt_fabric::FabricManagerMode::TERMINATE_FABRIC)) {
        const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
        if (tt::tt_fabric::is_tt_fabric_config(fabric_config)) {
            const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
            const auto& fabric_context = control_plane.get_fabric_context();
            const auto& builder_ctx = fabric_context.get_builder_context();
            auto [termination_signal_address, signal] = builder_ctx.get_fabric_router_termination_address_and_signal();
            std::vector<uint32_t> termination_signal(1, signal);

            // Terminate fabric tensix configs (mux cores) if enabled
            // TODO: issue #26855, move the termination process to device
            bool tensix_config_enabled = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
                                         tt::tt_fabric::FabricTensixConfig::DISABLED;
            if (tensix_config_enabled) {
                const auto& tensix_config = builder_ctx.get_tensix_config();

                for (const auto& dev : this->get_all_active_devices()) {
                    if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
                        continue;
                    }

                    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
                    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dev->id());
                    const auto& active_fabric_eth_channels =
                        control_plane.get_active_fabric_eth_channels(fabric_node_id);

                    for (const auto& [eth_chan_id, direction] : active_fabric_eth_channels) {
                        auto core_id = tensix_config.get_core_id_for_channel(dev->id(), eth_chan_id);
                        auto [tensix_termination_address, tensix_signal] =
                            tensix_config.get_termination_address_and_signal(core_id);
                        std::vector<uint32_t> tensix_termination_signal(1, tensix_signal);
                        auto mux_core = tensix_config.get_core_for_channel(dev->id(), eth_chan_id);

                        tt_metal::detail::WriteToDeviceL1(
                            dev, mux_core, tensix_termination_address, tensix_termination_signal, CoreType::WORKER);
                    }

                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(dev->id());
                }
            }

            for (const auto& dev : this->get_all_active_devices()) {
                if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
                    continue;
                }

                auto master_router_logical_core =
                    tt::tt_metal::MetalContext::instance()
                        .get_cluster()
                        .get_soc_desc(dev->id())
                        .get_eth_core_for_channel(
                            builder_ctx.get_fabric_master_router_chan(dev->id()), CoordSystem::LOGICAL);
                tt_metal::detail::WriteToDeviceL1(
                    dev, master_router_logical_core, termination_signal_address, termination_signal, CoreType::ETH);
            }
        }
    }

    if (dispatch_firmware_was_active) {
        for (const ChipId device_id : devices_to_close) {
            IDevice* device = this->get_active_device(device_id);
            if (lite_fabric_bootstrap_terminated && !device->is_mmio_capable()) {
                continue;
            }
            detail::ReadDeviceProfilerResults(device, ProfilerReadState::ONLY_DISPATCH_CORES);
        }
    }

    detail::ProfilerSync(ProfilerSyncState::CLOSE_DEVICE);

    bool pass = true;
    for (const auto& dev_id : devices_to_close) {
        auto* dev = this->get_active_device(dev_id);
        pass &= dev->close();
    }

    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);

    if (lite_fabric_bootstrap_terminated) {
        std::sort(devices_to_close.begin(), devices_to_close.end());
        devices_to_close.erase(std::unique(devices_to_close.begin(), devices_to_close.end()), devices_to_close.end());
        if (!devices_to_close.empty()) {
            std::vector<int> pci_device_ids;
            pci_device_ids.reserve(devices_to_close.size());
            for (const auto& dev_id : devices_to_close) {
                pci_device_ids.push_back(static_cast<int>(dev_id));
            }
            tt::umd::WarmReset::warm_reset(pci_device_ids);
        }
    }

    if (getDeviceProfilerState()) {
        // Device profiling data is dumped here instead of MetalContext::teardown() because MetalContext::teardown() is
        // called as a std::atexit() function, and ProfilerStateManager::cleanup_device_profilers() cannot be safely
        // called from a std::atexit() function because it creates new threads, which is unsafe during program
        // termination.
        tt::tt_metal::MetalContext::instance().profiler_state_manager()->cleanup_device_profilers();
    }

    return pass;
}

DeviceManager::~DeviceManager() {
    for (const auto& dev : this->devices_) {
        if (dev != nullptr and dev->is_initialized()) {
            // TODO: #13876, Was encountering issues with the DispatchMemMap being destroyed before the DeviceManager
            // destructor, which leads to device->close() hitting asserts. We need to move the ownership of
            // DispatchMemMap to the device, so it doesn't go out of scope before the device is closed.
            dev->close();
        }
    }
    this->devices_.clear();
}
}  // namespace tt_metal
}  // namespace tt
