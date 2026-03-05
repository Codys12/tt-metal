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

        // Must open all reachable devices in cluster to use fabric.
        // Exclude N-hop chips discovered via BFS that don't have lite fabric tunnels yet.
        if (any_remote_devices) {
            device_ids_to_open.clear();
            for (int id = 0; id < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); ++id) {
                if (!tt::tt_metal::MetalContext::instance().is_chip_unreachable(id)) {
                    device_ids_to_open.push_back(id);
                }
            }
        }
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

    if (has_flag(
            tt::tt_metal::MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        this->wait_for_fabric_router_sync(this->get_fabric_router_sync_timeout_ms());
    }
    log_trace(tt::LogMetal, "Fabric and Dispatch Firmware initialized");
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

    // Configure remote devices FIRST. Remote device configure_fabric() uses
    // WRITE_REG (through the lite fabric relay on ERISC1) to deassert ERISC0's
    // soft reset on remote ETH cores. MMIO configure_fabric() launches fabric
    // router kernels that overwrite the lite fabric relay on the MMIO-side ETH
    // cores, so remote devices must complete their WRITE_REG operations while
    // the relay is still running.
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (auto* dev : compiled_devices) {
        if (!cluster.mmio_chip_ids().count(dev->id())) {
            dev->configure_fabric();
        }
    }
    for (auto* dev : compiled_devices) {
        if (cluster.mmio_chip_ids().count(dev->id())) {
            dev->configure_fabric();
        }
    }
}

void DeviceManager::initialize_active_devices() {
    const auto& active_devices = this->get_all_active_devices();

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
            // then MMIO devices (which overwrite the lite fabric relay with fabric routers).
            init_fabric(active_devices);

            // Now that fabric routers are running on MMIO-side ETH cores, update UMD
            // bindings to exclude channels reserved for fabric routing.  This must happen
            // AFTER init_fabric because remote device configure_fabric() needs UMD bindings
            // to write L1 data and perform l1_barrier through the lite fabric relay.
            tt::tt_metal::MetalContext::instance().update_lite_fabric_bindings_for_fabric_routers();
            log_info(tt::LogMetal, "Fabric Initialized with config {}", fabric_config);
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
        return;
    }

    // When fabric is active, only configure dispatch for devices that are mapped in the control plane.
    // With lite fabric, the topology mapper may map only a subset of physical chips to the fabric mesh.
    const bool check_chip_mapped = tt_fabric::is_tt_fabric_config(fabric_config);
    const auto* control_plane_ptr =
        check_chip_mapped ? &tt::tt_metal::MetalContext::instance().get_control_plane() : nullptr;
    auto is_dispatch_device = [&](ChipId id) { return !check_chip_mapped || control_plane_ptr->is_chip_mapped(id); };

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
            for (const auto& tunnel : tunnels_from_mmio) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    if (!is_dispatch_device(mmio_controlled_device_id)) {
                        continue;
                    }
                    auto* device = get_device(mmio_controlled_device_id);
                    populate_cq_static_args(device);
                }
            }
        }
    }

    // Create command queue programs
    log_info(tt::LogMetal, "DEBUG: FD init: creating CQ programs");
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

        create_cq_program(dev);
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        if (not this->skip_remote_devices_) {
            for (const auto& tunnel : tunnels_from_mmio) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    if (!is_dispatch_device(mmio_controlled_device_id)) {
                        continue;
                    }
                    auto* device = get_device(mmio_controlled_device_id);
                    create_cq_program(device);
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
        // Diagnostic barrier: check if d2h desync already present after MMIO CQ init
        for (const auto& tunnel : tunnels_from_mmio) {
            for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                uint32_t rid = tunnel[ts];
                log_info(tt::LogMetal, "DEBUG: FD init: post-MMIO-CQ barrier for remote device {}", rid);
                tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(rid);
                log_info(tt::LogMetal, "DEBUG: FD init: post-MMIO-CQ barrier for remote device {} passed", rid);
            }
        }
        if (not this->skip_remote_devices_) {
            for (const auto& tunnel : tunnels_from_mmio) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnel.size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnel[ts];
                    log_info(
                        tt::LogMetal, "DEBUG: FD init: CQ device init for tunnel device {}", mmio_controlled_device_id);
                    if (!is_dispatch_device(mmio_controlled_device_id)) {
                        log_info(
                            tt::LogMetal, "DEBUG: FD init: skipping non-dispatch device {}", mmio_controlled_device_id);
                        continue;
                    }
                    auto* device = get_device(mmio_controlled_device_id);
                    if (!device) {
                        log_info(
                            tt::LogMetal, "DEBUG: FD init: device {} not found, skipping", mmio_controlled_device_id);
                        continue;
                    }
                    log_info(
                        tt::LogMetal, "DEBUG: FD init: pre-CQ barrier for tunnel device {}", mmio_controlled_device_id);
                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(mmio_controlled_device_id);
                    log_info(
                        tt::LogMetal,
                        "DEBUG: FD init: pre-CQ barrier for tunnel device {} passed",
                        mmio_controlled_device_id);
                    device->init_command_queue_device();
                    log_info(tt::LogMetal, "Command Queue initialized on Device {}", device->id());
                }
            }
        }
    }
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
        for (const auto& device_id : device_ids) {
            // Get list of all devices in the cluster connected to the passed in device_ids
            const auto& mmio_device_id =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
            for (const auto& mmio_controlled_device_id :
                 tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                     mmio_device_id)) {
                // Skip unreachable N-hop chips discovered via BFS
                if (tt::tt_metal::MetalContext::instance().is_chip_unreachable(mmio_controlled_device_id)) {
                    continue;
                }
                devices_to_activate.insert(mmio_controlled_device_id);
            }
        }
    }

    for (const auto& device_id : devices_to_activate) {
        if (not this->is_device_active(device_id)) {
            this->activate_device(device_id);
        }
    }

    // Only can launch Fabric if all reachable devices are active
    tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        for (int i = 0; i < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); i++) {
            // Skip unreachable N-hop chips that don't have Device objects
            if (tt::tt_metal::MetalContext::instance().is_chip_unreachable(i)) {
                continue;
            }
            // Fabric currently requires all reachable devices to be active
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
        if (tt_fabric::is_tt_fabric_config(fabric_config)) {
            // Only create FD kernels for devices that are mapped in the control plane.
            // With lite fabric, the topology mapper may map only a subset of physical chips.
            const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
            std::set<ChipId> mapped_devices;
            for (const auto& id : devices_to_activate) {
                if (control_plane.is_chip_mapped(id)) {
                    mapped_devices.insert(id);
                }
            }
            populate_fd_kernels(mapped_devices, this->num_hw_cqs_);
        } else {
            populate_fd_kernels(devices_to_activate, this->num_hw_cqs_);
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

    auto wait_for_handshake = [&](IDevice* dev) {
        if (!dev) {
            TT_THROW("Fabric router sync on null device. All devices must be opened for Fabric.");
        }
        if (builder_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
            return;
        }

        const auto master_router_chan = builder_context.get_fabric_master_router_chan(dev->id());
        const auto master_router_logical_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(dev->id()).get_eth_core_for_channel(
                master_router_chan, CoordSystem::LOGICAL);

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
        for (const auto& tunnel : tunnels_from_mmio) {
            // Need to poll on devices from farthest to the closest.
            for (auto j = tunnel.size() - 1; j > 0; j--) {
                if (!control_plane.is_chip_mapped(tunnel[j])) {
                    continue;
                }
                wait_for_handshake(get_device(tunnel[j]));
            }
        }

        wait_for_handshake(dev);
    }
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
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    log_debug(tt::LogMetal, "Tunnel {} Device {} Tunnel Stop: {}", t, mmio_controlled_device_id, ts);
                    auto* device = get_device(mmio_controlled_device_id);
                    this->initialize_host(device);
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

    // TODO(MO): Remove when legacy non-mesh device is removed
    for (const ChipId device_id : devices_to_close) {
        IDevice* device = get_active_device(device_id);
        detail::ReadDeviceProfilerResults(device, ProfilerReadState::LAST_FD_READ);
    }

    dispatch_firmware_active_ = false;
    teardown_fd(std::unordered_set<ChipId>(devices_to_close.begin(), devices_to_close.end()));
    // Terminate sent to each device. Wait for dispatch to finish. MMIO only to prevent clogging SD path.
    // Dispatch kernels internally have a sync at the end to ensure all credits are returned
    for (const auto& dev_id : devices_to_close) {
        auto* dev = get_active_device(dev_id);
        if (!dev->is_mmio_capable() || !using_fast_dispatch_) {
            continue;
        }

        auto dispatch_cores = tt::tt_metal::get_virtual_dispatch_cores(dev_id);
        tt::llrt::internal_::wait_until_cores_done(dev_id, dev_msgs::RUN_MSG_GO, dispatch_cores, 0);
    }

    // Process registered termination signals from topology
    for (const auto& dev_id : devices_to_close) {
        auto* dev = this->get_active_device(dev_id);
        const auto& info = tt::tt_metal::get_registered_termination_cores(dev_id);
        for (const auto& core_to_terminate : info) {
            std::vector<uint32_t> val{core_to_terminate.val};
            tt_metal::detail::WriteToDeviceL1(
                dev, core_to_terminate.logical_core, core_to_terminate.address, val, core_to_terminate.core_type);
        }
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(dev_id);
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

    for (const ChipId device_id : devices_to_close) {
        IDevice* device = this->get_active_device(device_id);
        detail::ReadDeviceProfilerResults(device, ProfilerReadState::ONLY_DISPATCH_CORES);
    }

    detail::ProfilerSync(ProfilerSyncState::CLOSE_DEVICE);

    bool pass = true;
    for (const auto& dev_id : devices_to_close) {
        auto* dev = this->get_active_device(dev_id);
        pass &= dev->close();
    }

    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);

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
