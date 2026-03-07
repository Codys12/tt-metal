// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <tuple>
#include <mutex>
#include <future>
#include <vector>
#include <unordered_set>

#include <enchantum/enchantum.hpp>
#include <tracy/Tracy.hpp>

#include "metal_context.hpp"
#include "core_coord.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "allocator/l1_banking_allocator.hpp"
#include "debug/dprint_server.hpp"
#include "debug/inspector/inspector.hpp"

#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "debug/inspector/data.hpp"
#include "debug/noc_logging.hpp"
#include "debug/watcher_server.hpp"
#include "dispatch/topology.hpp"
#include "profiler/profiler_state_manager.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/llrt.hpp"
#include <experimental/fabric/control_plane.hpp>
#include "device/device_manager.hpp"
#include <distributed_context.hpp>
#include <experimental/fabric/fabric.hpp>

#include <tt_metal.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/chip/remote_chip.hpp>
#include "lite_fabric/hal/lite_fabric_hal.hpp"
#include <umd/device/arch/blackhole_implementation.hpp>
#include "tt_metal/lite_fabric/hw/inc/blackhole/lf_dev_mem_map.hpp"
#include "lite_fabric/host_util.hpp"
#include "dispatch/data_collector.hpp"

#include <dispatch/dispatch_query_manager.hpp>
#include <dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>
#include <dispatch/dispatch_mem_map.hpp>
#include "common/executor.hpp"

namespace tt::tt_metal {

namespace {
// Helper function to validate worker_l1_size, also updates it if it's 0.
void validate_worker_l1_size(size_t& worker_l1_size, Hal& hal) {
    if (worker_l1_size == 0) {
        worker_l1_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    }
    size_t max_worker_l1_size = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                                hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) -
                                hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
    TT_FATAL(
        worker_l1_size <= max_worker_l1_size,
        "Worker L1 size {} is larger than max size {}",
        worker_l1_size,
        max_worker_l1_size);
}

// ETH core L1 register addresses for Blackhole SYSENG boot results.
// These are populated by the syseng firmware during POR, before Metal starts.
constexpr uint64_t ETH_PORT_STATUS_ADDR = 0x7CC04;           // uint8_t: port_status_e
constexpr uint64_t ETH_REMOTE_BOARD_ID_HI_ADDR = 0x7CFE4;    // uint32_t
constexpr uint64_t ETH_REMOTE_BOARD_ID_LO_ADDR = 0x7CFE8;    // uint32_t
constexpr uint64_t ETH_REMOTE_ASIC_LOCATION_ADDR = 0x7CFE1;  // uint8_t
constexpr uint64_t ETH_REMOTE_ETH_ID_ADDR = 0x7CFE2;         // uint8_t (remote channel)
constexpr uint8_t PORT_UP = 1;                               // blackhole::port_status_e::PORT_UP

// Compute ASIC unique ID from board_id and asic_location (matches UMD's mangle_asic_id).
inline uint64_t compute_asic_uid(uint64_t board_id, uint8_t asic_location) {
    return (board_id << 5) | (asic_location & 0x1F);
}

// Info about a chip discovered during BFS.
struct DiscoveredChipInfo {
    ChipId chip_id;                   // Newly assigned chip ID
    ChipId intermediate_chip;         // Chip via which this was discovered
    uint32_t downstream_eth_chan;     // ETH channel index on intermediate chip
    tt_xy_pair downstream_core_noc0;  // NOC0 coords of downstream ETH core on intermediate chip
    uint8_t remote_eth_id;            // ETH channel on the newly discovered chip
};

// BFS discovery of chips beyond 1-hop from MMIO via lite fabric.
// Reads each frontier chip's ETH core registers (populated by syseng FW at POR)
// to find trained links to undiscovered chips.  Adds discovered chips to the UMD
// cluster descriptor and creates RemoteChip objects so that subsequent Metal
// phases can communicate with them.
//
// Returns a vector of DiscoveredChipInfo with connection info for each new chip.
std::vector<DiscoveredChipInfo> discover_nhop_chips(
    Cluster& cluster,
    const std::set<ChipId>& frontier_chips,
    int max_hops,
    const std::set<ChipId>& reachable_chips = {}) {
    auto* driver = cluster.get_driver().get();
    auto* cluster_desc = cluster.get_cluster_desc();

    // Build set of known ASIC UIDs.
    std::set<uint64_t> known_uids;
    for (const auto& [chip_id, uid] : cluster_desc->get_chip_unique_ids()) {
        known_uids.insert(uid);
    }

    // Map from ASIC UID to chip_id for reverse lookup.
    auto uid_to_chip_id = [&](uint64_t uid) -> std::optional<ChipId> {
        for (const auto& [cid, u] : cluster_desc->get_chip_unique_ids()) {
            if (u == uid) {
                return cid;
            }
        }
        return std::nullopt;
    };

    ChipId next_chip_id = 0;
    for (ChipId cid : cluster_desc->get_all_chips()) {
        next_chip_id = std::max(next_chip_id, cid + 1);
    }
    // Also account for chips in the UMD driver that may have been removed from
    // the descriptor (e.g., unreachable chips from a previous init cycle).
    for (ChipId cid : driver->get_target_device_ids()) {
        next_chip_id = std::max(next_chip_id, cid + 1);
    }

    std::vector<DiscoveredChipInfo> all_discovered;
    std::set<ChipId> current_frontier = frontier_chips;

    // Sanity check: read boot_results from the MMIO chip to verify addresses work
    {
        ChipId mmio_chip = *cluster.mmio_chip_ids().begin();
        const auto& mmio_soc = cluster.get_soc_desc(mmio_chip);
        auto mmio_eth = mmio_soc.get_cores(CoreType::ETH, CoordSystem::TRANSLATED);
        log_info(
            tt::LogMetal,
            "BFS sanity: reading boot_results from MMIO chip {} ({} ETH cores)",
            mmio_chip,
            mmio_eth.size());
        for (size_t i = 0; i < mmio_eth.size(); i++) {
            const auto& c = mmio_eth[i];
            umd::CoreCoord cc(c.x, c.y, CoreType::ETH, CoordSystem::TRANSLATED);
            uint32_t ps = 0;
            try {
                driver->read_from_device(&ps, mmio_chip, cc, ETH_PORT_STATUS_ADDR, sizeof(ps));
                log_info(
                    tt::LogMetal,
                    "BFS sanity: MMIO chip {} chan {} core ({},{}) port_status_word={:#x}",
                    mmio_chip,
                    i,
                    c.x,
                    c.y,
                    ps);
            } catch (const std::exception& e) {
                log_info(tt::LogMetal, "BFS sanity: MMIO chip {} chan {} exception: {}", mmio_chip, i, e.what());
            }
        }
    }

    for (int hop = 0; hop < max_hops && !current_frontier.empty(); hop++) {
        std::set<ChipId> next_frontier;

        for (ChipId chip_id : current_frontier) {
            const auto& soc_desc = cluster.get_soc_desc(chip_id);
            // Use PHYSICAL coordinates for ETH core reads on remote chips.
            // Remote chips may not have NOC translation tables programmed yet
            // (that happens in Phase 3), so TRANSLATED coordinates would
            // route NOC reads to wrong physical tiles.  PHYSICAL coordinates
            // match the actual NOC0 grid positions and work regardless of
            // whether translation is active.
            auto eth_cores = soc_desc.get_cores(CoreType::ETH, CoordSystem::NOC0);

            log_info(
                tt::LogMetal, "BFS: scanning chip {} with {} ETH cores (PHYSICAL coords)", chip_id, eth_cores.size());

            for (size_t chan = 0; chan < eth_cores.size(); chan++) {
                const auto& core = eth_cores[chan];
                umd::CoreCoord core_coord(core.x, core.y, CoreType::ETH, CoordSystem::NOC0);

                // Read port status — read as uint32_t since the enum is 4 bytes
                // in the boot_results_t structure, then extract the low byte.
                uint32_t port_status_word = 0;
                try {
                    driver->read_from_device(
                        &port_status_word, chip_id, core_coord, ETH_PORT_STATUS_ADDR, sizeof(port_status_word));
                } catch (const std::exception& e) {
                    log_info(
                        tt::LogMetal,
                        "BFS: chip {} chan {} core ({},{}) read exception: {}",
                        chip_id,
                        chan,
                        core.x,
                        core.y,
                        e.what());
                    continue;
                }
                uint8_t port_status = static_cast<uint8_t>(port_status_word & 0xFF);

                if (port_status != PORT_UP) {
                    log_info(
                        tt::LogMetal,
                        "BFS: chip {} chan {} core ({},{}) port_status_word={:#x} port_status={} (not UP)",
                        chip_id,
                        chan,
                        core.x,
                        core.y,
                        port_status_word,
                        port_status);
                    continue;
                }

                // Read remote chip info
                uint32_t board_id_hi = 0, board_id_lo = 0;
                uint8_t asic_location = 0, remote_eth_id = 0;
                try {
                    driver->read_from_device(
                        &board_id_hi, chip_id, core_coord, ETH_REMOTE_BOARD_ID_HI_ADDR, sizeof(board_id_hi));
                    driver->read_from_device(
                        &board_id_lo, chip_id, core_coord, ETH_REMOTE_BOARD_ID_LO_ADDR, sizeof(board_id_lo));
                    driver->read_from_device(
                        &asic_location, chip_id, core_coord, ETH_REMOTE_ASIC_LOCATION_ADDR, sizeof(asic_location));
                    driver->read_from_device(
                        &remote_eth_id, chip_id, core_coord, ETH_REMOTE_ETH_ID_ADDR, sizeof(remote_eth_id));
                } catch (const std::exception& e) {
                    log_info(
                        tt::LogMetal,
                        "BFS: chip {} chan {} core ({},{}) remote info read exception: {}",
                        chip_id,
                        chan,
                        core.x,
                        core.y,
                        e.what());
                    continue;
                }

                uint64_t remote_board_id = (static_cast<uint64_t>(board_id_hi) << 32) | board_id_lo;
                uint64_t remote_uid = compute_asic_uid(remote_board_id, asic_location);

                log_info(
                    tt::LogMetal,
                    "BFS: chip {} chan {} core ({},{}) port_status=UP remote_uid={:#x} "
                    "board_id={:#x} asic_loc={} remote_eth={}",
                    chip_id,
                    chan,
                    core.x,
                    core.y,
                    remote_uid,
                    remote_board_id,
                    asic_location,
                    remote_eth_id);

                if (known_uids.count(remote_uid)) {
                    // Already known chip — record the connection if missing
                    auto existing_id = uid_to_chip_id(remote_uid);
                    if (existing_id.has_value()) {
                        log_info(
                            tt::LogMetal,
                            "BFS: chip {} chan {} -> already known as chip {}",
                            chip_id,
                            chan,
                            *existing_id);
                        // Only add the connection if neither side already has an
                        // entry for this channel.  Topology discovery populates
                        // ethernet_connections using patched (logical) channel
                        // numbers.  BFS uses raw remote_eth_id (physical port).
                        // Unconditionally calling add_ethernet_connection would
                        // overwrite existing entries with mismatched numbering,
                        // corrupting the tunnel-to-chip mapping on reinit.
                        {
                            const auto& existing_conns = cluster_desc->get_ethernet_connections();
                            bool a_exists = existing_conns.count(chip_id) &&
                                            existing_conns.at(chip_id).count(static_cast<uint32_t>(chan));
                            bool b_exists = existing_conns.count(*existing_id) &&
                                            existing_conns.at(*existing_id).count(remote_eth_id);
                            if (!a_exists && !b_exists) {
                                cluster_desc->add_ethernet_connection(
                                    chip_id, static_cast<uint32_t>(chan), *existing_id, remote_eth_id);
                            }
                        }

                        // If this chip is known but NOT yet reachable via lite fabric,
                        // treat it as a discovered chip so the BFS sets up n-hop tunnels.
                        // Skip if it's an MMIO chip or already reachable.
                        if (!reachable_chips.empty() && !reachable_chips.count(*existing_id) &&
                            !next_frontier.count(*existing_id)) {
                            log_info(
                                tt::LogMetal,
                                "BFS: chip {} chan {} -> chip {} is known but unreachable, adding to frontier",
                                chip_id,
                                chan,
                                *existing_id);
                            all_discovered.push_back(DiscoveredChipInfo{
                                .chip_id = *existing_id,
                                .intermediate_chip = chip_id,
                                .downstream_eth_chan = static_cast<uint32_t>(chan),
                                .downstream_core_noc0 = tt_xy_pair(core.x, core.y),
                                .remote_eth_id = remote_eth_id,
                            });
                            next_frontier.insert(*existing_id);
                        }
                    }
                    continue;
                }

                // New chip discovered!
                ChipId new_chip_id = next_chip_id++;
                known_uids.insert(remote_uid);

                ChipId gateway_id = cluster_desc->get_closest_mmio_capable_chip(chip_id);

                log_info(
                    tt::LogMetal,
                    "BFS discovery: found new chip {} (uid={:#x}) via chip {} ETH channel {} "
                    "(board_id={:#x} asic_loc={} remote_eth={} gateway={})",
                    new_chip_id,
                    remote_uid,
                    chip_id,
                    chan,
                    remote_board_id,
                    asic_location,
                    remote_eth_id,
                    gateway_id);

                // Register chip in the cluster descriptor with proxy metadata from gateway
                cluster_desc->register_chip(
                    new_chip_id,
                    remote_uid,
                    ARCH::BLACKHOLE,
                    gateway_id,
                    cluster_desc->get_board_type(gateway_id),
                    cluster_desc->get_noc_translation_table_en().at(gateway_id),
                    cluster_desc->get_harvesting_masks(gateway_id));

                // Record ethernet connections (bidirectional)
                cluster_desc->add_ethernet_connection(chip_id, static_cast<uint32_t>(chan), new_chip_id, remote_eth_id);

                // Create a RemoteChip in UMD with the gateway's ETH channels.
                // On reinit, the chip may already exist in the driver (if it was
                // discovered in a previous init cycle but later removed from the
                // cluster descriptor as unreachable).  In that case, reuse the
                // existing driver entry.
                if (!driver->get_target_device_ids().count(new_chip_id)) {
                    auto gateway_channels = cluster_desc->get_active_eth_channels(gateway_id);
                    if (gateway_channels.empty()) {
                        gateway_channels = cluster_desc->get_idle_eth_channels(gateway_id);
                    }
                    auto proxy_soc_desc = driver->get_soc_descriptor(gateway_id);
                    driver->register_remote_chip(new_chip_id, gateway_id, gateway_channels, proxy_soc_desc);

                    // Add Metal-layer SOC descriptor (proxy from gateway)
                    cluster.add_soc_descriptor(
                        new_chip_id,
                        metal_SocDescriptor(
                            driver->get_soc_descriptor(new_chip_id), cluster_desc->get_board_type(new_chip_id)));
                } else {
                    log_info(
                        tt::LogMetal, "BFS: chip {} already exists in driver, reusing existing entry", new_chip_id);
                }

                all_discovered.push_back(DiscoveredChipInfo{
                    .chip_id = new_chip_id,
                    .intermediate_chip = chip_id,
                    .downstream_eth_chan = static_cast<uint32_t>(chan),
                    .downstream_core_noc0 = tt_xy_pair(core.x, core.y),
                    .remote_eth_id = remote_eth_id,
                });
                next_frontier.insert(new_chip_id);
            }
        }

        current_frontier = next_frontier;
        log_info(
            tt::LogMetal,
            "BFS discovery hop {}: discovered {} new chips, next frontier size {}",
            hop + 1,
            next_frontier.size(),
            current_frontier.size());
    }

    return all_discovered;
}

// Construct compute-only distributed context by filtering out switch meshes
std::shared_ptr<distributed::multihost::DistributedContext> construct_compute_only_distributed_context(
    MetalContext& metal_context) {
    const auto& global_context = distributed::multihost::DistributedContext::get_current_world();
    if (*global_context->size() == 1) {
        return global_context;
    }

    // Get all compute mesh IDs (excludes switches) from control plane mesh graph
    const auto& mesh_graph = metal_context.get_control_plane().get_mesh_graph();

    // If there are no switch meshes, return the global context directly
    if (mesh_graph.get_switch_ids().empty()) {
        return global_context;
    }

    const auto& compute_mesh_ids = mesh_graph.get_mesh_ids();

    // Get global logical bindings to map ranks to mesh IDs
    const auto& global_logical_bindings = metal_context.get_control_plane().get_global_logical_bindings();

    // Collect all MPI ranks for compute meshes only
    std::unordered_set<int> compute_mpi_ranks;
    for (const auto& [rank, mesh_binding] : global_logical_bindings) {
        const auto& [mesh_id, _] = mesh_binding;
        // Check if this mesh_id is a compute mesh (not a switch)
        if (std::find(compute_mesh_ids.begin(), compute_mesh_ids.end(), mesh_id) != compute_mesh_ids.end()) {
            compute_mpi_ranks.insert(rank.get());
        }
    }

    // If no compute meshes found, fall back to host_local_context
    if (compute_mpi_ranks.empty()) {
        TT_THROW("No compute meshes found in mesh graph.");
    }

    // Convert to sorted vector for create_sub_context
    std::vector<int> compute_ranks_vec(compute_mpi_ranks.begin(), compute_mpi_ranks.end());
    std::sort(compute_ranks_vec.begin(), compute_ranks_vec.end());

    // Check if current rank is in compute ranks
    int current_rank = *global_context->rank();
    bool is_current_rank_in_compute =
        std::find(compute_ranks_vec.begin(), compute_ranks_vec.end(), current_rank) != compute_ranks_vec.end();

    // If current rank is not in compute ranks (e.g., host only has switches), return host_local_context
    if (!is_current_rank_in_compute) {
        return metal_context.get_control_plane().get_host_local_context();
    }

    // Create sub-context with only compute mesh ranks
    return global_context->create_sub_context(compute_ranks_vec);
}

}  // namespace

void MetalContext::initialize_device_manager(
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    const tt_metal::DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size,
    bool init_profiler,
    bool initialize_fabric_and_dispatch_fw) {
    initialize(dispatch_core_config, num_hw_cqs, {l1_bank_remap.begin(), l1_bank_remap.end()}, worker_l1_size);
    device_manager_->initialize(
        device_ids,
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        l1_bank_remap,
        worker_l1_size,
        init_profiler,
        initialize_fabric_and_dispatch_fw);
}

void MetalContext::initialize(
    const DispatchCoreConfig& dispatch_core_config,
    uint8_t num_hw_cqs,
    const BankMapping& l1_bank_remap,
    size_t worker_l1_size,
    bool minimal) {
    ZoneScoped;

    log_info(
        tt::LogMetal,
        "DEBUG: MetalContext::initialize() called, initialized_={}, force_reinit_={}",
        initialized_,
        force_reinit_);

    if (cluster_->get_target_device_type() == tt::TargetDevice::Mock) {
        TT_THROW(
            "Mock cluster cannot be initialized because there is no device. "
            "Mock clusters are only supported for testing control plane initialization without a device."
            "Please unset the TT_METAL_MOCK_CLUSTER_DESC_PATH environment variable.");
    }

    // Workaround for galaxy, need to always re-init
    if (rtoptions_.get_force_context_reinit() or cluster_->is_galaxy_cluster()) {
        force_reinit_ = true;
    }
    // Settings that affect FW build can also trigger a re-initialization
    const size_t fw_compile_hash = std::hash<std::string>{}(rtoptions_.get_compile_hash_string());
    validate_worker_l1_size(worker_l1_size, *hal_);
    if (initialized_) {
        if (dispatch_core_config_ != dispatch_core_config or num_hw_cqs != num_hw_cqs_ or
            worker_l1_size_ != worker_l1_size or l1_bank_remap != l1_bank_remap_ or
            fw_compile_hash != fw_compile_hash_) {
            log_warning(tt::LogAlways, "Closing and re-initializing MetalContext with new parameters.");
            teardown();
        } else {
            // Re-init request with the same parameters, do nothing unless force re-init requested.
            if (force_reinit_) {
                force_reinit_ = false;
                log_debug(
                    tt::LogAlways,
                    "Closing and re-initializing MetalContext with same parameters due to force_reinit flag.");
                teardown();
            } else {
                return;
            }
        }
    }

    // Clear force_reinit_ unconditionally before starting the full init.
    // set_fabric_config() sets this flag, but when called before the very
    // first MetalContext::initialize() (initialized_=false), the if(initialized_)
    // guard above is skipped and force_reinit_ is never cleared.  Without this,
    // the second initialize() call from initialize_device_manager() (which is
    // idempotent and should be a no-op) sees initialized_=true, force_reinit_=true
    // and tears down everything, forcing Phase 2b to run twice.
    force_reinit_ = false;

    initialized_ = true;
    dispatch_core_config_ = dispatch_core_config;
    num_hw_cqs_ = num_hw_cqs;
    worker_l1_size_ = worker_l1_size;
    l1_bank_remap_ = l1_bank_remap;
    fw_compile_hash_ = fw_compile_hash;
    std::uint32_t max_alignment = std::max(hal_->get_alignment(HalMemType::DRAM), hal_->get_alignment(HalMemType::L1));
    worker_l1_unreserved_start_ = tt::align(
        hal_->get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal_->get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size_,
        max_alignment);

    // Initialize inspector
    inspector_data_ = Inspector::initialize();
    // Set fw_compile_hash for Inspector RPC build environment info
    Inspector::set_build_env_fw_compile_hash(fw_compile_hash);

    // Reset timeout detection state
    dispatch_timeout_detection_processed_ = false;
    unreachable_chip_ids_.clear();
    remote_fabric_eth_channels_.clear();
    downstream_sender_cores_.clear();

    // Initialize dispatch state
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config, num_hw_cqs);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs);
    // Need DispatchMemMap for both dispatch core types
    tt_metal::DispatchSettings::initialize(*cluster_);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::WORKER)] =
        std::make_unique<DispatchMemMap>(CoreType::WORKER, num_hw_cqs);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::ETH)] =
        std::make_unique<DispatchMemMap>(CoreType::ETH, num_hw_cqs);
    // Initialize debug servers. Attaching individual devices done below
    if (rtoptions_.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        TT_FATAL(!rtoptions_.get_profiler_enabled(), "Both DPRINT and Profiler cannot be enabled at the same time.");
        rtoptions_.set_disable_dma_ops(true);  // DMA is not thread-safe
        dprint_server_ = std::make_unique<DPrintServer>(rtoptions_);
    }
    watcher_server_ =
        std::make_unique<WatcherServer>();  // Watcher server always created, since we use it to register kernels

    if (rtoptions_.get_profiler_enabled()) {
        profiler_state_manager_ = std::make_unique<ProfilerStateManager>();
    }

    data_collector_ = std::make_unique<DataCollector>();

    // Minimal setup, don't initialize FW/Dispatch/etc.
    if (minimal) {
        return;
    }

    // Clear state, build FW
    auto all_devices = cluster_->all_chip_ids();

    // For Blackhole multi-chip clusters, lite fabric must be running on the local MMIO chip's
    // ERISC1 before we can write to any remote chip registers.  Split initialization into:
    //   Phase 1 – FW build + device init + resets + FW launch for MMIO chips only
    //   Phase 2 – Start lite fabric (programs ERISC1, waits for READY)
    //   Phase 2b – Upgrade remote chip info (now reachable via lite fabric)
    //   Phase 3 – FW build + device init + resets + FW launch for remote chips
    // For all other cluster configurations we keep the original single-phase parallel init.
    const bool needs_lite_fabric =
        (cluster_->arch() == tt::ARCH::BLACKHOLE && cluster_->all_chip_ids().size() > cluster_->mmio_chip_ids().size());

    std::set<ChipId> remote_devices;
    if (needs_lite_fabric) {
        for (ChipId id : all_devices) {
            if (!cluster_->mmio_chip_ids().count(id)) {
                remote_devices.insert(id);
            }
        }
    }

    // Determine which devices to initialize before lite fabric is up.
    // Remote BH chips are not reachable yet, so defer them.
    const std::set<ChipId> initial_devices =
        needs_lite_fabric ? cluster_->mmio_chip_ids() : std::set<ChipId>(all_devices.begin(), all_devices.end());

    std::vector<std::shared_future<void>> futures;

    // Lambda: run FW builds and device init for a set of devices.
    // When sequential=true, devices are processed one at a time (required for remote
    // devices behind lite fabric whose tunnels are not safe for concurrent access).
    // When skip_eth=true, ETH core L1 clearing and launch message clearing are skipped
    // (remote devices: ERISC0 is not running, lite fabric ERISC1 is active on the link).
    auto build_and_init_devices = [&](const auto& device_set, bool sequential = false, bool skip_eth = false) {
        auto per_device_init = [this, fw_compile_hash, skip_eth](ChipId device_id) {
            log_info(tt::LogMetal, "build_and_init device {}: start", device_id);
            // Clear L1/DRAM if requested
            if (rtoptions_.get_clear_l1()) {
                log_info(tt::LogMetal, "build_and_init device {}: clear_l1_state (skip_eth={})", device_id, skip_eth);
                clear_l1_state(device_id, skip_eth);
            }
            if (rtoptions_.get_clear_dram()) {
                log_info(tt::LogMetal, "build_and_init device {}: clear_dram_state", device_id);
                clear_dram_state(device_id);
            }
            log_info(tt::LogMetal, "build_and_init device {}: get_device_aiclk", device_id);
            [[maybe_unused]] int ai_clk = cluster_->get_device_aiclk(device_id);
            log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", device_id, ai_clk);
            log_info(tt::LogMetal, "build_and_init device {}: generate_device_bank_to_noc_tables", device_id);
            generate_device_bank_to_noc_tables(device_id);
            log_info(tt::LogMetal, "build_and_init device {}: generate_worker_logical_to_virtual_map", device_id);
            generate_worker_logical_to_virtual_map(device_id);

            // Create build env for this device, and build FW if it's not built already
            log_info(tt::LogMetal, "build_and_init device {}: add_build_env", device_id);
            BuildEnvManager::get_instance().add_build_env(device_id, num_hw_cqs_);
            uint64_t fw_build_key =
                BuildEnvManager::get_instance().get_device_build_env(device_id).build_key() ^ fw_compile_hash;

            {
                std::lock_guard<std::mutex> lock(firmware_built_keys_mutex_);
                if (!firmware_built_keys_.contains(fw_build_key)) {
                    log_info(tt::LogMetal, "build_and_init device {}: build_firmware", device_id);
                    BuildEnvManager::get_instance().build_firmware(device_id);
                    firmware_built_keys_.insert(fw_build_key);
                }
            }

            // Clear the entire launch message ring buffer on ethernet cores before application firmware is
            // activated. This is required since ethernet cores context switch between application and routing
            // firmware. If ERISC application firmware is activated before the launch messages are cleared, it can
            // enter an undefined state by reading a corrupted launch message. Routing firmware will never run in
            // this case, causing UMD issued transactions to hang.
            //
            // Skipped for remote devices behind lite fabric: no ERISC0 base firmware is
            // running on remote ETH cores, and clearing the lite fabric ETH core's
            // launch messages is unnecessary (ERISC1 is actively servicing the link).
            if (!skip_eth) {
                log_info(tt::LogMetal, "build_and_init device {}: clear_launch_messages_on_eth_cores", device_id);
                clear_launch_messages_on_eth_cores(device_id);
            }
            log_info(tt::LogMetal, "build_and_init device {}: complete", device_id);
        };

        if (sequential) {
            for (ChipId device_id : device_set) {
                per_device_init(device_id);
            }
        } else {
            futures.clear();
            for (ChipId device_id : device_set) {
                futures.emplace_back(detail::async([per_device_init, device_id]() { per_device_init(device_id); }));
            }
            for (auto& fut : futures) {
                fut.get();
            }
        }
    };

    {
        ZoneScopedN("FW builds and Device Inits");
        futures.reserve(all_devices.size());
        build_and_init_devices(initial_devices);
    }

    // Populate FD topology across all devices
    if (rtoptions_.get_fast_dispatch()) {
        std::set<ChipId> all_devices_set(all_devices.begin(), all_devices.end());
        // TODO: enable this when dispatch init/teardown moves to MetalContext
        // populate_fd_kernels(all_devices_set, num_hw_cqs);
    }

    // Set internal routing for active ethernet cores, this is required for our FW to run.
    // Skip when needs_lite_fabric: on first init fabric_manager_ isn't set yet (no-op),
    // and on re-init this fires before lite fabric is relaunched, hanging on writes to
    // remote ETH cores.  Routing info will be set later by initialize_active_devices().
    if (!needs_lite_fabric &&
        has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        cluster_->set_internal_routing_info_for_ethernet_cores(true);
    }

    // Initialize debug tools, reset cores, init FW
    if (dprint_server_) {
        dprint_server_->attach_devices();
    }
    watcher_server_->init_devices();

    // Lambda: reset cores and launch FW for a set of devices
    auto launch_fw_for_devices = [&](const auto& device_set) {
        futures.clear();
        for (ChipId device_id : device_set) {
            futures.emplace_back(detail::async([this, device_id]() {
                log_info(tt::LogMetal, "launch_fw device {}: ClearNocData", device_id);
                ClearNocData(device_id);
                log_info(tt::LogMetal, "launch_fw device {}: reset_cores", device_id);
                reset_cores(device_id);
                log_info(tt::LogMetal, "launch_fw device {}: initialize_and_launch_firmware", device_id);
                initialize_and_launch_firmware(device_id);
                log_info(tt::LogMetal, "launch_fw device {}: complete", device_id);
            }));
        }
        for (auto& fut : futures) {
            fut.get();
        }
    };

    // Parallelize device initialization
    {
        ZoneScopedN("Resets and FW Launch");

        if (needs_lite_fabric) {
            // Phase 1: MMIO chips
            {
                ZoneScopedN("MMIO FW Launch");
                launch_fw_for_devices(cluster_->mmio_chip_ids());
            }

            // Phase 2: Initialize lite fabric (ERISC1 on MMIO chip talks to remote chip)
            {
                ZoneScopedN("Lite Fabric Init");
                lite_fabric_hal_ = lite_fabric::LiteFabricHal::create();
                lite_fabric::InitializeLiteFabric(lite_fabric_hal_);
            }

            // Filter remote_devices to only those reachable via lite fabric tunnels,
            // and bind each remote chip's UMD communication to the specific ETH
            // channel(s) that have active lite fabric tunnels to it.  Without this,
            // UMD round-robins through ALL active ETH channels on the MMIO chip,
            // which may send reads down a tunnel to the wrong remote chip and hang.
            {
                const auto& tunnels = lite_fabric_hal_->get_system_descriptor().tunnels_from_mmio;

                log_info(tt::LogMetal, "=== Lite fabric tunnel summary: {} surviving tunnels ===", tunnels.size());
                for (size_t i = 0; i < tunnels.size(); i++) {
                    const auto& t = tunnels[i];
                    log_info(
                        tt::LogMetal,
                        "  tunnel[{}]: mmio_chip={} mmio_core_logical={} -> connected_chip={} "
                        "connected_core_logical={}",
                        i,
                        t.mmio_id,
                        t.mmio_core_logical.str(),
                        t.connected_id,
                        t.connected_core_logical.str());
                }

                log_info(tt::LogMetal, "=== remote_devices before filter: {} devices ===", remote_devices.size());
                for (ChipId id : remote_devices) {
                    auto gateway = cluster_->get_cluster_desc()->get_closest_mmio_capable_chip(id);
                    log_info(tt::LogMetal, "  remote device {} (gateway mmio={})", id, gateway);
                }

                // Build per-remote-chip, per-mmio-chip set of MMIO-side ETH channels.
                // A remote chip may have tunnels from multiple MMIO chips; we only use
                // the channels from the MMIO chip that matches UMD's gateway for that
                // remote chip.
                // Key: (connected_id, mmio_id) -> set of ETH channel Y-coordinates
                std::map<std::pair<ChipId, ChipId>, std::set<uint32_t>> remote_chip_eth_channels;
                std::map<std::pair<ChipId, ChipId>, int> remote_chip_num_hops;
                for (const auto& tunnel : tunnels) {
                    remote_chip_eth_channels[{tunnel.connected_id, tunnel.mmio_id}].insert(tunnel.mmio_core_logical.y);
                    remote_chip_num_hops[{tunnel.connected_id, tunnel.mmio_id}] = tunnel.num_hops;
                }

                std::set<ChipId> reachable_remote_devices;
                for (ChipId chip_id : remote_devices) {
                    auto gateway = cluster_->get_cluster_desc()->get_closest_mmio_capable_chip(chip_id);
                    auto key = std::make_pair(chip_id, gateway);
                    auto it = remote_chip_eth_channels.find(key);

                    if (it == remote_chip_eth_channels.end()) {
                        // No tunnels from the matching MMIO chip for this remote device.
                        // Log which MMIO chips DO have tunnels (if any) for diagnostics.
                        std::vector<ChipId> other_mmios;
                        for (const auto& [k, v] : remote_chip_eth_channels) {
                            if (k.first == chip_id) {
                                other_mmios.push_back(k.second);
                            }
                        }
                        if (other_mmios.empty()) {
                            log_warning(tt::LogMetal, "  chip {}: no lite fabric tunnels found, skipping", chip_id);
                        } else {
                            log_warning(
                                tt::LogMetal,
                                "  chip {}: UMD gateway={} but tunnels only from mmio=[{}], skipping",
                                chip_id,
                                gateway,
                                fmt::join(other_mmios, ", "));
                        }
                        continue;
                    }

                    const auto& channels = it->second;
                    log_info(
                        tt::LogMetal,
                        "  chip {}: UMD gateway={}, channels=[{}]",
                        chip_id,
                        gateway,
                        fmt::join(channels, ", "));

                    reachable_remote_devices.insert(chip_id);

                    // Rebind UMD remote communication to only the channels with tunnels to this chip
                    auto* remote_chip = cluster_->get_driver()->get_remote_chip(chip_id);
                    remote_chip->set_remote_transfer_ethernet_cores(channels);
                    int nhops = remote_chip_num_hops[key];
                    if (nhops > 1) {
                        remote_chip->get_remote_communication()->set_num_hops(nhops);
                    }
                    log_info(
                        tt::LogMetal,
                        "Bound remote device {} to lite fabric ETH channel(s): [{}] ({} hops)",
                        chip_id,
                        fmt::join(channels, ", "),
                        nhops);
                }

                if (reachable_remote_devices.size() < remote_devices.size()) {
                    log_warning(
                        tt::LogMetal,
                        "Only {} of {} remote devices are reachable via lite fabric, skipping unreachable devices",
                        reachable_remote_devices.size(),
                        remote_devices.size());
                }
                log_info(
                    tt::LogMetal, "=== remote_devices after filter: {} devices ===", reachable_remote_devices.size());
                for (ChipId id : reachable_remote_devices) {
                    log_info(tt::LogMetal, "  reachable remote device {}", id);
                }
                remote_devices = std::move(reachable_remote_devices);
            }

            // Phase 2a: Upgrade remote BH chip firmware info providers now that lite fabric
            // is running and the remote ARC is accessible.  This replaces the proxy providers
            // (borrowed from the local gateway chip during topology discovery) with real ones
            // that read the actual harvesting masks, DRAM training status, and other chip
            // metadata from each remote chip.  Also refreshes the SocDescriptor and
            // ClusterDescriptor entries, then polls DRAM training to completion.
            //
            // This MUST run before BFS discovery because RemoteBlackholeTTDevice::read_from_device()
            // returns zeros until upgrade_firmware_info_provider() sets lite_fabric_running_ = true.
            {
                ZoneScopedN("Remote Chip Info Upgrade");
                for (ChipId id : remote_devices) {
                    log_info(tt::LogMetal, "Phase 2a: upgrade_remote_bh_chip_info for device {}", id);
                    cluster_->upgrade_remote_bh_chip_info(id);
                    // The UMD SoC descriptor now has the real harvesting masks from the
                    // remote chip's ARC.  Refresh the metal-layer SoC descriptor so that
                    // all subsequent coordinate translations (logical → virtual → translated)
                    // account for the remote chip's actual harvesting, not the gateway's.
                    cluster_->refresh_soc_desc_for_chip(id);
                    log_info(tt::LogMetal, "Phase 2a: upgrade complete for device {}", id);
                }
                // SOC descriptors changed — channel-to-logical-core mappings may differ.
                // Refresh the routing info so it uses the correct logical coordinates.
                cluster_->refresh_remote_ethernet_routing_info();
            }

            // Phase 2b: Iterative N-hop BFS discovery with downstream tunnel launch.
            // For each hop level: discover new chips, launch lite fabric on the
            // intermediate chip's downstream ETH core, configure forwarding on the
            // upstream receiver, create tunnel descriptors, bind UMD, and upgrade.
            // Track which MMIO ETH channels have forwarding configured per 1-hop
            // chip.  After Phase 2b, intermediate devices are rebound to exclude
            // these channels so that reads don't desync the downstream receiver's
            // forwarding_downstream_wr_idx.
            // Key: (intermediate chip_id, mmio_id) -> set of forwarded channel Y-coords
            std::map<std::pair<ChipId, ChipId>, std::set<uint32_t>> forwarded_channels_per_1hop;

            if (!remote_devices.empty()) {
                ZoneScopedN("N-Hop BFS Discovery");
                constexpr int max_extra_hops = 6;  // Up to 7-hop chips (8-chip ring)
                std::set<ChipId> current_frontier = remote_devices;
                std::set<ChipId> all_nhop_chips;
                // Track which chips are already reachable (MMIO + 1-hop remote)
                std::set<ChipId> reachable_chips = initial_devices;
                reachable_chips.insert(remote_devices.begin(), remote_devices.end());
                int current_hop_level = 1;
                const auto& binary_data = lite_fabric_hal_->get_binary_data();
                auto& sys_desc = lite_fabric_hal_->get_mutable_system_descriptor();

                // Fix up 1-hop tunnel descriptors.  UMD's patch_eth_connections
                // converts remote hardware port numbers to logical channels using
                // a pre-upgrade SoC descriptor (no harvesting info).  After Phase
                // 2a the remote SoC descriptors have correct harvesting, so we
                // re-derive the connected_core using the corrected mapping.
                // Tunnels whose remote ETH core is harvested are marked — the
                // connected_core_logical can't be correctly expressed and using
                // it for n-hop forwarding would address the wrong physical core.
                // Such tunnels are still valid for 1-hop UMD access.
                std::set<size_t> harvested_tunnel_indices;
                for (size_t ti = 0; ti < sys_desc.tunnels_from_mmio.size(); ti++) {
                    auto& tunnel = sys_desc.tunnels_from_mmio[ti];
                    if (tunnel.num_hops != 1) {
                        continue;
                    }
                    // Read remote_eth_id from MMIO ETH core's boot results (PCI).
                    // This gives the hardware port number on the remote chip.
                    uint32_t mmio_chan = static_cast<uint32_t>(tunnel.mmio_core_logical.y);
                    uint8_t remote_hw_port = 0;
                    auto mmio_cxy = tunnel.mmio_cxy_virtual();
                    cluster_->read_core(&remote_hw_port, sizeof(remote_hw_port), mmio_cxy, ETH_REMOTE_ETH_ID_ADDR);

                    // Convert remote HW port → NOC0 → logical on the corrected
                    // remote SoC descriptor (post-Phase 2a with real harvesting).
                    if (remote_hw_port >= tt::umd::blackhole::ETH_CORES_NOC0.size()) {
                        continue;
                    }
                    auto remote_noc0 = tt::umd::blackhole::ETH_CORES_NOC0[remote_hw_port];
                    const auto& remote_soc = cluster_->get_soc_desc(tunnel.connected_id);
                    CoreCoord remote_logical;
                    try {
                        remote_logical = remote_soc.translate_coord_to(
                            tt_xy_pair(remote_noc0.x, remote_noc0.y), CoordSystem::NOC0, CoordSystem::LOGICAL);
                    } catch (...) {
                        // Remote ETH core is harvested — the tunnel's
                        // connected_core_logical maps to a different physical
                        // core post-harvest.  Mark it so find_tunnel_to_chip
                        // skips it for n-hop forwarding, but keep it in the
                        // list for 1-hop UMD channel binding.
                        log_warning(
                            tt::LogMetal,
                            "1-hop tunnel fixup: marking MMIO chan {} -> chip {} as "
                            "harvested (remote HW port {} is harvested, NOC0 ({},{})). "
                            "Tunnel is valid for 1-hop but NOT for n-hop forwarding.",
                            mmio_chan,
                            tunnel.connected_id,
                            remote_hw_port,
                            remote_noc0.x,
                            remote_noc0.y);
                        harvested_tunnel_indices.insert(ti);
                        continue;
                    }

                    uint32_t correct_logical_chan = remote_logical.y;
                    uint32_t stored_logical_chan = static_cast<uint32_t>(tunnel.connected_core_logical.y);

                    // Always re-derive logical AND virtual from the post-harvest
                    // SoC descriptor.  Even when the logical channel happens to
                    // match (e.g. pre-harvest LOGICAL 5 = HW 5, post-harvest
                    // LOGICAL 5 = HW 6), the underlying NOC0 coordinate changes,
                    // so the virtual coordinate must be updated.  A stale virtual
                    // causes forwarding config writes to target the wrong core.
                    auto new_logical = remote_soc.get_eth_core_for_channel(correct_logical_chan, CoordSystem::LOGICAL);
                    auto new_virtual = cluster_->get_virtual_coordinate_from_logical_coordinates(
                        tunnel.connected_id, CoreCoord(new_logical.x, new_logical.y), tt::CoreType::ETH);
                    if (correct_logical_chan != stored_logical_chan ||
                        tunnel.connected_core_virtual.x != new_virtual.x ||
                        tunnel.connected_core_virtual.y != new_virtual.y) {
                        log_info(
                            tt::LogMetal,
                            "1-hop tunnel fixup: chip {} connected_core_logical "
                            "({},{}) -> ({},{}) virtual ({},{}) -> ({},{}) (HW port {}, MMIO chan {})",
                            tunnel.connected_id,
                            tunnel.connected_core_logical.x,
                            tunnel.connected_core_logical.y,
                            new_logical.x,
                            new_logical.y,
                            tunnel.connected_core_virtual.x,
                            tunnel.connected_core_virtual.y,
                            new_virtual.x,
                            new_virtual.y,
                            remote_hw_port,
                            mmio_chan);
                    }
                    tunnel.connected_core_logical = CoreCoord(new_logical.x, new_logical.y);
                    tunnel.connected_core_virtual = CoreCoord(new_virtual.x, new_virtual.y);
                }

                // Build lookup: for each remote chip, which tunnel(s) reach it and
                // from which MMIO ETH core.  Skips tunnels where the remote core
                // is harvested (marked during the 1-hop fixup above) since their
                // connected_core is wrong and can't be used for forwarding.
                auto find_tunnel_to_chip = [&](ChipId target_chip) -> const lite_fabric::TunnelDescriptor* {
                    for (size_t ti = 0; ti < sys_desc.tunnels_from_mmio.size(); ti++) {
                        const auto& t = sys_desc.tunnels_from_mmio[ti];
                        if (t.connected_id == target_chip && harvested_tunnel_indices.count(ti) == 0) {
                            return &t;
                        }
                    }
                    return nullptr;
                };

                for (int extra = 0; extra < max_extra_hops && !current_frontier.empty(); extra++) {
                    int next_hop_level = current_hop_level + 1;
                    log_info(
                        tt::LogMetal,
                        "Phase 2b: BFS iteration {}, frontier size {}, discovering {}-hop chips",
                        extra + 1,
                        current_frontier.size(),
                        next_hop_level);

                    auto discovered = discover_nhop_chips(*cluster_, current_frontier, /*max_hops=*/1, reachable_chips);
                    if (discovered.empty()) {
                        log_info(tt::LogMetal, "Phase 2b: no new chips at hop level {}", next_hop_level);
                        break;
                    }

                    log_info(
                        tt::LogMetal, "Phase 2b: discovered {} new {}-hop chips", discovered.size(), next_hop_level);

                    // Register new chips in system containers
                    std::set<ChipId> new_chip_ids;
                    for (const auto& info : discovered) {
                        new_chip_ids.insert(info.chip_id);
                        all_devices.insert(info.chip_id);
                        dram_bank_offset_map_.emplace(info.chip_id, std::vector<int32_t>{});
                        l1_bank_offset_map_.emplace(info.chip_id, std::vector<int32_t>{});
                        dram_bank_to_noc_xy_.emplace(info.chip_id, std::vector<uint16_t>{});
                        l1_bank_to_noc_xy_.emplace(info.chip_id, std::vector<uint16_t>{});
                        worker_logical_col_to_virtual_col_.emplace(info.chip_id, std::vector<uint8_t>{});
                        worker_logical_row_to_virtual_row_.emplace(info.chip_id, std::vector<uint8_t>{});
                    }

                    // Update UMD routing info for new chips
                    cluster_->update_routing_info_for_dynamic_chips(new_chip_ids);
                    cluster_->reassign_mem_channels();

                    // Fix-up tunnel descriptors for frontier chips.  The connected_core
                    // was derived from remote_eth_id (BFS boot results), which is a
                    // hardware port number that may NOT match the SoC descriptor's
                    // channel index.  Probe ALL ETH cores on the frontier chip for
                    // routing_enabled to find the actual upstream receiver.
                    //
                    // We cannot rely on the cluster descriptor's ethernet_connections
                    // because they were populated from the same incorrect remote_eth_id
                    // mapping (stored_ok would always be true for the wrong channel).
                    // Instead, directly probe every ETH core: at this point only the
                    // upstream receiver has routing_enabled=ENABLED (no downstream
                    // senders have been launched yet).
                    for (auto frontier_chip : current_frontier) {
                        auto tunnel_it = std::find_if(
                            sys_desc.tunnels_from_mmio.begin(),
                            sys_desc.tunnels_from_mmio.end(),
                            [frontier_chip](const lite_fabric::TunnelDescriptor& t) {
                                return t.connected_id == frontier_chip;
                            });
                        if (tunnel_it == sys_desc.tunnels_from_mmio.end() || tunnel_it->num_hops < 2) {
                            continue;
                        }
                        uint32_t stored_channel = static_cast<uint32_t>(tunnel_it->connected_core_logical.y);

                        const auto& frontier_soc = cluster_->get_soc_desc(frontier_chip);
                        auto eth_cores_noc0 = frontier_soc.get_cores(CoreType::ETH, CoordSystem::NOC0);
                        uint32_t re_addr = LITE_FABRIC_CONFIG_START +
                                           offsetof(lite_fabric::FabricLiteMemoryMap, config) +
                                           offsetof(lite_fabric::FabricLiteConfig, routing_enabled);

                        uint32_t correct_channel = stored_channel;
                        bool found = false;

                        // Collect ALL channels with routing_enabled (not just the first).
                        // On reinit, stale routing_enabled values from a previous run's
                        // downstream senders may still be in L1, causing multiple matches.
                        struct FixupCandidate {
                            uint32_t channel;
                            tt_cxy_pair cxy;
                        };
                        std::vector<FixupCandidate> candidates;

                        for (size_t ch = 0; ch < eth_cores_noc0.size(); ch++) {
                            auto core_noc0 = eth_cores_noc0[ch];
                            auto core_translated = frontier_soc.translate_coord_to(
                                tt_xy_pair(core_noc0.x, core_noc0.y), CoordSystem::NOC0, CoordSystem::TRANSLATED);
                            auto cxy =
                                tt_cxy_pair(static_cast<size_t>(frontier_chip), core_translated.x, core_translated.y);

                            uint32_t routing_val = 0;
                            try {
                                cluster_->read_core(&routing_val, sizeof(routing_val), cxy, re_addr);
                            } catch (...) {
                                continue;
                            }

                            if (static_cast<lite_fabric::RoutingEnabledState>(routing_val) ==
                                lite_fabric::RoutingEnabledState::ENABLED) {
                                candidates.push_back({static_cast<uint32_t>(ch), cxy});
                            }
                        }

                        if (candidates.size() == 1) {
                            correct_channel = candidates[0].channel;
                            found = true;
                        } else if (candidates.size() > 1) {
                            // Multiple channels have routing_enabled — stale values from
                            // previous run.  Disambiguate: the upstream receiver connects
                            // to a chip in reachable_chips.  Read boot_results from each
                            // candidate to find the one linking back to a known chip.
                            auto* cluster_desc = cluster_->get_cluster_desc();
                            const auto& uid_map = cluster_desc->get_chip_unique_ids();
                            std::map<uint64_t, ChipId> boardid_to_chip;
                            for (const auto& [cid, uid] : uid_map) {
                                boardid_to_chip[uid >> 5] = cid;
                            }

                            for (const auto& cand : candidates) {
                                uint32_t board_id_hi = 0, board_id_lo = 0;
                                try {
                                    cluster_->read_core(
                                        &board_id_hi, sizeof(board_id_hi), cand.cxy, ETH_REMOTE_BOARD_ID_HI_ADDR);
                                    cluster_->read_core(
                                        &board_id_lo, sizeof(board_id_lo), cand.cxy, ETH_REMOTE_BOARD_ID_LO_ADDR);
                                } catch (...) {
                                    log_info(
                                        tt::LogMetal,
                                        "Phase 2b: tunnel fixup: disambig chip {} chan {} "
                                        "board_id read exception",
                                        frontier_chip,
                                        cand.channel);
                                    continue;
                                }
                                uint64_t remote_board_id = (static_cast<uint64_t>(board_id_hi) << 32) | board_id_lo;

                                auto it = boardid_to_chip.find(remote_board_id);
                                log_info(
                                    tt::LogMetal,
                                    "Phase 2b: tunnel fixup: disambig chip {} chan {} "
                                    "board_id={:#x} (hi={:#x} lo={:#x}) -> {}",
                                    frontier_chip,
                                    cand.channel,
                                    remote_board_id,
                                    board_id_hi,
                                    board_id_lo,
                                    it != boardid_to_chip.end()
                                        ? ("chip " + std::to_string(it->second) +
                                           (reachable_chips.count(it->second) ? " (reachable)" : " (NOT reachable)"))
                                        : "unknown");
                                if (it != boardid_to_chip.end() && reachable_chips.count(it->second)) {
                                    correct_channel = cand.channel;
                                    found = true;
                                    log_info(
                                        tt::LogMetal,
                                        "Phase 2b: tunnel fixup: disambiguated {} candidates on "
                                        "chip {}, channel {} connects to reachable chip {}",
                                        candidates.size(),
                                        frontier_chip,
                                        cand.channel,
                                        it->second);
                                    break;
                                }
                            }

                            if (!found) {
                                // Fallback: pick a candidate that does NOT conflict
                                // with any downstream channel needed by discovered
                                // chips.  On reinit, the board_id reads may return
                                // stale/zero data, so disambiguation can fail.  But
                                // we can still avoid the worst outcome (picking the
                                // downstream channel as the upstream receiver, which
                                // causes the downstream tunnel to be skipped).
                                std::set<uint32_t> downstream_chans_for_frontier;
                                for (const auto& d : discovered) {
                                    if (d.intermediate_chip == frontier_chip) {
                                        downstream_chans_for_frontier.insert(d.downstream_eth_chan);
                                    }
                                }
                                auto format_downstream_chans = [&]() {
                                    std::string s;
                                    for (auto c : downstream_chans_for_frontier) {
                                        if (!s.empty()) {
                                            s += ",";
                                        }
                                        s += std::to_string(c);
                                    }
                                    return s;
                                };
                                auto is_port_up = [&](const FixupCandidate& cand) {
                                    uint32_t ps_word = 0;
                                    try {
                                        cluster_->read_core(&ps_word, sizeof(ps_word), cand.cxy, ETH_PORT_STATUS_ADDR);
                                    } catch (...) {
                                        return false;
                                    }
                                    return (static_cast<uint8_t>(ps_word & 0xFF) == PORT_UP);
                                };
                                // First pass: prefer a non-conflicting candidate whose
                                // ETH port is UP.  Stale routing_enabled from a previous
                                // run may exist on dead/non-UP ports; picking such a port
                                // as the upstream receiver breaks the reverse-forwarding
                                // path and causes wait_for_read_event timeouts.
                                for (const auto& cand : candidates) {
                                    if (!downstream_chans_for_frontier.count(cand.channel) && is_port_up(cand)) {
                                        correct_channel = cand.channel;
                                        found = true;
                                        log_warning(
                                            tt::LogMetal,
                                            "Phase 2b: tunnel fixup: couldn't disambiguate {} "
                                            "candidates on chip {}, picked channel {} "
                                            "(port UP, avoids downstream channels {})",
                                            candidates.size(),
                                            frontier_chip,
                                            cand.channel,
                                            format_downstream_chans());
                                        break;
                                    }
                                }
                                // Second pass: extended scan — probe ALL ETH channels on
                                // frontier_chip for port-UP links that connect back toward
                                // MMIO (board_id resolves to a known reachable chip).  The
                                // real upstream receiver may not have routing_enabled if it
                                // was freshly set up via ETH_INIT_NEIGHBOUR on the previous
                                // hop (e.g. on reinit after initialize_and_launch_firmware
                                // changed which ETH link is UP for reaching the frontier).
                                if (!found) {
                                    for (size_t ch = 0; ch < eth_cores_noc0.size(); ch++) {
                                        if (downstream_chans_for_frontier.count(static_cast<uint32_t>(ch))) {
                                            continue;  // skip downstream channels
                                        }
                                        auto core_noc0 = eth_cores_noc0[ch];
                                        auto core_translated = frontier_soc.translate_coord_to(
                                            tt_xy_pair(core_noc0.x, core_noc0.y),
                                            CoordSystem::NOC0,
                                            CoordSystem::TRANSLATED);
                                        auto cxy = tt_cxy_pair(
                                            static_cast<size_t>(frontier_chip), core_translated.x, core_translated.y);
                                        uint32_t ps_word = 0;
                                        try {
                                            cluster_->read_core(&ps_word, sizeof(ps_word), cxy, ETH_PORT_STATUS_ADDR);
                                        } catch (...) {
                                            continue;
                                        }
                                        if ((ps_word & 0xFF) != PORT_UP) {
                                            continue;
                                        }
                                        uint32_t hi = 0, lo = 0;
                                        try {
                                            cluster_->read_core(&hi, sizeof(hi), cxy, ETH_REMOTE_BOARD_ID_HI_ADDR);
                                            cluster_->read_core(&lo, sizeof(lo), cxy, ETH_REMOTE_BOARD_ID_LO_ADDR);
                                        } catch (...) {
                                            continue;
                                        }
                                        uint64_t remote_board_id = (static_cast<uint64_t>(hi) << 32) | lo;
                                        auto it = boardid_to_chip.find(remote_board_id);
                                        if (it == boardid_to_chip.end() || !reachable_chips.count(it->second)) {
                                            continue;
                                        }
                                        correct_channel = static_cast<uint32_t>(ch);
                                        found = true;
                                        log_warning(
                                            tt::LogMetal,
                                            "Phase 2b: tunnel fixup: extended scan found "
                                            "chip {} chan {} (port UP, connects to reachable "
                                            "chip {}, avoids downstream channels {})",
                                            frontier_chip,
                                            ch,
                                            it->second,
                                            format_downstream_chans());
                                        break;
                                    }
                                }
                                // Third pass: last resort from routing_enabled candidates —
                                // any non-conflicting channel regardless of port state.
                                if (!found) {
                                    for (const auto& cand : candidates) {
                                        if (!downstream_chans_for_frontier.count(cand.channel)) {
                                            correct_channel = cand.channel;
                                            found = true;
                                            log_warning(
                                                tt::LogMetal,
                                                "Phase 2b: tunnel fixup: couldn't disambiguate {} "
                                                "candidates on chip {}, picked channel {} "
                                                "(avoids downstream channels {})",
                                                candidates.size(),
                                                frontier_chip,
                                                cand.channel,
                                                format_downstream_chans());
                                            break;
                                        }
                                    }
                                }
                                if (!found) {
                                    // All candidates conflict — last resort
                                    correct_channel = candidates[0].channel;
                                    found = true;
                                    log_warning(
                                        tt::LogMetal,
                                        "Phase 2b: tunnel fixup: couldn't disambiguate {} "
                                        "candidates on chip {}, all conflict with downstream, "
                                        "falling back to channel {}",
                                        candidates.size(),
                                        frontier_chip,
                                        candidates[0].channel);
                                }
                            }
                        }

                        if (found && correct_channel != stored_channel) {
                            auto logical = frontier_soc.get_eth_core_for_channel(correct_channel, CoordSystem::LOGICAL);
                            auto virtual_cc = cluster_->get_virtual_coordinate_from_logical_coordinates(
                                frontier_chip, CoreCoord(logical.x, logical.y), tt::CoreType::ETH);

                            log_info(
                                tt::LogMetal,
                                "Phase 2b: tunnel fixup: correcting chip {} connected_core "
                                "from chan {} to chan {} (probed routing_enabled)",
                                frontier_chip,
                                stored_channel,
                                correct_channel);

                            // Update ALL tunnel descriptors pointing to this frontier chip
                            for (auto& t : sys_desc.tunnels_from_mmio) {
                                if (t.connected_id == frontier_chip && t.num_hops >= 2) {
                                    t.connected_core_logical = CoreCoord(logical.x, logical.y);
                                    t.connected_core_virtual = CoreCoord(virtual_cc.x, virtual_cc.y);
                                }
                            }
                        } else if (!found) {
                            log_warning(
                                tt::LogMetal,
                                "Phase 2b: tunnel fixup: could not find upstream receiver "
                                "with routing_enabled on chip {} (probed {} ETH cores), "
                                "keeping chan {}",
                                frontier_chip,
                                eth_cores_noc0.size(),
                                stored_channel);
                        }
                    }

                    // Launch downstream tunnels and configure forwarding
                    uint32_t config_addr =
                        LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config);
                    uint32_t host_iface_addr =
                        LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, host_interface);
                    uint32_t sender_buf_addr =
                        LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, sender_ch0_buffer);
                    uint32_t forwarding_offset = config_addr + offsetof(lite_fabric::FabricLiteConfig, forwarding);

                    for (auto& info : discovered) {
                        // Collect ALL upstream receiver channels on the intermediate
                        // chip (from all non-harvested tunnels). The downstream tunnel
                        // must NOT use any of these channels because they share ERISC1
                        // and launching the downstream would clobber the receiver.
                        std::set<uint32_t> upstream_rx_chans;
                        for (size_t ti = 0; ti < sys_desc.tunnels_from_mmio.size(); ti++) {
                            const auto& t = sys_desc.tunnels_from_mmio[ti];
                            if (t.connected_id == info.intermediate_chip && harvested_tunnel_indices.count(ti) == 0) {
                                upstream_rx_chans.insert(static_cast<uint32_t>(t.connected_core_logical.y));
                            }
                        }

                        // If the BFS-selected downstream channel conflicts with an
                        // upstream receiver, find an alternative downstream channel.
                        if (upstream_rx_chans.count(info.downstream_eth_chan)) {
                            auto* cdesc = cluster_->get_cluster_desc();
                            std::vector<std::tuple<int, int>> alt_pairs;
                            try {
                                alt_pairs = cdesc->get_directly_connected_ethernet_channels_between_chips(
                                    info.intermediate_chip, info.chip_id);
                            } catch (...) {
                            }

                            const auto& alt_soc = cluster_->get_soc_desc(info.intermediate_chip);
                            auto alt_eth_cores = alt_soc.get_cores(CoreType::ETH, CoordSystem::NOC0);

                            bool found_alt = false;
                            for (const auto& [local_chan, remote_chan] : alt_pairs) {
                                uint32_t lc = static_cast<uint32_t>(local_chan);
                                if (lc >= alt_eth_cores.size()) {
                                    continue;
                                }
                                if (upstream_rx_chans.count(lc)) {
                                    continue;  // still conflicts
                                }
                                log_info(
                                    tt::LogMetal,
                                    "Phase 2b: downstream chan {} on chip {} conflicts with "
                                    "upstream receiver, switching to chan {} core ({},{}) for chip {}",
                                    info.downstream_eth_chan,
                                    info.intermediate_chip,
                                    lc,
                                    alt_eth_cores[lc].x,
                                    alt_eth_cores[lc].y,
                                    info.chip_id);
                                info.downstream_eth_chan = lc;
                                info.downstream_core_noc0 = tt_xy_pair(alt_eth_cores[lc].x, alt_eth_cores[lc].y);
                                info.remote_eth_id = static_cast<uint8_t>(remote_chan);
                                found_alt = true;
                                break;
                            }
                            if (!found_alt) {
                                log_warning(
                                    tt::LogMetal,
                                    "Phase 2b: downstream chan {} on chip {} conflicts with upstream "
                                    "receiver and no alternative found, skipping chip {}",
                                    info.downstream_eth_chan,
                                    info.intermediate_chip,
                                    info.chip_id);
                                continue;
                            }
                        }

                        // Find existing tunnel to the intermediate chip.
                        // Prefer a tunnel whose connected core does NOT conflict with
                        // the (possibly updated) downstream channel.
                        const lite_fabric::TunnelDescriptor* upstream_tunnel = nullptr;
                        for (size_t ti = 0; ti < sys_desc.tunnels_from_mmio.size(); ti++) {
                            const auto& t = sys_desc.tunnels_from_mmio[ti];
                            if (t.connected_id == info.intermediate_chip && harvested_tunnel_indices.count(ti) == 0 &&
                                static_cast<uint32_t>(t.connected_core_logical.y) != info.downstream_eth_chan) {
                                upstream_tunnel = &t;
                                break;
                            }
                        }
                        if (!upstream_tunnel) {
                            // Fall back to any non-harvested tunnel
                            upstream_tunnel = find_tunnel_to_chip(info.intermediate_chip);
                        }
                        if (!upstream_tunnel) {
                            log_warning(
                                tt::LogMetal,
                                "Phase 2b: no tunnel to intermediate chip {} for new chip {}, skipping",
                                info.intermediate_chip,
                                info.chip_id);
                            continue;
                        }

                        log_info(
                            tt::LogMetal,
                            "Phase 2b: launching downstream tunnel on chip {} ETH chan {} "
                            "core ({},{}) for {}-hop chip {} (upstream rx chan {})",
                            info.intermediate_chip,
                            info.downstream_eth_chan,
                            info.downstream_core_noc0.x,
                            info.downstream_core_noc0.y,
                            next_hop_level,
                            info.chip_id,
                            upstream_tunnel->connected_core_logical.y);

                        // Write lite fabric config to downstream ETH core
                        lite_fabric::FabricLiteConfig ds_config{};
                        ds_config.is_primary = true;
                        ds_config.is_mmio = true;
                        ds_config.initial_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
                        ds_config.current_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
                        ds_config.binary_addr = LITE_FABRIC_TEXT_START;
                        ds_config.binary_size = (binary_data.size() + 15) & ~0xF;
                        ds_config.eth_chans_mask = 0x3;  // Needs ≥2 bits for routing_init assert
                        ds_config.routing_enabled = lite_fabric::RoutingEnabledState::ENABLED;

                        // Configure reverse forwarding on downstream receiver so read
                        // responses from the downstream chip get relayed upstream to the
                        // upstream core's sender buffer (and eventually back to MMIO).
                        {
                            const auto& interm_soc = cluster_->get_soc_desc(info.intermediate_chip);
                            auto upstream_eth_cores_noc0 = interm_soc.get_cores(CoreType::ETH, CoordSystem::NOC0);
                            auto upstream_core_noc0 =
                                upstream_eth_cores_noc0[upstream_tunnel->connected_core_logical.y];
                            // Use TRANSLATED coordinates for NOC addressing on the
                            // remote chip.  The ARC programs NOC translation tables
                            // at POR, so ERISC1's NOC0 writes go through the
                            // translation layer.  NOC0/physical coords (e.g. (12,1))
                            // would be translated to the wrong physical tile.
                            auto upstream_core_translated = interm_soc.translate_coord_to(
                                tt_xy_pair(upstream_core_noc0.x, upstream_core_noc0.y),
                                CoordSystem::NOC0,
                                CoordSystem::TRANSLATED);
                            using HostIface = lite_fabric::HostToFabricLiteInterface<
                                lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0],
                                lite_fabric::CHANNEL_BUFFER_SIZE>;
                            // Start with forwarding disabled.  We enable it AFTER all
                            // reads through the upstream sender complete so that
                            // initial_wr_idx is computed from a stable d2h (avoids
                            // TOCTOU: lite-fabric reads advance d2h between the
                            // snapshot and the first reverse-forwarding use).
                            ds_config.forwarding.enabled = 0;
                            ds_config.forwarding.is_reverse_relay = 1;
                            ds_config.forwarding.downstream_noc_x = static_cast<uint8_t>(upstream_core_translated.x);
                            ds_config.forwarding.downstream_noc_y = static_cast<uint8_t>(upstream_core_translated.y);
                            ds_config.forwarding.downstream_num_buffers = lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0];
                            ds_config.forwarding.downstream_sender_buf_addr = sender_buf_addr;
                            ds_config.forwarding.downstream_h2d_addr = host_iface_addr + offsetof(HostIface, h2d);
                            ds_config.forwarding.downstream_buffer_size = lite_fabric::CHANNEL_BUFFER_SIZE;
                            // Set initial_wr_idx to 0xFF sentinel.  The FW uses
                            // initial_wr_idx != 0xFF (not forwarding_config->enabled)
                            // as the activation signal, because the enabled field may
                            // not be visible to the FW due to BH ERISC L1 caching.
                            ds_config.forwarding.initial_wr_idx = 0xFF;

                            log_info(
                                tt::LogMetal,
                                "Phase 2b: reverse forwarding prepared on downstream core ({},{}) "
                                "-> upstream ({},{}) TRANSLATED (deferred activation)",
                                info.downstream_core_noc0.x,
                                info.downstream_core_noc0.y,
                                upstream_core_translated.x,
                                upstream_core_translated.y);
                        }

                        // Convert NOC0 physical coords to TRANSLATED for host-side operations
                        const auto& ds_soc = cluster_->get_soc_desc(info.intermediate_chip);
                        auto ds_core_translated = ds_soc.translate_coord_to(
                            tt_xy_pair(info.downstream_core_noc0.x, info.downstream_core_noc0.y),
                            CoordSystem::NOC0,
                            CoordSystem::TRANSLATED);
                        auto ds_cxy = tt_cxy_pair(
                            static_cast<size_t>(info.intermediate_chip), ds_core_translated.x, ds_core_translated.y);

                        // Reset ERISC1 only (keep ERISC0 running for ETH keepalive).
                        // On reinit, ERISC1 may still be running the old downstream
                        // sender FW; resetting it prevents stale FW from corrupting
                        // the config/binary we're about to write.  ERISC0 stays alive
                        // so the ETH MAC link doesn't time out during the multi-hop
                        // write sequence (which can take >1s through 3+ hops).
                        constexpr uint32_t kSoftResetAddr = 0xFFB121B0;
                        constexpr uint32_t kErisc1InResetErisc0Running = 0x47000;
                        cluster_->write_core(
                            &kErisc1InResetErisc0Running, sizeof(kErisc1InResetErisc0Running), ds_cxy, kSoftResetAddr);

                        // Zero entire host interface (d2h, pad, AND h2d) on downstream
                        // core.  The FW zeros h2d during object_init, but on reinit
                        // stale h2d values from the previous iteration can cause phantom
                        // sends before the FW init runs.  Zero 8 bytes: d2h (2B) +
                        // _d2h_h2d_pad (2B) + h2d (2B) + 2 extra for alignment.
                        uint64_t zero = 0;
                        cluster_->write_core(&zero, sizeof(zero), ds_cxy, host_iface_addr);

                        // Write config and binary (ERISC0 still running, these L1
                        // addresses don't overlap with syseng FW).
                        cluster_->write_core(&ds_config, sizeof(ds_config), ds_cxy, config_addr);
                        cluster_->write_core(binary_data.data(), binary_data.size(), ds_cxy, LITE_FABRIC_TEXT_START);

                        // Set PC for ERISC1
                        uint32_t pc = LITE_FABRIC_TEXT_START;
                        cluster_->write_core(&pc, sizeof(pc), ds_cxy, LITE_FABRIC_RESET_PC);

                        // Ensure all L1 writes (config, binary, PC) have been consumed
                        // by the lite fabric sender before deasserting ERISC1.  Without
                        // this barrier, a fast deassert could cause the FW to start
                        // before the config is fully written to L1.
                        cluster_->l1_barrier(info.intermediate_chip);

                        // Now kill ERISC0 and deassert ERISC1 in quick succession.
                        // These two writes travel through the same forwarding chain and
                        // are processed back-to-back on the target chip, minimizing the
                        // time without ETH keepalive (microseconds vs the ~1.3s it took
                        // when ERISC0 was killed before the config/binary writes).
                        constexpr uint32_t kBothEriscsInReset = 0x47800;
                        cluster_->write_core(&kBothEriscsInReset, sizeof(kBothEriscsInReset), ds_cxy, kSoftResetAddr);
                        constexpr uint32_t kErisc1OutErisc0InReset = 0x46800;
                        cluster_->write_core(
                            &kErisc1OutErisc0InReset, sizeof(kErisc1OutErisc0InReset), ds_cxy, kSoftResetAddr);

                        log_info(
                            tt::LogMetal,
                            "Phase 2b: downstream tunnel launched on chip {} core ({},{}), waiting for READY",
                            info.intermediate_chip,
                            info.downstream_core_noc0.x,
                            info.downstream_core_noc0.y);

                        // Wait for downstream lite fabric to reach READY.
                        // Deeper hops need more time: each hop adds relay latency
                        // for the config/binary writes and init handshake.
                        int k_MaxPolls = 20 + 10 * next_hop_level;
                        constexpr int k_PollMs = 100;
                        uint32_t state_addr = config_addr + offsetof(lite_fabric::FabricLiteConfig, current_state);
                        bool ready = false;
                        for (int p = 0; p < k_MaxPolls; p++) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(k_PollMs));
                            uint32_t state = 0;
                            cluster_->read_core(&state, sizeof(state), ds_cxy, state_addr);
                            if (static_cast<lite_fabric::InitState>(state) == lite_fabric::InitState::READY) {
                                ready = true;
                                break;
                            }
                        }
                        if (!ready) {
                            // Read diagnostic fields from the downstream core's config
                            uint32_t diag_state = 0, diag_handshake = 0, diag_loop = 0;
                            uint32_t diag_routing = 0;
                            cluster_->read_core(
                                &diag_state,
                                sizeof(diag_state),
                                ds_cxy,
                                config_addr + offsetof(lite_fabric::FabricLiteConfig, current_state));
                            cluster_->read_core(
                                &diag_handshake,
                                sizeof(diag_handshake),
                                ds_cxy,
                                config_addr + offsetof(lite_fabric::FabricLiteConfig, primary_local_handshake));
                            cluster_->read_core(
                                &diag_loop,
                                sizeof(diag_loop),
                                ds_cxy,
                                config_addr + offsetof(lite_fabric::FabricLiteConfig, neighbour_handshake));
                            cluster_->read_core(
                                &diag_routing,
                                sizeof(diag_routing),
                                ds_cxy,
                                config_addr + offsetof(lite_fabric::FabricLiteConfig, routing_enabled));
                            log_warning(
                                tt::LogMetal,
                                "Phase 2b: downstream tunnel on chip {} core ({},{}) failed to reach "
                                "READY after {} polls, skipping chip {}. "
                                "Diagnostics: current_state={}, breadcrumb=0x{:x}, loop_counter={}, routing_enabled={}",
                                info.intermediate_chip,
                                info.downstream_core_noc0.x,
                                info.downstream_core_noc0.y,
                                k_MaxPolls,
                                info.chip_id,
                                diag_state,
                                diag_handshake,
                                diag_loop,
                                diag_routing);
                            // Clean up: put ERISC1 back in reset on the failed core
                            cluster_->assert_risc_reset_at_core(ds_cxy, tt::umd::RiscType::ERISC1);
                            continue;
                        }

                        log_info(tt::LogMetal, "Phase 2b: downstream tunnel READY, configuring forwarding");

                        // Track this core so initialize_remote_eth_cores_for_fabric
                        // knows ERISC1 is running here as a downstream sender.
                        downstream_sender_cores_.insert({info.intermediate_chip, info.downstream_eth_chan});

                        // Configure forwarding on the upstream receiver (on intermediate chip)
                        // The upstream receiver is on the connected_core of the tunnel to
                        // intermediate_chip.
                        lite_fabric::FabricLiteConfig::ForwardingConfig fwd{};
                        fwd.enabled = 1;
                        fwd.downstream_noc_x = static_cast<uint8_t>(ds_core_translated.x);
                        fwd.downstream_noc_y = static_cast<uint8_t>(ds_core_translated.y);
                        fwd.downstream_num_buffers = lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0];
                        fwd.downstream_sender_buf_addr = sender_buf_addr;
                        // offsetof with template types containing commas breaks the macro;
                        // use a typedef to avoid the issue.
                        using HostIface = lite_fabric::HostToFabricLiteInterface<
                            lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0],
                            lite_fabric::CHANNEL_BUFFER_SIZE>;
                        fwd.downstream_h2d_addr = host_iface_addr + offsetof(HostIface, h2d);
                        fwd.downstream_buffer_size = lite_fabric::CHANNEL_BUFFER_SIZE;

                        // Write forwarding config to the upstream receiver core on
                        // intermediate chip (the connected end of the existing tunnel)
                        auto upstream_rx_cxy = upstream_tunnel->connected_cxy_virtual();
                        cluster_->write_core(&fwd, sizeof(fwd), upstream_rx_cxy, forwarding_offset);

                        auto mmio_cxy = tt_cxy_pair(
                            upstream_tunnel->mmio_id,
                            upstream_tunnel->mmio_core_virtual.x,
                            upstream_tunnel->mmio_core_virtual.y);

                        // Ensure the forwarding config write has been consumed by the
                        // MMIO sender FW.  l1_barrier for remote chips calls
                        // wait_for_non_mmio_flush which waits for d2h.sender ==
                        // h2d.sender.  It does NOT generate a receiver response.
                        cluster_->l1_barrier(info.intermediate_chip);

                        // Ensure d2h.receiver on the MMIO core is stable before using
                        // it for initial_wr_idx.  The FW writes a diagnostic word with
                        // wr_sent (bits 31-24) and completion (bits 23-16) counters.
                        // When they match, all pending completions have fired and
                        // d2h.receiver is accurate.
                        {
                            uint32_t diag_addr = LITE_FABRIC_CONFIG_START +
                                                 offsetof(lite_fabric::FabricLiteMemoryMap, config) +
                                                 offsetof(lite_fabric::FabricLiteConfig, padding1);
                            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                            while (true) {
                                uint32_t diag = 0;
                                cluster_->read_core(&diag, sizeof(diag), mmio_cxy, diag_addr);
                                uint8_t wr_sent = (diag >> 24) & 0xFF;
                                uint8_t completion = (diag >> 16) & 0xFF;
                                if (wr_sent == completion) {
                                    break;
                                }
                                if (std::chrono::steady_clock::now() > timeout) {
                                    TT_THROW(
                                        "Phase 2b: timeout waiting for MMIO "
                                        "receiver completion to catch up on "
                                        "core ({},{}) wr_sent={} completion={}",
                                        mmio_cxy.x,
                                        mmio_cxy.y,
                                        wr_sent,
                                        completion);
                                }
                            }
                        }

                        log_info(
                            tt::LogMetal,
                            "Phase 2b: forwarding configured on chip {} core ({},{}) -> "
                            "downstream ({},{})",
                            info.intermediate_chip,
                            upstream_rx_cxy.x,
                            upstream_rx_cxy.y,
                            info.downstream_core_noc0.x,
                            info.downstream_core_noc0.y);

                        // Record that this MMIO channel has forwarding configured
                        // through the intermediate (1-hop) chip.  After Phase 2b,
                        // the 1-hop chip will be rebound to exclude this channel.
                        forwarded_channels_per_1hop[{info.intermediate_chip, upstream_tunnel->mmio_id}].insert(
                            static_cast<uint32_t>(upstream_tunnel->mmio_core_logical.y));

                        // Deferred reverse-forwarding activation: compute the correct
                        // initial_wr_idx for the downstream receiver's return forwarding.
                        // This must match the upstream sender's current d2h.sender so
                        // that forwarding_downstream_wr_idx starts at the right slot.
                        {
                            uint8_t upstream_sender_d2h = 0;

                            if (next_hop_level <= 2) {
                                // 2-hop: upstream sender is on chip 1, 1:1 with MMIO receiver.
                                // Read MMIO d2h locally (PCI, no side effects).
                                uint32_t mmio_d2h_word = 0;
                                cluster_->read_core(&mmio_d2h_word, sizeof(mmio_d2h_word), mmio_cxy, host_iface_addr);
                                // d2h.fabric_receiver_channel_index is byte 1.
                                upstream_sender_d2h = (mmio_d2h_word >> 8) & 0xFF;
                            } else {
                                // 3+ hop: upstream sender is on the intermediate chip
                                // (not MMIO). Read d2h.sender from the actual upstream
                                // sender core via lite fabric.  The read itself advances
                                // that sender's d2h by 1 (response travels back), so
                                // compensate: initial_wr_idx = (read_value + 1) % NUM_BUFS.
                                uint32_t upstream_d2h_word = 0;
                                cluster_->read_core(
                                    &upstream_d2h_word, sizeof(upstream_d2h_word), upstream_rx_cxy, host_iface_addr);
                                uint8_t read_d2h_sender = upstream_d2h_word & 0xFF;
                                upstream_sender_d2h = (read_d2h_sender + 1) % lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0];
                                log_info(
                                    tt::LogMetal,
                                    "Phase 2b: read upstream sender d2h.sender={} from "
                                    "chip {} core ({},{}), compensated initial_wr_idx={}",
                                    read_d2h_sender,
                                    info.intermediate_chip,
                                    upstream_rx_cxy.x,
                                    upstream_rx_cxy.y,
                                    upstream_sender_d2h);
                            }

                            ds_config.forwarding.initial_wr_idx = upstream_sender_d2h;
                            ds_config.forwarding.enabled = 1;
                            cluster_->write_core(
                                &ds_config.forwarding, sizeof(ds_config.forwarding), ds_cxy, forwarding_offset);
                            // Wait for the sender to consume the activation write.
                            // l1_barrier checks d2h.sender == h2d.sender across ALL
                            // bound channels for this chip, so it works regardless of
                            // which MMIO channel UMD chose for this particular write.
                            cluster_->l1_barrier(info.intermediate_chip);

                            log_info(
                                tt::LogMetal,
                                "Phase 2b: reverse forwarding enabled on downstream core ({},{}) "
                                "initial_wr_idx={} (hop_level={})",
                                info.downstream_core_noc0.x,
                                info.downstream_core_noc0.y,
                                upstream_sender_d2h,
                                next_hop_level);
                        }

                        // Create tunnel descriptor for the new chip
                        // Reuse the same MMIO ETH core as the upstream tunnel
                        const auto& new_soc = cluster_->get_soc_desc(info.chip_id);
                        auto connected_logical_cc =
                            new_soc.get_eth_core_for_channel(info.remote_eth_id, CoordSystem::LOGICAL);
                        auto connected_virtual_cc = cluster_->get_virtual_coordinate_from_logical_coordinates(
                            info.chip_id, CoreCoord(connected_logical_cc.x, connected_logical_cc.y), tt::CoreType::ETH);

                        // Save upstream_tunnel fields before push_back.  push_back may
                        // reallocate the vector, invalidating the upstream_tunnel pointer.
                        auto saved_mmio_id = upstream_tunnel->mmio_id;
                        auto saved_mmio_core_virtual = upstream_tunnel->mmio_core_virtual;
                        auto saved_mmio_core_logical = upstream_tunnel->mmio_core_logical;
                        auto saved_connected_id = upstream_tunnel->connected_id;
                        auto saved_connected_core_virtual = upstream_tunnel->connected_core_virtual;

                        sys_desc.tunnels_from_mmio.push_back(lite_fabric::TunnelDescriptor{
                            .mmio_id = saved_mmio_id,
                            .mmio_core_virtual = saved_mmio_core_virtual,
                            .mmio_core_logical = saved_mmio_core_logical,
                            .connected_id = info.chip_id,
                            .connected_core_virtual = CoreCoord(connected_virtual_cc.x, connected_virtual_cc.y),
                            .connected_core_logical = CoreCoord(connected_logical_cc.x, connected_logical_cc.y),
                            .num_hops = next_hop_level,
                        });
                        // upstream_tunnel is now potentially dangling — use saved locals.

                        // Bind UMD communication for the new chip
                        std::set<uint32_t> channels = {static_cast<uint32_t>(saved_mmio_core_logical.y)};
                        auto* remote_chip = cluster_->get_driver()->get_remote_chip(info.chip_id);
                        remote_chip->set_remote_transfer_ethernet_cores(channels);
                        remote_chip->get_remote_communication()->set_num_hops(next_hop_level);

                        log_info(
                            tt::LogMetal,
                            "Phase 2b: bound chip {} via MMIO ETH chan {} ({} hops)",
                            info.chip_id,
                            saved_mmio_core_logical.y,
                            next_hop_level);

                        // Upgrade remote chip info with retry.  Multi-hop reads
                        // can intermittently time out if the response is lost in
                        // the forwarding chain (e.g. race between forwarding lazy-init
                        // and the first forwarded packet).  Re-syncing h2d/d2h and
                        // retrying recovers from both "FW didn't see h2d" and
                        // "response lost in chain" failure modes.
                        constexpr int k_MaxUpgradeRetries = 2;
                        bool upgrade_succeeded = false;
                        for (int attempt = 0; attempt <= k_MaxUpgradeRetries; attempt++) {
                            try {
                                cluster_->upgrade_remote_bh_chip_info(info.chip_id);
                                upgrade_succeeded = true;
                                break;
                            } catch (const std::runtime_error& e) {
                                if (attempt < k_MaxUpgradeRetries) {
                                    log_warning(
                                        tt::LogMetal,
                                        "Phase 2b: upgrade_remote_bh_chip_info({}) attempt {} failed: {}. "
                                        "Re-syncing h2d/d2h and retrying...",
                                        info.chip_id,
                                        attempt + 1,
                                        e.what());

                                    // Diagnostic: read intermediate core state before resync.
                                    // Upstream receiver and downstream core on intermediate chip
                                    // are 1-hop reachable.  First resync chip 1's communication
                                    // so reads work, then dump state from both cores.
                                    try {
                                        auto* interm_chip =
                                            cluster_->get_driver()->get_remote_chip(info.intermediate_chip);
                                        std::set<uint32_t> interm_channels = {
                                            static_cast<uint32_t>(saved_mmio_core_logical.y)};
                                        interm_chip->set_remote_transfer_ethernet_cores(interm_channels);

                                        uint32_t handshake_addr =
                                            config_addr +
                                            offsetof(lite_fabric::FabricLiteConfig, primary_local_handshake);
                                        uint32_t padding1_addr =
                                            config_addr + offsetof(lite_fabric::FabricLiteConfig, padding1);
                                        uint32_t loop_addr =
                                            config_addr + offsetof(lite_fabric::FabricLiteConfig, neighbour_handshake);

                                        uint32_t padding2_addr =
                                            config_addr + offsetof(lite_fabric::FabricLiteConfig, padding2);

                                        uint32_t padding0_addr =
                                            config_addr + offsetof(lite_fabric::FabricLiteConfig, padding0);
                                        auto dump_core = [&](const char* label, tt_cxy_pair cxy) {
                                            uint32_t d2h_word = 0, h2d_word = 0;
                                            uint32_t sdiag = 0, rdiag = 0, loop_cnt = 0;
                                            uint32_t fwd_diag = 0, fwd_tgt = 0;
                                            lite_fabric::FabricLiteConfig::ForwardingConfig fwd_cfg{};
                                            cluster_->read_core(&d2h_word, 4, cxy, host_iface_addr);
                                            using HostIface = lite_fabric::HostToFabricLiteInterface<
                                                lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0],
                                                lite_fabric::CHANNEL_BUFFER_SIZE>;
                                            cluster_->read_core(
                                                &h2d_word, 4, cxy, host_iface_addr + offsetof(HostIface, h2d));
                                            cluster_->read_core(&sdiag, 4, cxy, handshake_addr);
                                            cluster_->read_core(&rdiag, 4, cxy, padding1_addr);
                                            cluster_->read_core(&loop_cnt, 4, cxy, loop_addr);
                                            cluster_->read_core(&fwd_diag, 4, cxy, padding2_addr);
                                            cluster_->read_core(&fwd_tgt, 4, cxy, padding0_addr);
                                            cluster_->read_core(&fwd_cfg, sizeof(fwd_cfg), cxy, forwarding_offset);
                                            log_warning(
                                                tt::LogMetal,
                                                "Phase 2b diag {}: chip {} core ({},{}) | "
                                                "d2h.s={} d2h.r={} h2d.s={} h2d.r={} | "
                                                "sender: nfs={} comp={} unsent={} can={} | "
                                                "receiver: wr_sent={} comp={} d2h_idx={} h2d_idx={} | "
                                                "fwd_cfg: en={} noc=({},{}) init_wr={} | "
                                                "fwd_fw: wr_idx={} en={} mmio={} routing={} | "
                                                "fwd_tgt: noc=({},{}) wr={} addr_lo=0x{:02x} | loop={}",
                                                label,
                                                cxy.chip,
                                                cxy.x,
                                                cxy.y,
                                                d2h_word & 0xFF,
                                                (d2h_word >> 8) & 0xFF,
                                                h2d_word & 0xFF,
                                                (h2d_word >> 8) & 0xFF,
                                                (sdiag >> 24) & 0xFF,
                                                (sdiag >> 16) & 0xFF,
                                                (sdiag >> 8) & 0xFF,
                                                sdiag & 0xFF,
                                                (rdiag >> 24) & 0xFF,
                                                (rdiag >> 16) & 0xFF,
                                                (rdiag >> 8) & 0xFF,
                                                rdiag & 0xFF,
                                                fwd_cfg.enabled,
                                                fwd_cfg.downstream_noc_x,
                                                fwd_cfg.downstream_noc_y,
                                                fwd_cfg.initial_wr_idx,
                                                fwd_diag & 0xFF,
                                                (fwd_diag >> 8) & 0xFF,
                                                (fwd_diag >> 16) & 0xFF,
                                                (fwd_diag >> 24) & 0x3,
                                                (fwd_tgt >> 24) & 0xFF,
                                                (fwd_tgt >> 16) & 0xFF,
                                                (fwd_tgt >> 8) & 0xFF,
                                                fwd_tgt & 0xFF,
                                                loop_cnt);
                                        };

                                        auto up_rx_cxy = tt_cxy_pair(
                                            saved_connected_id,
                                            saved_connected_core_virtual.x,
                                            saved_connected_core_virtual.y);
                                        dump_core("upstream_rx", up_rx_cxy);
                                        dump_core("downstream", ds_cxy);
                                    } catch (const std::exception& diag_ex) {
                                        log_warning(
                                            tt::LogMetal, "Phase 2b: diagnostic reads failed: {}", diag_ex.what());
                                    }

                                    // Re-sync h2d/d2h counters with device state
                                    remote_chip->set_remote_transfer_ethernet_cores(channels);
                                    remote_chip->get_remote_communication()->set_num_hops(next_hop_level);
                                } else {
                                    log_warning(
                                        tt::LogMetal,
                                        "Phase 2b: upgrade_remote_bh_chip_info({}) failed after {} retries, "
                                        "skipping chip (remaining chips in this hop level will still be processed)",
                                        info.chip_id,
                                        k_MaxUpgradeRetries + 1);
                                    break;
                                }
                            }
                        }
                        if (upgrade_succeeded) {
                            cluster_->refresh_soc_desc_for_chip(info.chip_id);

                            // Add to remote_devices for Phase 3 FW launch
                            remote_devices.insert(info.chip_id);
                            all_nhop_chips.insert(info.chip_id);
                            reachable_chips.insert(info.chip_id);
                        }
                    }

                    // Only use successfully upgraded chips as frontier for
                    // the next BFS iteration.  Failed chips can't be read
                    // (their forwarding chain is broken), so including them
                    // causes ~30s of read timeouts per chip and prevents
                    // discovering deeper-hop chips beyond them.
                    current_frontier.clear();
                    for (auto cid : new_chip_ids) {
                        if (reachable_chips.count(cid)) {
                            current_frontier.insert(cid);
                        }
                    }
                    current_hop_level = next_hop_level;
                }

                if (!all_nhop_chips.empty()) {
                    cluster_->refresh_remote_ethernet_routing_info();
                    control_plane_.reset();
                    log_info(
                        tt::LogMetal,
                        "Phase 2b: total {} N-hop chips discovered and made reachable",
                        all_nhop_chips.size());
                } else {
                    log_info(tt::LogMetal, "Phase 2b: no additional chips discovered beyond 1-hop");
                }
            }

            // Rebind intermediate devices to exclude MMIO ETH channels that
            // have forwarding configured for deeper n-hop chips.  If a read to
            // an intermediate chip goes through a forwarding-enabled channel,
            // the upstream sender's d2h advances without updating the deeper
            // downstream receiver's forwarding_downstream_wr_idx.  When a
            // subsequent deeper read response arrives, the downstream receiver
            // writes to the wrong upstream sender slot → response lost.
            // This applies to ALL intermediate hops, not just 1-hop chips.
            std::map<std::pair<ChipId, ChipId>, std::set<uint32_t>> remote_chip_eth_channels;
            for (const auto& t : lite_fabric_hal_->get_system_descriptor().tunnels_from_mmio) {
                remote_chip_eth_channels[{t.connected_id, t.mmio_id}].insert(t.mmio_core_logical.y);
            }
            for (const auto& [key, fwd_channels] : forwarded_channels_per_1hop) {
                auto [chip_id, mmio_id] = key;
                auto orig_it = remote_chip_eth_channels.find(key);
                if (orig_it == remote_chip_eth_channels.end()) {
                    continue;
                }
                const auto& orig_channels = orig_it->second;

                std::set<uint32_t> clean_channels;
                for (uint32_t ch : orig_channels) {
                    if (fwd_channels.find(ch) == fwd_channels.end()) {
                        clean_channels.insert(ch);
                    }
                }

                if (clean_channels.empty()) {
                    log_warning(
                        tt::LogMetal,
                        "Phase 2b: all channels for chip {} have forwarding, "
                        "cannot rebind (deeper reads may conflict)",
                        chip_id);
                    continue;
                }

                if (clean_channels != orig_channels) {
                    auto* remote_chip = cluster_->get_driver()->get_remote_chip(chip_id);
                    remote_chip->set_remote_transfer_ethernet_cores(clean_channels);
                    log_info(
                        tt::LogMetal,
                        "Phase 2b: rebound chip {} to non-forwarded channels [{}] "
                        "(excluded forwarded [{}])",
                        chip_id,
                        fmt::join(clean_channels, ", "),
                        fmt::join(fwd_channels, ", "));
                }
            }

            // Refresh dispatch_core_manager now that BFS discovery may have added
            // N-hop chips to the UMD cluster.  The manager was created at the top
            // of initialize() with only the initially-known chip IDs (MMIO +
            // 1-hop).  Without this, any N-hop chip (2+ hops from MMIO) would be
            // missing from available_dispatch_cores_by_device and trigger
            // "Invalid device ID to assign dispatch cores" later.
            dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config_, num_hw_cqs_);

            // Phase 3: FW builds, device init, resets, and FW launch for remote chips
            // (now reachable via lite fabric)
            {
                ZoneScopedN("Remote Device Init and FW Launch");
                log_info(
                    tt::LogMetal, "Phase 3: build_and_init_devices for {} remote device(s)", remote_devices.size());
                // Process devices deepest-first (descending hop count).  This prevents
                // reads to intermediate chips from desyncing forwarding_downstream_wr_idx
                // on downstream receivers before deeper reads are processed.
                // Build a vector sorted by hop count (descending), breaking ties by chip ID (descending).
                auto& tunnels = lite_fabric_hal_->get_system_descriptor().tunnels_from_mmio;
                std::vector<ChipId> remote_devices_ordered(remote_devices.begin(), remote_devices.end());
                std::sort(remote_devices_ordered.begin(), remote_devices_ordered.end(), [&tunnels](ChipId a, ChipId b) {
                    int hops_a = 1, hops_b = 1;
                    for (const auto& t : tunnels) {
                        if (t.connected_id == a) {
                            hops_a = t.num_hops;
                        }
                        if (t.connected_id == b) {
                            hops_b = t.num_hops;
                        }
                    }
                    return hops_a != hops_b ? hops_a > hops_b : a > b;
                });
                // Run sequentially and skip ETH cores for remote devices behind
                // lite fabric.  All writes to remote chips go through lite fabric
                // tunnels which share sender/receiver buffers on the MMIO-side
                // ETH core and are not safe for concurrent access from multiple
                // threads.  ETH cores are skipped because ERISC0 is not running on
                // remote ETH cores (all in POR reset), and the lite fabric ETH core
                // has ERISC1 actively servicing the link.
                // Process each remote device individually with lite fabric
                // re-sync between devices.  Multiple N-hop devices share the
                // same MMIO ETH core (e.g., 2-hop and 4-hop devices both route
                // through channel 7).  When processed sequentially, each
                // device's reads advance the shared MMIO-side ch1 ring state,
                // leaving the next device's cached recv_ch1 stale and pointing
                // to the wrong receiver slot (0xdeadbeef sentinel timeout).
                // set_remote_transfer_ethernet_cores re-syncs the active
                // channel's ch1 position from device state and clears stale
                // event IDs before the next device uses that path.
                for (ChipId device_id : remote_devices_ordered) {
                    auto gateway = cluster_->get_cluster_desc()->get_closest_mmio_capable_chip(device_id);
                    auto ch_it = remote_chip_eth_channels.find({device_id, gateway});
                    if (ch_it != remote_chip_eth_channels.end()) {
                        auto* remote_chip = cluster_->get_driver()->get_remote_chip(device_id);
                        remote_chip->set_remote_transfer_ethernet_cores(ch_it->second);
                    }
                    build_and_init_devices(std::vector<ChipId>{device_id}, /*sequential=*/true, /*skip_eth=*/true);
                }
                log_info(tt::LogMetal, "Phase 3: build_and_init_devices complete, launching FW for remote devices");
                // Skip reset_cores for remote BH chips: their Tensix cores are
                // already in POR reset, and reset_cores would kill the lite
                // fabric by resetting ERISC1 on the active ETH core (the remote
                // end of the lite fabric link).  ClearNocData is also a NO-OP
                // when NOC recording is disabled (the default).
                //
                // Launch FW sequentially for remote devices because all writes
                // go through lite fabric tunnels which share sender/receiver
                // buffers on the MMIO-side ETH core and are not safe for
                // concurrent access from multiple threads.
                for (ChipId device_id : remote_devices_ordered) {
                    // Re-sync lite fabric interface (same sharing issue as
                    // build_and_init above — FW launch may read from device).
                    auto gateway = cluster_->get_cluster_desc()->get_closest_mmio_capable_chip(device_id);
                    auto ch_it = remote_chip_eth_channels.find({device_id, gateway});
                    if (ch_it != remote_chip_eth_channels.end()) {
                        auto* remote_chip = cluster_->get_driver()->get_remote_chip(device_id);
                        remote_chip->set_remote_transfer_ethernet_cores(ch_it->second);
                    }
                    log_info(tt::LogMetal, "launch_fw device {} (remote): initialize_and_launch_firmware", device_id);
                    initialize_and_launch_firmware(device_id);
                    log_info(tt::LogMetal, "launch_fw device {} (remote): complete", device_id);
                }
                log_info(tt::LogMetal, "Phase 3: remote FW launch complete");
            }

            // Phase 3b: Ensure build environments exist for unreachable remote
            // devices.  These chips were skipped in Phase 3 because their lite
            // fabric tunnels failed, but the DeviceManager will still create
            // Device objects for them (fabric requires all devices to be
            // active), and fabric compilation needs build environments.
            for (ChipId id : all_devices) {
                if (initial_devices.count(id) || remote_devices.count(id)) {
                    continue;  // Already initialized in Phase 1 or 3
                }
                log_info(tt::LogMetal, "Phase 3b: adding build env for unreachable device {}", id);
                unreachable_chip_ids_.insert(id);
                generate_device_bank_to_noc_tables(id);
                generate_worker_logical_to_virtual_map(id);
                BuildEnvManager::get_instance().add_build_env(id, num_hw_cqs_);
            }

            // Remove unreachable chips from the cluster descriptor so the
            // control plane (PhysicalSystemDescriptor / TopologyMapper) won't
            // include them in the fabric mesh.  Without this, dispatch tries to
            // create routing tables for chips that have no valid forwarding path.
            for (ChipId id : unreachable_chip_ids_) {
                log_info(tt::LogMetal, "Phase 3b: removing unreachable chip {} from cluster descriptor", id);
                cluster_->get_cluster_desc()->remove_chip(id);
            }

            // Phase 3c: Rebuild UMD tunnel structure with flat [MMIO, remote]
            // tunnels for all reachable remote devices.  The original
            // discover_tunnels_from_mmio_device() only found 1-hop remotes
            // because ETH connections for multi-hop chips weren't in the cluster
            // descriptor yet.  Dispatch topology (generate_nodes,
            // GetUpstreamDeviceId, etc.) needs every remote device to appear in
            // a tunnel.
            for (ChipId mmio_id : cluster_->mmio_chip_ids()) {
                std::vector<std::vector<ChipId>> flat_tunnels;
                for (ChipId remote_id : remote_devices) {
                    if (!unreachable_chip_ids_.contains(remote_id)) {
                        flat_tunnels.push_back({mmio_id, remote_id});
                    }
                }
                if (!flat_tunnels.empty()) {
                    cluster_->set_tunnels_from_mmio(mmio_id, std::move(flat_tunnels));
                    log_info(
                        tt::LogMetal,
                        "Phase 3c: updated UMD tunnels for MMIO {} with {} flat tunnels",
                        mmio_id,
                        cluster_->get_tunnels_from_mmio_device(mmio_id).size());
                }
            }
        } else {
            launch_fw_for_devices(initial_devices);
        }
    }
    // Watcher needs to init before FW since FW needs watcher mailboxes to be set up, and needs to attach after FW
    // starts since it also writes to watcher mailboxes.
    watcher_server_->attach_devices();

    // Register teardown function, but only once.
    if (not teardown_registered_) {
        std::atexit([]() { MetalContext::instance().teardown(); });
        teardown_registered_ = true;
    }
}

// IMPORTANT: This function is registered as an atexit handler. Creating threads during program termination may cause
// undefined behavior. Do not create threads in this function or any functions it calls.
void MetalContext::teardown() {
    ZoneScoped;

    if (!initialized_) {
        return;
    }
    initialized_ = false;

    auto all_devices = cluster_->all_chip_ids();
    // If simulator is enabled, force a teardown of active ethernet cores for WH
    if (rtoptions_.get_simulator_enabled()) {
        if (hal_->get_eth_fw_is_cooperative()) {
            for (ChipId device_id : all_devices) {
                for (const auto& logical_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
                    CoreCoord virtual_core = cluster_->get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, CoreType::ETH);
                    erisc_send_exit_signal(device_id, virtual_core, false);
                    while (erisc_app_still_running(device_id, virtual_core)) {
                    }
                }
            }
        }
    }

    // Set internal routing to false to exit active ethernet FW & go back to base FW
    // Must happen before lite fabric termination since writes to remote ETH cores go through lite fabric
    cluster_->set_internal_routing_info_for_ethernet_cores(false);

    // Terminate lite fabric (stops ERISC1 on MMIO chip) after routing info is cleared on remote devices
    if (lite_fabric_hal_) {
        lite_fabric_hal_->terminate();
        lite_fabric_hal_.reset();
    }

    // Reset lite_fabric_running_ on all remote chips so that UMD silently drops
    // any subsequent reads/writes (e.g. from watcher init during reinit) until
    // upgrade_remote_bh_chip_info() re-enables them after lite fabric relaunch.
    {
        auto mmio_ids = cluster_->mmio_chip_ids();
        for (ChipId id : all_devices) {
            if (!mmio_ids.contains(id)) {
                auto* remote_chip = cluster_->get_driver()->get_remote_chip(id);
                if (remote_chip) {
                    remote_chip->downgrade_after_lite_fabric_teardown();
                }
            }
        }
    }

    if (data_collector_) {
        data_collector_->DumpData();
        data_collector_.reset();
    }

    if (dprint_server_) {
        dprint_server_->detach_devices();
        dprint_server_.reset();
        rtoptions_.set_disable_dma_ops(false);
    }

    watcher_server_->detach_devices();
    watcher_server_.reset();

    // Assert cores on MMIO devices only.  Remote devices are unreachable after lite fabric
    // termination, and their cores will be reset during the next init cycle.
    auto mmio_ids = cluster_->mmio_chip_ids();
    for (ChipId device_id : all_devices) {
        if (unreachable_chip_ids_.contains(device_id)) {
            continue;
        }
        if (!mmio_ids.contains(device_id)) {
            continue;
        }
        assert_cores(device_id);

        cluster_->l1_barrier(device_id);
    }

    if (profiler_state_manager_) {
        profiler_state_manager_.reset();
    }

    for (auto& mem_map : dispatch_mem_map_) {
        if (mem_map) {
            mem_map.reset();
        }
    }

    dispatch_query_manager_.reset();
    dispatch_core_manager_.reset();
    // Reset fabric manager mode so that the next initialize() doesn't
    // try to send exit signals / wait for heartbeats on ETH cores that
    // are in POR after a board reset.
    // NOTE: FabricManagerMode::DEFAULT = (INIT_FABRIC | TERMINATE_FABRIC) = 3,
    // which still has INIT_FABRIC set. We need to fully clear all flags so
    // reset_cores() won't enter the active ETH core heartbeat-wait block.
    fabric_manager_ = static_cast<tt_fabric::FabricManagerMode>(0);
    tt::tt_metal::reset_topology_state();

    // Clear dispatch, dispatch_s and prefetcher core info in inspector data
    Inspector::clear_all_core_info();
    // Deinitialize inspector
    inspector_data_.reset();

    control_plane_.reset();
}

MetalContext& MetalContext::instance() {
    static tt::stl::Indestructible<MetalContext> inst;
    return inst.get();
}

void MetalContext::teardown_base_objects() {
    // Teardown in backward order of dependencies to avoid dereferencing uninitialized objects
    distributed_context_.reset();
    // Destroy inspector before cluster to prevent RPC handlers from accessing destroyed cluster
    inspector_data_.reset();
    cluster_.reset();
    hal_.reset();
}

MetalContext::MetalContext() {
    // If a custom fabric mesh graph descriptor is specified as an RT Option, use it by default
    // to initialize the control plane.
    if (rtoptions_.is_custom_fabric_mesh_graph_desc_path_specified()) {
        custom_mesh_graph_desc_path_ = rtoptions_.get_custom_fabric_mesh_graph_desc_path();
    }

    const bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
    const auto platform_arch = get_platform_architecture(rtoptions_);

    const auto initialize_objects = [&]() {
        hal_ = std::make_unique<Hal>(
            platform_arch,
            is_base_routing_fw_enabled,
            rtoptions_.get_enable_2_erisc_mode(),
            get_profiler_dram_bank_size_per_risc_bytes(rtoptions_));
        rtoptions_.ParseAllFeatureEnv(*hal_);
        cluster_ = std::make_unique<Cluster>(rtoptions_, *hal_);
        distributed_context_ = distributed::multihost::DistributedContext::get_current_world();
    };

    initialize_objects();

    // Requires reinit with features disabled
    // This will maintain backward compatibility with clusters that have legacy firmware but it will cause a slowdown
    // during the first init
    if (!cluster_->verify_eth_fw_capability()) {
        rtoptions_.set_enable_2_erisc_mode(false);
        teardown_base_objects();
        initialize_objects();
    }

    // Initialize some container members to allow threadsafe operations on them later
    dram_bank_offset_map_.reserve(cluster_->all_chip_ids().size());
    l1_bank_offset_map_.reserve(cluster_->all_chip_ids().size());
    dram_bank_to_noc_xy_.reserve(cluster_->all_chip_ids().size());
    l1_bank_to_noc_xy_.reserve(cluster_->all_chip_ids().size());
    worker_logical_col_to_virtual_col_.reserve(cluster_->all_chip_ids().size());
    worker_logical_row_to_virtual_row_.reserve(cluster_->all_chip_ids().size());
    for (ChipId device_id : cluster_->all_chip_ids()) {
        dram_bank_offset_map_.emplace(device_id, std::vector<int32_t>{});
        l1_bank_offset_map_.emplace(device_id, std::vector<int32_t>{});
        dram_bank_to_noc_xy_.emplace(device_id, std::vector<uint16_t>{});
        l1_bank_to_noc_xy_.emplace(device_id, std::vector<uint16_t>{});
        worker_logical_col_to_virtual_col_.emplace(device_id, std::vector<uint8_t>{});
        worker_logical_row_to_virtual_row_.emplace(device_id, std::vector<uint8_t>{});
    }

    device_manager_ = std::make_unique<DeviceManager>();

    // We do need to call Cluster teardown at the end of the program, use atexit temporarily until we have clarity on
    // how MetalContext lifetime will work through the API.
    std::atexit([]() { MetalContext::instance().~MetalContext(); });
}

const distributed::multihost::DistributedContext& MetalContext::full_world_distributed_context() const {
    TT_FATAL(distributed_context_, "Distributed context not initialized.");
    return *distributed_context_;
}

const distributed::multihost::DistributedContext& MetalContext::global_distributed_context() {
    // If control plane is not initilazed, return the global distributed context
    if (!control_plane_) {
        return *distributed_context_;
    }
    // Lazy initilazation of compute only distributed context
    if (!compute_only_distributed_context_) {
        compute_only_distributed_context_ = construct_compute_only_distributed_context(*this);
    }
    return *compute_only_distributed_context_;
}

std::shared_ptr<distributed::multihost::DistributedContext> MetalContext::get_distributed_context_ptr() {
    TT_FATAL(distributed_context_, "Distributed context not initialized.");
    return distributed_context_;
}

MetalContext::~MetalContext() {
    device_manager_.reset();
    teardown_base_objects();
}

llrt::RunTimeOptions& MetalContext::rtoptions() { return rtoptions_; }

Cluster& MetalContext::get_cluster() {
    TT_FATAL(cluster_, "Trying to get cluster before initializing it.");
    return *cluster_;
}

const llrt::RunTimeOptions& MetalContext::rtoptions() const { return rtoptions_; }

const Cluster& MetalContext::get_cluster() const {
    TT_FATAL(cluster_, "Trying to get cluster before initializing it.");
    return *cluster_;
}

const Hal& MetalContext::hal() const {
    TT_FATAL(hal_, "Trying to get hal before initializing it.");
    return *hal_;
}

dispatch_core_manager& MetalContext::get_dispatch_core_manager() {
    TT_FATAL(dispatch_core_manager_, "Trying to get dispatch_core_manager before initializing it.");
    return *dispatch_core_manager_;
}

DispatchQueryManager& MetalContext::get_dispatch_query_manager() {
    TT_FATAL(dispatch_query_manager_, "Trying to get dispatch_query_manager before initializing it.");
    return *dispatch_query_manager_;
}

const DispatchMemMap& MetalContext::dispatch_mem_map() const {
    return dispatch_mem_map(dispatch_core_config_.get_core_type());
}

const DispatchMemMap& MetalContext::dispatch_mem_map(const CoreType& core_type) const {
    const auto& mem_map = dispatch_mem_map_[enchantum::to_underlying(core_type)];
    TT_FATAL(mem_map, "Tried to get dispatch_mem_map for {} before initializing it.", core_type);
    return *mem_map;
}

void MetalContext::clear_l1_state(ChipId device_id, bool skip_eth_cores) {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", device_id);
    // Clear all clearable Tensix and Eth L1
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t l1_size_per_core = cluster_->get_soc_desc(device_id).worker_l1_size;
    TT_ASSERT(l1_size_per_core % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(l1_size_per_core / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            auto virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            cluster_->write_core(device_id, virtual_core, zero_vec, start_address);
        }
    }

    if (!skip_eth_cores) {
        // Clear erisc unreserved L1
        // Skipped for remote devices behind lite fabric: ERISC0 is not running on
        // remote ETH cores (all in POR reset), and the lite fabric ETH core has
        // ERISC1 actively servicing the link — writing to its L1 is unnecessary.
        for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
            static uint32_t zero_vec_size = hal::get_erisc_l1_unreserved_size();
            auto zero_vec_addr = hal::get_erisc_l1_unreserved_base();

            static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);

            CoreCoord virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
            cluster_->write_core(device_id, virtual_core, zero_vec, zero_vec_addr);
        }
        // TODO: clear idle eriscs as well
    }
    cluster_->l1_barrier(device_id);
}

void MetalContext::clear_dram_state(ChipId device_id) {
    log_debug(tt::LogMetal, "Clearing DRAM for device {}", device_id);

    auto dram_size_per_channel = cluster_->get_soc_desc(device_id).dram_view_size;
    auto num_dram_channels = cluster_->get_soc_desc(device_id).get_num_dram_views();
    constexpr uint32_t start_address = 0;
    std::vector<uint8_t> zero_vec(dram_size_per_channel, 0);
    for (int channel = 0; channel < num_dram_channels; ++channel) {
        cluster_->write_dram_vec(zero_vec.data(), zero_vec.size(), device_id, channel, start_address);

        cluster_->dram_barrier(device_id);
    }
}

void MetalContext::clear_launch_messages_on_eth_cores(ChipId device_id) {
    auto clear_ethernet_core = [&](const CoreCoord& logical_eth_core, HalProgrammableCoreType programmable_core_type) {
        auto factory = hal_->get_dev_msgs_factory(programmable_core_type);
        std::vector<std::byte> init_launch_msg_data(
            dev_msgs::launch_msg_buffer_num_entries * factory.size_of<dev_msgs::launch_msg_t>(), std::byte{0});
        dev_msgs::go_msg_t go_msg = factory.create<dev_msgs::go_msg_t>();
        go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

        CoreCoord virtual_eth_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
        cluster_->write_core(
            init_launch_msg_data.data(),
            init_launch_msg_data.size(),
            tt_cxy_pair(device_id, virtual_eth_core),
            hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
        cluster_->write_core(
            go_msg.data(),
            go_msg.size(),
            {static_cast<size_t>(device_id), virtual_eth_core},
            hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG));
    };

    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        clear_ethernet_core(eth_core, HalProgrammableCoreType::ACTIVE_ETH);
    }
    for (const auto& eth_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
        if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        clear_ethernet_core(eth_core, HalProgrammableCoreType::IDLE_ETH);
    }

    cluster_->l1_barrier(device_id);
}

tt::tt_fabric::ControlPlane& MetalContext::get_control_plane() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    if (!control_plane_) {
        this->initialize_control_plane_impl();
    }
    return *control_plane_;
}

void MetalContext::set_custom_fabric_topology(
    const std::string& mesh_graph_desc_file,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    TT_FATAL(
        !device_manager_->is_initialized() || device_manager_->get_all_active_devices().empty(),
        "Modifying control plane requires no devices to be active");
    // Set the user specified mesh graph descriptor file and FabricNodeID to physical chip mapping.
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_ = logical_mesh_chip_id_to_physical_chip_id_mapping;
    custom_mesh_graph_desc_path_ = mesh_graph_desc_file;
    this->set_fabric_config(fabric_config_, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

void MetalContext::set_default_fabric_topology() {
    TT_FATAL(
        !device_manager_->is_initialized() || device_manager_->get_all_active_devices().empty(),
        "Modifying control plane requires no devices to be active");
    // Reset the control plane, since it was initialized with custom parameters.
    control_plane_.reset();
    // Set the mesh graph descriptor file to the default value and clear the custom FabricNodeId to physical chip
    // mapping.
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.clear();

    if (rtoptions_.is_custom_fabric_mesh_graph_desc_path_specified()) {
        custom_mesh_graph_desc_path_ = rtoptions_.get_custom_fabric_mesh_graph_desc_path();
    } else {
        custom_mesh_graph_desc_path_ = std::nullopt;
    }
    this->set_fabric_config(fabric_config_, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

void MetalContext::teardown_fabric_config() {
    this->fabric_config_ = tt_fabric::FabricConfig::DISABLED;
    this->cluster_->configure_ethernet_cores_for_fabric_routers(this->fabric_config_);
    this->num_fabric_active_routing_planes_ = 0;
    // if (!rtoptions_.get_erisc_iram_env_var_enabled()) {
    //     rtoptions_.set_erisc_iram_enabled(false);
    // }
    this->get_control_plane().clear_fabric_context();
}

void MetalContext::set_fabric_config(
    const tt_fabric::FabricConfig fabric_config,
    tt_fabric::FabricReliabilityMode reliability_mode,
    std::optional<uint8_t> num_routing_planes,
    tt_fabric::FabricTensixConfig fabric_tensix_config,
    tt_fabric::FabricUDMMode fabric_udm_mode,
    tt_fabric::FabricManagerMode fabric_manager) {
    // Changes to fabric force a re-init. TODO: We should supply the fabric config in the same way as the dispatch
    // config, not through this function exposed in the detail API.
    force_reinit_ = true;

    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED ||
        fabric_config == tt_fabric::FabricConfig::DISABLED) {
        this->fabric_config_ = fabric_config;
        this->fabric_reliability_mode_ = reliability_mode;
    } else {
        TT_FATAL(
            this->fabric_config_ == fabric_config,
            "Tried to override previous value of fabric config: {}, with: {}",
            this->fabric_config_,
            fabric_config);
    }

    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        if (num_routing_planes.has_value()) {
            log_warning(
                tt::LogMetal,
                "Got num_routing_planes while disabling fabric, ignoring it and disabling all active routing planes");
        }

        this->teardown_fabric_config();
        return;
    }

    bool enable_erisc_iram =
        !rtoptions_.get_erisc_iram_env_var_enabled() || !rtoptions_.get_erisc_iram_env_var_disabled();
    rtoptions_.set_erisc_iram_enabled(enable_erisc_iram);

    if (num_routing_planes.has_value() && num_routing_planes.value() < this->num_fabric_active_routing_planes_) {
        log_warning(
            tt::LogMetal,
            "Got num_routing_planes: {}, which is less than current value: {}, ignoring the override",
            num_routing_planes.value(),
            this->num_fabric_active_routing_planes_);
        return;
    }

    // if num_routing_planes is not specified, use max available number of routing planes
    // ideally the highest value should be the maximum number of eth cores in a direction across all chips
    const auto new_val = std::max(
        this->num_fabric_active_routing_planes_, num_routing_planes.value_or(std::numeric_limits<uint8_t>::max()));
    if (new_val != this->num_fabric_active_routing_planes_ && this->num_fabric_active_routing_planes_ > 0) {
        log_info(
            tt::LogMetal,
            "Overriding the number of routing planes to activate from {} to {}",
            this->num_fabric_active_routing_planes_,
            new_val);
    }
    this->num_fabric_active_routing_planes_ = new_val;

    // Set the fabric tensix config
    this->set_fabric_tensix_config(fabric_tensix_config);
    this->fabric_udm_mode_ = fabric_udm_mode;
    this->fabric_manager_ = fabric_manager;
}

void MetalContext::initialize_fabric_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    log_info(tt::LogMetal, "DEBUG: initialize_fabric_config: configuring eth cores for fabric routers");
    this->cluster_->configure_ethernet_cores_for_fabric_routers(
        this->fabric_config_, this->num_fabric_active_routing_planes_);
    log_info(tt::LogMetal, "DEBUG: initialize_fabric_config: getting control plane");
    auto& control_plane = this->get_control_plane();
    log_info(tt::LogMetal, "DEBUG: initialize_fabric_config: control plane created");
    if (tt::tt_fabric::is_tt_fabric_config(this->fabric_config_)) {
        log_info(tt::LogMetal, "DEBUG: initialize_fabric_config: initializing fabric context");
        control_plane.initialize_fabric_context(this->fabric_config_);
    }
    log_info(tt::LogMetal, "DEBUG: initialize_fabric_config: configuring routing tables");
    control_plane.configure_routing_tables_for_fabric_ethernet_channels(
        this->fabric_config_, this->fabric_reliability_mode_);
    log_info(tt::LogMetal, "DEBUG: initialize_fabric_config: done");
}

void MetalContext::update_lite_fabric_bindings_for_fabric_routers() {
    if (!lite_fabric_hal_) {
        return;
    }

    const auto& tunnels = lite_fabric_hal_->get_system_descriptor().tunnels_from_mmio;
    if (tunnels.empty()) {
        return;
    }

    // Identify MMIO channels used for multi-hop forwarding.  These channels
    // have forwarding enabled on the 1-hop receiver — routing 1-hop reads
    // through them causes forwarding_downstream_wr_idx desync and corrupts
    // the entire forwarding chain (devices 2+ hops away become unreachable).
    // Phase 2b already excluded these channels via rebinding; we must NOT
    // re-include them here.
    std::set<uint32_t> forwarding_mmio_channels;
    for (const auto& t : tunnels) {
        if (t.num_hops > 1) {
            forwarding_mmio_channels.insert(t.mmio_core_logical.y);
        }
    }

    // Identify 1-hop chips that have at least one forwarding channel.
    // For these chips, Phase 2b already bound them to non-forwarded channels
    // and we must preserve that binding.
    std::set<ChipId> chips_with_forwarding;
    for (const auto& t : tunnels) {
        if (t.num_hops == 1 && forwarding_mmio_channels.count(t.mmio_core_logical.y)) {
            chips_with_forwarding.insert(t.connected_id);
        }
    }

    // Group channels by remote chip, but only include channels whose remote
    // peers have fabric routers.  Skip chips that have forwarding chains —
    // their Phase 2b bindings to non-forwarded channels must be preserved.
    std::map<ChipId, std::set<uint32_t>> channels_per_remote;
    for (const auto& t : tunnels) {
        if (t.num_hops != 1) {
            continue;  // Only 1-hop tunnels are candidates for direct UMD binding
        }
        if (chips_with_forwarding.count(t.connected_id)) {
            log_info(
                tt::LogMetal,
                "Preserving Phase 2b binding for remote device {} — "
                "MMIO channel {} has forwarding, skipping rebind",
                t.connected_id,
                t.mmio_core_logical.y);
            continue;
        }
        auto it = remote_fabric_eth_channels_.find(t.connected_id);
        if (it != remote_fabric_eth_channels_.end() && it->second.count(t.connected_core_logical.y)) {
            channels_per_remote[t.connected_id].insert(t.mmio_core_logical.y);
        } else {
            log_info(
                tt::LogMetal,
                "Excluding MMIO channel {} for remote device {} — remote peer core {} has no fabric router",
                t.mmio_core_logical.y,
                t.connected_id,
                t.connected_core_logical.str());
        }
    }

    for (auto& [chip_id, channels] : channels_per_remote) {
        // Drain all pending lite fabric writes before re-syncing the host-side
        // h2d/d2h counters.  set_remote_transfer_ethernet_cores reads d2h from
        // the device and sets host h2d = d2h.  If the relay hasn't finished
        // processing a write (d2h < h2d on device), the sync picks up stale d2h,
        // and when the relay eventually advances d2h, it permanently disagrees
        // with the host's h2d, causing wait_for_all_writes_consumed to hang.
        cluster_->l1_barrier(chip_id);

        cluster_->get_driver()->get_remote_chip(chip_id)->set_remote_transfer_ethernet_cores(channels);
        log_info(
            tt::LogMetal,
            "Updated UMD binding for remote device {}: lite fabric channels=[{}]",
            chip_id,
            fmt::join(channels, ", "));
    }
}

void MetalContext::initialize_fabric_tensix_datamover_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    // Initialize fabric tensix config after routing tables are configured and devices are available
    if (tt::tt_fabric::is_tt_fabric_config(this->fabric_config_)) {
        auto& control_plane = this->get_control_plane();
        control_plane.initialize_fabric_tensix_datamover_config();
    }
}

tt_fabric::FabricConfig MetalContext::get_fabric_config() const { return fabric_config_; }

tt_fabric::FabricReliabilityMode MetalContext::get_fabric_reliability_mode() const { return fabric_reliability_mode_; }

void MetalContext::set_fabric_tensix_config(tt_fabric::FabricTensixConfig fabric_tensix_config) {
    fabric_tensix_config_ = fabric_tensix_config;
}

tt_fabric::FabricTensixConfig MetalContext::get_fabric_tensix_config() const { return fabric_tensix_config_; }

tt_fabric::FabricUDMMode MetalContext::get_fabric_udm_mode() const { return fabric_udm_mode_; }

tt_fabric::FabricManagerMode MetalContext::get_fabric_manager() const { return fabric_manager_; }

void MetalContext::construct_control_plane(const std::filesystem::path& mesh_graph_desc_path) {
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping_.empty()) {
        log_info(tt::LogDistributed, "Using custom Fabric Node Id to physical chip mapping.");
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(
            mesh_graph_desc_path.string(), logical_mesh_chip_id_to_physical_chip_id_mapping_);
    } else {
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
    }
}

void MetalContext::construct_control_plane() {
    // Use auto-discovery to generate mesh graph from physical system descriptor
    // This uses MeshGraph::generate_from_physical_system_descriptor which internally
    // uses map_mesh_to_physical to find a valid mapping
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping_.empty()) {
        log_warning(
            tt::LogDistributed,
            "Custom Fabric Node Id to physical chip mapping provided but no mesh graph descriptor path. "
            "Mapping will be ignored. Please provide a custom mesh graph descriptor path for custom logical to "
            "physical mapping.");
    }
    log_info(tt::LogDistributed, "Constructing control plane using auto-discovery (no mesh graph descriptor).");
    control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>();
}

void MetalContext::initialize_control_plane() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    initialize_control_plane_impl();
}

void MetalContext::initialize_control_plane_impl() {
    if (custom_mesh_graph_desc_path_.has_value()) {
        log_debug(tt::LogDistributed, "Using custom mesh graph descriptor: {}", custom_mesh_graph_desc_path_.value());
        std::filesystem::path mesh_graph_desc_path = std::filesystem::path(custom_mesh_graph_desc_path_.value());
        TT_FATAL(
            std::filesystem::exists(mesh_graph_desc_path),
            "Custom mesh graph descriptor file not found: {}",
            mesh_graph_desc_path.string());

        log_info(tt::LogDistributed, "Using custom mesh graph descriptor: {}", mesh_graph_desc_path.string());
        this->construct_control_plane(mesh_graph_desc_path);
        return;
    }
    // If no custom mesh graph descriptor use auto discovery to generate mesh graph
    log_info(tt::LogDistributed, "Using auto discovery to generate mesh graph.");

    if (*distributed_context_->size() == 1) {
        this->construct_control_plane();
    } else {
        auto cluster_type = cluster_->get_cluster_type();
        auto fabric_type = tt::tt_fabric::get_fabric_type(this->fabric_config_);
        std::filesystem::path mesh_graph_desc_path =
            tt::tt_fabric::MeshGraph::get_mesh_graph_descriptor_path_for_cluster_type(
                cluster_type, rtoptions_.get_root_dir(), fabric_type);

        log_debug(tt::LogMetal, "Using mesh graph descriptor: {}", mesh_graph_desc_path);

        TT_FATAL(!mesh_graph_desc_path.empty(), "No mesh graph descriptor found for cluster type");
        TT_FATAL(
            std::filesystem::exists(mesh_graph_desc_path),
            "Mesh graph descriptor file not found: {}",
            mesh_graph_desc_path.string());
        this->construct_control_plane(mesh_graph_desc_path);
    }
}

void MetalContext::reset_cores(ChipId device_id) {
    ZoneScoped;
    // Assert worker cores + dispatch cores, in case they were in a bad state from before.
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> device_to_early_exit_cores;

    if (has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        // Active ethernet
        if (hal_->get_eth_fw_is_cooperative()) {
            for (const auto& logical_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
                CoreCoord virtual_core =
                    cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
                if (erisc_app_still_running(device_id, virtual_core)) {
                    log_info(
                        tt::LogMetal,
                        "While initializing device {}, active ethernet dispatch core {} detected as still "
                        "running, issuing exit signal.",
                        device_id,
                        virtual_core.str());
                    erisc_send_exit_signal(device_id, virtual_core, false /* is_idle_eth */);
                    device_to_early_exit_cores[device_id].insert(virtual_core);
                }
            }
        } else {
            for (const auto& logical_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
                // Ensure exit to base firmware. Send this before assertion subordinate cores otherwise if we stop the
                // subordinates we could hang waiting for subordinates to finish
                CoreCoord virtual_core =
                    cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
                if (rtoptions_.get_enable_2_erisc_mode()) {
                    erisc_send_exit_signal(
                        device_id, virtual_core, false /* is_idle_eth */);  // Stop any running erisc kernels
                    llrt::internal_::return_to_base_firmware_and_wait_for_heartbeat(device_id, virtual_core);
                }
                // Only send reset to subordinate cores
                // Assert all cores except ERISC0, which is running base firmware.
                tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
            }
        }
    }
    // Early exiting dispatch cores should show RUN_MSG_DONE when they exit.
    for (auto& id_and_cores : device_to_early_exit_cores) {
        const int timeout_ms = 10000;  // 10 seconds for now
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(
                    id_and_cores.first, dev_msgs::RUN_MSG_GO, id_and_cores.second, timeout_ms);
            } catch (std::runtime_error& e) {
                log_warning(
                    tt::LogAlways,
                    "Detected dispatch kernels still running but failed to complete an early exit. This may happen "
                    "from time to time following a reset, continuing to FW initialization...");
            }
        }
    }

    // Reset Tensix cores
    CoreCoord grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
        }
    }

    if (has_flag(
            tt::tt_metal::MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        // Reset idle ethernet cores
        for (const auto& logical_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
            cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), tt::umd::RiscType::ALL);
        }
    }
    cluster_->l1_barrier(device_id);
}

void MetalContext::assert_cores(ChipId device_id) {
    auto dispatch_cores = get_virtual_dispatch_cores(device_id);
    auto routing_cores = get_virtual_dispatch_routing_cores(device_id);

    // Assert riscs on Tensix
    CoreCoord grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);

            if (!dispatch_cores.contains(worker_core) && !routing_cores.contains(worker_core)) {
                if (!hal_->get_eth_fw_is_cooperative() &&
                    this->get_control_plane().get_active_ethernet_cores(device_id, false).contains(logical_core)) {
                    // Cannot put these cores into reset because they are running base FW
                    // Below will return to base FW
                    continue;
                }
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), device_id);
            }
        }
    }

    if (!hal_->get_eth_fw_is_cooperative()) {
        // Assert riscs on active eth
        const auto assert_eth_core = [&](const CoreCoord& logical_eth_core) {
            CoreCoord virtual_eth_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
            if (rtoptions_.get_enable_2_erisc_mode()) {
                // In 2-erisc mode, ERISC0 may be stuck in service_eth_msg() after lite fabric
                // termination killed ERISC1. Skip the heartbeat wait and assert reset on ALL
                // cores including ERISC0. The next init cycle will re-boot everything.
                // NOTE: RiscType::ALL_TENSIX excludes ERISC0/ERISC1 on BH (get_soft_reset_reg_value
                // silently ignores ERISC bits). Use direct register write instead.
                if (cluster_->arch() == ARCH::BLACKHOLE) {
                    constexpr uint32_t kSoftResetAddr = 0xFFB121B0;
                    constexpr uint32_t kBothEriscsInReset = 0x47800;  // bits 11+12 set
                    cluster_->write_core(
                        &kBothEriscsInReset,
                        sizeof(kBothEriscsInReset),
                        tt_cxy_pair(device_id, virtual_eth_core),
                        kSoftResetAddr);
                } else {
                    cluster_->assert_risc_reset_at_core(
                        tt_cxy_pair(device_id, virtual_eth_core), tt::umd::RiscType::ALL_TENSIX);
                }
            } else {
                // Stop subordinate
                // Assert all cores except ERISC0, which is running base firmware.
                tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_eth_core), reset_val);
            }
        };

        for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
            assert_eth_core(eth_core);
        }
    }
}

CoreCoord MetalContext::virtual_noc0_coordinate(ChipId device_id, uint8_t noc_index, CoreCoord coord) {
    const auto& grid_size = cluster_->get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y || cluster_->arch() == ARCH::BLACKHOLE) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        // Coordinate in Physical NOC0 Space. Convert to Virtual.
        coord = cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, coord);
        // Derive virtual coord in noc_index space.
        CoreCoord virtual_coord = {
            hal_->noc_coordinate(noc_index, grid_size.x, coord.x),
            hal_->noc_coordinate(noc_index, grid_size.y, coord.y)};
        return virtual_coord;
    }
}

void MetalContext::generate_device_bank_to_noc_tables(ChipId device_id) {
    // Create a dummp allocator to generatoe the bank/noc tables. Specifically, these depend on l1_bank_remap.
    auto config = L1BankingAllocator::generate_config(
        device_id,
        num_hw_cqs_,
        DEFAULT_L1_SMALL_SIZE,      // Not required for noc table gen
        DEFAULT_TRACE_REGION_SIZE,  // Not required for noc table gen
        worker_l1_unreserved_start_,
        l1_bank_remap_);
    const auto allocator = L1BankingAllocator(config);
    const auto& soc_d = cluster_->get_soc_desc(device_id);
    const size_t num_dram_banks = allocator.get_num_banks(BufferType::DRAM);
    dram_bank_offset_map_[device_id].clear();
    dram_bank_offset_map_[device_id].resize(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        dram_bank_offset_map_[device_id][bank_id] = allocator.get_bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = allocator.get_num_banks(BufferType::L1);
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    l1_bank_offset_map_[device_id].clear();
    l1_bank_offset_map_[device_id].resize(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] = cluster_->get_virtual_coordinate_from_logical_coordinates(
            device_id, allocator.get_logical_core_from_bank_id(bank_id), CoreType::WORKER);
        l1_bank_offset_map_[device_id][bank_id] = allocator.get_bank_offset(BufferType::L1, bank_id);
    }

    dram_bank_to_noc_xy_[device_id].clear();
    dram_bank_to_noc_xy_[device_id].reserve(hal_->get_num_nocs() * num_dram_banks);
    bool noc_translation_enabled = cluster_->get_cluster_desc()->get_noc_translation_table_en().at(device_id);
    bool dram_is_virtualized =
        noc_translation_enabled && (hal_->get_virtualized_core_types().contains(dev_msgs::AddressableCoreType::DRAM));
    for (unsigned int noc = 0; noc < hal_->get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < num_dram_banks; bank_id++) {
            uint16_t noc_x, noc_y;
            CoreCoord dram_noc_coord =
                soc_d.get_preferred_worker_core_for_dram_view(allocator.get_dram_channel_from_bank_id(bank_id), noc);
            if (dram_is_virtualized) {
                noc_x = dram_noc_coord.x;
                noc_y = dram_noc_coord.y;
            } else {
                noc_x = hal_->noc_coordinate(noc, soc_d.grid_size.x, dram_noc_coord.x);
                noc_y = hal_->noc_coordinate(noc, soc_d.grid_size.y, dram_noc_coord.y);
            }
            uint16_t xy = ((noc_y << hal_->get_noc_addr_node_id_bits()) | noc_x) << hal_->get_noc_coord_reg_offset();
            dram_bank_to_noc_xy_[device_id].push_back(xy);
        }
    }

    l1_bank_to_noc_xy_[device_id].clear();
    l1_bank_to_noc_xy_[device_id].reserve(hal_->get_num_nocs() * l1_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < hal_->get_num_nocs(); noc++) {
        for (const auto& noc_coord : l1_noc_coord_per_bank) {
            auto l1_noc_coords = virtual_noc0_coordinate(device_id, noc, noc_coord);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << hal_->get_noc_addr_node_id_bits()) | noc_x) << hal_->get_noc_coord_reg_offset();
            l1_bank_to_noc_xy_[device_id].push_back(xy);
        }
    }
}

void MetalContext::generate_worker_logical_to_virtual_map(ChipId device_id) {
    // Generate logical to virtual map for DRAM and L1 banks
    const auto& soc_desc = cluster_->get_soc_desc(device_id);
    auto tensix_grid_size = soc_desc.get_grid_size(CoreType::TENSIX);

    worker_logical_col_to_virtual_col_[device_id].clear();
    worker_logical_row_to_virtual_row_[device_id].clear();
    worker_logical_col_to_virtual_col_[device_id].reserve(tensix_grid_size.x);
    worker_logical_row_to_virtual_row_[device_id].reserve(tensix_grid_size.y);

    for (size_t x = 0; x < tensix_grid_size.x; x++) {
        worker_logical_col_to_virtual_col_[device_id].push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{x, 0}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .x);
    }
    for (size_t y = 0; y < tensix_grid_size.y; y++) {
        worker_logical_row_to_virtual_row_[device_id].push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{0, y}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .y);
    }
}

void MetalContext::initialize_device_bank_to_noc_tables(
    ChipId device_id, const HalProgrammableCoreType& core_type, CoreCoord virtual_core) {
    const uint32_t dram_to_noc_sz_in_bytes = dram_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t l1_to_noc_sz_in_bytes = l1_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t dram_offset_sz_in_bytes = dram_bank_offset_map_[device_id].size() * sizeof(int32_t);
    const uint32_t l1_offset_sz_in_bytes = l1_bank_offset_map_[device_id].size() * sizeof(int32_t);

    const uint64_t mem_bank_to_noc_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t mem_bank_to_noc_size = hal_->get_dev_size(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    TT_ASSERT(
        (dram_to_noc_sz_in_bytes + l1_to_noc_sz_in_bytes + dram_offset_sz_in_bytes + l1_offset_sz_in_bytes) <=
            mem_bank_to_noc_size,
        "Size of bank_to_noc table is greater than available space");

    cluster_->write_core(
        dram_bank_to_noc_xy_[device_id].data(),
        dram_to_noc_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        mem_bank_to_noc_addr);
    uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
    cluster_->write_core(
        l1_bank_to_noc_xy_[device_id].data(), l1_to_noc_sz_in_bytes, tt_cxy_pair(device_id, virtual_core), l1_noc_addr);

    uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
    cluster_->write_core(
        dram_bank_offset_map_[device_id].data(),
        dram_offset_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        dram_offset_addr);
    uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
    cluster_->write_core(
        l1_bank_offset_map_[device_id].data(),
        l1_offset_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        l1_offset_addr);
}

void MetalContext::initialize_worker_logical_to_virtual_tables(
    ChipId device_id, const HalProgrammableCoreType& core_type, CoreCoord virtual_core) {
    // Generate logical to virtual map for DRAM and L1 banks
    const auto& soc_desc = cluster_->get_soc_desc(device_id);
    const uint32_t logical_col_to_virtual_col_sz_in_bytes =
        worker_logical_col_to_virtual_col_[device_id].size() * sizeof(uint8_t);
    const uint8_t firmware_grid_size_x =
        tt::round_up(soc_desc.grid_size.x, 4);  // Ensure multiple of 4 for uint32_t alignment
    const uint32_t logical_row_to_virtual_row_sz_in_bytes =
        worker_logical_row_to_virtual_row_[device_id].size() * sizeof(uint8_t);
    const uint64_t logical_to_virtual_map_addr =
        hal_->get_dev_addr(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    const uint32_t logical_to_virtual_map_size =
        hal_->get_dev_size(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);

    TT_ASSERT(
        (firmware_grid_size_x + logical_row_to_virtual_row_sz_in_bytes) <= logical_to_virtual_map_size,
        "Size of logical to virtual map is greater than available space");

    uint64_t logical_col_to_virtual_col_addr = logical_to_virtual_map_addr;
    cluster_->write_core(
        worker_logical_col_to_virtual_col_[device_id].data(),
        logical_col_to_virtual_col_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        logical_col_to_virtual_col_addr);

    // Size of the data in the firmware is the full size of the grid, not the harvested size.
    // Therefore, we must adjust the address to account for the full grid size.
    uint64_t logical_row_to_virtual_row_addr = logical_to_virtual_map_addr + (firmware_grid_size_x * sizeof(uint8_t));
    cluster_->write_core(
        worker_logical_row_to_virtual_row_[device_id].data(),
        logical_row_to_virtual_row_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        logical_row_to_virtual_row_addr);
}

void MetalContext::initialize_firmware(
    ChipId device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    dev_msgs::launch_msg_t::View launch_msg,
    dev_msgs::go_msg_t::ConstView go_msg,
    bool assert_reset) {
    ZoneScoped;

    initialize_device_bank_to_noc_tables(device_id, core_type, virtual_core);

    if (core_type == HalProgrammableCoreType::TENSIX) {
        // Only need to generate logical to virtual tables for Tensix cores, as only they run the firmware that
        // requires it.
        initialize_worker_logical_to_virtual_tables(device_id, core_type, virtual_core);
    }

    uint32_t core_type_idx = hal_->get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal_->get_processor_classes_count(core_type);
    auto jit_build_config =
        hal_->get_jit_build_config(core_type_idx, 0, 0);  // Only the first risc needs to be programmed

    // Initialize each entry in the launch_msg ring buffer with the correct dispatch mode - Cores that don't get a valid
    // launch_message during program execution need to at least have the correct dispatch mode.
    // When using Fast Dispatch on Tensix:
    // dispatch cores (Tensix) configured with DISPATCH_MODE_HOST
    // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // Idle Eth cores configured with DISPATCH_MODE_HOST but not used
    // When using Fast Dispatch on Idle Eth:
    // dispatch cores (Idle Eth) configured with DISPATCH_MODE_HOST
    // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // When using Slow Dispatch, all cores initialized with DISPATCH_MODE_HOST
    const auto write_initial_go_launch_msg = [&]() {
        size_t launch_msg_size = launch_msg.size();
        std::vector<std::byte> init_launch_msg_data(
            dev_msgs::launch_msg_buffer_num_entries * launch_msg_size, std::byte{0});
        for (size_t i = 0; i < dev_msgs::launch_msg_buffer_num_entries; ++i) {
            std::copy(
                launch_msg.data(),
                launch_msg.data() + launch_msg_size,
                init_launch_msg_data.data() + (i * launch_msg_size));
        }
        auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
        cluster_->write_core(
            init_launch_msg_data.data(),
            init_launch_msg_data.size(),
            tt_cxy_pair(device_id, virtual_core),
            hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
        uint32_t go_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG);
        cluster_->write_core(go_msg.data(), go_msg.size(), tt_cxy_pair(device_id, virtual_core), go_addr);
        uint64_t launch_msg_buffer_read_ptr_addr =
            hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
        uint32_t zero = 0;
        cluster_->write_core(
            &zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), launch_msg_buffer_read_ptr_addr);
        uint32_t go_message_index_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG_INDEX);
        cluster_->write_core(&zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), go_message_index_addr);
    };

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [build_idx, num_build_states] =
                    BuildEnvManager::get_instance().get_build_index_and_state_count(core_type_idx, processor_class);
                for (uint32_t riscv_id = 0; riscv_id < num_build_states; riscv_id++) {
                    auto fw_path = BuildEnvManager::get_instance()
                                       .get_firmware_build_state(device_id, core_type_idx, processor_class, riscv_id)
                                       .get_target_out_path("");
                    const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                    uint32_t fw_size = binary_mem.get_text_size();
                    hal_->set_iram_text_size(
                        launch_msg, core_type, static_cast<HalProcessorClassType>(processor_class), riscv_id, fw_size);
                    log_debug(
                        tt::LogMetal,
                        "RISC {} DM{} fw {} binary size: {} in bytes",
                        virtual_core.str(),
                        riscv_id,
                        fw_path,
                        fw_size);

                    if (not rtoptions_.get_skip_loading_fw()) {
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, riscv_id);
                    }
                }
            }

            if (!rtoptions_.get_fast_dispatch()) {
                // Host always writes launch messages
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
            } else {
                std::unordered_set<CoreCoord> virtual_dispatch_cores;
                if (dispatch_core_manager_->get_dispatch_core_type() == CoreType::WORKER) {
                    for (const auto& logical_core : dispatch_core_manager_->get_all_logical_dispatch_cores(device_id)) {
                        virtual_dispatch_cores.insert(cluster_->get_virtual_coordinate_from_logical_coordinates(
                            device_id, logical_core, CoreType::WORKER));
                    }
                }
                if (virtual_dispatch_cores.contains(virtual_core)) {
                    // Dispatch cores - Host writes launch messages
                    launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
                } else {
                    // Worker cores - Dispatcher will write launch messages
                    launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_DEV;
                }
            }

            write_initial_go_launch_msg();
            cluster_->write_core(
                &jit_build_config.fw_launch_addr_value,
                sizeof(uint32_t),
                tt_cxy_pair(device_id, virtual_core),
                jit_build_config.fw_launch_addr);

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
                log_info(
                    tt::LogMetal,
                    "Device {} init_fw ETH {}: INIT_FABRIC not set, skipping",
                    device_id,
                    virtual_core.str());
                break;
            }
            const bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            const bool is_active_eth = !is_idle_eth;
            log_info(
                tt::LogMetal,
                "Device {} init_fw ETH {}: is_active={}, assert_reset={}, cooperative={}, 2erisc={}",
                device_id,
                virtual_core.str(),
                is_active_eth,
                assert_reset,
                hal_->get_eth_fw_is_cooperative(),
                rtoptions_.get_enable_2_erisc_mode());
            tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX;
            if (is_active_eth) {
                // On active eth, don't assert ERISC0, which is running base firmware.
                reset_val &= ~tt::umd::RiscType::ERISC0;
            }
            if (assert_reset && (is_idle_eth or !hal_->get_eth_fw_is_cooperative())) {
                log_info(
                    tt::LogMetal,
                    "Device {} init_fw ETH {}: asserting reset (reset_val=0x{:x})",
                    device_id,
                    virtual_core.str(),
                    static_cast<uint64_t>(reset_val));
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
            }
            if (not rtoptions_.get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto num_build_states = hal_->get_processor_types_count(core_type_idx, processor_class);
                    for (uint32_t eriscv_id = 0; eriscv_id < num_build_states; eriscv_id++) {
                        auto fw_path =
                            BuildEnvManager::get_instance()
                                .get_firmware_build_state(device_id, core_type_idx, processor_class, eriscv_id)
                                .get_target_out_path("");
                        const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                        [[maybe_unused]] uint32_t fw_size = binary_mem.get_text_size();
                        log_debug(
                            tt::LogMetal,
                            "{} ERISC {} DM{} fw {} binary size: {} in bytes",
                            is_active_eth ? "Active" : "Idle",
                            virtual_core.str(),
                            eriscv_id,
                            fw_path,
                            fw_size);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, eriscv_id);
                    }
                }
            }
            // Ethernet worker core. Launch messages will be sent by FD infra if it's enabled
            // Idle ethernet core. Used by FD infra. Host will write launch messages during init.
            launch_msg.kernel_config().mode() = (!rtoptions_.get_fast_dispatch() or is_idle_eth)
                                                    ? dev_msgs::DISPATCH_MODE_HOST
                                                    : dev_msgs::DISPATCH_MODE_DEV;
            // For eth, write the go and launch message before initializing because when using the ETH FW API
            // it will launch immediately. DM0 is not in a reset state as it is running base FW.
            write_initial_go_launch_msg();
            if (core_type == HalProgrammableCoreType::ACTIVE_ETH) {
                // Clear the ncrisc_halt message
                DeviceAddr mailbox_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::MAILBOX);
                auto factory = hal_->get_dev_msgs_factory(core_type);
                DeviceAddr ncrisc_halt_addr =
                    mailbox_addr + factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::ncrisc_halt);
                std::vector<uint8_t> data(factory.size_of<dev_msgs::ncrisc_halt_msg_t>(), 0);
                cluster_->write_core(data.data(), data.size(), tt_cxy_pair(device_id, virtual_core), ncrisc_halt_addr);
            }

            // Write firmware main to primary erisc (DM0)
            // Using classic ASSERT/DEASSERT PC method for 1 erisc mode because erisc1 has no base firmware
            if (hal_->get_eth_fw_is_cooperative() || core_type != HalProgrammableCoreType::ACTIVE_ETH ||
                !rtoptions_.get_enable_2_erisc_mode()) {
                // PC
                log_info(
                    tt::LogMetal,
                    "Device {} init_fw ETH {}: writing PC directly (fw_launch_addr={:#x}, fw_launch_addr_value={:#x})",
                    device_id,
                    virtual_core.str(),
                    jit_build_config.fw_launch_addr,
                    jit_build_config.fw_launch_addr_value);
                cluster_->write_core(
                    &jit_build_config.fw_launch_addr_value,
                    sizeof(uint32_t),
                    tt_cxy_pair(device_id, virtual_core),
                    jit_build_config.fw_launch_addr);
            } else {
                // Active ethernet firmware launched immediately. Set the enable flag to 1 so FW doesn't exit
                // immediately.
                // Wait for ack not required because we wait for the done message
                log_info(
                    tt::LogMetal,
                    "Device {} init_fw ETH {}: sending ETH_MSG_RELEASE_CORE via mailbox (fw_launch_addr_value={:#x})",
                    device_id,
                    virtual_core.str(),
                    jit_build_config.fw_launch_addr_value);
                constexpr uint32_t mailbox_index = 0;
                tt::llrt::internal_::send_msg_to_eth_mailbox(
                    device_id,
                    virtual_core,
                    tt_metal::FWMailboxMsg::ETH_MSG_RELEASE_CORE,
                    mailbox_index,
                    {/*l1 addr to exec*/ jit_build_config.fw_launch_addr_value},
                    false);
                log_info(
                    tt::LogMetal, "Device {} init_fw ETH {}: ETH_MSG_RELEASE_CORE sent", device_id, virtual_core.str());
            }

            break;
        }
        default:
            TT_THROW(
                "Unsupported programable core type {} to initialize build states", enchantum::to_string(core_type));
    }
}

dev_msgs::core_info_msg_t MetalContext::populate_core_info_msg(
    ChipId device_id, HalProgrammableCoreType programmable_core_type) const {
    const metal_SocDescriptor& soc_d = cluster_->get_soc_desc(device_id);
    // Use architecture-defined supported PCIe address bounds
    auto factory = hal_->get_dev_msgs_factory(programmable_core_type);
    dev_msgs::core_info_msg_t buffer = factory.create<dev_msgs::core_info_msg_t>();
    auto core_info = buffer.view();
    core_info.noc_pcie_addr_base() = hal_->get_pcie_addr_lower_bound();
    core_info.noc_pcie_addr_end() = hal_->get_pcie_addr_upper_bound();
    core_info.noc_dram_addr_base() = 0;
    core_info.noc_dram_addr_end() = soc_d.dram_core_size;
    core_info.l1_unreserved_start() = align(worker_l1_unreserved_start_, hal_->get_alignment(HalMemType::DRAM));
    if (programmable_core_type == HalProgrammableCoreType::TENSIX) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::WORKER;
    } else if (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::ACTIVE_ETH;
    } else {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::IDLE_ETH;
    }
    const std::vector<tt::umd::CoreCoord>& pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    // There are multiple NoC endpoints for DRAM, but not all are exposed through the API. Watcher will flag endpoints
    // that are not exposed as invalid transactions. This helps to avoid BH issue highlighted by SYS-592 where writing
    // to multiple DRAM endpoints can hang the card.
    std::unordered_set<tt::umd::CoreCoord> dram_cores;
    auto num_dram_channels = cluster_->get_soc_desc(device_id).get_num_dram_views();
    for (uint32_t dram_channel = 0; dram_channel < num_dram_channels; dram_channel++) {
        for (uint32_t noc = 0; noc < hal_->get_num_nocs(); noc++) {
            auto worker_dram_ep = soc_d.get_preferred_worker_core_for_dram_view(dram_channel, noc);
            auto eth_dram_ep = soc_d.get_preferred_eth_core_for_dram_view(dram_channel, noc);
            auto physical_worker_dram_ep =
                soc_d.translate_coord_to(worker_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            auto physical_eth_dram_ep =
                soc_d.translate_coord_to(eth_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            dram_cores.insert(physical_worker_dram_ep);
            dram_cores.insert(physical_eth_dram_ep);
        }
    }

    const std::vector<tt::umd::CoreCoord>& eth_cores =
        soc_d.get_cores(CoreType::ETH, CoordSystem::NOC0);  // make these translated and then convert to physical

    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= core_info.non_worker_cores().size(),
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= core_info.virtual_non_worker_cores().size(),
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    auto set_addressable_core =
        [](dev_msgs::addressable_core_t::View core, const CoreCoord& core_coord, dev_msgs::AddressableCoreType type) {
            core.x() = core_coord.x;
            core.y() = core_coord.y;
            core.type() = type;
        };
    for (auto non_worker_core : core_info.non_worker_cores()) {
        set_addressable_core(
            non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    for (auto virtual_non_worker_core : core_info.virtual_non_worker_cores()) {
        set_addressable_core(
            virtual_non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    // On Blackhole, virtualized Tensix coordinates overlap with NoC1 physical DRAM and PCIe coordinates beause
    // virtualized Tensix coordinates == NoC0 Tensix physical coordinates. This causes false negative Watcher
    // sanitization errors because it appears as a mixed use of physical and virtual To workaround this, skip over
    // populating `non_worker_cores` for BH DRAM when virtualization is enabled
    int non_worker_cores_idx = 0;
    bool skip_physical = cluster_->arch() == ARCH::BLACKHOLE and hal_->is_coordinate_virtualization_enabled();
    if (not skip_physical) {
        for (tt::umd::CoreCoord core : pcie_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::PCIE);
        }
        for (tt::umd::CoreCoord core : dram_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::DRAM);
        }
        for (tt::umd::CoreCoord core : eth_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::ETH);
        }
    }

    if (hal_->is_coordinate_virtualization_enabled()) {
        // Track Virtual Non Worker Cores (In this case only Eth) separately
        uint32_t virtual_non_worker_cores_idx = 0;
        for (tt::umd::CoreCoord core : eth_cores) {
            auto virtual_core = cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
            set_addressable_core(
                core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                virtual_core,
                dev_msgs::AddressableCoreType::ETH);
        }

        if (cluster_->arch() == ARCH::BLACKHOLE) {
            for (const CoreCoord& core : pcie_cores) {
                auto virtual_core =
                    cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::PCIE);
            }

            for (const CoreCoord& core : dram_cores) {
                auto virtual_core =
                    cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::DRAM);
            }
        }
    }

    // Determine which noc-coords are harvested
    std::vector<uint32_t> harvested_axis_coord;
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t harvested_noc_coords = umd::CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        cluster_->get_soc_desc(device_id).arch, cluster_->get_harvesting_mask(device_id));
    uint32_t max_along_axis =
        hal_->get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW ? soc_d.grid_size.y : soc_d.grid_size.x;
    for (uint32_t idx = 0; idx < max_along_axis; idx++) {
        bool harvested_axis = (harvested_noc_coords >> idx) & 0x1;
        if (harvested_axis) {
            harvested_axis_coord.push_back(idx);
        }
    }
    TT_ASSERT(
        harvested_axis_coord.size() <= core_info.harvested_coords().size(),
        "Detected more harvested rows than fit in mailbox.");
    for (size_t idx = 0; idx < core_info.harvested_coords().size(); idx++) {
        core_info.harvested_coords()[idx] =
            (idx < harvested_axis_coord.size()) ? harvested_axis_coord[idx] : dev_msgs::CORE_COORD_INVALID;
        // Populate harvested rows/cols in virtual coordinate space if virtualization is supported by HW.
        // Harvested rows/cols in the virtual space are placed at the end of the worker grid,
        if (hal_->is_coordinate_virtualization_enabled() and idx < harvested_axis_coord.size()) {
            // On BH virtual coordinates are not contiguous
            uint32_t end_virtual_grid;
            if (hal_->get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW) {
                end_virtual_grid = hal_->get_virtual_worker_start_y() + logical_grid_size.y;
            } else if (cluster_->arch() == ARCH::BLACKHOLE) {
                end_virtual_grid = max_along_axis - 1;
            } else {
                end_virtual_grid = hal_->get_virtual_worker_start_x() + logical_grid_size.x;
            }

            // BH translated tensix cores are same as noc0 physical
            core_info.virtual_harvested_coords()[idx] = end_virtual_grid + harvested_axis_coord.size() - (idx + 1);
        } else {
            core_info.virtual_harvested_coords()[idx] = dev_msgs::CORE_COORD_INVALID;
        }
    }

    core_info.noc_size_x() = soc_d.grid_size.x;
    core_info.noc_size_y() = soc_d.grid_size.y;
    core_info.worker_grid_size_x() = logical_grid_size.x;  // Grid size as virtual coords see it (workers only)
    core_info.worker_grid_size_y() = logical_grid_size.y;

    return buffer;
}

int MetalContext::get_lite_fabric_hop_count(ChipId chip_id) const {
    if (!lite_fabric_hal_) {
        return 0;
    }
    for (const auto& t : lite_fabric_hal_->get_system_descriptor().tunnels_from_mmio) {
        if (t.connected_id == chip_id) {
            return t.num_hops;
        }
    }
    return 0;
}

bool MetalContext::is_lite_fabric_mmio_core(ChipId mmio_id, CoreCoord virtual_core) const {
    if (!lite_fabric_hal_) {
        return false;
    }
    for (const auto& t : lite_fabric_hal_->get_system_descriptor().tunnels_from_mmio) {
        if (t.mmio_id == mmio_id && t.mmio_core_virtual == virtual_core) {
            return true;
        }
    }
    return false;
}

bool MetalContext::has_fabric_routers_launched(ChipId device_id) const {
    const auto& mmio_ids = cluster_->mmio_chip_ids();
    if (mmio_ids.count(device_id)) {
        return true;  // MMIO devices launch fabric routers directly
    }
    auto it = remote_fabric_eth_channels_.find(device_id);
    return it != remote_fabric_eth_channels_.end() && !it->second.empty();
}

void MetalContext::initialize_remote_eth_cores_for_fabric(
    ChipId device_id, const std::vector<CoreCoord>& logical_eth_cores) {
    if (logical_eth_cores.empty()) {
        return;
    }

    log_info(
        tt::LogMetal,
        "Device {} init remote ETH: initializing {} ETH core(s) for fabric",
        device_id,
        logical_eth_cores.size());

    auto core_type = HalProgrammableCoreType::ACTIVE_ETH;
    auto core_info = populate_core_info_msg(device_id, core_type);
    auto dev_msgs_factory = hal_->get_dev_msgs_factory(core_type);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    static std::vector<uint32_t> zero_vec_erisc_init(
        hal_->get_dev_size(core_type, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t), 0);

    // Soft reset register values for BH ETH tiles.  Bits 13, 14, 18 control
    // internal ETH tile subsystems and must always remain set.
    // Bit 11 = ERISC0 reset, bit 12 = ERISC1 reset.
    // These values match the lite fabric FW (risc_interface.hpp).
    constexpr uint32_t SOFT_RESET_REG_ADDR = 0xFFB121B0;
    constexpr uint32_t AERISC_RESET_PC_ADDR = 0xFFB14000;    // debug reg: ERISC0 boot PC
    // MMIO-peering cores: ERISC1 already running lite fabric receiver.
    // 0x46000 = bits 13/14/18 set, bits 11/12 clear (both ERISCs deasserted).
    constexpr uint32_t SOFT_RESET_ERISC0_ERISC1_RUNNING = 0x46000;
    // Non-MMIO-peering cores: ERISC1 in POR reset, no firmware loaded.
    // 0x47000 = bits 12/13/14/18 set, bit 11 clear (ERISC0 deasserted, ERISC1 stays in reset).
    constexpr uint32_t SOFT_RESET_ERISC0_ONLY = 0x47000;

    for (const auto& logical_core : logical_eth_cores) {
        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);

        // Look up the peer ETH core to check if the peer is MMIO-reachable.
        auto [peer_chip, peer_logical] = cluster_->get_connected_ethernet_core({device_id, logical_core});

        // Skip ETH cores whose peer is an unreachable N-hop chip.  The WRITE_REG
        // to deassert ERISC0 would need to go through the peer chip's relay, which
        // doesn't exist for unreachable chips.  The fabric router on this core will
        // also be unable to handshake with the unreachable peer.
        if (is_chip_unreachable(peer_chip)) {
            log_info(
                tt::LogMetal,
                "Device {} init remote ETH: skipping core {} — peer chip {} is unreachable",
                device_id,
                logical_core.str(),
                peer_chip);
            continue;
        }

        bool peer_is_mmio = cluster_->mmio_chip_ids().count(peer_chip) > 0;

        // Per-core ERISC1 detection: ERISC1 only runs on cores that are part
        // of a lite fabric tunnel (Phase 2 receivers, Phase 2b downstream
        // senders, and Phase 2b tunnel endpoints).  Non-tunnel cores on remote
        // chips have ERISC1 in POR reset — deasserting with 0x46000 would
        // cause ERISC1 to boot from POR RESET_PC (syseng subordinate FW or
        // garbage), conflicting with ERISC0's fabric router.
        bool erisc1_running = false;
        if (lite_fabric_hal_) {
            // 1. MMIO-peering cores: Phase 2 deployed ERISC1 via ETH_INIT_NEIGHBOUR
            if (peer_is_mmio) {
                erisc1_running = true;
            }
            // 2. Downstream sender cores: Phase 2b launched ERISC1 explicitly
            if (downstream_sender_cores_.count({device_id, static_cast<uint32_t>(logical_core.y)})) {
                erisc1_running = true;
            }
            // 3. Tunnel endpoint cores: Phase 2b ETH_INIT_NEIGHBOUR deployed ERISC1
            if (!erisc1_running) {
                const auto& tunnels = lite_fabric_hal_->get_system_descriptor().tunnels_from_mmio;
                for (const auto& t : tunnels) {
                    if (t.connected_id == device_id &&
                        static_cast<uint32_t>(t.connected_core_logical.y) == static_cast<uint32_t>(logical_core.y)) {
                        erisc1_running = true;
                        break;
                    }
                }
            }
        }

        log_info(
            tt::LogMetal,
            "Device {} init remote ETH: core {} (virtual {}) peer chip {} peer_logical {} (MMIO={}, ERISC1={})",
            device_id,
            logical_core.str(),
            virtual_core.str(),
            peer_chip,
            peer_logical.str(),
            peer_is_mmio,
            erisc1_running);

        // Signal ERISC1 to switch from TXQ0 to TXQ2 before we launch ERISC0
        // (fabric router uses TXQ0/TXQ1).  ERISC1 polls active_txq_request in
        // its main loop and switches on the next iteration.
        if (erisc1_running) {
            uint32_t txq_req_addr = LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config) +
                                    offsetof(lite_fabric::FabricLiteConfig, forwarding) +
                                    offsetof(lite_fabric::FabricLiteConfig::ForwardingConfig, active_txq_request);
            uint8_t txq2 = 2;
            // Signal the remote-side ERISC1
            cluster_->write_core(&txq2, sizeof(txq2), tt_cxy_pair(device_id, virtual_core), txq_req_addr);
            // Also signal the MMIO-side ERISC1 (peer) — it runs lite fabric too
            if (peer_is_mmio) {
                CoreCoord peer_virtual =
                    cluster_->get_virtual_coordinate_from_logical_coordinates(peer_chip, peer_logical, CoreType::ETH);
                cluster_->write_core(&txq2, sizeof(txq2), tt_cxy_pair(peer_chip, peer_virtual), txq_req_addr);
            }
            log_info(
                tt::LogMetal,
                "Device {} init remote ETH: wrote active_txq_request=2 to core {} and peer {}:{} at {:#x}",
                device_id,
                virtual_core.str(),
                peer_chip,
                peer_logical.str(),
                txq_req_addr);
        }

        // Track MMIO-peering cores where the fabric router is launched.
        // Used by update_lite_fabric_bindings_for_fabric_routers()
        // and has_fabric_routers_launched().
        if (peer_is_mmio) {
            remote_fabric_eth_channels_[device_id].insert(logical_core.y);
        }

        // Clear erisc app sync info
        cluster_->write_core(
            zero_vec_erisc_init.data(),
            zero_vec_erisc_init.size() * sizeof(uint32_t),
            tt_cxy_pair(device_id, virtual_core),
            hal_->get_dev_addr(core_type, HalL1MemAddrType::APP_SYNC_INFO));

        // Write core info
        core_info.view().absolute_logical_x() = logical_core.x;
        core_info.view().absolute_logical_y() = logical_core.y;
        cluster_->write_core_immediate(
            core_info.data(),
            core_info.size(),
            {static_cast<size_t>(device_id), virtual_core},
            hal_->get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));

        // On remote BH devices behind lite fabric, ERISC0 is held in POR
        // reset — syseng base FW was never loaded or run.  The standard
        // ETH_MSG_RELEASE_CORE mechanism (used on MMIO devices) therefore
        // does not work.
        //
        // Instead we load Metal's FW binaries and messages into L1 while
        // ERISC0 is in reset, write the boot PC to AERISC_RESET_PC, and
        // deassert ERISC0 via the soft reset register so it boots directly
        // into Metal's active erisc FW.
        //
        // Debug registers (0xFFBxxxxx) including the soft reset register
        // are written via write_core, which routes through UMD's default
        // channel for this remote device.  On Blackhole, the NOC can deliver
        // writes to tile register addresses from the lite fabric receiver.

        // 1. ERISC0 is already in reset:
        //    - MMIO-peering cores: lite fabric setup wrote 0x46800 (ERISC0 in
        //      reset, ERISC1 running lite fabric receiver).
        //    - Non-MMIO-peering cores: POR state (both ERISCs in reset, 0x47800).

        // 2. Clear the syseng FW mailbox so that send_msg_to_eth_mailbox
        //    (called inside initialize_firmware) doesn't timeout on the
        //    residual POR/CALL marker.
        {
            constexpr uint32_t mailbox_index = 0;
            auto mailbox_addr = hal_->get_eth_fw_mailbox_address(mailbox_index);
            uint32_t zero = 0;
            cluster_->write_core(&zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), mailbox_addr);
        }

        // 3. Load FW binaries and write go/launch messages via initialize_firmware.
        //    Pass assert_reset=false since we already set the soft reset state.
        initialize_firmware(
            device_id, core_type, virtual_core, launch_msg.view(), go_msg.view(), /*assert_reset=*/false);

        // 3b. Populate the syseng base FW API table at MEM_SYSENG_ETH_API_TABLE
        //     (0x7CF00) with a no-op stub.  The active erisc FW calls
        //     service_eth_msg() on every main-loop iteration, which reads a
        //     function pointer from this table and calls it.  On MMIO devices
        //     the syseng base FW initializes the table; on remote devices
        //     behind lite fabric the base FW was never loaded, so the table
        //     contains zeros/garbage.  Calling a NULL pointer crashes ERISC0.
        //     Fix: write a RISC-V `ret` instruction at 0x7CF10 (right after
        //     the API table) and point all entries there.  Do NOT use L1[0xC]
        //     — that address is MEM_L1_BARRIER and gets overwritten with 0
        //     by l1_barrier(), destroying the ret and crashing service_eth_msg.
        {
            constexpr uint32_t API_TABLE_ADDR = 0x7CF00;  // MEM_SYSENG_ETH_API_TABLE
            constexpr uint32_t API_TABLE_ENTRIES = 4;     // send, service, link_status, dynamic_noc
            constexpr uint32_t RET_STUB_ADDR = API_TABLE_ADDR + API_TABLE_ENTRIES * sizeof(uint32_t);  // 0x7CF10
            constexpr uint32_t RISCV_RET_INSN = 0x00008067;  // jalr x0, ra, 0  (ret)
            // Write ret instruction right after the API table
            uint32_t ret_insn = RISCV_RET_INSN;
            cluster_->write_core(&ret_insn, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), RET_STUB_ADDR);
            uint32_t api_table[API_TABLE_ENTRIES];
            for (uint32_t i = 0; i < API_TABLE_ENTRIES; i++) {
                api_table[i] = RET_STUB_ADDR;
            }
            cluster_->write_core(api_table, sizeof(api_table), tt_cxy_pair(device_id, virtual_core), API_TABLE_ADDR);
            log_info(
                tt::LogMetal,
                "Device {} init remote ETH: wrote syseng API table stubs at {:#x} -> ret@{:#x} on core {}",
                device_id,
                API_TABLE_ADDR,
                RET_STUB_ADDR,
                virtual_core.str());
        }

        // 4. Write a boot trampoline at L1 address 0x0 that initializes SP
        //    then jumps to the FW entry point.  ERISC0 boots from the
        //    address in AERISC_RESET_PC after soft reset deassert.  On MMIO
        //    devices, syseng FW has
        //    already initialized SP, but on remote devices behind lite fabric,
        //    ERISC0 was never running (POR state) so SP is uninitialized.
        //    The FW entry starts with `addi sp, sp, -16` which crashes if
        //    SP=0.  We write: LUI sp + ADDI sp + JAL to fw_base.
        //    Set AERISC_RESET_PC (0xFFB14000) to 0x0 so ERISC0 boots from
        //    the trampoline, not from fw_base directly.
        uint32_t core_type_idx = hal_->get_programmable_core_type_index(core_type);
        auto jit_build_config = hal_->get_jit_build_config(core_type_idx, 0, 0);
        uint32_t fw_base = jit_build_config.fw_launch_addr_value;  // MEM_AERISC_FIRMWARE_BASE
        // Boot trampoline: 3 words at L1[0x0..0xB]
        //   [0x0] lui  sp, 0xFFB02      # sp = 0xFFB02000
        //   [0x4] addi sp, sp, -16      # sp = 0xFFB01FF0 (matches crt0 __stack_top - 16)
        //   [0x8] jal  x0, fw_base      # jump to FW entry (offset from PC=0x8)
        // Note: L1[0xC] is MEM_L1_BARRIER — do not place code there.
        uint32_t trampoline[3];
        trampoline[0] = 0xFFB02137;  // lui sp, 0xFFB02
        trampoline[1] = 0xFF010113;  // addi sp, sp, -16
        TT_FATAL(fw_base > 0x8, "fw_base must be > 0x8 for trampoline JAL offset");
        trampoline[2] = generate_risc_startup_addr(fw_base - 0x8);  // jal x0, (fw_base - 0x8) from PC=0x8
        log_info(
            tt::LogMetal,
            "Device {} init remote ETH: writing SP-init trampoline to L1[0x0..0xB] on core {}: "
            "lui={:#010x} addi={:#010x} jal({:#x})={:#010x}",
            device_id,
            virtual_core.str(),
            trampoline[0],
            trampoline[1],
            fw_base,
            trampoline[2]);
        cluster_->write_core(trampoline, sizeof(trampoline), tt_cxy_pair(device_id, virtual_core), 0x0);
        // Set AERISC_RESET_PC to 0x0 (trampoline address) so ERISC0 boots from
        // the trampoline which initializes SP before jumping to fw_base.
        // On remote devices ERISC0 was never running (POR state), so SP=0;
        // booting directly at fw_base would crash on the first stack access.
        {
            uint32_t reset_pc_val = 0x0;
            cluster_->write_core(
                &reset_pc_val, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), AERISC_RESET_PC_ADDR);
        }

        // Verify writes by reading back key addresses.
        cluster_->l1_barrier(device_id);
        {
            uint32_t readback_tramp[3] = {};
            cluster_->read_core(readback_tramp, sizeof(readback_tramp), tt_cxy_pair(device_id, virtual_core), 0x0);
            uint32_t readback_fw = 0;
            cluster_->read_core(&readback_fw, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), fw_base);
            log_info(
                tt::LogMetal,
                "Device {} init remote ETH: readback trampoline L1[0x0]={:#010x} [0x4]={:#010x} [0x8]={:#010x}, "
                "L1[{:#x}]={:#010x} (expect non-zero if FW loaded)",
                device_id,
                readback_tramp[0],
                readback_tramp[1],
                readback_tramp[2],
                fw_base,
                readback_fw);
        }

        // 5. Write skip-dance flag to ncrisc_halt.resume_addr (mailbox_base + 0).
        //    On remote devices, ERISC1 is either running the lite fabric relay
        //    (MMIO-peering cores) or in POR reset (non-MMIO-peering cores) — in
        //    neither case is it running Metal's subordinate FW.  The firmware
        //    checks this flag at boot and skips deassert_all_reset()/enter_reset().
        {
            DeviceAddr mailbox_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::MAILBOX);
            uint32_t skip_dance_flag = 1;
            cluster_->write_core(
                &skip_dance_flag, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), mailbox_addr);
        }

        // 5b. Write a sentinel to subordinate_sync so we can detect if ERISC0 boots.
        //     FW line 218 writes 0 to subordinate_sync.all; if we see the sentinel
        //     after the delay, ERISC0 never reached that point.
        {
            DeviceAddr mailbox_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::MAILBOX);
            uint32_t sentinel = 0xDEADBEEF;
            // subordinate_sync is at offset 8 from mailbox base (after ncrisc_halt which is 8 bytes)
            cluster_->write_core(&sentinel, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), mailbox_addr + 8);
        }

        // 6. Deassert ERISC0 via write_core (NOC unicast write to the soft
        //    reset register on the target ETH tile).  On Blackhole, the
        //    0xFFBxxxxx tile register space is NOC-addressable, so the lite
        //    fabric receiver can deliver the write to any tile on the chip.
        //
        //    With lite fabric active, ERISC1 runs on all cores (erisc1_running=true),
        //    so we always use 0x46000 (both ERISCs deasserted).
        //    Without lite fabric, ERISC1 is in POR — use 0x47000 (ERISC0 only).
        cluster_->l1_barrier(device_id);

        uint32_t soft_reset_val = erisc1_running ? SOFT_RESET_ERISC0_ERISC1_RUNNING : SOFT_RESET_ERISC0_ONLY;
        log_info(
            tt::LogMetal,
            "Device {} init remote ETH: deasserting ERISC0 ({:#x}) on core {} via write_core (erisc1_running={})",
            device_id,
            soft_reset_val,
            virtual_core.str(),
            erisc1_running);
        cluster_->write_core(
            &soft_reset_val, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), SOFT_RESET_REG_ADDR);
    }

    // Barrier to ensure deassert has reached the device
    cluster_->l1_barrier(device_id);

    // Wait for ERISC0 to boot, skip the 2-erisc dance (via the skip flag),
    // pass through wait_subordinate_eriscs(), and reach the go-signal loop.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Diagnostic: check if ERISC0 booted by reading back sentinel and go signal.
    for (const auto& logical_core : logical_eth_cores) {
        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
        auto [peer_chip, peer_logical] = cluster_->get_connected_ethernet_core({device_id, logical_core});
        if (is_chip_unreachable(peer_chip)) {
            continue;
        }
        DeviceAddr mailbox_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::MAILBOX);
        uint32_t diag[8] = {};
        cluster_->read_core(diag, sizeof(diag), tt_cxy_pair(device_id, virtual_core), mailbox_addr);
        // go_messages[0] is at go_addr; read it too
        uint32_t go_area[4] = {};
        DeviceAddr go_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::GO_MSG);
        if (go_addr == 0) {
            go_addr = 0x410;  // fallback: known BH ACTIVE_ETH go_messages address
        }
        cluster_->read_core(go_area, sizeof(go_area), tt_cxy_pair(device_id, virtual_core), go_addr);
        log_info(
            tt::LogMetal,
            "Device {} init remote ETH DIAG: core {} mailbox[0..7]="
            "[{:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}] "
            "go[0..3]=[{:#010x}, {:#010x}, {:#010x}, {:#010x}] "
            "(sentinel at +8: {} booted={})",
            device_id,
            virtual_core.str(),
            diag[0],
            diag[1],
            diag[2],
            diag[3],
            diag[4],
            diag[5],
            diag[6],
            diag[7],
            go_area[0],
            go_area[1],
            go_area[2],
            go_area[3],
            diag[2] == 0xDEADBEEF ? "SENTINEL" : "cleared",
            diag[2] != 0xDEADBEEF ? "YES" : "NO");
    }

    log_info(tt::LogMetal, "Device {} init remote ETH: {} core(s) initialized", device_id, logical_eth_cores.size());
}

void MetalContext::initialize_and_launch_firmware(ChipId device_id) {
    ZoneScoped;

    // On a remote BH device behind lite fabric, ERISC0 is not running on any
    // ETH core (all are in POR reset) and the lite fabric ETH core has ERISC1
    // actively servicing the lite fabric link.
    //
    // The normal 2-erisc init path for active ETH cores relies on ERISC0's
    // base firmware being alive to process ETH_MSG_RELEASE_CORE and deassert
    // ERISC1.  That assumption does not hold on remote devices, so we skip
    // ALL ETH core initialization here:
    //   - Lite fabric ETH core: ERISC1 is running — touching it kills the link.
    //   - Other active ETH cores: ERISC0 isn't running, so the 2-erisc mailbox
    //     handshake would never complete and wait_until_cores_done would hang.
    //   - Idle ETH cores: no base firmware loaded, deasserting would boot
    //     garbage and hang the wait loop.
    //
    // Only Tensix cores are initialized.  ETH cores on the remote device will
    // be brought up later by the fabric initialization path if needed.
    const bool skip_eth_cores = lite_fabric_hal_ && !cluster_->mmio_chip_ids().count(device_id);

    // Download to worker cores
    std::unordered_set<CoreCoord> not_done_cores;
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    log_info(
        tt::LogMetal,
        "Device {} init_fw: grid={}x{}, skip_eth={}",
        device_id,
        logical_grid_size.x,
        logical_grid_size.y,
        skip_eth_cores);

    // Remote device: Tensix cores are in POR (Power-On Reset) state because
    // reset_cores() is skipped (it would kill the lite fabric ERISC1 link).
    // In POR the NOC NIU can accept writes to L1 but may not generate read
    // responses, causing wait_until_cores_done to hang.  Assert software
    // reset on all worker cores to transition them to a clean reset state
    // where the NIU is fully functional.
    if (skip_eth_cores) {
        log_info(tt::LogMetal, "Device {} init_fw: asserting reset on Tensix worker cores (POR->reset)", device_id);
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            for (uint32_t x = 0; x < logical_grid_size.x; x++) {
                CoreCoord logical_core(x, y);
                CoreCoord worker_core = cluster_->get_virtual_coordinate_from_logical_coordinates(
                    device_id, logical_core, CoreType::WORKER);
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
            }
        }
        cluster_->l1_barrier(device_id);
    }

    auto dev_msgs_factory = hal_->get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::TENSIX);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    log_info(
        tt::LogMetal,
        "Device {} init_fw: writing Tensix core info ({} B) + FW to {}x{} cores, harvest={:#x}",
        device_id,
        core_info.size(),
        logical_grid_size.x,
        logical_grid_size.y,
        cluster_->get_soc_desc(device_id).harvesting_masks.tensix_harvesting_mask);
    for (uint32_t y = 0; y < logical_grid_size.y; y++) {
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            // Setup the absolute logical coordinates of this worker which are relative to true origin. not the sub
            // device. When running the user kernel, which potentially is on a sub device, send that info using the
            // launch message using dispatch.
            core_info.view().absolute_logical_x() = logical_core.x;
            core_info.view().absolute_logical_y() = logical_core.y;
            // Must write to core before starting it
            cluster_->write_core_immediate(
                core_info.data(),
                core_info.size(),
                {static_cast<size_t>(device_id), worker_core},
                hal_->get_dev_addr(llrt::get_core_type(device_id, worker_core), HalL1MemAddrType::CORE_INFO));
            initialize_firmware(
                device_id, HalProgrammableCoreType::TENSIX, worker_core, launch_msg.view(), go_msg.view());
            not_done_cores.insert(worker_core);
        }
    }
    log_info(tt::LogMetal, "Device {} init_fw: Tensix writes done ({} cores)", device_id, not_done_cores.size());

    std::unordered_set<CoreCoord> multi_risc_active_eth_cores;

    if (!skip_eth_cores) {
        // Clear erisc sync info
        for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
            static std::vector<uint32_t> zero_vec_erisc_init(
                hal_->get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) /
                    sizeof(uint32_t),
                0);

            CoreCoord virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);

            cluster_->write_core_immediate(
                device_id,
                virtual_core,
                zero_vec_erisc_init,
                hal_->get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
        }

        // Load erisc app base FW to eth cores on WH and active_erisc FW on second risc of BH active eth cores
        log_debug(tt::LogMetal, "Initializing active ethernet cores");
        dev_msgs_factory = hal_->get_dev_msgs_factory(HalProgrammableCoreType::ACTIVE_ETH);
        core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::ACTIVE_ETH);
        launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
        go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
        go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

        for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
            core_info.view().absolute_logical_x() = eth_core.x;
            core_info.view().absolute_logical_y() = eth_core.y;
            cluster_->write_core_immediate(
                core_info.data(),
                core_info.size(),
                {static_cast<size_t>(device_id), virtual_core},
                hal_->get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
            initialize_firmware(
                device_id, HalProgrammableCoreType::ACTIVE_ETH, virtual_core, launch_msg.view(), go_msg.view());
            if (!hal_->get_eth_fw_is_cooperative()) {
                multi_risc_active_eth_cores.insert(virtual_core);
                not_done_cores.insert(virtual_core);
            }
        }

        log_debug(tt::LogMetal, "Initializing idle ethernet cores");
        dev_msgs_factory = hal_->get_dev_msgs_factory(HalProgrammableCoreType::IDLE_ETH);
        core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::IDLE_ETH);
        launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
        go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
        go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;
        for (const auto& eth_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
            core_info.view().absolute_logical_x() = eth_core.x;
            core_info.view().absolute_logical_y() = eth_core.y;
            cluster_->write_core_immediate(
                core_info.data(),
                core_info.size(),
                {static_cast<size_t>(device_id), virtual_core},
                hal_->get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
            initialize_firmware(
                device_id, HalProgrammableCoreType::IDLE_ETH, virtual_core, launch_msg.view(), go_msg.view());
            not_done_cores.insert(virtual_core);
        }
    } else {
        log_info(
            tt::LogMetal,
            "Device {} init_fw: remote device — skipping ETH core init (Tensix only, {} cores)",
            device_id,
            not_done_cores.size());
    }

    // Barrier between L1 writes above and deassert below
    log_info(tt::LogMetal, "Device {} init_fw: l1_barrier", device_id);
    cluster_->l1_barrier(device_id);

    // Deassert worker cores
    log_info(tt::LogMetal, "Device {} init_fw: deasserting {} worker cores", device_id, not_done_cores.size());
    for (const auto& worker_core : not_done_cores) {
        if (multi_risc_active_eth_cores.contains(worker_core) && rtoptions_.get_enable_2_erisc_mode()) {
            // In 2-erisc mode, ERISC0 (primary) boots base FW and then deasserts ERISC1
            // (subordinate). On first init, ERISC0 is already running from base FW load.
            // On reinit, assert_cores put ERISC0 in reset — we must deassert it here.
            // Use direct register write because deassert_risc_reset_at_core doesn't handle
            // ERISC bits correctly on BH (get_soft_reset_reg_value ignores ERISC bits).
            if (cluster_->arch() == ARCH::BLACKHOLE) {
                constexpr uint32_t kSoftResetAddr = 0xFFB121B0;
                constexpr uint32_t kErisc0OutErisc1InReset = 0x47000;  // bit 12 set, bit 11 clear
                cluster_->write_core(
                    &kErisc0OutErisc1InReset,
                    sizeof(kErisc0OutErisc1InReset),
                    tt_cxy_pair(device_id, worker_core),
                    kSoftResetAddr);
            }
            // ERISC0 will deassert ERISC1 once base FW processes the release message.
            continue;
        }

        tt::umd::RiscType reset_val;
        if (cluster_->arch() == ARCH::QUASAR) {
            reset_val = tt::umd::RiscType::ALL_NEO_DMS;
        } else {
            reset_val = tt::umd::RiscType::BRISC;
            if (multi_risc_active_eth_cores.contains(worker_core)) {
                // bit 12 needs to be deasserted to run second erisc on BH
                reset_val |= tt::umd::RiscType::ERISC1;
            }
        }
        cluster_->deassert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), reset_val);
    }
    log_info(tt::LogMetal, "Device {} init_fw: deassert done, waiting for FW init", device_id);

    // Flush deassert writes through lite fabric before polling
    cluster_->l1_barrier(device_id);

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    if (skip_eth_cores) {
        // DIAGNOSTIC: The lite fabric NOC_READ path to Tensix L1 on remote
        // devices currently hangs (remote ERISC's noc_async_read never
        // completes).  Use a fixed delay instead of read-based polling.
        // Tensix FW init takes microseconds; 500ms is extremely conservative.
        log_info(
            tt::LogMetal,
            "Device {} init_fw: remote device — using fixed delay instead of read-based polling ({} cores)",
            device_id,
            not_done_cores.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        log_info(tt::LogMetal, "Device {} init_fw: FW init assumed complete (500ms delay)", device_id);
    } else {
        const int timeout_ms = 10000;  // 10 seconds for now
        try {
            llrt::internal_::wait_until_cores_done(device_id, dev_msgs::RUN_MSG_INIT, not_done_cores, timeout_ms);
        } catch (std::runtime_error& e) {
            log_error(
                tt::LogMetal,
                "Device {} init_fw TIMEOUT: {} cores not done. Inner: {}",
                device_id,
                not_done_cores.size(),
                e.what());
            TT_THROW(
                "Device {} init: failed to initialize FW ({} cores not done). Inner: {}",
                device_id,
                not_done_cores.size(),
                e.what());
        }
    }
    log_info(tt::LogMetal, "Device {} init_fw: FW init complete", device_id);
}

// Command queue id stack for thread
thread_local MetalContext::CommandQueueIdStack MetalContext::command_queue_id_stack_for_thread_;

MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() {
    return MetalContext::command_queue_id_stack_for_thread_;
}
const MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() const {
    return MetalContext::command_queue_id_stack_for_thread_;
}

uint32_t MetalContext::get_active_erisc_launch_flag_addr() {
    auto core_type_idx = hal_->get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    std::uint32_t launch_erisc_addr = hal_->get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr;
    return launch_erisc_addr;
};

bool MetalContext::erisc_app_still_running(ChipId device_id, CoreCoord virtual_core) {
    // Check if the kernel/erisc_app is still running on a ethernet core with context switching enabled
    // The LAUNCH_ERISC_APP_FLAG is reset to 0 after reset/reboot, and set to 1 when Metal runtime launches erisc
    // app FW Only applicable to WORMHOLE ethernet cores today, but could in theory extend to other cores, remove
    // assert if so
    if (cluster_->arch() != ARCH::WORMHOLE_B0) {
        return false;
    }
    TT_ASSERT(
        cluster_->is_ethernet_core(virtual_core, device_id),
        "Invalid core {} for context switch check",
        virtual_core.str());
    std::uint32_t launch_erisc_addr = get_active_erisc_launch_flag_addr();
    auto data = cluster_->read_core(device_id, virtual_core, launch_erisc_addr, sizeof(std::uint32_t));
    return (data[0] != 0);
};

// Send exit_erisc_kernel to the launch message
void MetalContext::erisc_send_exit_signal(ChipId device_id, CoreCoord virtual_core, bool is_idle_eth) {
    HalProgrammableCoreType programmable_core_type =
        is_idle_eth ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    auto dev_msgs_factory = hal_->get_dev_msgs_factory(programmable_core_type);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    DeviceAddr launch_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH);

    cluster_->read_core(
        launch_msg.data(), launch_msg.size(), {static_cast<size_t>(device_id), virtual_core}, launch_addr);

    launch_msg.view().kernel_config().exit_erisc_kernel() = 1;
    llrt::write_launch_msg_to_core(device_id, virtual_core, launch_msg.view(), go_msg.view(), false);

    if (!is_idle_eth) {
        // Active
        std::vector<uint32_t> clear_flag_data = {0};
        cluster_->write_core_immediate(device_id, virtual_core, clear_flag_data, get_active_erisc_launch_flag_addr());
    }
};

bool MetalContext::is_coord_in_range(CoreCoord coord, CoreType core_type) {
    ChipId id = *cluster_->all_chip_ids().begin();
    if (core_type == CoreType::ACTIVE_ETH || core_type == CoreType::IDLE_ETH) {
        core_type = CoreType::ETH;
    }

    CoreCoord virtual_coord = cluster_->get_virtual_coordinate_from_logical_coordinates(id, coord, core_type);
    return cluster_->is_ethernet_core(virtual_coord, id) || cluster_->is_worker_core(virtual_coord, id);
}

void MetalContext::on_dispatch_timeout_detected() {
    std::lock_guard<std::mutex> lock(dispatch_timeout_detection_mutex_);

    if (!dispatch_timeout_detection_processed_) {
        dispatch_timeout_detection_processed_ = true;
        log_error(tt::LogMetal, "Timeout detected");
        // Serialize Inspector RPC data if enabled
        if (rtoptions_.get_serialize_inspector_on_dispatch_timeout()) {
            log_info(tt::LogMetal, "Serializing Inspector RPC data");
            Inspector::serialize_rpc();
        }

        // Execute command if specified (mostly used to call tt-triage when a timeout occurs)
        std::string command = rtoptions_.get_dispatch_timeout_command_to_execute();
        if (!command.empty()) {
            log_info(tt::LogMetal, "Executing command: {}", command);

            int result = std::system(command.c_str());

            if (result != 0) {
                log_warning(
                    tt::LogMetal, "Timeout command '{}' returned non-zero exit code: {}", command, WEXITSTATUS(result));
            }
        }
    }
}

}  // namespace tt::tt_metal
