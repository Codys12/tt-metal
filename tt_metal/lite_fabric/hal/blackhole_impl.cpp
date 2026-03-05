// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <thread>

#include "blackhole_impl.hpp"
#include "hw/inc/host_interface.hpp"
#include "tt_metal/lite_fabric/hw/inc/blackhole/lf_dev_mem_map.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace {

constexpr uint32_t GetStateAddress() {
    return LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config) +
           offsetof(lite_fabric::FabricLiteConfig, current_state);
}

uint32_t GetConfigAddress() { return LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config); }

}  // namespace

namespace lite_fabric {

void BlackholeLiteFabricHal::set_reset_state(tt_cxy_pair virtual_core, bool assert_reset) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (assert_reset) {
        // Assert ALL cores including ERISC0.  Phase 1 loads base firmware on
        // ERISC0 and deasserts it, so ERISC0 is running when we get here.
        // ERISC0 and ERISC1 share the ethernet TX queue on the same ETH core;
        // if ERISC0 is left running, its base firmware can:
        //   - Send heartbeats / mailbox responses that stall ERISC1's
        //     eth_txq_is_busy() loops, blocking lite fabric packet sends.
        //   - Assert ERISC1's reset as part of a context-switch, killing
        //     the lite fabric link mid-operation.
        // Putting ERISC0 in reset for the duration of lite fabric avoids
        // both issues.  ERISC0 is restored when Metal re-initializes
        // (Phase 1 of the next init cycle).
        cluster.assert_risc_reset_at_core(virtual_core, tt::umd::RiscType::ALL_TENSIX);
    } else {
        // Deassert only ERISC1.  ERISC0 stays in reset to avoid TX queue
        // contention for the entire lite fabric lifetime.
        tt::umd::RiscType reset_val = tt::umd::RiscType::ERISC1;
        cluster.deassert_risc_reset_at_core(virtual_core, reset_val);
    }
}

void BlackholeLiteFabricHal::set_pc(tt_cxy_pair virtual_core, uint32_t pc_val) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    cluster.write_core(reinterpret_cast<void*>(&pc_val), sizeof(uint32_t), virtual_core, LITE_FABRIC_RESET_PC);
}

tt::umd::semver_t BlackholeLiteFabricHal::get_binary_version() { return tt::umd::semver_t{0, 0, 0}; }

void BlackholeLiteFabricHal::launch(const std::filesystem::path& bin_path) {
    constexpr uint32_t k_FirmwareStart = LITE_FABRIC_TEXT_START;

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto config_addr = GetConfigAddress();

    // Read binary once
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (!bin_file) {
        throw std::runtime_error(fmt::format("Failed to open binary file: {}", bin_path));
    }
    bin_file.seekg(0, std::ios::end);
    size_t bin_size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    binary_data_.resize(bin_size);
    bin_file.read(reinterpret_cast<char*>(binary_data_.data()), bin_size);
    bin_file.close();
    log_info(tt::LogMetal, "Loaded lite fabric binary {} size {} B", bin_path, bin_size);
    const auto& binary_data = binary_data_;

    // Diagnostic: read ETH port status and TXQ0 registers for each tunnel's MMIO core
    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        try {
            uint8_t port_status = 0xFF;
            cluster.read_core(&port_status, sizeof(port_status), tunnel_1x.mmio_cxy_virtual(), 0x7CC04);
            const char* status_str = (port_status == 1)   ? "UP"
                                     : (port_status == 2) ? "DOWN"
                                     : (port_status == 0) ? "UNKNOWN/TRAINING"
                                                          : "UNUSED/OTHER";
            log_info(
                tt::LogMetal,
                "ETH port status: mmio chip={} core={} (virtual={}) -> remote chip={}: port_status={} ({})",
                tunnel_1x.mmio_id,
                tunnel_1x.mmio_core_logical.str(),
                tunnel_1x.mmio_core_virtual.str(),
                tunnel_1x.connected_id,
                port_status,
                status_str);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "ETH port status: mmio chip={} core={} -> remote chip={}: read failed: {}",
                tunnel_1x.mmio_id,
                tunnel_1x.mmio_core_logical.str(),
                tunnel_1x.connected_id,
                e.what());
        }
        // Read TXQ0 registers: CTRL (offset 0x0), CMD (0x4), STATUS (0x8),
        // TRANSFER_CNT (0x30), PKT_START_CNT (0x34), PKT_END_CNT (0x3C)
        try {
            constexpr uint32_t TXQ0_BASE = 0xFFB90000;
            uint32_t ctrl = 0, status = 0, xfer_cnt = 0, pkt_start = 0, pkt_end = 0;
            cluster.read_core(&ctrl, sizeof(ctrl), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x00);
            cluster.read_core(&status, sizeof(status), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x08);
            cluster.read_core(&xfer_cnt, sizeof(xfer_cnt), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x30);
            cluster.read_core(&pkt_start, sizeof(pkt_start), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x34);
            cluster.read_core(&pkt_end, sizeof(pkt_end), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x3C);
            log_info(
                tt::LogMetal,
                "TXQ0 diag: mmio chip={} core={}: CTRL=0x{:x} (KEEPALIVE={}), STATUS=0x{:x} (CMD_ONGOING={}), "
                "TRANSFER_CNT={}, PKT_START={}, PKT_END={}",
                tunnel_1x.mmio_id,
                tunnel_1x.mmio_core_logical.str(),
                ctrl,
                (ctrl & 1) ? "YES" : "NO",
                status,
                ((status >> 16) & 1) ? "YES" : "NO",
                xfer_cnt,
                pkt_start,
                pkt_end);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "TXQ0 diag: mmio chip={} core={}: read failed: {}",
                tunnel_1x.mmio_id,
                tunnel_1x.mmio_core_logical.str(),
                e.what());
        }
    }

    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        auto mmio_mask = system_descriptor_.enabled_eth_channels.at(tunnel_1x.mmio_id);
        log_info(
            tt::LogMetal,
            "Launching lite fabric: mmio chip={} core={} (virtual={}) -> remote chip={} core={} (virtual={}), "
            "eth_chans_mask=0x{:x}",
            tunnel_1x.mmio_id,
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str(),
            tunnel_1x.connected_id,
            tunnel_1x.connected_core_logical.str(),
            tunnel_1x.connected_core_virtual.str(),
            mmio_mask);

        // Host writes firmware + config to MMIO side; firmware handles remote via ethernet
        lite_fabric::FabricLiteConfig config{};
        config.is_primary = true;
        config.is_mmio = true;
        config.initial_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
        config.current_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
        config.binary_addr = LITE_FABRIC_TEXT_START;
        config.binary_size = (bin_size + 15) & ~0xF;
        config.eth_chans_mask = mmio_mask;
        config.routing_enabled = lite_fabric::RoutingEnabledState::ENABLED;

        set_reset_state(tunnel_1x.mmio_cxy_virtual(), true);
        set_pc(tunnel_1x.mmio_cxy_virtual(), k_FirmwareStart);

        // Zero the host interface counters (d2h + h2d = 4 bytes) on device before
        // starting the firmware.  Phase 1's clear_l1_state only clears the "unreserved"
        // portion of ETH L1 which doesn't include the lite fabric memory area.
        // Without this, the firmware starts with stale counter values from previous
        // runs or undefined post-reset state, causing wait_for_all_writes_consumed to
        // deadlock on a counter mismatch.
        uint32_t host_iface_addr =
            LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, host_interface);
        uint32_t zero_counters = 0;
        cluster.write_core(
            reinterpret_cast<void*>(&zero_counters),
            sizeof(zero_counters),
            tunnel_1x.mmio_cxy_virtual(),
            host_iface_addr);

        cluster.write_core(
            (void*)&config, sizeof(lite_fabric::FabricLiteConfig), tunnel_1x.mmio_cxy_virtual(), config_addr);
        cluster.write_core(binary_data.data(), bin_size, tunnel_1x.mmio_cxy_virtual(), LITE_FABRIC_TEXT_START);

        log_info(
            tt::LogMetal,
            "Wrote lite fabric. Chip: {}, Core: {}, Config: {:#x}, Binary: {:#x}, Size: {} B",
            tunnel_1x.mmio_id,
            tunnel_1x.mmio_core_logical,
            config_addr,
            LITE_FABRIC_TEXT_START,
            bin_size);
    }

    cluster.l1_barrier(0);

    for (auto tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        set_reset_state(tunnel_1x.mmio_cxy_virtual(), false);
    }

    // Wait for MMIO side to reach READY (firmware sends binary to remote and handshakes).
    // Tunnels that fail to come up are reset and removed so the system can proceed with
    // whatever connectivity is available.
    auto it = system_descriptor_.tunnels_from_mmio.begin();
    while (it != system_descriptor_.tunnels_from_mmio.end()) {
        const auto& tunnel_1x = *it;
        log_info(
            tt::LogMetal,
            "Waiting for lite fabric: mmio chip={} core={} (virtual={}) -> remote chip={} core={} (virtual={})",
            tunnel_1x.mmio_id,
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str(),
            tunnel_1x.connected_id,
            tunnel_1x.connected_core_logical.str(),
            tunnel_1x.connected_core_virtual.str());
        try {
            wait_for_state(tunnel_1x.mmio_cxy_virtual(), lite_fabric::InitState::READY);
        } catch (...) {
            // Try reading the remote chip's config for additional diagnostics
            try {
                std::vector<uint32_t> remote_readback(sizeof(lite_fabric::FabricLiteConfig) / sizeof(uint32_t));
                cluster.read_core(
                    remote_readback,
                    sizeof(lite_fabric::FabricLiteConfig),
                    tunnel_1x.connected_cxy_virtual(),
                    GetConfigAddress());
                auto* rcfg = reinterpret_cast<lite_fabric::FabricLiteConfig*>(remote_readback.data());
                log_warning(
                    tt::LogMetal,
                    "Lite fabric tunnel mmio chip={} core={} -> remote chip={} core={} failed to reach READY, "
                    "skipping. Remote config: current_state={}, initial_state={}, "
                    "is_mmio={}, is_primary={}, routing_enabled={}, eth_chans_mask=0x{:x}, "
                    "binary_addr=0x{:x}, binary_size={}, "
                    "primary_local_handshake=0x{:x}, neighbour_handshake=0x{:x}",
                    tunnel_1x.mmio_id,
                    tunnel_1x.mmio_core_logical.str(),
                    tunnel_1x.connected_id,
                    tunnel_1x.connected_core_logical.str(),
                    static_cast<uint32_t>(rcfg->current_state),
                    static_cast<uint32_t>(rcfg->initial_state),
                    rcfg->is_mmio,
                    rcfg->is_primary,
                    static_cast<uint32_t>(rcfg->routing_enabled),
                    rcfg->eth_chans_mask,
                    static_cast<uint32_t>(rcfg->binary_addr),
                    static_cast<uint32_t>(rcfg->binary_size),
                    rcfg->primary_local_handshake,
                    rcfg->neighbour_handshake);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "Lite fabric tunnel mmio chip={} core={} -> remote chip={} failed to reach READY and "
                    "could not read remote config: {}. Skipping tunnel.",
                    tunnel_1x.mmio_id,
                    tunnel_1x.mmio_core_logical.str(),
                    tunnel_1x.connected_id,
                    e.what());
            }
            // Post-timeout TXQ0 diagnostic on MMIO core
            try {
                constexpr uint32_t TXQ0_BASE = 0xFFB90000;
                uint32_t ctrl = 0, status = 0, xfer_cnt = 0, pkt_start = 0, pkt_end = 0;
                cluster.read_core(&ctrl, sizeof(ctrl), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x00);
                cluster.read_core(&status, sizeof(status), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x08);
                cluster.read_core(&xfer_cnt, sizeof(xfer_cnt), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x30);
                cluster.read_core(&pkt_start, sizeof(pkt_start), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x34);
                cluster.read_core(&pkt_end, sizeof(pkt_end), tunnel_1x.mmio_cxy_virtual(), TXQ0_BASE + 0x3C);
                log_warning(
                    tt::LogMetal,
                    "Post-timeout TXQ0: mmio chip={} core={}: CTRL=0x{:x} (KEEPALIVE={}), STATUS=0x{:x} "
                    "(CMD_ONGOING={}), "
                    "TRANSFER_CNT={}, PKT_START={}, PKT_END={}",
                    tunnel_1x.mmio_id,
                    tunnel_1x.mmio_core_logical.str(),
                    ctrl,
                    (ctrl & 1) ? "YES" : "NO",
                    status,
                    ((status >> 16) & 1) ? "YES" : "NO",
                    xfer_cnt,
                    pkt_start,
                    pkt_end);
            } catch (...) {
            }
            // Reset the failed MMIO-side core and remove the tunnel
            set_reset_state(tunnel_1x.mmio_cxy_virtual(), true);
            it = system_descriptor_.tunnels_from_mmio.erase(it);
            continue;
        }
        log_info(
            tt::LogMetal,
            "Lite Fabric {} (virtual={}) is ready",
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str());
        ++it;
    }

    if (system_descriptor_.tunnels_from_mmio.empty()) {
        TT_THROW("All lite fabric tunnels failed to initialize. No remote devices are reachable.");
    }
}

void BlackholeLiteFabricHal::terminate() {
    uint32_t routing_enabled_address = LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config) +
                                       offsetof(lite_fabric::FabricLiteConfig, routing_enabled);
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Signal STOP (not STOPPED) so the firmware properly shuts down the remote side.
    // The STOP handler in service_lite_fabric() asserts reset on the remote ERISC1
    // before transitioning to STOPPED.  Without this, the remote ERISC1 keeps running
    // after the MMIO side is killed, leaving stale state (possibly a busy TXQ) that
    // causes the next init to fail.
    uint32_t stop_val = static_cast<uint32_t>(lite_fabric::RoutingEnabledState::STOP);
    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        log_info(
            tt::LogMetal,
            "Host to terminate lite fabric on device {} {} (virtual={})",
            tunnel_1x.mmio_id,
            tunnel_1x.mmio_core_logical,
            tunnel_1x.mmio_core_virtual);
        cluster.write_core((void*)&stop_val, sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), routing_enabled_address);
    }
    cluster.l1_barrier(0);

    // Wait for firmware to process STOP and transition to STOPPED.
    // This confirms the remote ERISC1 has been reset.
    std::vector<uint32_t> readback(1);
    constexpr int k_TerminateMaxPolls = 100;
    constexpr int k_TerminatePollMs = 10;
    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        for (int i = 0; i < k_TerminateMaxPolls; i++) {
            cluster.read_core(readback, sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), routing_enabled_address);
            if (static_cast<lite_fabric::RoutingEnabledState>(readback[0]) ==
                lite_fabric::RoutingEnabledState::STOPPED) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(k_TerminatePollMs));
        }
    }

    LiteFabricHal::set_reset_state(true);
}

void BlackholeLiteFabricHal::wait_for_state(tt_cxy_pair virtual_core, lite_fabric::InitState state) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<uint32_t> readback{static_cast<uint32_t>(lite_fabric::InitState::UNKNOWN)};
    constexpr int k_MaxPolls = 20;  // 2 seconds – handshake is near-instant on a healthy link
    constexpr int k_PollIntervalMs = 100;
    int poll_count = 0;
    while (static_cast<lite_fabric::InitState>(readback[0]) != state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(k_PollIntervalMs));
        cluster.read_core(readback, sizeof(uint32_t), virtual_core, GetStateAddress());
        poll_count++;
        if (poll_count % 10 == 0) {
            log_warning(
                tt::LogMetal,
                "wait_for_state: core {} still in state {} (waiting for {}), poll #{}",
                virtual_core.str(),
                readback[0],
                static_cast<uint32_t>(state),
                poll_count);
        }
        if (poll_count >= k_MaxPolls) {
            // Read the full config struct for diagnostics
            std::vector<uint32_t> config_readback(sizeof(lite_fabric::FabricLiteConfig) / sizeof(uint32_t));
            cluster.read_core(config_readback, sizeof(lite_fabric::FabricLiteConfig), virtual_core, GetConfigAddress());
            auto* cfg = reinterpret_cast<lite_fabric::FabricLiteConfig*>(config_readback.data());
            log_error(
                tt::LogMetal,
                "wait_for_state: TIMEOUT on core {}. current_state={}, initial_state={}, "
                "is_mmio={}, is_primary={}, routing_enabled={}, eth_chans_mask=0x{:x}, "
                "binary_addr=0x{:x}, binary_size={}, "
                "primary_local_handshake=0x{:x}, neighbour_handshake=0x{:x}, "
                "padding1=[0x{:x},0x{:x},0x{:x}], padding2=[0x{:x}]",
                virtual_core.str(),
                static_cast<uint32_t>(cfg->current_state),
                static_cast<uint32_t>(cfg->initial_state),
                cfg->is_mmio,
                cfg->is_primary,
                static_cast<uint32_t>(cfg->routing_enabled),
                cfg->eth_chans_mask,
                static_cast<uint32_t>(cfg->binary_addr),
                static_cast<uint32_t>(cfg->binary_size),
                cfg->primary_local_handshake,
                cfg->neighbour_handshake,
                cfg->padding1[0],
                cfg->padding1[1],
                cfg->padding1[2],
                cfg->padding2[0]);
            TT_THROW(
                "Lite fabric core {} failed to reach state {} (stuck at {})",
                virtual_core.str(),
                static_cast<uint32_t>(state),
                readback[0]);
        }
    }
}

std::vector<std::filesystem::path> BlackholeLiteFabricHal::build_includes(const std::filesystem::path& root_dir) {
    return {
        root_dir / "tt_metal/hw/inc/internal/tt-1xx/blackhole",
        root_dir / "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc",
        root_dir / "tt_metal/hw/inc/internal/tt-1xx",
        root_dir / "tt_metal/hw/inc/internal",
        root_dir / "tt_metal/hw/inc/internal/dataflow",
        root_dir / "tt_metal/hw/inc/internal/ethernet",
        root_dir / "tt_metal/hw/ckernels/blackhole/metal/common",
        root_dir / "tt_metal/hw/ckernels/blackhole/metal/llk_io",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
        root_dir / "tt_metal/lite_fabric/hw/inc",
        root_dir / "tt_metal/lite_fabric/hw/inc/blackhole",
    };
}

std::vector<std::string> BlackholeLiteFabricHal::build_defines() {
    return {
        "ARCH_BLACKHOLE",
        "TENSIX_FIRMWARE",
        "LOCAL_MEM_EN=0",
        "COMPILE_FOR_ERISC",  // This is needed to enable the ethernet APIs
        "ERISC",
        "RISC_B0_HW",
        "FW_BUILD",
        // Must be NOC 0 to match the noc_index in the packet header (set by UMD) and
        // edm_to_local_chip_noc in constants.hpp.  UMD uses TRANSLATED coordinates which
        // are NOC 0 coordinates on Blackhole; NOC 1 has a mirrored coordinate system.
        "NOC_INDEX=0",
        "DISPATCH_MESSAGE_ADDR=0",
        "COMPILE_FOR_LITE_FABRIC=1",
        "ROUTING_FW_ENABLED",
        // This is needed to get things to compile
        "NUM_DRAM_BANKS=1",
        "NUM_L1_BANKS=1",
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=0",
        "LOG_BASE_2_OF_NUM_L1_BANKS=0",
        // We do not access the PCIe cores
        "PCIE_NOC_X=0",
        "PCIE_NOC_Y=0",
        // Lite Fabric is intended to run on risc1
        "PROCESSOR_INDEX=1",
    };
}

std::vector<std::filesystem::path> BlackholeLiteFabricHal::build_linker(const std::filesystem::path& root_dir) {
    return {
        root_dir / "runtime/hw/lib/blackhole/tmu-crt0.o",
        root_dir / "runtime/hw/lib/blackhole/substitutes.o",
    };
}

}  // namespace lite_fabric
