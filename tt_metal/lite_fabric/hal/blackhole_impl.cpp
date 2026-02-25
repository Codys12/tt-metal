// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

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
    // We run on ERISC1. Don't touch ERISC0. It is running base firmware
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (assert_reset) {
        // Assert all cores except ERISC0.
        tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
        cluster.assert_risc_reset_at_core(virtual_core, reset_val);
    } else {
        // Deassert only ERISC1.
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
    std::vector<uint8_t> binary_data(bin_size);
    bin_file.read(reinterpret_cast<char*>(binary_data.data()), bin_size);
    bin_file.close();
    log_info(tt::LogMetal, "Loaded lite fabric binary {} size {} B", bin_path, bin_size);

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
        config.binary_size = (LITE_FABRIC_TEXT_SIZE + 15) & ~0xF;
        config.eth_chans_mask = mmio_mask;
        config.routing_enabled = lite_fabric::RoutingEnabledState::ENABLED;

        set_reset_state(tunnel_1x.mmio_cxy_virtual(), true);
        set_pc(tunnel_1x.mmio_cxy_virtual(), k_FirmwareStart);
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

    // Wait for MMIO side to reach READY (firmware sends binary to remote and handshakes)
    for (auto tunnel_1x : system_descriptor_.tunnels_from_mmio) {
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
                log_error(
                    tt::LogMetal,
                    "Remote chip {} core {} (virtual={}) config: current_state={}, initial_state={}, "
                    "is_mmio={}, is_primary={}, routing_enabled={}, eth_chans_mask=0x{:x}, "
                    "binary_addr=0x{:x}, binary_size={}, "
                    "primary_local_handshake=0x{:x}, neighbour_handshake=0x{:x}",
                    tunnel_1x.connected_id,
                    tunnel_1x.connected_core_logical.str(),
                    tunnel_1x.connected_core_virtual.str(),
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
                log_error(tt::LogMetal, "Failed to read remote chip {} config: {}", tunnel_1x.connected_id, e.what());
            }
            throw;
        }
        log_info(
            tt::LogMetal,
            "Lite Fabric {} (virtual={}) is ready",
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str());
    }
}

void BlackholeLiteFabricHal::terminate() {
    uint32_t routing_enabled_address = LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config) +
                                       offsetof(lite_fabric::FabricLiteConfig, routing_enabled);
    uint32_t enabled = 0;
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Signal MMIO-side ERISC1 to stop; firmware propagates STOP to remote via ethernet
    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        log_info(
            tt::LogMetal,
            "Host to terminate lite fabric on device {} {} (virtual={})",
            tunnel_1x.mmio_id,
            tunnel_1x.mmio_core_logical,
            tunnel_1x.mmio_core_virtual);
        cluster.write_core((void*)&enabled, sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), routing_enabled_address);
    }
    cluster.l1_barrier(0);

    LiteFabricHal::set_reset_state(true);
}

void BlackholeLiteFabricHal::wait_for_state(tt_cxy_pair virtual_core, lite_fabric::InitState state) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<uint32_t> readback{static_cast<uint32_t>(lite_fabric::InitState::UNKNOWN)};
    int poll_count = 0;
    while (static_cast<lite_fabric::InitState>(readback[0]) != state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
        if (poll_count >= 100) {
            // Read the full config struct for diagnostics
            std::vector<uint32_t> config_readback(sizeof(lite_fabric::FabricLiteConfig) / sizeof(uint32_t));
            cluster.read_core(config_readback, sizeof(lite_fabric::FabricLiteConfig), virtual_core, GetConfigAddress());
            auto* cfg = reinterpret_cast<lite_fabric::FabricLiteConfig*>(config_readback.data());
            log_error(
                tt::LogMetal,
                "wait_for_state: TIMEOUT on core {}. current_state={}, initial_state={}, "
                "is_mmio={}, is_primary={}, routing_enabled={}, eth_chans_mask=0x{:x}, "
                "binary_addr=0x{:x}, binary_size={}, "
                "primary_local_handshake=0x{:x}, neighbour_handshake=0x{:x}",
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
                cfg->neighbour_handshake);
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
        "NOC_INDEX=1",
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
