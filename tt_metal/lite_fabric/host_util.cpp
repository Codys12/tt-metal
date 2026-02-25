// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/entries.hpp>
#include <tt-logger/tt-logger.hpp>
#include "hal/lite_fabric_hal.hpp"
#include "tt_metal/lite_fabric/build.hpp"
#include <tt-metalium/hal_types.hpp>
#include <memory>

namespace lite_fabric {

void InitializeLiteFabric(std::shared_ptr<lite_fabric::LiteFabricHal>& lite_fabric_hal) {
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    if (tt_metal_home == nullptr) {
        throw std::runtime_error("TT_METAL_HOME environment variable is not set; required for lite fabric compilation");
    }
    auto home_directory = std::filesystem::path(tt_metal_home);
    auto output_directory = home_directory / "lite_fabric";

    if (lite_fabric::CompileFabricLite(lite_fabric_hal, home_directory, output_directory)) {
        throw std::runtime_error("Failed to compile lite fabric");
    }
    auto bin_path = lite_fabric::LinkFabricLite(lite_fabric_hal, home_directory, output_directory);
    if (!bin_path) {
        throw std::runtime_error("Failed to link lite fabric");
    }

    lite_fabric_hal->launch(*bin_path);
}

}  // namespace lite_fabric
