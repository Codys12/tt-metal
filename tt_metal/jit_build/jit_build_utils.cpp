// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "impl/context/metal_context.hpp"

namespace tt::jit_build::utils {

namespace {

std::string summarize_command(const std::string& cmd) {
    constexpr std::size_t kMaxCommandPreview = 160;
    if (cmd.size() <= kMaxCommandPreview) {
        return cmd;
    }
    return cmd.substr(0, kMaxCommandPreview - 3) + "...";
}

}  // namespace

bool run_command(const std::string& cmd, const std::string& log_file, const bool verbose) {
    // ZoneScoped;
    // ZoneText( cmd.c_str(), cmd.length());
    int ret;
    static std::mutex io_mutex;
    // Use cached env var from rtoptions instead of calling getenv() on every invocation
    const bool dump_commands = tt::tt_metal::MetalContext::instance().rtoptions().get_dump_build_commands();
    if (dump_commands || verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:\n";
            std::cout << cmd << "\n" << std::endl;
        }
        ret = system(cmd.c_str());
    } else {
        std::string redirected_cmd = cmd + " >> " + log_file + " 2>&1";
        std::atomic<bool> command_finished = false;
        std::thread heartbeat_thread([&command_finished, &cmd, &log_file] {
            constexpr auto kHeartbeatInterval = std::chrono::seconds(30);
            const auto command_preview = summarize_command(cmd);
            const auto log_path = std::filesystem::path(log_file).filename().string();
            while (!command_finished.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(kHeartbeatInterval);
                if (command_finished.load(std::memory_order_relaxed)) {
                    break;
                }
                std::lock_guard<std::mutex> lk(io_mutex);
                std::cout << "===== BUILD COMMAND STILL RUNNING";
                if (!log_path.empty()) {
                    std::cout << " [" << log_path << "]";
                }
                std::cout << ":\n" << command_preview << "\n" << std::endl;
            }
        });
        ret = system(redirected_cmd.c_str());
        command_finished.store(true, std::memory_order_relaxed);
        heartbeat_thread.join();
    }

    return (ret == 0);
}

void create_file(const std::string& file_path_str) {
    namespace fs = std::filesystem;

    fs::path file_path(file_path_str);
    fs::create_directories(file_path.parent_path());

    std::ofstream ofs(file_path);
    ofs.close();
}

}  // namespace tt::jit_build::utils
