// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include <tt-metalium/global_circular_buffer.hpp>
#include "dram_prefetcher_program_factory.hpp"

namespace ttnn::prim {

using std::vector;

using namespace tt::tt_metal;

std::pair<uint32_t, uint32_t> get_max_page_size_and_num_pages(
    uint32_t max_page_size, uint32_t num_tiles, uint32_t num_datums_per_tile) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * num_datums_per_tile;

    uint32_t page_size = (max_page_size / num_datums_per_tile) * num_datums_per_tile;
    while (total_size % page_size != 0 && page_size >= num_datums_per_tile) {
        page_size -= num_datums_per_tile;
    }
    uint32_t num_pages = total_size / page_size;

    return {page_size, num_pages};
}

DramPrefetcherProgramFactory::cached_program_t DramPrefetcherProgramFactory::create(
    const DramPrefetcherParams& operation_attributes,
    const DramPrefetcherInputs& tensor_args,
    Tensor& /*output_tensor*/) {
    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "Must have at least one input tensor");
    TT_FATAL(operation_attributes.global_cb.has_value(), "Global circular buffer must be provided");
    const auto& global_cb = *(operation_attributes.global_cb);
    const uint32_t num_layers = operation_attributes.num_layers;
    const bool enable_performance_mode = operation_attributes.enable_performance_mode;

    /* Buffers */
    // tensors that with addresses
    const ttnn::Tensor& tensor_addrs = input_tensors.back();  // Last tensor is tensor_addrs
    Buffer* tensor_addrs_buffer = tensor_addrs.buffer();
    std::vector<Buffer*> tensor_buffers;
    // tensors that with actual data
    std::vector<Tensor> tensors;
    tensors.resize(input_tensors.size() - 1);
    std::copy(input_tensors.begin(), input_tensors.end() - 1, tensors.begin());
    tensor_buffers.reserve(tensors.size());
    std::transform(
        tensors.begin(), tensors.end(), std::back_inserter(tensor_buffers), [](const auto& t) { return t.buffer(); });

    /* Tiles */
    std::vector<tt::tt_metal::Tile> tensor_tiles;
    tensor_tiles.reserve(tensors.size());
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(tensor_tiles), [](const auto& t) {
        return t.tensor_spec().tile();
    });

    /* Dataformats */
    tt::DataFormat tensor_addrs_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_addrs.dtype());
    std::vector<tt::DataFormat> tensor_data_formats;
    tensor_data_formats.reserve(tensors.size());
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(tensor_data_formats), [](const auto& t) {
        return tt::tt_metal::datatype_to_dataformat_converter(t.dtype());
    });

    Program program{};

    // In validate we make sure that all tensors are on the same device
    uint32_t num_tensors = tensors.size();
    auto sender_receiver_core_mapping = global_cb.sender_receiver_core_mapping()[0];
    uint32_t max_receivers_per_reader = 0;
    for (const auto& [sender, receivers] : global_cb.sender_receiver_core_mapping()) {
        max_receivers_per_reader = std::max(max_receivers_per_reader, receivers.num_cores());
    }

    uint32_t num_dram_banks = tensors[0].shard_spec()->grid.num_cores();
    uint32_t num_readers = global_cb.sender_receiver_core_mapping().size();
    TT_FATAL(num_readers % num_dram_banks == 0, "Reader count must be divisible by DRAM bank count");
    uint32_t readers_per_bank = num_readers / num_dram_banks;
    uint32_t num_blocks = global_cb.receiver_cores().num_cores();
    TT_FATAL(num_blocks % num_dram_banks == 0, "Receiver count must be divisible by DRAM bank count");
    uint32_t receivers_per_bank = num_blocks / num_dram_banks;

    std::vector<uint32_t> tensor_block_num_tiles;
    std::vector<std::vector<uint32_t>> tensor_shapes(num_tensors, std::vector<uint32_t>(2));
    std::vector<uint32_t> tensor_tile_sizes;
    std::vector<uint32_t> tensor_full_row_bytes;
    std::vector<uint32_t> tensor_block_heights;
    tensor_block_num_tiles.reserve(num_tensors);
    tensor_tile_sizes.reserve(num_tensors);
    for (uint32_t t = 0; t < num_tensors; t++) {
        uint32_t height_in_tiles = tensor_buffers[t]->shard_spec().shape()[0] / tensor_tiles[t].get_tile_shape()[0];
        uint32_t width_in_tiles = tensor_buffers[t]->shard_spec().shape()[1] / tensor_tiles[t].get_tile_shape()[1];

        height_in_tiles = tt::round_up(height_in_tiles, num_blocks);
        TT_FATAL(
            width_in_tiles % receivers_per_bank == 0,
            "DRAM shard width {} must be divisible by {} receivers per bank",
            width_in_tiles,
            receivers_per_bank);
        uint32_t reader_width_in_tiles = width_in_tiles / receivers_per_bank * max_receivers_per_reader;
        tensor_shapes[t][0] = height_in_tiles;
        tensor_shapes[t][1] = reader_width_in_tiles;
        uint32_t tile_size = tensor_tiles[t].get_tile_size(tensor_data_formats[t]);
        tensor_block_num_tiles.push_back(height_in_tiles * reader_width_in_tiles / num_blocks);
        tensor_tile_sizes.push_back(tile_size);
        tensor_full_row_bytes.push_back(width_in_tiles * tile_size);
        tensor_block_heights.push_back(height_in_tiles / num_blocks);
    }
    uint32_t max_block_tiles = *std::max_element(tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());
    auto max_tile_size_iterator = std::max_element(tensor_tile_sizes.begin(), tensor_tile_sizes.end());
    uint32_t max_tile_size = *max_tile_size_iterator;
    uint32_t max_tile_size_tensor_idx = std::distance(tensor_tile_sizes.begin(), max_tile_size_iterator);
    tt::DataFormat max_tile_size_df = tensor_data_formats[max_tile_size_tensor_idx];

    // Keep the local reader CB uniformly strided by the maxima below, but do
    // not combine maxima from different tensors when checking the remote GCB.
    // Mixed BFP8/BFP4 streams can have their widest block in BFP4 and their
    // largest tile in BFP8; that Cartesian product is not a real tensor.
    uint32_t max_actual_block_size_per_receiver_core = 0;
    for (uint32_t t = 0; t < num_tensors; t++) {
        max_actual_block_size_per_receiver_core = std::max(
            max_actual_block_size_per_receiver_core,
            tensor_block_num_tiles[t] * tensor_tile_sizes[t] / max_receivers_per_reader);
    }
    uint32_t max_block_size_per_reader_core = max_tile_size * max_block_tiles;

    TT_FATAL(
        2 * max_actual_block_size_per_receiver_core <= global_cb.size(),
        "two receiver pages {} must fit in global cb {}",
        2 * max_actual_block_size_per_receiver_core,
        global_cb.size());

    /* Cores setup */
    const auto& all_reader_core_range = global_cb.sender_cores();
    auto reader_core_range_vec = corerange_to_cores(all_reader_core_range, std::nullopt, true);
    std::vector<CoreRange> active_reader_core_range_vec;
    for (uint32_t i = 0; i < num_readers; ++i) {
        auto core = reader_core_range_vec[i];
        active_reader_core_range_vec.push_back(CoreRange{core, core});
    }
    auto reader_core_range = CoreRangeSet{active_reader_core_range_vec};

    /* read cb setup */
    uint32_t reader_cb_single_tile_size = max_tile_size;
    const uint32_t total_num_blocks_in_buffer = 3;  // reader cb is triple buffered
    uint32_t reader_cb_size = max_block_size_per_reader_core * total_num_blocks_in_buffer;

    uint32_t reader_cb_index = tt::CBIndex::c_0;
    CircularBufferConfig reader_cb_config = CircularBufferConfig(reader_cb_size, {{reader_cb_index, max_tile_size_df}})
                                                .set_page_size(reader_cb_index, reader_cb_single_tile_size);

    CreateCircularBuffer(program, reader_core_range, reader_cb_config);

    uint32_t sync_cb_index = tt::CBIndex::c_3;
    uint32_t sync_cb_page_size = hal::get_l1_alignment();
    CircularBufferConfig sync_cb_confg =
        CircularBufferConfig(sync_cb_page_size, {{sync_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(sync_cb_index, sync_cb_page_size);

    CreateCircularBuffer(program, reader_core_range, sync_cb_confg);

    /* tensor addresses cb setup */
    uint32_t tensor_addrs_single_tile_size = sizeof(uint32_t);
    uint32_t tensor_addrs_cb_size = num_layers * num_tensors * tensor_addrs_single_tile_size;

    uint32_t tensor_addrs_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig tensor_addrs_cb_config =
        CircularBufferConfig(tensor_addrs_cb_size, {{tensor_addrs_cb_index, tensor_addrs_data_format}})
            .set_page_size(tensor_addrs_cb_index, tensor_addrs_single_tile_size)
            .set_globally_allocated_address(*tensor_addrs_buffer);
    auto tensor_addrs_cb = CreateCircularBuffer(program, reader_core_range, tensor_addrs_cb_config);

    /* remote cb setup */
    uint32_t remote_cb_size = global_cb.size();
    uint32_t remote_cb_page_size = global_cb.page_size();
    TT_FATAL(remote_cb_page_size != 0, "Prefetcher requires a fixed-page global circular buffer");
    TT_FATAL(
        remote_cb_page_size >= max_actual_block_size_per_receiver_core,
        "Global CB page {} is smaller than the largest receiver block {}",
        remote_cb_page_size,
        max_actual_block_size_per_receiver_core);
    uint32_t remote_cb_index = tt::CBIndex::c_31;
    CircularBufferConfig remote_cb_config = CircularBufferConfig(remote_cb_size);
    remote_cb_config.remote_index(remote_cb_index).set_page_size(remote_cb_page_size).set_data_format(max_tile_size_df);
    tt::tt_metal::experimental::CreateCircularBuffer(program, reader_core_range, remote_cb_config, global_cb);

    /* Compile time args */

    // Reader kernel
    std::vector<uint32_t> reader_ct_args = {
        num_layers,
        num_tensors,
        num_blocks,
        reader_cb_size,
        max_block_tiles,
        max_block_size_per_reader_core,
        reader_cb_index,
        tensor_addrs_cb_index,
        sync_cb_index,
    };

    // Configs to enable for performance mode
    reader_ct_args.push_back((uint32_t)enable_performance_mode /* skip_ptr_update */);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/reader_dram.cpp",
        reader_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = reader_ct_args});

    // Writer kernel
    std::vector<uint32_t> writer_ct_args = {
        num_layers,
        num_tensors,
        num_blocks,
        max_receivers_per_reader,
        max_block_tiles,
        reader_cb_index,
        remote_cb_index,
        sync_cb_index,
    };

    // Configs to enable for performance mode
    writer_ct_args.push_back((uint32_t)enable_performance_mode /* posted_payload */);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp",
        reader_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            // Keep receiver fanout off the DRAM reader's NOC so the two
            // data-movement processors can sustain traffic concurrently on
            // Blackhole. Preserve the established assignment elsewhere.
            .noc = global_cb.get_device()->arch() == tt::ARCH::BLACKHOLE ? tt::tt_metal::NOC::RISCV_1_default
                                                                         : tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = writer_ct_args});

    /* Runtime args */
    std::vector<uint32_t> coalesced_page_sizes;
    std::vector<uint32_t> coalesced_num_pages;

    // Blackhole's NOC supports 16 KiB bursts.  Keeping the Wormhole-era
    // 8 KiB cap can select much smaller common divisors (for example 5,184 B
    // instead of 15,552 B for Muse BFP4 blocks), needlessly multiplying DRAM
    // read commands.
    uint32_t max_page_size = global_cb.get_device()->arch() == tt::ARCH::BLACKHOLE ? 16384 : 8192;

    for (uint32_t t = 0; t < num_tensors; t++) {
        uint32_t block_width_in_tiles = tensor_shapes[t][1];
        auto [coalesced_page_size, coalesced_num_page] = get_max_page_size_and_num_pages(
            max_page_size, block_width_in_tiles / max_receivers_per_reader, tt::tile_size(tensor_data_formats[t]));
        coalesced_page_sizes.push_back(coalesced_page_size);
        coalesced_num_pages.push_back(coalesced_num_page);
    }

    const auto& ordered_sender_mapping = global_cb.sender_receiver_core_mapping();

    // Runtime args for the reader cores
    for (uint32_t core_index = 0; core_index < ordered_sender_mapping.size(); core_index++) {
        const auto& core = ordered_sender_mapping[core_index].first;

        /* reader kernel */
        uint32_t bank_id = core_index / readers_per_bank;
        uint32_t reader_lane = core_index % readers_per_bank;
        uint32_t vc = (reader_lane & 0x1) + 2;

        uint32_t receiver_start = 0;
        uint32_t bank_group_start = bank_id * readers_per_bank;
        for (uint32_t prior = bank_group_start; prior < core_index; ++prior) {
            receiver_start += ordered_sender_mapping[prior].second.num_cores();
        }
        uint32_t receiver_count = ordered_sender_mapping[core_index].second.num_cores();
        std::vector<uint32_t> reader_page_sizes;
        std::vector<uint32_t> reader_pages_per_row;
        reader_page_sizes.reserve(num_tensors);
        reader_pages_per_row.reserve(num_tensors);
        for (uint32_t t = 0; t < num_tensors; ++t) {
            uint32_t tile_size = tensor_tile_sizes[t];
            uint32_t per_receiver_width_bytes = tensor_full_row_bytes[t] / receivers_per_bank;
            uint32_t width_bytes = per_receiver_width_bytes * receiver_count;
            auto [page_size, pages] =
                get_max_page_size_and_num_pages(max_page_size, width_bytes / tile_size, tile_size);
            reader_page_sizes.push_back(page_size);
            reader_pages_per_row.push_back(pages);
        }

        std::vector<uint32_t> reader_rt_args = {
            bank_id, vc, total_num_blocks_in_buffer, receiver_start, receivers_per_bank};
        reader_rt_args.insert(reader_rt_args.end(), reader_page_sizes.begin(), reader_page_sizes.end());
        reader_rt_args.insert(reader_rt_args.end(), reader_pages_per_row.begin(), reader_pages_per_row.end());
        reader_rt_args.insert(reader_rt_args.end(), tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());
        reader_rt_args.insert(reader_rt_args.end(), tensor_full_row_bytes.begin(), tensor_full_row_bytes.end());
        reader_rt_args.insert(reader_rt_args.end(), tensor_block_heights.begin(), tensor_block_heights.end());

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        /* writer kernel */
        std::vector<uint32_t> writer_rt_args;
        writer_rt_args.insert(writer_rt_args.end(), coalesced_page_sizes.begin(), coalesced_page_sizes.end());
        writer_rt_args.insert(writer_rt_args.end(), coalesced_num_pages.begin(), coalesced_num_pages.end());
        for (auto tensor_shape : tensor_shapes) {  // block_height_in_itles
            writer_rt_args.push_back(tensor_shape[0] / num_blocks);
        }

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
    }

    return cached_program_t{std::move(program), {tensor_addrs_cb}};
}

void DramPrefetcherProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DramPrefetcherParams& /*operation_attributes*/,
    const DramPrefetcherInputs& tensor_args,
    Tensor& /*output_tensor*/) {
    auto& program = cached_program.program;
    const auto& tensor_addrs_cb = cached_program.shared_variables.tensor_addrs_cb;
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& tensor_addrs = input_tensors.back();  // Last tensor is tensor_addrs
    auto* tensor_addrs_buffer = tensor_addrs.buffer();
    UpdateDynamicCircularBufferAddress(program, tensor_addrs_cb, *tensor_addrs_buffer);
}

}  // namespace ttnn::prim
