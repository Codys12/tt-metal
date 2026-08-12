// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <optional>

namespace ttnn::prim {

void DramPrefetcherOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "Must have at least one input tensor");
    TT_FATAL(args.num_layers > 0, "Prefetcher must run for at least 1 layer");
    TT_FATAL(args.global_cb.has_value(), "Global circular buffer must be provided");
    const ttnn::Tensor& tensor_addrs = input_tensors.back();  // Last tensor is tensor_addrs

    auto global_cb = *(args.global_cb);

    const auto& sender_receiver_core_mapping = global_cb.sender_receiver_core_mapping();
    uint32_t num_readers = sender_receiver_core_mapping.size();
    uint32_t num_dram_banks = input_tensors[0].shard_spec()->grid.num_cores();
    uint32_t num_receivers = 0;
    for (uint32_t i = 0; i < num_readers; ++i) {
        const auto& [sender_core, receiver_core_range] = sender_receiver_core_mapping[i];
        TT_FATAL(receiver_core_range.num_cores() > 0, "Every sender core must have at least one receiver");
        num_receivers += receiver_core_range.num_cores();
    }
    TT_FATAL(num_receivers % num_dram_banks == 0, "Receiver count must be divisible by DRAM bank count");
    uint32_t num_receivers_per_bank = num_receivers / num_dram_banks;

    TT_FATAL(num_readers > 0, "Number of reader cores must be greater than zero");

    for (size_t i = 0; i < input_tensors.size() - 1; ++i) {
        const auto& tensor = input_tensors[i];
        // Check that all tensors are on the same device
        TT_FATAL(tensor.device() == input_tensors[0].device(), "All tensors must be on the same device");
        TT_FATAL(tensor.layout() == Layout::TILE, "All tensors must be tilized");
        TT_FATAL(
            tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensors must be width sharded");
        TT_FATAL(tensor.memory_config().buffer_type() == BufferType::DRAM, "Input tensors must be in DRAM");

        // Each DRAM-bank shard is split across all receivers assigned to that
        // bank, even when two reader cores own unequal receiver groups.
        TT_FATAL(
            tensor.buffer()->shard_spec().shape()[1] % num_receivers_per_bank == 0,
            "All tensors' padded shard size (in last dim) {} must be divisible by the number of receiver cores per "
            "DRAM bank {}.",
            tensor.buffer()->shard_spec().shape()[1],
            num_receivers_per_bank);

        tt::DataFormat tensor_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
        TT_FATAL(
            tensor_data_format == tt::DataFormat::Bfp4_b || tensor_data_format == tt::DataFormat::Bfp8_b ||
                tensor_data_format == tt::DataFormat::Float16_b,
            "Input tensors must be of type Bfp4_b, Bfp8_b, or Float16_b");
    }

    TT_FATAL(
        tensor_addrs.device() == input_tensors[0].device(),
        "tensors_addrs must be on the same device as the input tensors");
    TT_FATAL(tensor_addrs.layout() == Layout::ROW_MAJOR, "Tensor containing addresses must be row major");
    TT_FATAL(
        tensor_addrs.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Tensor containing addresses must be height sharded");
    TT_FATAL(tensor_addrs.memory_config().buffer_type() == BufferType::L1, "Tensor containing addresses must be in L1");

    tt::DataFormat tensor_addrs_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_addrs.dtype());
    TT_FATAL(tensor_addrs_data_format == tt::DataFormat::UInt32, "Tensor containing addresses must be of type UInt32");
}

TensorSpec DramPrefetcherOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return TensorSpec(
        ttnn::Shape{32, 32},
        tt::tt_metal::TensorLayout(
            tensor_args.input_tensors[0].dtype(),
            tt::tt_metal::PageConfig(tensor_args.input_tensors[0].layout()),
            MemoryConfig{}));
}

DramPrefetcherOperation::tensor_return_value_t DramPrefetcherOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensors[0].device());
}

ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    const uint32_t num_layers,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const bool enable_performance_mode) {
    auto operation_attributes = DramPrefetcherParams{
        .num_layers = num_layers,
        .enable_performance_mode = enable_performance_mode,
        .global_cb = global_cb,
    };
    auto tensor_args = DramPrefetcherInputs{.input_tensors = tensors};

    return ttnn::device_operation::launch<DramPrefetcherOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
