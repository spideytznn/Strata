#pragma once
#include "strata/weights/safetensors.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace strata::weights {
// All four source tensors share the WeightSource lifetime. The canonical name
// describes the existing projection; logical_shape is [output, input].
struct Nvfp4Projection {
    std::string canonical_name;
    std::array<uint64_t, 2> logical_shape;
    const TensorDesc *weight, *micro_scale, *weight_scale, *input_scale;
};
Nvfp4Projection nvfp4_projection(const WeightSource&, const std::string& hf_prefix,
                                 const std::string& canonical, uint64_t rows, uint64_t cols);
// Pure byte permutation. Never quantizes, rounds, transposes, or folds scales.
// Nonfinite/negative E4M3 scales are refused because the old UE4M3 kernels would
// silently reinterpret them. Zero micro scales and signed FP4 zero are retained.
void pack_nvfp4(std::span<const uint8_t> weights, std::span<const uint8_t> scales,
                uint64_t rows, uint64_t cols, std::span<uint8_t> destination);
void unpack_nvfp4(std::span<const uint8_t> packed, uint64_t rows, uint64_t cols,
                  std::span<uint8_t> weights, std::span<uint8_t> scales);
struct NativeExpert {
    strata::kernels::NativeExpertLayout layout;
    std::vector<uint8_t> bytes;
    std::array<float, 3> input_scales; // gate, up, down; never collapsed by assumption
    uint64_t roundtrip_bytes = 0;
};
NativeExpert load_nvfp4_expert(WeightSource&, const std::array<Nvfp4Projection, 3>&,
                               int layer, bool verify_roundtrip);
} // namespace strata::weights
