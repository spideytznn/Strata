#pragma once
#include "strata/weights/dense.hpp"
#include "ggml.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace strata::weights {
// Same ten Q8_0 matrices as the existing MTP pack. The router, HC mixers,
// original FP8 experts and main-model weights do not pass through this adapter.
inline bool mtp_q8_projection(std::string_view name) {
    constexpr std::string_view names[] = {
        "fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight",
        "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.indexer.index_qk_proj.weight", "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (auto n : names) if (n == name) return true;
    return false;
}
inline uint64_t mtp_q8_row_bytes(uint64_t columns) {
    if (!columns || columns % 32) throw std::runtime_error("MTP Q8 columns must be a multiple of 32");
    return columns / 32 * 34;
}
// Row-aligned streaming bounds CPU scratch independently of the matrix size.
// Quantize through ggml's reference implementation used by the old Q8 pack.
inline void stream_mtp_q8(WeightSource& source, const DenseBinding& b,
                         const std::function<void(uint64_t, std::span<const uint8_t>)>& emit,
                         uint64_t tile_bytes = 4 * 1024 * 1024) {
    if (b.output != DType::BF16 || b.math != DenseMath::Identity)
        throw std::runtime_error("MTP Q8 requires an identity BF16 binding");
    const auto row_bytes = mtp_q8_row_bytes(b.columns);
    std::vector<float> values;
    std::vector<uint8_t> quantized;
    stream_dense(source, b, [&](uint64_t off, std::span<const uint8_t> data) {
        const auto rows = data.size() / (b.columns * 2);
        values.resize(data.size() / 2);
        for (size_t i = 0; i < values.size(); ++i) {
            uint16_t bits;
            std::memcpy(&bits, data.data() + 2 * i, 2);
            values[i] = std::bit_cast<float>(uint32_t(bits) << 16);
            if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite MTP Q8 input: " + b.name);
        }
        quantized.resize(rows * row_bytes);
        const auto written = ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), quantized.data(),
                                                  0, int64_t(rows), int64_t(b.columns), nullptr);
        if (written != quantized.size()) throw std::runtime_error("MTP Q8 quantized extent mismatch");
        for (size_t block = 0; block < quantized.size(); block += 34) {
            uint16_t scale;
            std::memcpy(&scale, quantized.data() + block, 2);
            if ((scale & 0x7c00) == 0x7c00) throw std::runtime_error("nonfinite MTP Q8 scale: " + b.name);
        }
        emit(off / (b.columns * 2) * row_bytes, quantized);
    }, tile_bytes);
}
} // namespace strata::weights
