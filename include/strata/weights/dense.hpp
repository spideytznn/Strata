#pragma once
#include "strata/weights/qwen4.hpp"
#include <functional>

namespace strata::weights {
enum class DenseMath { Identity, AddOne, NegativeExp };
struct DenseBinding {
    std::string name;
    const TensorDesc* source = nullptr;
    uint64_t rows = 0, columns = 0, first_row = 0;
    // HF [16 key heads, 3 value heads/key] -> engine [3, 16].
    uint64_t row_head_size = 0, row_prefix = 0, column_head_size = 0;
    DType output = DType::BF16;
    DenseMath math = DenseMath::Identity;
    bool native_projection = false;
    uint64_t bytes() const { return rows * columns * dtype_bytes(output); }
};
class Qwen4DensePlan {
public:
    explicit Qwen4DensePlan(SafetensorsSource&);
    std::vector<DenseBinding> tensors;
    const DenseBinding& find(const std::string&) const;
};
// Emits contiguous destination tiles. The callback must consume each span before
// returning. BF16 identities/permutations preserve every bit, including -0.
// Norm arithmetic and A_log use FP32, matching the existing forward contract.
void stream_dense(WeightSource&, const DenseBinding&,
                  const std::function<void(uint64_t, std::span<const uint8_t>)>&,
                  uint64_t tile_bytes = 4 * 1024 * 1024);
} // namespace strata::weights
