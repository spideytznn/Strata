#pragma once
#include "strata/kernels/iq_kernels.hpp"
namespace strata::kernels {
// Opt-in SM120 FP4 MMA decode. terms=1 uses checkpoint activation scales;
// terms=2/3 add independently block-scaled FP4 residuals, sharing the global scale.
// Residual modes extend the checkpoint scale by powers of two for outlier rows.
void nvfp4_tensor_grouped(const NativeExpertLayout& layout,
    const unsigned long long* ptr, const int32_t* starts, const int32_t* groups,
    const int32_t* dst, const int32_t* tok, int64_t cap_groups, int64_t cap_entries,
    const float* x, void* scratch, float* out, void* stream, int terms);
}
