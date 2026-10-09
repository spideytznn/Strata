#pragma once
#include <cstdint>

namespace strata::kernels {
// Original E4M3FN codes plus BF16 [ceil(rows/128),ceil(cols/128)]
// multiplicative scales; gate/up/down use the same fixed byte extent.
inline constexpr uint64_t kMtpFp8Matrix = 640ull * 2560;
inline constexpr uint64_t kMtpFp8Projection = kMtpFp8Matrix + 100 * 2;
inline constexpr uint64_t kMtpFp8Expert = 3 * kMtpFp8Projection;
void mtp_fp8_experts(const uint8_t* experts, const int32_t* ids, const float* x,
                     float* hidden, float* parts, int tokens, int topk, void* stream);
}
