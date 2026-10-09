#include "strata/weights/nvfp4.hpp"
#include "json_checked.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <future>
#include <set>
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace strata::weights {
using namespace detail;
Nvfp4Projection nvfp4_projection(const WeightSource& s, const std::string& prefix,
                                 const std::string& canonical, uint64_t rows, uint64_t cols) {
    require(rows > 0 && cols > 0 && cols % 64 == 0, "invalid NVFP4 projection geometry");
    auto get = [&](const char* suffix, DType dtype, std::vector<uint64_t> shape) -> const TensorDesc* {
        const auto& t = s.tensor(prefix + suffix);
        require(t.dtype == dtype && t.physical_shape == shape, "NVFP4 dtype/shape mismatch: " + t.name);
        return &t;
    };
    return {canonical, {rows, cols}, get(".weight", DType::U8, {rows, cols / 2}),
            get(".weight_scale", DType::F8_E4M3, {rows, cols / 16}),
            get(".weight_scale_2", DType::F32, {}), get(".input_scale", DType::F32, {})};
}
namespace {
uint64_t check_sizes(uint64_t rows, uint64_t cols, size_t w, size_t s, size_t p) {
    require(rows > 0 && cols > 0 && cols % 64 == 0, "NVFP4 requires rows > 0 and columns divisible by 64");
    const auto n = mul(rows, cols);
    require(w == n / 2 && s == n / 16 && p == mul(n / 64, 36), "NVFP4 buffer size mismatch");
    return n / 64;
}
float scalar(const std::vector<uint8_t>& b) {
    require(b.size() == 4, "FP32 scalar byte size mismatch");
    float value;
    std::memcpy(&value, b.data(), 4);
    require(std::isfinite(value) && value > 0, "NVFP4 global/input scale must be finite and positive");
    return value;
}
}
void pack_nvfp4(std::span<const uint8_t> w, std::span<const uint8_t> s,
                uint64_t rows, uint64_t cols, std::span<uint8_t> dst) {
    const auto blocks = check_sizes(rows, cols, w.size(), s.size(), dst.size());
    // Validate first, so failure never leaves a partially valid destination.
    for (auto scale : s) if (scale > 0x7e)
        throw std::runtime_error("NVFP4 micro scale is negative or NaN; UE4M3 cannot preserve it");
    for (uint64_t b = 0; b < blocks; ++b) {
        auto* out = dst.data() + b * 36;
        std::memcpy(out, s.data() + b * 4, 4);
#if defined(_M_X64) || defined(__SSE2__)
        // Two adjacent 16-value subblocks per register. Shuffle bytes only:
        // no floating-point decode or requantization enters the owned layout.
        const auto mask=_mm_set1_epi8(15);
        for(int half=0;half<2;++half) {
            const auto input=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.data()+b*32+half*16));
            const auto lo=_mm_shuffle_epi32(input,_MM_SHUFFLE(2,0,2,0));
            const auto hi=_mm_shuffle_epi32(input,_MM_SHUFFLE(3,1,3,1));
            const auto even=_mm_or_si128(_mm_and_si128(lo,mask),_mm_slli_epi16(_mm_and_si128(hi,mask),4));
            const auto odd=_mm_or_si128(_mm_and_si128(_mm_srli_epi16(lo,4),mask),_mm_andnot_si128(mask,hi));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out+4+half*16),_mm_unpacklo_epi8(even,odd));
        }
#else
        for (uint64_t sub = 0; sub < 4; ++sub) {
            const auto* in = w.data() + b * 32 + sub * 8;
            for (uint64_t j = 0; j < 8; ++j) {
                const uint8_t lo = (in[j / 2] >> (4 * (j % 2))) & 15;
                const uint8_t hi = (in[4 + j / 2] >> (4 * (j % 2))) & 15;
                out[4 + sub * 8 + j] = lo | (hi << 4);
            }
        }
#endif
    }
}
void unpack_nvfp4(std::span<const uint8_t> p, uint64_t rows, uint64_t cols,
                  std::span<uint8_t> w, std::span<uint8_t> s) {
    const auto blocks = check_sizes(rows, cols, w.size(), s.size(), p.size());
    for (uint64_t b = 0; b < blocks; ++b) {
        const auto* in = p.data() + b * 36;
        std::memcpy(s.data() + b * 4, in, 4);
        for (uint64_t sub = 0; sub < 4; ++sub) {
            for (uint64_t j = 0; j < 8; ++j) {
                const uint64_t a = j * 2, c = a + 1;
                const auto lo = (in[4 + sub * 8 + a % 8] >> (a / 8 * 4)) & 15;
                const auto hi = (in[4 + sub * 8 + c % 8] >> (c / 8 * 4)) & 15;
                w[b * 32 + sub * 8 + j] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }
    }
}
NativeExpert load_nvfp4_expert(WeightSource& source, const std::array<Nvfp4Projection, 3>& projections,
                               int layer, bool verify) {
    const auto ff = projections[0].logical_shape[0], hidden = projections[0].logical_shape[1];
    require(ff == 640 && hidden == 2560 && projections[1].logical_shape == projections[0].logical_shape &&
            projections[2].logical_shape == std::array<uint64_t, 2>{hidden, ff}, "unsupported expert geometry");
    NativeExpert result;
    auto& l = result.layout;
    l.gu_type = l.d_type = 40; // ggml NVFP4 id, existing NativeExpertLayout ABI
    l.n_embd = hidden; l.n_ff = ff; l.layer = layer;
    l.gu_row = hidden / 64 * 36; l.d_row = ff / 64 * 36;
    l.up_off = ff * l.gu_row; l.down_off = 2 * l.up_off;
    l.tail_off = l.down_off + hidden * l.d_row; l.bytes = l.tail_off + 16;
    result.bytes.resize(l.bytes);
    // Read all 12 ranges together; the source sorts/coalesces them. This is a
    // bounded single-expert adapter, not the eventual full-arena streaming loader.
    std::array<std::array<std::vector<uint8_t>, 4>, 3> raw;
    std::vector<ReadRequest> reads;
    for (size_t i = 0; i < 3; ++i) {
        const auto& p = projections[i];
        const TensorDesc* parts[] = {p.weight, p.micro_scale, p.weight_scale, p.input_scale};
        for (size_t j = 0; j < 4; ++j) {
            raw[i][j].resize(static_cast<size_t>(parts[j]->bytes));
            reads.push_back({parts[j], 0, raw[i][j]});
        }
    }
    source.read_many(reads);
    const size_t offsets[] = {0, l.up_off, l.down_off, l.tail_off};
    for (size_t i = 0; i < 3; ++i) {
        const auto [rows, cols] = projections[i].logical_shape;
        const auto dst = std::span<uint8_t>(result.bytes).subspan(offsets[i], offsets[i+1] - offsets[i]);
        const float global = scalar(raw[i][2]);
        result.input_scales[i] = scalar(raw[i][3]);
        pack_nvfp4(raw[i][0], raw[i][1], rows, cols, dst);
        std::memcpy(result.bytes.data() + l.tail_off + i * 4, &global, 4);
        if (verify) {
            std::vector<uint8_t> w(raw[i][0].size()), s(raw[i][1].size());
            unpack_nvfp4(dst, rows, cols, w, s);
            require(w == raw[i][0] && s == raw[i][1], "NVFP4 lossless roundtrip failed");
            require(std::memcmp(result.bytes.data() + l.tail_off + i*4, raw[i][2].data(), 4) == 0 &&
                    std::memcmp(&result.input_scales[i], raw[i][3].data(), 4) == 0, "FP32 scale bits changed");
            result.roundtrip_bytes += w.size() + s.size() + 8;
        }
    }
    // The current optional TC ABI has one gate/up activation scale. Keep both
    // originals above and reject dispatch to that ABI if they differ.
    if (result.input_scales[0] == result.input_scales[1]) l.input_scale_gu = result.input_scales[0];
    l.input_scale_down = result.input_scales[2];
    return result;
}

Nvfp4BatchStats load_nvfp4_expert_batch(WeightSource& source,
        std::span<const Nvfp4ExpertBuffer> targets, unsigned workers, Nvfp4LoadScratch* scratch) {
    require(!targets.empty() && targets.size() <= 128, "NVFP4 startup batch must contain 1..128 experts");
    require(workers > 0 && workers <= 16, "NVFP4 startup workers must be in [1,16]");
    constexpr uint64_t projection_bytes = 2560ull * 640 / 64 * 36;
    constexpr uint64_t blob_bytes = 3 * projection_bytes + 16;
    // Check all output ranges before reading or launching workers. Overlapping
    // destinations would race, even if the input checkpoint itself were valid.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    std::set<std::array<float,3>*> scale_outputs;
    for (const auto& target : targets) {
        require(target.projections && target.input_scales && target.bytes.data() && target.bytes.size() == blob_bytes,
                "invalid NVFP4 batch destination");
        const uintptr_t begin = reinterpret_cast<uintptr_t>(target.bytes.data());
        require(begin <= UINTPTR_MAX - blob_bytes, "NVFP4 destination address overflow");
        ranges.emplace_back(begin, begin + blob_bytes);
        require(scale_outputs.insert(target.input_scales).second, "overlapping NVFP4 scale destinations");
        for (size_t p = 0; p < 3; ++p) {
            const auto& projection = (*target.projections)[p];
            require(projection.logical_shape == (p == 2 ? std::array<uint64_t,2>{2560,640} :
                                                        std::array<uint64_t,2>{640,2560}), "unsupported expert geometry");
            require(projection.weight && projection.micro_scale && projection.weight_scale && projection.input_scale,
                    "null NVFP4 projection tensor");
            check_sizes(projection.logical_shape[0], projection.logical_shape[1],
                        projection.weight->bytes, projection.micro_scale->bytes, projection_bytes);
            require(projection.weight_scale->bytes == 4 && projection.input_scale->bytes == 4,
                    "invalid NVFP4 scalar byte size");
        }
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
        require(ranges[i-1].second <= ranges[i].first, "overlapping NVFP4 batch destinations");

    Nvfp4LoadScratch local;
    auto& raw = scratch ? scratch->experts : local.experts;
    raw.resize(targets.size());
    std::vector<ReadRequest> reads;
    Nvfp4BatchStats stats;
    reads.reserve(targets.size() * 12);
    for (size_t i = 0; i < targets.size(); ++i) for (size_t p = 0; p < 3; ++p) {
        const auto& projection = (*targets[i].projections)[p];
        const TensorDesc* parts[] = {projection.weight, projection.micro_scale, projection.weight_scale, projection.input_scale};
        for (size_t j = 0; j < 4; ++j) {
            raw[i][p][j].resize(static_cast<size_t>(parts[j]->bytes));
            stats.scratch_bytes += parts[j]->bytes;
            reads.push_back({parts[j], 0, raw[i][p][j]});
        }
    }
    using Clock = std::chrono::steady_clock;
    const auto read_start = Clock::now();
    source.read_many(reads); // the source sorts and coalesces only adjacent ranges of this family
    const auto pack_start = Clock::now();
    stats.read_ms = std::chrono::duration<double,std::milli>(pack_start-read_start).count();
    workers = std::min(workers, static_cast<unsigned>(targets.size()));
    auto pack = [&](unsigned worker) {
        for (size_t i = worker; i < targets.size(); i += workers) {
            const auto& target = targets[i];
            for (size_t p = 0; p < 3; ++p) {
                const auto& projection = (*target.projections)[p];
                const float global = scalar(raw[i][p][2]);
                (*target.input_scales)[p] = scalar(raw[i][p][3]);
                pack_nvfp4(raw[i][p][0], raw[i][p][1], projection.logical_shape[0], projection.logical_shape[1],
                           target.bytes.subspan(p * projection_bytes, projection_bytes));
                std::memcpy(target.bytes.data() + 3 * projection_bytes + p * 4, &global, 4);
            }
            // Match the zero-initialized legacy NativeExpert vector's unused tail.
            std::memset(target.bytes.data() + 3 * projection_bytes + 12, 0, 4);
        }
    };
    if (workers == 1) pack(0);
    else {
        std::vector<std::future<void>> jobs;
        for (unsigned worker = 0; worker < workers; ++worker)
            jobs.push_back(std::async(std::launch::async, pack, worker));
        // Future destruction joins the other workers even if a worker throws.
        // No reader thread touches WeightSource or its counters.
        for (auto& job : jobs) job.get();
    }
    stats.pack_ms = std::chrono::duration<double,std::milli>(Clock::now()-pack_start).count();
    return stats;
}
} // namespace strata::weights
