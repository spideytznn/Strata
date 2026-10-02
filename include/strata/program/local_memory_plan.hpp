#pragma once
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace strata::program {
struct PrefillBufferCandidate { int64_t chunk; uint64_t bytes; };

// Use only free VRAM after the existing cache is filled. Equal-size own buffers
// avoid removing and refilling cached experts; never shrink the borrowed batch.
inline int64_t independent_prefill_chunk(std::initializer_list<PrefillBufferCandidate> candidates,
                                         uint64_t free_bytes, uint64_t reserve_bytes,
                                         int64_t max_chunk, int64_t borrowed_chunk) {
    if (free_bytes <= reserve_bytes || max_chunk <= 0) return 0;
    const uint64_t available = free_bytes - reserve_bytes;
    int64_t best = 0;
    for (const auto& c : candidates)
        if (c.chunk > 0 && c.chunk <= max_chunk && c.chunk >= borrowed_chunk &&
            c.bytes > 0 && c.bytes <= available && c.chunk > best) best = c.chunk;
    return best;
}

// Parked images and resident experts share physical RAM. A disabled cache keeps
// upstream headroom exactly; an enabled one reserves its budget and RAM floor.
inline uint64_t conversation_expert_headroom(uint64_t expert_headroom, uint64_t cache_budget,
                                             uint64_t cache_floor, bool cache_enabled) {
    if (!cache_enabled || cache_budget == 0) return expert_headroom;
    constexpr uint64_t margin = 256ull << 20;
    constexpr uint64_t limit = std::numeric_limits<uint64_t>::max();
    if (cache_budget > limit - cache_floor || cache_budget + cache_floor > limit - margin) return limit;
    const uint64_t required = cache_budget + cache_floor + margin;
    return required > expert_headroom ? required : expert_headroom;
}
} // namespace strata::program
