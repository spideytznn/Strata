#pragma once
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <vector>
namespace strata::core {
// The slot is local to the cache of the GPU that owns the expert's layer.
struct ResidentSwapTarget { ExpertCache* cache; int device; cudaStream_t stream; };
template <class Swap, class Resolve>
bool resident_stage_swaps(strata::core::FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert, std::vector<Swap>& swaps,
                          Resolve resolve) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<Swap> kept;
    std::vector<ResidentSwapTarget> targets;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const ResidentSwapTarget target = resolve(s.layer);
        if (!target.cache || !target.cache->device_slot(slot)) return false;
        const strata::core::OnDevice on(target.device);
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || (target.device >= 0 && current != target.device)) return false;
        if (cudaMemcpyAsync(src.exchange_buffer(q), target.cache->device_slot(slot),
                            (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer), cudaMemcpyDeviceToHost,
                            target.stream) != cudaSuccess)
            return false;
        bool seen_target = false;
        for (const auto& t : targets)
            if (t.device == target.device && t.stream == target.stream) seen_target = true;
        if (!seen_target) targets.push_back(target);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const auto& target : targets) {
            const strata::core::OnDevice on(target.device);
            if (cudaStreamSynchronize(target.stream) != cudaSuccess) return false;
        }
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) return false;
    }
    swaps.swap(kept);
    return true;
}

} // namespace strata::core
