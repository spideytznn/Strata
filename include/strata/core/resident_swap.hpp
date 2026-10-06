#pragma once
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
namespace strata::core {
// A slot belongs to its owning card; shared by production and the dual-GPU regression.
struct SwapHome {
    strata::core::ExpertCache* cache;
    cudaStream_t stream;
    int dev;
};
template <class Swap, class Locate>
bool resident_stage_swaps(strata::core::FileExpertSource& src, const std::vector<int32_t>& host_res,
                          int64_t n_expert, std::vector<Swap>& swaps, Locate locate) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<SwapHome> used;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const SwapHome home = locate(s.layer);
        if (!home.cache || !home.cache->device_slot(slot)) return false;
        const strata::core::OnDevice on(home.dev);
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || (home.dev >= 0 && current != home.dev)) return false;
        if (const cudaError_t e = cudaMemcpyAsync(src.exchange_buffer(q), home.cache->device_slot(slot),
                            (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer), cudaMemcpyDeviceToHost,
                            home.stream); e != cudaSuccess) {
            std::fprintf(stderr, "strata: copy back of layer %d slot %d (device %d) failed: %s\n", (int) s.layer,
                         (int) slot, home.dev, cudaGetErrorString(e));
            return false;
        }
        bool seen = false;
        for (const SwapHome& u : used) seen = seen || (u.dev == home.dev && u.stream == home.stream);
        if (!seen) used.push_back(home);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const SwapHome& u : used) {
            const strata::core::OnDevice on(u.dev);
            if (const cudaError_t e = cudaStreamSynchronize(u.stream); e != cudaSuccess) {
                std::fprintf(stderr, "strata: copy back sync (device %d) failed: %s\n", u.dev, cudaGetErrorString(e));
                return false;
            }
        }
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) {
                std::fprintf(stderr, "strata: stage_exchange refused layer %d in %d out %d\n", (int) x.layer,
                             (int) x.in, (int) x.out);
                return false;
            }
    }
    swaps.swap(kept);
    return true;
}

} // namespace strata::core
