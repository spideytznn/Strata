// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/core/pinned.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    n_expert_ = n_expert;
    blobs_ = n_layers * n_expert;
    const uint64_t want = (uint64_t) blobs_ * (uint64_t) strata::kernels::cpu::BLOB;
    const std::string path = pack_dir + "/experts.bin";

#if defined(_WIN32)
    // UTF-8 -> UTF-16: the pack may live under a path with non-ASCII characters, and `CreateFileA` would
    // silently mangle it into a file-not-found.
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    // **`FILE_FLAG_RANDOM_ACCESS` WAS HERE AND IT COST 14x.**
    //
    // The design depends on the OS page cache holding the whole 34 GB expert set, because this machine has
    // 64 GB of DDR5 and `L9` measured the CPU path at 44.14 GB/s from DRAM.  `FILE_FLAG_RANDOM_ACCESS` tells
    // the cache manager the opposite: it disables read-ahead AND it lets the manager drop the pages again
    // quickly, on the assumption that a large randomly-accessed file will not be re-read.  Measured, on
    // `strata generate --max-new 24`: **1.93 GB/s** - disk speed, 344 ms/token, and it never warmed up over 25
    // tokens, because the pages were being evicted as fast as they were faulted in.
    //
    // The correct flag is NO flag.  The access pattern IS random (10 of 512 experts per layer, a different 10
    // each layer), but every byte read is read again on the next token, so retention is the whole game.
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = "FileExpertSource: cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        err = "FileExpertSource: cannot size " + path;
        return false;
    }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but %lld layers x %lld experts x %d B is %llu B - this "
                      "is not the pack this geometry came from",
                      path.c_str(), (unsigned long long) sz.QuadPart, (long long) n_layers,
                      (long long) n_expert, (int) strata::kernels::cpu::BLOB, (unsigned long long) want);
        CloseHandle(f);
        err = buf;
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        err = "FileExpertSource: CreateFileMapping failed on " + path;
        return false;
    }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        err = "FileExpertSource: MapViewOfFile failed on " + path;
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if ((uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but %lld layers x %lld experts x %d B is %llu B - this "
                      "is not the pack this geometry came from",
                      path.c_str(), (unsigned long long) st.st_size, (long long) n_layers, (long long) n_expert,
                      (int) strata::kernels::cpu::BLOB, (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
#endif
    return true;
}

void FileExpertSource::close() {
#if defined(_WIN32)
    if (base_ != nullptr) UnmapViewOfFile((LPCVOID) base_);
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (base_ != nullptr) munmap((void*) base_, (size_t) blobs_ * (size_t) strata::kernels::cpu::BLOB);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ = nullptr;
    blobs_ = 0;
    reads_ = 0;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0) return nullptr;
    // **`expert >= n_expert_` IS CHECKED SEPARATELY, AND THE FLAT INDEX ALONE DOES NOT CATCH IT.**  The blob
    // index is `layer * n_expert + expert`, so `blob(0, 512)` has flat index 512 - which is in range, and is
    // `blob(1, 0)`.  A router id one past the end of a layer would then read the NEXT LAYER's first expert:
    // finite, correctly sized, and wrong.  Layer and expert are separate axes and are validated as such.
    if (expert >= n_expert_) return nullptr;
    const int64_t i = layer * n_expert_ + expert;
    if (i >= blobs_) return nullptr;
    ++reads_;
    return base_ + (size_t) i * strata::kernels::cpu::BLOB;
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    ++d.layers;
    d.experts += k;
}

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[128];                     // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= 128 && n <= d.plan->cap) {
        int64_t distinct[128], first_of[128];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0) ++nmiss;
            }
        }
        // the tier fix: probe the LAYER's slice, not expert 0 - the VRAM-only skip set may have excluded
        // expert 0 from the arena, and a null alias here would silently turn the PCIe share off
        const bool pcie_ok = d.pcie_num > 0 && d.src->device_alias_layer(d.layers) != nullptr;
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src = d.src->blob(d.layers, e);
                        if (src != nullptr && d.src->pinned(d.layers, e)) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P.publish) P.publish(P.ctx);
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0 : -1;
        }
    }
    const auto c1 = std::chrono::steady_clock::now();
    if (native && lay.fmt[(size_t) d.layers].gu_type == 42)   // a native Q2_0 pack: the Q2_0 kernels' activations
        for (int64_t t = 0; t < n_tok; ++t) act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    else if (native)
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
    else
        for (int64_t t = 0; t < n_tok; ++t) act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    const auto c2 = std::chrono::steady_clock::now();
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // the GPU computes this entry (a VRAM hit or a PCIe read)
                if (kind[i] == 0) ++d.cache_hits;
                std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
    else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) cudaEventRecord((cudaEvent_t) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

namespace {

// Plan v0.3 P6: a layer's experts may sit in another shard of the model (native_experts.txt v3): a name beside
// `gguf`.  Shared by the dense loader, the sparse loader and the VRAM-only preload so all three agree on the path.
std::string gguf_file_of(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, int64_t l) {
    if (lay.gguf_file.empty() || lay.gguf_file[(size_t) l].empty()) return gguf;
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    return dir + lay.gguf_file[(size_t) l];
}

// The VRAM-only tier: ONE expert's three GGUF slices (gate / up / down) read into `dst`, which must hold
// `lay.blob_bytes(l)` bytes at the blob's internal layout [gate rows | up rows | down rows].  This is the inner
// loop of `load_experts_gguf` extracted, byte for byte: a sparse arena load and a VRAM-only preload through
// this helper read exactly what the dense loader would have written.
bool read_expert_gguf(std::ifstream& f, std::string& open_name, const std::string& gguf,
                      const strata::kernels::cpu::ExpertLayout& lay, int64_t l, int64_t e, uint8_t* dst) {
    const std::string name = gguf_file_of(gguf, lay, l);
    if (name != open_name) {
        f.close();
        f.clear();
        f.open(name, std::ios::binary);
        if (!f) return false;
        open_name = name;
    }
    const auto& fm = lay.fmt[(size_t) l];
    const uint64_t blob = lay.bytes[(size_t) l];
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        f.seekg((std::streamoff) (lay.gguf_off[(size_t) (3 * l + r)] + (uint64_t) e * per[r]));
        f.read((char*) (dst + at[r]), (std::streamsize) per[r]);
        if ((uint64_t) f.gcount() != per[r]) return false;
    }
    return true;
}

}  // namespace

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            if (!f.is_open() || gguf_file_of(gguf, lay, l) != open_name) {
                const std::string name = gguf_file_of(gguf, lay, l);
                f.close();
                f.clear();
                f.open(name, std::ios::binary);
                if (!f) { bad = true; return; }
                open_name = name;
            }
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    f.seekg((std::streamoff) (src + done));
                    f.read((char*) buf.data(), (std::streamsize) n);
                    if ((uint64_t) f.gcount() != n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

namespace {

// The VRAM-only tier's sparse GGUF load: only the RETAINED experts are read, each through the same
// three-slice `read_expert_gguf` the dense loader's inner loop performs, landing at the layer's COMPACT
// offset (`kept_off[l] + compact_idx * blob_bytes`).  `kept` is [l * n_expert + e] -> compact index or -1.
LoadStats load_experts_gguf_sparse(const std::string& gguf, uint8_t* dst,
                                   const strata::kernels::cpu::ExpertLayout& lay,
                                   const std::vector<int32_t>& kept, const std::vector<uint64_t>& kept_off,
                                   const std::vector<int64_t>& kept_cnt, int threads) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    uint64_t bytes = 0;
    for (int64_t l = 0; l < lay.n_layers; ++l) bytes += (uint64_t) kept_cnt[(size_t) l] * lay.blob_bytes(l);
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const uint64_t bb = lay.blob_bytes(l);
            for (int64_t e = 0; e < lay.n_expert; ++e) {
                const int32_t c = kept[(size_t) (l * lay.n_expert + e)];
                if (c < 0) continue;   // VRAM-only: its bytes are preloaded into the cache, not the arena
                if (!read_expert_gguf(f, open_name, gguf, lay, l, e, dst + kept_off[(size_t) l] + (uint64_t) c * bb)) {
                    bad = true;
                    return;
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        return st;
    }
    st.bytes = bytes;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

// The VRAM-only tier's sparse experts.bin load: the retained experts of a layer form contiguous runs in the
// FILE, and each run is one (source offset, arena offset, bytes) copy.  (`load_experts_ranges` cannot express
// this: its file offset and destination offset are the same number, which is only true of the dense layout.)
LoadStats load_experts_runs(const std::string& path, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            const std::vector<int32_t>& kept, const std::vector<uint64_t>& kept_off,
                            int threads, uint64_t chunk) {
    struct Run { uint64_t src, dst, bytes; };
    std::vector<Run> runs;
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const uint64_t bb = lay.blob_bytes(l);
        for (int64_t e = 0; e < lay.n_expert;) {
            const int32_t c0 = kept[(size_t) (l * lay.n_expert + e)];
            if (c0 < 0) { ++e; continue; }
            int64_t e1 = e;
            while (e1 < lay.n_expert && kept[(size_t) (l * lay.n_expert + e1)] == c0 + (e1 - e)) ++e1;
            runs.push_back({lay.blob_offset(l, e), kept_off[(size_t) l] + (uint64_t) c0 * bb, (uint64_t) (e1 - e) * bb});
            e = e1;
        }
    }
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    for (const Run& r : runs) st.bytes += r.bytes;
    if (threads < 1) threads = 1;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    auto worker = [&]() {
        std::vector<uint8_t> buf;
        std::ifstream f(path, std::ios::binary);
        if (!f) { bad = true; return; }
        for (;;) {
            const size_t r = next.fetch_add(1);
            if (r >= runs.size() || bad) break;
            if (buf.size() < chunk) buf.resize((size_t) chunk);
            uint64_t pos = 0;
            while (pos < runs[r].bytes) {
                const uint64_t n = std::min<uint64_t>(chunk, runs[r].bytes - pos);
                f.seekg((std::streamoff) (runs[r].src + pos));
                f.read((char*) buf.data(), (std::streamsize) n);
                if ((uint64_t) f.gcount() != n) { bad = true; return; }
                std::memcpy(dst + runs[r].dst + pos, buf.data(), (size_t) n);
                pos += n;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        return st;
    }
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

}  // namespace

ArenaExpertSource::~ArenaExpertSource() { close(); }

void ArenaExpertSource::set_skip(const std::vector<std::pair<int32_t, int32_t>>* skip) {
    skip_.clear();
    if (skip != nullptr) skip_ = *skip;
}

bool ArenaExpertSource::skipped(int64_t layer, int64_t expert) const {
    if (!sparse_ || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return kept_[(size_t) (layer * n_expert_ + expert)] < 0;
}

bool ArenaExpertSource::read_expert_blob(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) const {
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        err = "ArenaExpertSource: expert (layer " + std::to_string(layer) + ", expert " + std::to_string(expert) +
              ") is out of range";
        return false;
    }
    if (from_gguf_) {
        // the same three gate/up/down reads the (dense or sparse) arena load makes
        std::ifstream f;
        std::string open_name;
        if (!read_expert_gguf(f, open_name, gguf_, lay, layer, expert, dst)) {
            err = "ArenaExpertSource: cannot read expert (layer " + std::to_string(layer) + ", expert " +
                  std::to_string(expert) + ") from " + gguf_file_of(gguf_, lay, layer);
            return false;
        }
        return true;
    }
    // an experts.bin pack: the blob sits at its dense offset in the file whether or not the arena kept it
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
        err = "ArenaExpertSource: cannot open " + path_;
        return false;
    }
    const uint64_t bb = lay.blob_bytes(layer);
    f.seekg((std::streamoff) lay.blob_offset(layer, expert));
    f.read((char*) dst, (std::streamsize) bb);
    if ((uint64_t) f.gcount() != bb) {
        err = "ArenaExpertSource: short read of expert (layer " + std::to_string(layer) + ", expert " +
              std::to_string(expert) + ") from " + path_;
        return false;
    }
    return true;
}

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;

    // ---- the VRAM-only tier's compaction tables.  `ExpertLayout` keeps describing the FILE; this remap is
    // private to the source.  With no skip set (the default) every table stays empty and every path below is
    // bit-for-bit the dense one.
    sparse_ = !skip_.empty();
    kept_.clear();
    kept_off_.clear();
    kept_cnt_.clear();
    slice_idx_.clear();
    uint64_t want = lay.total;
    if (sparse_) {
        kept_.assign((size_t) (n_layers * n_expert), 0);
        kept_off_.assign((size_t) n_layers, 0);
        kept_cnt_.assign((size_t) n_layers, 0);
        for (const auto& p : skip_) {
            if (p.first < 0 || p.first >= n_layers || p.second < 0 || p.second >= n_expert) {
                err = "ArenaExpertSource: the skip set holds (layer " + std::to_string(p.first) + ", expert " +
                      std::to_string(p.second) + "), out of range";
                return false;
            }
            const size_t at = (size_t) (p.first * n_expert + p.second);
            kept_[at] = -1;   // exclusion set; duplicate pairs remain excluded
        }
        want = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            for (int64_t e = 0; e < n_expert; ++e) {
                int32_t& c = kept_[(size_t) (l * n_expert + e)];
                if (c >= 0) c = (int32_t) kept_cnt_[(size_t) l]++;
            }
            kept_off_[(size_t) l] = want;
            want += (uint64_t) kept_cnt_[(size_t) l] * lay.blob_bytes(l);
        }
    }

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.  (The check is against the FILE's dense size `lay.total`, not the
    // compact `want`: it validates the pack, and only the arena is compacted.)
    if (!from_gguf) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != lay.total) {
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) lay.total);
            err = buf;
            return false;
        }
    }

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    // (Sparse: one slice per layer WITH retained experts; an emptied layer would register zero bytes, which
    // cudaHostRegister refuses, so `slice_idx_` maps layers to their slice instead of assuming layer order.)
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (sparse_ && kept_cnt_[(size_t) l] == 0) continue;
        const uint64_t off = sparse_ ? kept_off_[(size_t) l] : lay.layer_offset(l);
        const uint64_t bytes = sparse_ ? (uint64_t) kept_cnt_[(size_t) l] * lay.blob_bytes(l)
                                       : lay.blob_bytes(l) * (uint64_t) n_expert;
        bounds.push_back(off);
        loff.push_back(off);
        lbytes.push_back(bytes);
    }
    if (bounds.empty()) bounds.push_back(0);
    bounds.push_back(want);
    PinnedArena* a = new PinnedArena(want + (uint64_t) blob, bounds);
    if (!a->valid()) {
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" + std::to_string(want) + " B)";
        return false;
    }
    const LoadStats st = sparse_
        ? (from_gguf ? load_experts_gguf_sparse(gguf_, a->data(), lay, kept_, kept_off_, kept_cnt_, threads)
                     : load_experts_runs(path, a->data(), lay, kept_, kept_off_, threads, /*chunk=*/8u << 20))
        : (from_gguf ? load_experts_gguf(gguf_, a->data(), lay, threads)
                     : load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20));
    if (st.seconds < 0 || st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    arena_ = a;
    resident_bytes_ = want;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    // the sparse remap of layer -> registration slice (the dense layout needs none: slice l IS layer l)
    if (sparse_) {
        slice_idx_.assign((size_t) n_layers, -1);
        if (a->slice_bytes > 0)
            for (int64_t l = 0; l < n_layers; ++l)
                for (size_t i = 0; i < a->slice_starts.size(); ++i)
                    if (a->slice_starts[i] == kept_off_[(size_t) l]) slice_idx_[(size_t) l] = (int32_t) i;
    }
    blobs_ = n_layers * n_expert;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    path_ = path;
    from_gguf_ = from_gguf;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.gib_per_second();
    return true;
}

void ArenaExpertSource::close() {
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    resident_bytes_ = 0;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    sparse_ = false;
    kept_.clear();
    kept_off_.clear();
    kept_cnt_.clear();
    slice_idx_.clear();
    path_.clear();
    from_gguf_ = false;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_ || layer >= n_layers_) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (sparse_) {
        const int32_t c = kept_[(size_t) (layer * n_expert_ + expert)];
        if (c < 0) return false;   // VRAM-only: no arena row, so nothing is pinned
        return kept_off_[(size_t) layer] + (uint64_t) (c + 1) * lay.blob_bytes(layer) <= pinned_bytes_;
    }
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (sparse_) {
        const int32_t c = kept_[(size_t) (layer * n_expert_ + expert)];
        if (slice_bytes_ == 0) return dev_slice_[0] + kept_off_[(size_t) layer] + (uint64_t) c * lay.blob_bytes(layer);
        // one registration slice per (non-empty) layer
        const int32_t si = slice_idx_[(size_t) layer];
        if (si < 0 || (size_t) si >= dev_slice_.size()) return nullptr;
        return dev_slice_[(size_t) si] + (uint64_t) c * lay.blob_bytes(layer);
    }
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::device_alias_layer(int64_t layer) const {
    if (dev_slice_.empty() || layer < 0 || layer >= n_layers_) return nullptr;
    if (sparse_ && kept_cnt_[(size_t) layer] == 0) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0)
        return dev_slice_[0] + (sparse_ ? kept_off_[(size_t) layer] : lay.layer_offset(layer));
    const int32_t si = sparse_ ? slice_idx_[(size_t) layer] : (int32_t) layer;
    if (si < 0 || (size_t) si >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) si];
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (sparse_) {
        const int32_t c = kept_[(size_t) idx];
        if (c < 0) return nullptr;   // VRAM-only: it lives in the cache, not here (and no read is counted)
        ++reads_;
        return base_ + kept_off_[(size_t) layer] + (uint64_t) c * lay.blob_bytes(layer);
    }
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return base_ + lay.blob_offset(layer, expert);
}

}  // namespace strata::core
