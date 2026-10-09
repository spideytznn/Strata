// Bounded, model-free reproduction of the native residency allocation order.
// Run only when no inference engine is alive: this locks the full expert arena.
#include "strata/platform/memory.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {
constexpr size_t GiB = size_t(1) << 30;
constexpr size_t arena_bytes = 67948118016ull;
char device_luid[8]{};
void check(cudaError_t e, const char* step) {
    size_t free_b = 0, total_b = 0;
    const auto mem = cudaMemGetInfo(&free_b, &total_b);
    std::printf("step=%s status=%d message=\"%s\" free_bytes=%zu total_bytes=%zu mem_status=%d\n",
                step, int(e), cudaGetErrorString(e), free_b, total_b, int(mem));
    uint64_t budget = 0, usage = 0;
    std::string why;
    if (strata::platform::gpu_shared_memory_budget(device_luid, budget, usage, why))
        std::printf("shared_budget_bytes=%llu shared_usage_bytes=%llu\n",
                    (unsigned long long) budget, (unsigned long long) usage);
#ifdef _WIN32
    MEMORYSTATUSEX ram{};
    ram.dwLength = sizeof ram;
    if (GlobalMemoryStatusEx(&ram))
        std::printf("available_ram_bytes=%llu available_commit_bytes=%llu\n",
                    (unsigned long long) ram.ullAvailPhys, (unsigned long long) ram.ullAvailPageFile);
#endif
    std::fflush(stdout);
    if (e != cudaSuccess) throw std::runtime_error(step);
}
__global__ void read_registered(const unsigned char* p, size_t last, unsigned* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = unsigned(p[0]) + unsigned(p[last]);
}
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: native-registration-probe REGISTER_GIB (0..44) [late|chunks|alloc]; chunks registers the whole arena in pieces; alloc uses cudaHostAlloc for all 63.282 GiB; uses 20 GiB VRAM\n");
        return 2;
    }
    char* end = nullptr;
    const long gib = std::strtol(argv[1], &end, 10);
    if (!*argv[1] || *end || gib < 0 || gib > 44) return 2;
    const bool late = argc == 3 && std::strcmp(argv[2], "late") == 0;
    const bool chunks = argc == 3 && std::strcmp(argv[2], "chunks") == 0;
    const bool allocated = argc == 3 && std::strcmp(argv[2], "alloc") == 0;
    if ((argc == 3 && !late && !chunks && !allocated) || (chunks && !gib)) return 2;
    void* arena = nullptr;
    uint64_t locked = 0;
    std::vector<void*> registrations;
    std::vector<std::pair<void*,size_t>> aliases;
    int status = 1;
    try {
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "device");
        std::memcpy(device_luid, prop.luid, sizeof device_luid);
        std::printf("device=\"%s\" registered_gib=%ld arena_bytes=%zu late=%d chunks=%d\n", prop.name, gib, arena_bytes, int(late), int(chunks));
        void* dense = nullptr;
        check(cudaMalloc(&dense, 8 * GiB), "dense_allocate");
        check(cudaMemset(dense, 0, 8 * GiB), "dense_zero");
        check(cudaDeviceSynchronize(), "dense_commit");
        void* cache = nullptr;
        auto allocate_cache = [&] {
            check(cudaMalloc(&cache, 12 * GiB), "cache_allocate");
            check(cudaMemset(cache, 0, 12 * GiB), "cache_zero");
            check(cudaDeviceSynchronize(), "cache_commit");
        };
        if (late) allocate_cache();
        if (allocated) check(cudaHostAlloc(&arena, arena_bytes, cudaHostAllocMapped), "host_allocate_pinned");
        else arena = std::malloc(arena_bytes);
        if (!arena) throw std::runtime_error("host_allocate");
        if (!allocated) {
            const auto lock = strata::platform::lock_resident(arena, arena_bytes);
            locked = lock.locked_bytes;
            std::printf("locked_bytes=%llu lock_ok=%d note=\"%s\"\n",
                        (unsigned long long) locked, int(lock.ok), lock.note.c_str());
            std::fflush(stdout);
            if (!lock.ok || locked != arena_bytes) throw std::runtime_error("host_lock");
        }
        auto* bytes = static_cast<unsigned char*>(arena);
        std::memset(bytes, 19, 4 << 20);
        void* mapped = nullptr;
        if (allocated) {
            bytes[arena_bytes-1]=23;
            check(cudaHostGetDevicePointer(&mapped, arena, 0), "host_alias_allocated");
            aliases.emplace_back(mapped,arena_bytes);
        } else if (gib) {
            const size_t total = chunks ? arena_bytes : size_t(gib) * GiB;
            for (size_t off=0; off<total; off+=size_t(gib)*GiB) {
                const size_t n = total-off < size_t(gib)*GiB ? total-off : size_t(gib)*GiB;
                bytes[off] = 19; bytes[off+n-1] = 23;
                check(cudaHostRegister(bytes+off, n, cudaHostRegisterMapped), "host_register");
                registrations.push_back(bytes+off);
                void* alias=nullptr;
                check(cudaHostGetDevicePointer(&alias, bytes+off, 0), "host_alias");
                if (!mapped) mapped=alias;
                std::printf("alias_offset=%zu contiguous=%d\n",off,int(static_cast<unsigned char*>(mapped)+off==alias));
                aliases.emplace_back(alias,n);
            }
        }
        if (!late) allocate_cache();
        check(cudaMemcpy(cache, arena, 4 << 20, cudaMemcpyHostToDevice), "cache_fill");
        check(cudaDeviceSynchronize(), "fill_commit");
        for (const auto& span:aliases) {
            read_registered<<<1, 32>>>(static_cast<unsigned char*>(span.first), span.second - 1,
                                      static_cast<unsigned*>(cache));
            check(cudaGetLastError(), "mapped_launch");
            check(cudaDeviceSynchronize(), "mapped_commit");
            unsigned sum = 0;
            check(cudaMemcpy(&sum, cache, sizeof(sum), cudaMemcpyDeviceToHost), "mapped_result");
            if (sum != 42) throw std::runtime_error("mapped_value");
        }
        status = 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "probe failed at %s\n", e.what());
    }
    for (void* p:registrations) (void) cudaHostUnregister(p);
    if (allocated && arena) (void) cudaFreeHost(arena);
    // Tear down all GPU references before returning the owned host pages.
    (void) cudaDeviceReset();
    if (locked) strata::platform::unlock_resident(arena, locked);
    if (!allocated) std::free(arena);
    return status;
}
