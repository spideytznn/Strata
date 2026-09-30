#include "strata/core/expert_source.hpp"
#include "strata/core/device.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>

static void check(bool b, const std::string& why) { if (!b) throw std::runtime_error(why); }
static void cu(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
int main() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("strata-resident-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove_all(p, ec); } } cleanup{dir};
    try {
        std::string err;
        check(expert_layout_load(dir.string(), 1, 3, err), err);
        std::vector<std::vector<uint8_t>> truth(3, std::vector<uint8_t>(BLOB));
        std::ofstream file(dir / "experts.bin", std::ios::binary);
        for (int e = 0; e < 3; ++e) {
            for (size_t j = 0; j < truth[e].size(); ++j) truth[e][j] = uint8_t(e * 79 + j * 37 + j / 257);
            file.write((const char*)truth[e].data(), truth[e].size());
        }
        file.close();
        int devices = 0; cu(cudaGetDeviceCount(&devices)); check(devices > 0, "no GPU");
        for (int dev = 0; dev < devices; ++dev) for (bool pin : {false, true}) {
            cu(cudaSetDevice(dev));
            ExpertCache cache; check(cache.open(1, 1, 3, BLOB, err), err);
            const int slot = cache.admit(0, 0); check(slot >= 0, "admit");
            check(cache.fill_slot_blocking(slot, truth[0].data(), err), err);
            FileExpertSource source; check(source.open(dir.string(), 1, 3, err), err);
            // Production allocates the portable host arena on device 0, then
            // both stages read it. Exercise that cross-device mapping too.
            cu(cudaSetDevice(0));
            check(source.pin_cache_complement(cache, err, pin, {}, 1ull << 30), err);
            cu(cudaSetDevice(dev));
            check(source.resident_bytes() == 2ull * BLOB, "compact size");
            cudaStream_t stream; cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            uint8_t* alias_copy = nullptr; cu(cudaMalloc(&alias_copy, BLOB));
            std::vector<uint8_t> got(BLOB);
            int victim = 0;
            for (int turn = 0; turn < 128; ++turn) {
                const int incoming = (victim + 1) % 3;
                check(source.pinned(0, incoming) == pin, "incoming pinned status");
                check(!source.pinned(0, victim), "GPU expert has no pinned RAM duplicate");
                if (pin) {
                    auto alias = source.device_alias(0, incoming); check(alias != nullptr, "alias missing");
                    cu(cudaMemcpyAsync(alias_copy, alias, BLOB, cudaMemcpyDeviceToDevice, stream));
                    cu(cudaStreamSynchronize(stream));
                    cu(cudaMemcpy(got.data(), alias_copy, BLOB, cudaMemcpyDeviceToHost));
                    check(got == truth[incoming], "mapped alias bytes");
                }
                check(source.exchange_resident(0, incoming, victim, cache.device_slot(slot), stream, err), err);
                cu(cudaMemcpy(got.data(), cache.device_slot(slot), BLOB, cudaMemcpyDeviceToHost));
                check(got == truth[incoming], "GPU bytes changed after exchange");
                auto host = source.blob(0, victim);
                check(host && std::equal(truth[victim].begin(), truth[victim].end(), host), "victim RAM bytes");
                check(source.pinned(0, victim) == pin && !source.pinned(0, incoming), "ownership publication");
                victim = incoming;
            }
            check(!source.exchange_resident(0, victim, victim, cache.device_slot(slot), stream, err), "invalid exchange accepted");
            check(source.resident_exchanges() == 128, "exchange count");
            cu(cudaFree(alias_copy)); cu(cudaStreamDestroy(stream));
            source.close();
            check(!source.pinned(0, 0) && source.device_alias(0, 0) == nullptr, "close kept stale mapping");
            std::cout << "PASS device=" << dev << " pinned=" << pin << " exchanges=128 byte-exact\n";
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
