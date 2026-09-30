#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace fs = std::filesystem;
using Pair = std::pair<int32_t, int32_t>;
using namespace strata::core;
using namespace strata::kernels::cpu;

static void check(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
static std::vector<uint8_t> pattern(int l, int e, size_t n) {
    std::vector<uint8_t> b(n);
    for (size_t i = 0; i < n; ++i) b[i] = (uint8_t) (i * 37 + (i >> 9) + l * 71 + e * 19);
    return b;
}
static void fixture(const fs::path& dir, bool native) {
    fs::create_directories(dir);
    std::string err;
    if (!native) {
        std::ofstream f(dir / "experts.bin", std::ios::binary);
        for (int l = 0; l < 3; ++l) for (int e = 0; e < 4; ++e) {
            auto b = pattern(l, e, BLOB);
            f.write((const char*) b.data(), b.size());
        }
        return;
    }
    std::ofstream meta(dir / "native_experts.txt");
    meta << "# strata native experts v3 (n_expert 4)\n";
    uint64_t dense = 0;
    for (int l = 0; l < 3; ++l) {
        NativeFmt fm;
        const int gt = l == 0 ? 18 : 22, dt = 20;
        check(native_fmt(gt, dt, H, FF, fm, err), err);
        const std::string name = "shard" + std::to_string(l) + ".gguf";
        const uint64_t per[] = {fm.up_off, fm.up_off, fm.bytes - fm.down_off};
        const uint64_t inside[] = {0, fm.up_off, fm.down_off};
        const uint64_t off[] = {64, 64 + 4 * per[0], 64 + 4 * (per[0] + per[1])};
        meta << l << ' ' << gt << ' ' << dt << ' ' << dense << ' ' << fm.bytes;
        for (auto o : off) meta << ' ' << o;
        meta << ' ' << name << '\n';
        std::ofstream f(dir / name, std::ios::binary);
        for (int r = 0; r < 3; ++r) for (int e = 0; e < 4; ++e) {
            auto b = pattern(l, e, fm.bytes);
            f.seekp(off[r] + e * per[r]);
            f.write((const char*) b.data() + inside[r], per[r]);
        }
        dense += 4 * fm.bytes;
    }
}
static void run(const fs::path& dir, bool native) {
    fixture(dir, native);
    std::string err;
    check(expert_layout_load(dir.string(), 3, 4, err), err);
    const auto& lay = expert_layout();
    std::vector<Pair> all;
    for (int l = 0; l < 3; ++l) for (int e = 0; e < 4; ++e) all.emplace_back(l, e);
    const std::vector<std::vector<Pair>> cases = {
        {}, {{0,0}}, {{0,1},{1,0},{1,2},{2,3}},
        {{0,0},{0,1},{0,2},{0,3},{2,1}}, {{1,1},{1,1}}, all, {}
    };
    ArenaExpertSource src;
    if (native) src.set_gguf((dir / "shard0.gguf").string());
    for (const auto& skip : cases) {
        src.set_skip(&skip);
        check(src.open(dir.string(), 3, 4, 2, err), err);
        uint64_t expected_bytes = 0;
        for (int l = 0; l < 3; ++l) {
            bool any = false;
            for (int e = 0; e < 4; ++e) {
                const bool excluded = std::find(skip.begin(), skip.end(), Pair{l,e}) != skip.end();
                const auto expected = pattern(l, e, lay.blob_bytes(l));
                const uint8_t* b = src.blob(l, e);
                check(src.skipped(l, e) == excluded, "wrong skipped membership");
                check((b == nullptr) == excluded, "RAM must contain exactly the complement of skip");
                std::vector<uint8_t> direct(expected.size());
                check(src.read_expert_blob(l, e, direct.data(), err), err);
                check(direct == expected, "direct file read byte mismatch");
                if (excluded) {
                    check(!src.pinned(l,e) && !src.device_alias(l,e), "excluded expert has host mapping");
                } else {
                    expected_bytes += expected.size();
                    any = true;
                    check(std::equal(expected.begin(), expected.end(), b), "compact arena byte mismatch");
                    check(src.pinned(l,e), "small fixture unexpectedly unpinned");
                    const auto* d = src.device_alias(l,e);
                    check(d != nullptr, "retained expert missing CUDA alias");
                    check(cudaMemcpy(direct.data(), d, direct.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                          "mapped alias copy failed");
                    check(direct == expected, "CUDA alias points to wrong expert");
                }
            }
            check((src.device_alias_layer(l) != nullptr) == any, "empty/nonempty layer alias mismatch");
        }
        check(src.resident_bytes() == expected_bytes, "incorrect compact allocation size");
        src.close();
        check(src.resident_bytes() == 0, "close did not reset byte count");
    }
    std::vector<Pair> bad{{3,0}};
    src.set_skip(&bad);
    check(!src.open(dir.string(), 3, 4, 2, err), "invalid exclusion was accepted");
    if (native) {
        // A short GGUF must not produce a partially initialized successful arena.
        fs::resize_file(dir / "shard1.gguf", 70);
        std::vector<Pair> skip{{0,0}};
        src.set_skip(&skip);
        check(!src.open(dir.string(), 3, 4, 2, err), "truncated GGUF accepted");
    }
    std::cout << (native ? "native GGUF" : "canonical experts.bin") << ": all ownership/bytes/alias cases PASS\n";
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: strata-tier-test NEW_FIXTURE_DIRECTORY");
        const fs::path root(argv[1]);
        check(!fs::exists(root), "fixture directory must be new");
        check(cudaSetDevice(0) == cudaSuccess, "CUDA device unavailable");
        run(root / "canonical", false);
        run(root / "native", true);
        ExpertCache cache;
        std::string err;
        check(cache.open_sized({1024,2048}, 1, 2, err), err);
        cache.pin_prefix(1);
        check(cache.pinned_prefix() == 1, "pin prefix missing");
        cache.close();
        check(cache.pinned_prefix() == 0, "cache close retained pin prefix");
        std::cout << "cache lifetime: PASS\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
