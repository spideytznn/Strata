#include "strata/core/native_draft_vocab.hpp"

#include <array>
#include <cstdio>

int main() {
    std::vector<int32_t> output;
    std::string error;
    auto check = [&](std::vector<int32_t> input, int64_t vocabulary, bool expected) {
        const auto bytes = std::span(reinterpret_cast<const uint8_t*>(input.data()), input.size() * sizeof(int32_t));
        const bool ok = strata::core::decode_native_draft_vocab(bytes, vocabulary, output, error);
        if (ok != expected || (ok && output != input) || (!ok && (error.empty() || !output.empty()))) return false;
        return true;
    };
    if (!check({248319, 0, 17}, 248320, true) || !check({0}, 1, true) ||
        !check({}, 248320, false) || !check({0, 0}, 248320, false) ||
        !check({-1}, 248320, false) || !check({248320}, 248320, false) ||
        !check({2147483647}, 248320, false) || !check({0}, 0, false) ||
        !check({0}, -1, false) || !check({0, 1}, 1, false)) return 1;
    const std::array<uint8_t, 3> truncated = {0, 0, 0};
    if (strata::core::decode_native_draft_vocab(truncated, 248320, output, error) || !output.empty()) return 1;
    std::puts("PASS: original ID order, vocabulary edges, missing/truncated IDs, duplicate and range rejection");
    return 0;
}
