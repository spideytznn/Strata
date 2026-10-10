#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace strata::core {

// Validate before a token ID can index the full checkpoint's GPU head.
inline bool decode_native_draft_vocab(std::span<const uint8_t> raw, int64_t vocabulary,
                                     std::vector<int32_t>& ids, std::string& err) {
    ids.clear();
    if (vocabulary <= 0 || vocabulary > std::numeric_limits<int32_t>::max() || raw.empty() ||
        raw.size() % sizeof(int32_t) != 0 || raw.size() / sizeof(int32_t) > (uint64_t) vocabulary) {
        err = "mtp: native draft vocabulary must be a nonempty int32 token-ID list, at most the full vocabulary";
        return false;
    }
    ids.resize(raw.size() / sizeof(int32_t));
    std::memcpy(ids.data(), raw.data(), raw.size());
    std::vector<bool> seen((size_t) vocabulary, false);
    for (int32_t id : ids) {
        if (id < 0 || id >= vocabulary || seen[(size_t) id]) {
            ids.clear();
            err = "mtp: native draft vocabulary has a duplicate or out-of-range token ID";
            return false;
        }
        seen[(size_t) id] = true;
    }
    return true;
}

}  // namespace strata::core
