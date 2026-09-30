#pragma once
#include <charconv>
#include <cstdint>
#include <string_view>

namespace strata::core {
// Global tensors are retained. A stage may omit only a well-formed blk.N tensor
// outside its execution range; unknown names remain loaded conservatively.
struct WeightStage {
    int64_t begin = 0, end = 0;
    bool owns(std::string_view name) const {
        if (!name.starts_with("blk.")) return true;
        const auto dot = name.find('.', 4);
        if (dot == std::string_view::npos) return true;
        int64_t layer = -1;
        const auto parsed = std::from_chars(name.data() + 4, name.data() + dot, layer);
        if (parsed.ec != std::errc{} || parsed.ptr != name.data() + dot || layer < 0) return true;
        return begin <= layer && layer < end;
    }
};
}
