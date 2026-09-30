#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace strata::core::detail {
// Caller excludes all source readers for the transaction. transfer must save the
// device victim into scratch, upload the incoming RAM bytes, and finish BOTH
// transfers before returning true. A transport failure is fatal to this session;
// the host mapping is not published on failure (device rollback is not promised).
template<class Transfer>
bool exchange_complement(std::vector<uint64_t>& offsets, uint8_t* arena, uint64_t arena_bytes,
                         size_t incoming, size_t outgoing, uint64_t blob_bytes,
                         std::vector<uint8_t>& scratch, Transfer transfer, std::string& err) {
    constexpr auto absent = std::numeric_limits<uint64_t>::max();
    if (!arena || !blob_bytes || incoming >= offsets.size() || outgoing >= offsets.size() ||
        incoming == outgoing || offsets[incoming] == absent || offsets[outgoing] != absent ||
        offsets[incoming] > arena_bytes || blob_bytes > arena_bytes - offsets[incoming] ||
        blob_bytes > std::numeric_limits<size_t>::max()) {
        err = "resident expert exchange: invalid CPU/GPU ownership or slot bounds";
        return false;
    }
    const uint64_t slot = offsets[incoming];
    scratch.resize((size_t) blob_bytes);
    if (!transfer(arena + slot, scratch.data(), (size_t) blob_bytes)) return false;
    std::memcpy(arena + slot, scratch.data(), (size_t) blob_bytes);
    offsets[outgoing] = slot;
    offsets[incoming] = absent;
    return true;
}
}
