#include "strata/program/local_memory_plan.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
static void require(bool ok, const char* name) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", name); std::exit(1); }
}
int main() {
    using namespace strata::program;
    constexpr uint64_t GiB = 1ull << 30;
    const auto c = {PrefillBufferCandidate{8192, 4 * GiB}, {4096, 2 * GiB}, {2048, GiB}};
    require(independent_prefill_chunk(c, 6 * GiB, 2 * GiB, 8192, 4096) == 8192, "larger batch");
    require(independent_prefill_chunk(c, 6 * GiB - 1, 2 * GiB, 8192, 4096) == 4096, "reserve boundary");
    require(independent_prefill_chunk(c, 4 * GiB - 1, 2 * GiB, 8192, 4096) == 0, "do not shrink borrowed batch");
    require(independent_prefill_chunk(c, GiB, 2 * GiB, 8192, 0) == 0, "no unsigned underflow");
    require(independent_prefill_chunk(c, 10 * GiB, GiB, 4096, 0) == 4096, "requested cap");
    require(independent_prefill_chunk(c, 10 * GiB, GiB, 0, 0) == 0, "disabled prefill");
    require(conversation_expert_headroom(4 * GiB, 2 * GiB, 2560ull << 20, true) == (4864ull << 20), "budget and floor");
    require(conversation_expert_headroom(4 * GiB, GiB, 2560ull << 20, true) == 4 * GiB, "existing larger headroom");
    require(conversation_expert_headroom(4 * GiB, 2 * GiB, 2560ull << 20, false) == 4 * GiB, "disabled cache unchanged");
    require(conversation_expert_headroom(4 * GiB, 0, 2560ull << 20, true) == 4 * GiB, "zero budget unchanged");
    require(conversation_expert_headroom(4 * GiB, std::numeric_limits<uint64_t>::max(), 1, true) == std::numeric_limits<uint64_t>::max(), "overflow saturates");
    std::puts("local memory planner: 11 checks passed");
}
