#pragma once
#include "strata/weights/nvfp4.hpp"

namespace strata::weights {
struct NgramSegment {
    const TensorDesc* tensor;
    uint64_t first_row, row_count;
};
struct Qwen4Config {
    uint64_t layers = 48, hidden = 2560, experts = 512, top_k = 10, ff = 640, vocab = 248320;
    std::vector<std::string> layer_types;
    static Qwen4Config load(SafetensorsSource&);
};
// P1 binding plan. No dense/GDN/HC transforms are silently claimed here: these
// descriptors bind only experts and segmented PLE storage for later forward use.
class Qwen4WeightPlan {
public:
    explicit Qwen4WeightPlan(SafetensorsSource&);
    Qwen4Config config;
    std::vector<std::array<Nvfp4Projection, 3>> experts;
    std::vector<NgramSegment> ngram;
    const TensorDesc *ngram_scale, *ngram_offsets, *ngram_vocab_sizes;
    uint64_t expert_arena_bytes = 0, ngram_rows = 0;
    const std::array<Nvfp4Projection, 3>& expert(size_t layer, size_t id) const;
    ReadRequest ngram_row(uint64_t row, std::span<uint8_t> destination) const;
    // Read checkpoint metadata; never regenerate offsets from a random seed.
    void validate_ngram_constants(WeightSource&) const;
};
} // namespace strata::weights
