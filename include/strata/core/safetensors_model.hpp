#pragma once
#include "strata/weights/dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/ngram.hpp"

namespace strata::core {
class FileExpertSource;
// Lifetime: descriptors/source must outlive the forward and all expert workers.
// All startup transformations are the independently checked weights adapters.
class SafetensorsModel {
public:
    explicit SafetensorsModel(const std::string& path);
    ~SafetensorsModel();
    uint64_t dense_bytes() const;
    bool load_dense(WeightTable&, void* arena, uint64_t capacity, std::string& err);
    bool load_embed(NativeEmbed&, std::string& err);
    bool load_head(NativeHead&, std::string& err);
    bool load_experts(FileExpertSource&, std::string& err);
    bool open_ple(strata::kernels::PleTable&, const strata::kernels::PleIoOptions&, std::string& err);
    strata::kernels::PleConsts ple_constants();
    const weights::IoStats& io_stats() const { return source_.io_stats(); }
    uint64_t session_identity() const { return session_identity_; }
    weights::SafetensorsSource& source() { return source_; }
private:
    weights::SafetensorsSource source_;
    weights::Qwen4WeightPlan plan_;
    weights::Qwen4DensePlan dense_;
    void* scratch_ = nullptr;
    std::vector<std::array<float,3>> input_scales_;
    uint64_t session_identity_ = 0, sealed_expert_bytes_ = 0, sealed_mtp_bytes_ = 0;
};
}
