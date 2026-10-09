#include "strata/weights/qwen4.hpp"
#include "json_checked.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::weights {
using namespace detail;
Qwen4Config Qwen4Config::load(SafetensorsSource& source) {
    const auto j = parse(source.metadata_text("config.json"));
    require(j.at("model_type") == "qwen4_exp" &&
            j.at("architectures") == Json::array({"Qwen4ExpForConditionalGeneration"}), "unsupported model architecture");
    const auto& t = j.at("text_config");
    const auto expect = [&](const char* key, uint64_t value) {
        require(uint64(t.at(key)) == value, std::string("unsupported qwen4_exp config: ") + key);
    };
    expect("num_hidden_layers",48); expect("hidden_size",2560); expect("num_experts",512);
    expect("num_experts_per_tok",10); expect("moe_intermediate_size",640);
    expect("shared_expert_intermediate_size",640); expect("vocab_size",248320);
    expect("num_attention_heads",24); expect("num_key_value_heads",2); expect("head_dim",256);
    expect("hc_count",4); expect("hc_lowrank",320); expect("full_attention_interval",4);
    expect("linear_conv_kernel_dim",4); expect("linear_key_head_dim",128); expect("linear_value_head_dim",128);
    expect("linear_num_key_heads",16); expect("linear_num_value_heads",48);
    expect("indexer_budget",2048); expect("indexer_compress_ratio",4); expect("indexer_head_dim",128);
    expect("indexer_kv_heads",1); expect("indexer_n_heads",4);
    expect("split_ngram_parts",128); expect("ple_embed_dim",2560); expect("heads_per_ngram",8);
    expect("ngram_size",3); expect("ple_conv_kernel_size",4); expect("mtp_num_hidden_layers",1);
    require(t.at("mamba_ssm_dtype") == "float32" && t.at("dtype") == "bfloat16" &&
            t.at("hidden_act") == "silu" && t.at("norm_topk_prob") == true &&
            t.at("output_gate_type") == "sigmoid" && t.at("tie_word_embeddings") == false &&
            t.at("ple_layer_ids") == Json::array({2}) && t.at("rms_norm_eps") == 1e-6 &&
            t.at("partial_rotary_factor") == 0.25, "unsupported qwen4_exp precision/PLE configuration");
    const auto& rope = t.at("rope_parameters");
    require(rope.at("rope_theta") == 10000000 && rope.at("partial_rotary_factor") == 0.25 &&
            rope.at("rope_type") == "default", "unsupported RoPE configuration");
    Qwen4Config c;
    c.layer_types = t.at("layer_types").get<std::vector<std::string>>();
    require(c.layer_types.size() == 48, "invalid layer_types count");
    for (size_t i = 0; i < c.layer_types.size(); ++i)
        require(c.layer_types[i] == (i % 4 == 3 ? "full_attention" : "linear_attention"), "unsupported layer order");
    const auto quant = parse(source.metadata_text("hf_quant_config.json"));
    require(quant.at("producer").at("name") == "modelopt" &&
            quant.at("quantization").at("quant_algo") == "MIXED_PRECISION" &&
            uint64(quant.at("quantization").at("group_size")) == 16, "unsupported quantization recipe");
    return c;
}
Qwen4WeightPlan::Qwen4WeightPlan(SafetensorsSource& source) : config(Qwen4Config::load(source)) {
    experts.reserve(48 * 512);
    const char* hf[] = {"gate_proj", "up_proj", "down_proj"};
    const char* canonical[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    for (size_t l = 0; l < 48; ++l) for (size_t e = 0; e < 512; ++e) {
        std::array<Nvfp4Projection, 3> ex;
        for (size_t p = 0; p < 3; ++p) {
            ex[p] = nvfp4_projection(source, "model.language_model.layers." + std::to_string(l) +
                ".mlp.experts." + std::to_string(e) + "." + hf[p],
                "blk." + std::to_string(l) + "." + canonical[p], p == 2 ? 2560 : 640, p == 2 ? 640 : 2560);
        }
        experts.push_back(std::move(ex));
    }
    expert_arena_bytes = 48ull * 512 * (3ull * 2560 * 640 / 64 * 36 + 16);
    const std::string ple = "model.language_model.layers.1.ple.ple_embedding.";
    for (uint64_t i = 0; i < 128; ++i) {
        const auto& tensor = source.tensor(ple + "ngram_embedding.shard_" + std::to_string(i) + ".weight");
        require(tensor.dtype == DType::F8_E4M3 && tensor.physical_shape == std::vector<uint64_t>{2500012, 160},
                "unsupported ngram segment shape/dtype");
        ngram.push_back({&tensor, ngram_rows, tensor.physical_shape[0]});
        ngram_rows += tensor.physical_shape[0];
    }
    ngram_scale = &source.tensor(ple + "ngram_embedding.weight_scale");
    ngram_offsets = &source.tensor(ple + "ngram_heads_offsets");
    ngram_vocab_sizes = &source.tensor(ple + "ngram_heads_vocab_sizes");
    ngram_multipliers = &source.tensor(ple + "layer_multipliers");
    require(ngram_multipliers->dtype == DType::I64 && ngram_multipliers->physical_shape == std::vector<uint64_t>{3},
            "invalid ngram multiplier descriptor");
    require(ngram_scale->dtype == DType::BF16 && ngram_scale->physical_shape == std::vector<uint64_t>{1},
            "invalid ngram scale descriptor");
    for (const auto* t : {ngram_offsets, ngram_vocab_sizes})
        require(t->dtype == DType::I64 && t->physical_shape == std::vector<uint64_t>{16}, "invalid ngram head descriptor");
}
const std::array<Nvfp4Projection, 3>& Qwen4WeightPlan::expert(size_t layer, size_t id) const {
    require(layer < config.layers && id < config.experts, "expert id out of range");
    return experts.at(layer * config.experts + id);
}
ReadRequest Qwen4WeightPlan::ngram_row(uint64_t row, std::span<uint8_t> destination) const {
    require(row < ngram_rows && destination.size() == 160, "ngram row or destination out of range");
    auto it = std::upper_bound(ngram.begin(), ngram.end(), row,
                              [](uint64_t r, const NgramSegment& s) { return r < s.first_row; });
    --it;
    require(row - it->first_row < it->row_count, "ngram segment gap");
    return {it->tensor, (row - it->first_row) * 160, destination};
}
void Qwen4WeightPlan::validate_ngram_constants(WeightSource& source) const {
    const auto offsets = source.read(*ngram_offsets), sizes = source.read(*ngram_vocab_sizes), scale = source.read(*ngram_scale);
    int64_t expected = 0;
    for (size_t i = 0; i < 16; ++i) {
        int64_t offset, size;
        std::memcpy(&offset, offsets.data() + i * 8, 8); std::memcpy(&size, sizes.data() + i * 8, 8);
        require(offset == expected && size > 0 && static_cast<uint64_t>(size) <= ngram_rows - expected,
                "ngram checkpoint offsets/vocab sizes inconsistent");
        expected += size;
    }
    require(static_cast<uint64_t>(expected) <= ngram_rows, "ngram head range exceeds table");
    uint16_t bits; std::memcpy(&bits, scale.data(), 2);
    const uint32_t fp32 = static_cast<uint32_t>(bits) << 16;
    float value; std::memcpy(&value, &fp32, 4);
    require(std::isfinite(value) && value > 0, "invalid ngram BF16 scale");
}
} // namespace strata::weights
