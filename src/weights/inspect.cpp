#include "strata/weights/qwen4.hpp"
#include "json_checked.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace w = strata::weights;
using w::detail::Json;
using w::detail::require;
namespace {
Json io_json(const w::IoStats& s) {
    Json j{{"header_bytes",s.header_bytes},{"metadata_bytes",s.metadata_bytes},
           {"data_read_calls",s.data_calls},{"staging_peak_bytes",s.staging_peak_bytes}};
    for (size_t i = 0; i < 5; ++i) j["source_read_bytes"][w::family_name(static_cast<w::Family>(i))] = s.data_bytes[i];
    return j;
}
uint64_t number(const std::string& s) {
    require(!s.empty() && s.find_first_not_of("0123456789") == s.npos, "expected unsigned decimal argument");
    return std::stoull(s);
}
std::string hex_bytes(std::span<const uint8_t> bytes) {
    std::ostringstream hex;
    for (auto b : bytes) hex << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(b);
    return hex.str();
}
int run(const std::vector<std::string>& args) {
    try {
        require(args.size() >= 3, "usage: strata-safetensors headers|inspect MODEL | verify MODEL LAYER EXPERT [TEST_BLOB] | verify-set MODEL [TEST_DIRECTORY] | read MODEL TENSOR OFFSET BYTES");
        const auto start = std::chrono::steady_clock::now();
        const auto& mode = args[1];
        require(mode == "headers" || mode == "inspect" || mode == "verify" || mode == "verify-set" || mode == "read", "unknown command");
        w::SafetensorsSource source(w::detail::utf8_path(args[2]));
        const auto opened = std::chrono::steady_clock::now();
        Json out{{"schema_version",1},{"mode",mode},{"tensor_count",source.tensors().size()},
                 {"scope","weight source validation; not full model inference"}};
        for (const auto& s : source.shards()) {
            const auto u8 = s.path.filename().u8string();
            out["files"].push_back({{"name",std::string(u8.begin(),u8.end())},{"file_bytes",s.file_bytes},
                {"header_bytes",s.header_bytes},{"data_offset",s.data_offset},{"tensor_count",s.tensor_count}});
        }
        std::array<uint64_t,5> bytes{}, counts{};
        for (const auto& [name,t] : source.tensors()) {
            bytes[static_cast<size_t>(t.family)] += t.bytes;
            ++counts[static_cast<size_t>(t.family)];
        }
        for (size_t i = 0; i < 5; ++i) out["families"][w::family_name(static_cast<w::Family>(i))] =
            {{"tensor_count",counts[i]},{"bytes",bytes[i]}};
        if (mode == "inspect" || mode == "verify" || mode == "verify-set") {
            w::Qwen4WeightPlan plan(source);
            plan.validate_ngram_constants(source);
            out["model"] = {{"type","qwen4_exp"},{"layers",48},{"linear_layers",36},{"full_attention_layers",12},
                {"experts",plan.experts.size()},{"nvfp4_projections",plan.experts.size()*3},
                {"expert_arena_planned_bytes",plan.expert_arena_bytes},{"expert_arena_committed_bytes",0},
                {"input_scales_planned_bytes",plan.experts.size()*3*4},{"ngram_segments",plan.ngram.size()},
                {"ngram_rows",plan.ngram_rows},{"ngram_first_absolute_offset",plan.ngram[0].tensor->offset},
                {"vision_forward_supported",false},{"text_forward_connected",false}};
            if (mode == "verify" || mode == "verify-set") {
                std::vector<std::pair<uint64_t,uint64_t>> picks;
                if (mode == "verify") {
                    require(args.size() == 5 || args.size() == 6, "verify needs LAYER EXPERT [TEST_BLOB]");
                    picks.emplace_back(number(args[3]),number(args[4]));
                } else {
                    require(args.size() == 3 || args.size() == 4, "verify-set takes MODEL [TEST_OUTPUT_DIRECTORY]");
                    picks = {{0,0},{0,7},{12,100},{24,300},{40,5},{47,511}};
                }
                for (const auto& [layer,id] : picks) {
                const auto& descriptors = plan.expert(layer,id);
                auto expert = w::load_nvfp4_expert(source, descriptors, static_cast<int>(layer), true);
                out["verification"].push_back({{"layer",layer},{"expert",id},{"roundtrip_bytes",expert.roundtrip_bytes},
                    {"blob_bytes",expert.bytes.size()},{"input_scales",expert.input_scales},
                    {"byte_exact",true},{"physical_ram_locked",false}});
                if ((mode == "verify" && args.size() == 6) || (mode == "verify-set" && args.size() == 4)) {
                    // A small, optional test artifact. Never a required converted model.
                    auto path = mode == "verify" ? w::detail::utf8_path(args[5]) :
                        w::detail::utf8_path(args[3]) / ("expert-"+std::to_string(layer)+"-"+std::to_string(id)+".bin");
                    const auto dest = std::filesystem::weakly_canonical(path);
                    const auto root = std::filesystem::canonical(w::detail::utf8_path(args[2]));
                    auto rel = dest.lexically_relative(root);
                    require(dest.root_name() != root.root_name() || (!rel.empty() && *rel.begin() == ".."),
                            "test blob must be outside original model directory");
                    require(!std::filesystem::exists(dest), "refusing to overwrite test blob");
                    std::ofstream f(dest, std::ios::binary);
                    require(static_cast<bool>(f.write(reinterpret_cast<const char*>(expert.bytes.data()), expert.bytes.size())),
                            "cannot write test blob");
                }
                }
                source.seal_expert_reads();
                bool sealed = false;
                try { source.read(*plan.expert(0,0)[0].input_scale); }
                catch (const std::runtime_error& e) { sealed = std::string(e.what()).find("sealed") != std::string::npos; }
                require(sealed, "sealed expert source unexpectedly read data");
                out["expert_read_guard"] = sealed;
                // Include a row crossing a 4 KiB boundary, logical segment
                // transitions, a nonadjacent shard and the very last row.
                const uint64_t rows[] = {0, (4096-plan.ngram[0].tensor->offset%4096)/160,
                    2500011,2500012,2500013,2500012ull*10-1,2500012ull*10,plan.ngram_rows-1};
                std::array<std::array<uint8_t,160>,8> buffers{};
                std::vector<w::ReadRequest> reads;
                for (size_t i=0;i<8;++i) reads.push_back(plan.ngram_row(rows[i],buffers[i]));
                const auto expert_bytes_before=source.io_stats().data_bytes[static_cast<size_t>(w::Family::Expert)];
                source.read_many(reads);
                require(source.io_stats().data_bytes[static_cast<size_t>(w::Family::Expert)]==expert_bytes_before,
                        "ngram lookup read expert data");
                for (size_t i=0;i<8;++i) out["ngram_row_samples"].push_back({{"row",rows[i]},
                    {"tensor",reads[i].tensor->name},{"absolute_offset",reads[i].tensor->offset+reads[i].relative_offset},
                    {"hex",hex_bytes(buffers[i])}});
            } else require(args.size() == 3, "inspect takes only MODEL");
        } else if (mode == "read") {
            require(args.size() == 6, "read needs TENSOR OFFSET BYTES");
            const auto offset = number(args[4]), n = number(args[5]);
            require(n <= 4096, "diagnostic read limited to 4096 bytes");
            std::vector<uint8_t> buffer(static_cast<size_t>(n));
            const w::ReadRequest r{&source.tensor(args[3]),offset,buffer};
            source.read_many({&r,1});
            out["hex"] = hex_bytes(buffer);
        } else require(args.size() == 3, "headers takes only MODEL");
        out["io"] = io_json(source.io_stats());
        out["container_seconds"] = std::chrono::duration<double>(opened-start).count();
        out["elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        out["ok"] = true;
        std::cout << out.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        // Parser errors may quote invalid UTF-8 bytes; reporting the error must
        // not itself throw while serializing those bytes.
        std::cerr << Json{{"ok",false},{"error",e.what()}}.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
        return 1;
    }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i=0;i<argc;++i) {
        const auto u8 = std::filesystem::path(argv[i]).u8string();
        args.emplace_back(u8.begin(),u8.end());
    }
    return run(args);
}
#else
int main(int argc, char** argv) { return run({argv,argv+argc}); }
#endif
