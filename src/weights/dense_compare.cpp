// Test-only comparison against a previously validated artifact. The loader and
// production engine do not depend on GGUF or write intermediate weight files.
#include "strata/weights/dense.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "json_checked.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>

namespace w = strata::weights;
using w::detail::Json;
using w::detail::require;
int run(const std::vector<std::string>& args) {
    try {
        require(args.size() == 4,"usage: safetensors_dense_compare MODEL MAIN_ORACLE.gguf DENSE_ORACLE.gguf");
        const auto start = std::chrono::steady_clock::now();
        w::SafetensorsSource source(w::detail::utf8_path(args[1]));
        w::Qwen4DensePlan plan(source);
        strata::GgufFile main(args[2]), dense(args[3]);
        Json report{{"scope","all dense weights, no forward or throughput claim"},{"ok",true}};
        uint64_t elements = 0, bytes = 0, exact = 0, exp_ulp = 0;
        for (const auto& b : plan.tensors) {
            const auto* t = dense.find(b.name);
            const auto* file = &dense;
            if (!t) { t = main.find(b.name); file = &main; }
            require(t && (t->type == 0 || t->type == 30),"missing unquantized oracle: "+b.name);
            require(t->elements() == b.rows*b.columns,"oracle element mismatch: "+b.name);
            const auto* reference = file->tensor_data(*t);
            uint64_t different = 0; uint32_t max_ulp = 0;
            w::stream_dense(source,b,[&](uint64_t offset,std::span<const uint8_t> data) {
                const auto width = w::dtype_bytes(b.output);
                for (size_t i = 0; i < data.size()/width; ++i) {
                    const auto idx = offset/width+i;
                    uint32_t actual = 0, expected = 0;
                    if (width == 2) { uint16_t h; std::memcpy(&h,data.data()+i*2,2); actual=uint32_t(h)<<16; }
                    else std::memcpy(&actual,data.data()+i*4,4);
                    if (t->type == 30) { uint16_t h; std::memcpy(&h,reference+idx*2,2); expected=uint32_t(h)<<16; }
                    else std::memcpy(&expected,reference+idx*4,4);
                    if (actual != expected) {
                        ++different;
                        const auto ulp = actual > expected ? actual-expected : expected-actual;
                        max_ulp=std::max(max_ulp,ulp);
                        require(b.math == w::DenseMath::NegativeExp && ulp <= 2,
                            "dense mismatch: "+b.name+" element "+std::to_string(idx)+
                            " actual="+std::to_string(std::bit_cast<float>(actual))+
                            " expected="+std::to_string(std::bit_cast<float>(expected)));
                    }
                }
            });
            elements += b.rows*b.columns; bytes += b.bytes(); exact += b.rows*b.columns-different;
            exp_ulp += different;
            report["tensors"].push_back({{"name",b.name},{"elements",b.rows*b.columns},
                {"output_dtype",w::dtype_name(b.output)},{"exact_elements",b.rows*b.columns-different},
                {"exp_differing_elements",different},{"max_ulp",max_ulp}});
            std::cerr << b.name << " OK\n";
        }
        report["tensor_count"]=plan.tensors.size(); report["elements"]=elements;
        report["output_bytes"]=bytes; report["exact_elements"]=exact;
        report["exp_rounding_differences"]=exp_ulp;
        report["expert_bytes_read"]=source.io_stats().data_bytes[static_cast<size_t>(w::Family::Expert)];
        report["ngram_bytes_read"]=source.io_stats().data_bytes[static_cast<size_t>(w::Family::Ngram)];
        report["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout << report.dump(2) << '\n'; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
    std::vector<std::string> args;
    for (int i=0;i<argc;++i) { const auto u8=std::filesystem::path(argv[i]).u8string(); args.emplace_back(u8.begin(),u8.end()); }
    return run(args);
}
#else
int main(int argc,char** argv) { return run({argv,argv+argc}); }
#endif
