// CPU-only, bounded startup sample. Never allocates a GPU context or a full
// expert arena; never writes into the model directory. Warm samples are not a
// measurement of cold full-engine startup.
#include "strata/weights/qwen4.hpp"
#include "json_checked.hpp"
#include <chrono>
#include <cstring>
#include <iostream>

namespace w = strata::weights;
using w::detail::require;
using w::detail::Json;
int run(const std::vector<std::string>& args) {
    try {
        require(args.size() == 5, "usage: safetensors_batch_bench MODEL LAYER COUNT WORKERS");
        auto number = [](const std::string& text) {
            require(!text.empty() && text.find_first_not_of("0123456789") == text.npos, "expected unsigned decimal");
            return std::stoul(text);
        };
        const auto layer=number(args[2]),count=number(args[3]),workers=number(args[4]);
        require(layer<48 && count>=1 && count<=128 && workers>=1 && workers<=16,"sample outside bounded range");
        w::SafetensorsSource source(w::detail::utf8_path(args[1]));
        w::Qwen4WeightPlan plan(source);
        constexpr size_t bytes=2764816;
        std::vector<uint8_t> legacy(count*bytes),batch(count*bytes,0xcd);
        std::vector<std::array<float,3>> legacy_scales(count),batch_scales(count);
        std::vector<w::Nvfp4ExpertBuffer> targets;
        for (size_t i=0;i<count;++i) targets.push_back({&plan.expert(layer,i),
            {batch.data()+i*bytes,bytes},&batch_scales[i]});
        using Clock=std::chrono::steady_clock;
        w::Nvfp4LoadScratch scratch;
        auto run = [&](bool grouped) {
            const auto before=source.io_stats();
            const auto start=Clock::now();
            w::Nvfp4BatchStats stats;
            if (grouped) stats=w::load_nvfp4_expert_batch(source,targets,static_cast<unsigned>(workers),&scratch);
            else for (size_t i=0;i<count;++i) {
                auto expert=w::load_nvfp4_expert(source,plan.expert(layer,i),static_cast<int>(layer),false);
                std::memcpy(legacy.data()+i*bytes,expert.bytes.data(),bytes);
                legacy_scales[i]=expert.input_scales;
            }
            const auto after=source.io_stats();
            return Json{{"mode",grouped?"batch":"legacy"},
                {"ms",std::chrono::duration<double,std::milli>(Clock::now()-start).count()},
                {"source_read_calls",after.data_calls-before.data_calls},
                {"expert_source_bytes",after.data_bytes[1]-before.data_bytes[1]},
                {"ngram_source_bytes",after.data_bytes[2]-before.data_bytes[2]},
                {"scratch_bytes",stats.scratch_bytes},{"read_ms",stats.read_ms},{"pack_ms",stats.pack_ms}};
        };
        Json out{{"scope","CPU-only bounded warm expert-loading sample; not cold full-engine startup"},
                 {"layer",layer},{"experts",count},{"workers",workers},{"rounds",Json::array()}};
        out["warmup"]=Json::array({run(false),run(true)});
        require(legacy==batch && legacy_scales==batch_scales,"sample differs from legacy bytes/scales");
        for (int round=0;round<3;++round) {
            if (round%2) out["rounds"].push_back(Json::array({run(true),run(false)}));
            else out["rounds"].push_back(Json::array({run(false),run(true)}));
            require(legacy==batch && legacy_scales==batch_scales,"repeated sample differs from legacy bytes/scales");
        }
        source.seal_expert_reads();
        bool refused=false;
        try { w::load_nvfp4_expert_batch(source,targets,static_cast<unsigned>(workers)); }
        catch (const std::runtime_error& e) { refused=std::string(e.what()).find("sealed")!=std::string::npos; }
        require(refused,"batch bypassed expert read barrier");
        out["byte_exact"]=true;out["sealed_expert_reads"]=true;
        std::cout<<out.dump(2)<<'\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i=0;i<argc;++i) {
        const auto u8=std::filesystem::path(argv[i]).u8string();
        args.emplace_back(u8.begin(),u8.end());
    }
    return run(args);
}
#else
int main(int argc,char** argv) { return run({argv,argv+argc}); }
#endif
