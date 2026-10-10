// CPU-only comparison of equal FP8 row reads; no model forward or GPU access.
#include "strata/ngram/ple_reader.hpp"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace {
using Json=nlohmann::json;
int run(const std::filesystem::path& spec,size_t count) {
    try {
        if(count<1 || count>131072) throw std::runtime_error("rows must be 1..131072");
        std::ifstream file(spec);const auto j=Json::parse(file);
        std::vector<strata::ngram::PleReader::Segment> native;
        const std::string path=j.at("native").at(0).at("file");
        for(const auto& s:j.at("native")) {
            if(s.at("file")!=path) throw std::runtime_error("benchmark expects one native shard");
            native.push_back({s.at("first_row"),s.at("rows"),s.at("file_offset")});
        }
        const uint64_t total=j.at("rows");
        std::mt19937 rng(42);std::vector<uint32_t> rows(count);
        for(auto& r:rows) r=uint32_t(rng()%total);
        std::vector<uint8_t> reference(count*160),current(count*160);
        for(int round=0;round<3;++round) for(int order=0;order<2;++order) {
            const bool sidecar=(round%2==0?order:1-order)==1;
            strata::ngram::PleReader reader;std::string err;
            if(!reader.open(sidecar?j.at("sidecar").get<std::string>():path,
                            sidecar?j.at("sidecar_offset").get<uint64_t>():0,total,256,0,err,true,160,
                            sidecar?std::vector<strata::ngram::PleReader::Segment>{}:native) ||
               !reader.set_batch_readers(8,err)) throw std::runtime_error(err);
            const auto start=std::chrono::steady_clock::now();
            if(!reader.read_batch(rows.data(),rows.size(),current.data(),err)) throw std::runtime_error(err);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(round==0 && order==0) reference=current;
            else if(reference!=current) throw std::runtime_error("ngram row bytes differ");
            const auto stats=reader.snapshot();
            Json out={{"round",round},{"source",sidecar?"G_sidecar":"D_original"},{"rows",count},
                      {"ms",ms},{"reads",stats.reads},{"bytes",stats.bytes},
                      {"p50_us",stats.percentile(0.5)},{"p99_us",stats.percentile(0.99)},
                      {"equal",true},{"cache_rows",0},{"inflight",256},{"readers",8},{"GPU",false}};
            std::puts(out.dump().c_str());std::fflush(stdout);
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) { return argc>=2?run(argv[1],argc>=3?std::stoull(argv[2]):131072):2; }
#else
int main(int argc,char** argv) { return argc>=2?run(argv[1],argc>=3?std::stoull(argv[2]):131072):2; }
#endif
