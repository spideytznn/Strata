#include "strata/ngram/ple_reader.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

using strata::ngram::PleReader;
void check(bool b,const std::string& message) { if (!b) throw std::runtime_error(message); }
int main() {
    const auto path=std::filesystem::temp_directory_path()/("strata-segments-"+std::to_string(std::random_device{}())+".bin");
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove(p,ec); } } cleanup{path};
    try {
        std::vector<uint8_t> file(32768,0xA5);
        // Disordered physical locations, gaps, and a row crossing a 4 KiB page.
        std::vector<PleReader::Segment> segments{{0,17,16380},{17,23,4090},{40,19,24570}};
        for (const auto& s:segments) for (uint64_t r=0;r<s.rows;++r)
            for (uint64_t c=0;c<160;++c) file[s.file_offset+r*160+c]=static_cast<uint8_t>((s.first_row+r)*13+c*7);
        { std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<char*>(file.data()),file.size()); }
        const auto u8=path.u8string(); const std::string name(u8.begin(),u8.end());
        for (bool threaded : {false,true}) for (unsigned batch : {0u,3u}) {
            PleReader reader; std::string err;
            check(reader.open(name,0,59,8,256,err,threaded,160,segments),err);
            check(reader.set_batch_readers(batch,err),err);
            std::vector<uint32_t> rows{0,16,17,18,39,40,58,3,41,16,17};
            std::vector<uint8_t> out(rows.size()*160);
            check(reader.read_batch(rows.data(),rows.size(),out.data(),err),err);
            for (size_t i=0;i<rows.size();++i) for (size_t c=0;c<160;++c)
                check(out[i*160+c]==static_cast<uint8_t>(rows[i]*13+c*7),"segmented row differs");
            const auto reads=reader.snapshot().reads;
            check(reader.read_batch(rows.data(),rows.size(),out.data(),err),err);
            check(reader.snapshot().reads==reads,"repeated rows performed disk reads");
            reader.close();
            auto bad=segments; bad[1].first_row=18;
            check(!reader.open(name,0,59,8,0,err,true,160,bad),"accepted segment gap");
            bad=segments; bad.back().file_offset=32767;
            check(!reader.open(name,0,59,8,0,err,true,160,bad),"accepted segment past EOF");
        }
        std::cout << "segmented direct I/O, cross-page, batch/worker and row-cache reuse passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
