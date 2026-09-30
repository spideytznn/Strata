#include "strata/core/weights.hpp"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
static void check(bool ok,const char* msg) { if(!ok){std::cerr<<msg<<'\n';std::exit(1);} }
int main(int argc,char** argv) {
    check(argc==2,"supply an empty fixture directory");
    const std::filesystem::path dir(argv[1]);
    check(std::filesystem::create_directory(dir),"fixture must be new");
    const std::vector<std::string> names={"output_hc_norm.weight","blk.0.foo.weight","blk.1.ple_value.weight",
        "blk.23.foo.weight","blk.24.foo.weight","blk.47.foo.weight","blk.0.attn_q.weight","blk.24.attn_q.weight"};
    std::ofstream idx(dir/"index.txt"),data(dir/"dense.bin",std::ios::binary);
    idx<<"# align 64 pool 512 tensors 8\n";
    for(size_t i=0;i<names.size();++i) {
        uint32_t values[4]={(uint32_t)i,0x3f800000u,0xc0800000u,0x7f123456u};
        data.write((char*)values,sizeof values);
        idx<<names[i]<<" 0 2 "<<i*16<<" 16 "<<i*64<<" 16 4 0 0 0 1 0 0 0 0 0 0 0\n";
    }
    idx.close();data.close();
    std::string err;
    std::vector<uint8_t> full(512);
    strata::core::WeightTable reference;
    check(reference.load(dir.string(),full.data(),full.size(),err),err.c_str());
    std::set<std::string> native={names[6],names[7]};
    for(const auto range : {strata::core::WeightStage{0,24},strata::core::WeightStage{24,48}}) {
        uint64_t n=0;
        check(strata::core::WeightTable::pool_bytes(dir.string(),n,err,&native,&range),err.c_str());
        std::vector<uint8_t> compact(n+32,0xa5);
        strata::core::WeightTable table;
        check(table.load(dir.string(),compact.data(),n,err,&native,&range),err.c_str());
        for(const auto& name:names) {
            const auto* w=table.find(name);const auto* ref=reference.find(name);
            check(w && ref,"all metadata retained");
            const bool own=range.owns(name),uploaded=own&&!native.count(name);
            check(w->stage_resident==own && w->resident==uploaded,"native-skip vs other-stage distinction");
            check(w->ne0==ref->ne0 && w->src_off==ref->src_off && w->bytes==ref->bytes,"shape/source metadata unchanged");
            if(uploaded) check(w->data && !std::memcmp(w->data,ref->data,w->bytes),"retained bytes exact");
            else check(w->data==nullptr,"skipped tensor has no allocation");
        }
        for(size_t i=n;i<compact.size();++i) check(compact[i]==0xa5,"no arena overrun");
    }
    std::cout<<"PASS: production loader stage filtering, native skip metadata, exact bytes and arena canaries\n";
}
