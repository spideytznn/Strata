#include "strata/core/weight_stage.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/complement_exchange.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <fstream>

static void check(bool ok, const char* msg) {
    if (!ok) { std::cerr << "FAIL: " << msg << '\n'; std::exit(1); }
}
int main(int argc, char** argv) {
    using strata::core::WeightStage;
    for (int split = 2; split < 48; ++split) {
        WeightStage a{0,split}, b{split,48};
        for (int l=0; l<48; ++l) {
            const auto name="blk."+std::to_string(l)+".attn_q.weight";
            check(a.owns(name) != b.owns(name), "each layer belongs to exactly one stage");
        }
        for (const auto* name : {"output.weight", "token_embd.weight", "blk.bad.weight", "blk.99999999999999999999.w"})
            check(a.owns(name) && b.owns(name), "global/unknown tensors retained");
        check(a.owns("blk.1.ple_key.weight") && !b.owns("blk.1.ple_key.weight"), "PLE stays on first stage");
    }
    // One authoritative byte pattern per expert; varying layer sizes, repeated
    // eviction/re-admission, with transfers simulated by ordinary host buffers.
    constexpr size_t layers=3, experts=17;
    constexpr uint64_t absent=UINT64_MAX;
    std::vector<uint64_t> offsets(layers*experts,absent);
    std::vector<std::vector<uint8_t>> truth(layers*experts), gpu(layers*experts);
    std::vector<uint8_t> arena, scratch;
    for (size_t l=0;l<layers;++l) for(size_t e=0;e<experts;++e) {
        size_t i=l*experts+e, n=97+l*134;
        truth[i].resize(n);
        for(size_t k=0;k<n;++k) truth[i][k]=(uint8_t)(i*31+k*19);
        if(e%2) gpu[i]=truth[i];
        else { offsets[i]=arena.size(); arena.insert(arena.end(),truth[i].begin(),truth[i].end()); }
    }
    const auto arena_size=arena.size();
    std::mt19937 rng(417);
    std::string err;
    for(int step=0;step<10000;++step) {
        size_t l=rng()%layers, in=0, out=0;
        do { in=l*experts+rng()%experts; } while(offsets[in]==absent);
        do { out=l*experts+rng()%experts; } while(offsets[out]!=absent);
        auto transfer=[&](const uint8_t* src,uint8_t* dst,size_t n) {
            check(gpu[out]==truth[out],"victim bytes intact before download");
            std::copy(gpu[out].begin(),gpu[out].end(),dst);
            gpu[in].assign(src,src+n); gpu[out].clear();
            return true;
        };
        check(strata::core::detail::exchange_complement(offsets,arena.data(),arena.size(),in,out,truth[in].size(),scratch,transfer,err),"exchange succeeds");
        check(arena.size()==arena_size,"RAM capacity constant");
        for(size_t i=0;i<truth.size();++i) {
            if(offsets[i]==absent) check(gpu[i]==truth[i],"GPU retains exact original bytes");
            else {
                check(gpu[i].empty(),"no duplicate GPU/RAM expert");
                check(std::equal(truth[i].begin(),truth[i].end(),arena.data()+offsets[i]),"RAM retains exact original bytes");
            }
        }
    }
    auto old_offsets=offsets; auto old_arena=arena;
    size_t in=0,out=0;
    while(offsets[in]==absent) ++in;
    while(offsets[out]!=absent) ++out;
    auto fail=[](const uint8_t*,uint8_t*,size_t){return false;};
    check(!strata::core::detail::exchange_complement(offsets,arena.data(),arena.size(),in,out,truth[in].size(),scratch,fail,err),"transport failure rejected");
    check(offsets==old_offsets && arena==old_arena,"failure does not publish host metadata");
    int transfers=0;
    auto never=[&](const uint8_t*,uint8_t*,size_t){++transfers;return true;};
    for(auto bad : {in,offsets.size()})
        check(!strata::core::detail::exchange_complement(offsets,arena.data(),arena.size(),in,bad,truth[in].size(),scratch,never,err),"invalid owner/index rejected");
    check(!strata::core::detail::exchange_complement(offsets,arena.data(),arena.size(),in,out,UINT64_MAX,scratch,never,err),"bounds overflow rejected");
    check(!transfers,"invalid plans cannot call transport");
    if(argc>1) {
        std::string pack=argv[1];
        uint64_t full=0,left=0,right=0,global=0;
        std::ifstream idx(pack+"/index.txt"); check(bool(idx),"index opens");
        std::set<std::string> skip;
        std::string line;
        // This native pack's absent canonical matrices have zero bytes. Keep
        // all metadata: pool_bytes can be tested without reading dense.bin.
        while(std::getline(idx,line)) {
            if(line.empty() || line[0]=='#') continue;
            std::istringstream row(line);std::string name; uint64_t f,k,so,sn,d,bytes;
            row>>name>>f>>k>>so>>sn>>d>>bytes;
            if(!name.starts_with("blk.")) global+=(bytes+63)/64*64;
        }
        WeightStage a{0,24},b{24,48};
        check(strata::core::WeightTable::pool_bytes(pack,full,err,&skip),"full pool plan");
        check(strata::core::WeightTable::pool_bytes(pack,left,err,&skip,&a),"first stage pool plan");
        check(strata::core::WeightTable::pool_bytes(pack,right,err,&skip,&b),"second stage pool plan");
        check(left+right==full+global,"only global weights duplicated");
        check(left<full && right<full,"both stages save memory");
        uint64_t no_skip=0;
        check(strata::core::WeightTable::pool_bytes(pack,no_skip,err,nullptr,&a) && no_skip==left,"stage filter works without native skip set");
        std::cout<<"pool bytes full="<<full<<" stage0="<<left<<" stage1="<<right<<" globals="<<global<<'\n';
    }
    std::cout<<"PASS: 46 splits, 10000 byte-exact exchanges, ownership/bounds/transport failures\n";
}
