// Real safetensors expert exports; compare only the existing FP32 loop layouts.
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace k=strata::kernels;
namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T>T* alloc(size_t n) { T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));return p; }
template<class T>void upload(T* p,const std::vector<T>& v) {ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
void run(const std::vector<uint8_t>& blobs,int groups,int nt,cudaStream_t s) {
    const auto L=k::native_expert_layout(40,40,2560,640);
    const int entries=groups*nt;
    uint8_t* db=alloc<uint8_t>(groups*L.bytes);
    for(int g=0;g<groups;++g)ck(cudaMemcpy(db+g*L.bytes,blobs.data()+(g%6)*L.bytes,L.bytes,cudaMemcpyHostToDevice));
    float *dx=alloc<float>(8*2560),*out=alloc<float>(entries*2560+16);
    uint8_t* scratch=alloc<uint8_t>(k::native_expert_scratch_bytes(entries,640));
    auto* ptr=alloc<unsigned long long>(groups);
    auto *starts=alloc<int32_t>(groups+1),*ng=alloc<int32_t>(1),*dst=alloc<int32_t>(entries),*tok=alloc<int32_t>(entries);
    std::mt19937 rng(9950);std::normal_distribution<float> normal;
    std::vector<float> x(8*2560);for(auto& v:x)v=normal(rng);upload(dx,x);
    std::vector<unsigned long long> hp(groups);std::vector<int32_t> hs(groups+1),hd(entries),ht(entries);
    for(int g=0;g<groups;++g) {hp[g]=reinterpret_cast<unsigned long long>(db+g*L.bytes);hs[g]=g*nt;}
    hs[groups]=entries;
    for(int e=0;e<entries;++e) {hd[e]=(e*17)%entries;ht[e]=(e+3*(e/nt))%8;}
    upload(ptr,hp);upload(starts,hs);upload(ng,std::vector<int32_t>{groups});upload(dst,hd);upload(tok,ht);
    auto launch=[&](bool reuse,int grid) {
        k::native_nvfp4_f32_set_reuse(reuse);
        k::native_expert_grouped(L,ptr,starts,ng,dst,tok,groups,entries,nullptr,scratch,out,s,grid,dx);
    };
    std::vector<float> reference(entries*2560+16),candidate(reference.size());
    auto collect=[&](bool reuse,int grid,std::vector<float>& result) {
        ck(cudaMemsetAsync(out,0xA5,result.size()*sizeof(float),s));
        launch(reuse,grid);
        ck(cudaMemcpyAsync(result.data(),out,result.size()*sizeof(float),cudaMemcpyDeviceToHost,s));
        ck(cudaStreamSynchronize(s));
    };
    collect(false,groups,reference);collect(true,groups,candidate);
    if(std::memcmp(reference.data(),candidate.data(),reference.size()*sizeof(float)))throw std::runtime_error("reuse changed output bits or canary");
    collect(true,1,candidate);
    if(std::memcmp(reference.data(),candidate.data(),reference.size()*sizeof(float)))throw std::runtime_error("strided groups changed output bits");
    upload(ng,std::vector<int32_t>{0});collect(true,groups,candidate);
    std::vector<uint8_t> canary(candidate.size()*sizeof(float),0xA5);
    if(std::memcmp(canary.data(),candidate.data(),canary.size()))throw std::runtime_error("empty group wrote output");
    // Uneven groups, including empty ones, and a deliberately overprovisioned
    // grid. Output entries not addressed remain canaries in both layouts.
    std::vector<int32_t> uneven(groups+1);int used=0;
    for(int g=0;g<groups;++g) {uneven[g]=used;used+=(g%3==0?0:std::max(1,nt-g%nt));}uneven[groups]=used;
    upload(starts,uneven);upload(ng,std::vector<int32_t>{groups});
    collect(false,groups+3,reference);collect(true,groups+3,candidate);
    if(std::memcmp(reference.data(),candidate.data(),reference.size()*sizeof(float)))throw std::runtime_error("uneven groups changed output bits");
    upload(starts,hs);
    cudaGraph_t graph[2]{};cudaGraphExec_t exec[2]{};
    for(int q=0;q<2;++q) {
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        for(int repeat=0;repeat<10;++repeat)launch(q!=0,groups);
        ck(cudaStreamEndCapture(s,&graph[q]));ck(cudaGraphInstantiate(&exec[q],graph[q],nullptr,nullptr,0));
        ck(cudaGraphLaunch(exec[q],s));ck(cudaStreamSynchronize(s));
    }
    cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    std::vector<double> timing[2];
    for(int round=0;round<5;++round)for(int step=0;step<2;++step) {
        const int q=(round+step)%2;
        ck(cudaEventRecord(begin,s));ck(cudaGraphLaunch(exec[q],s));ck(cudaEventRecord(end,s));ck(cudaEventSynchronize(end));
        float ms=0;ck(cudaEventElapsedTime(&ms,begin,end));timing[q].push_back(ms*1000/10);
    }
    for(auto& v:timing)std::sort(v.begin(),v.end());
    const char* tile_env=std::getenv("STRATA_NVFP4_F32_REUSE_TILE");
    const int tile=tile_env && std::strcmp(tile_env,"8")==0 ? 8 : tile_env && std::strcmp(tile_env,"4")==0 ? 4 : 3;
    std::printf("{\"groups\":%d,\"tokens_per_group\":%d,\"tile\":%d,\"rolled_us\":%.6f,\"reuse_us\":%.6f,\"bit_equal\":true,\"empty_uneven_strided\":true}\n",
                groups,nt,tile,timing[0][2],timing[1][2]);std::fflush(stdout);
    for(int q=0;q<2;++q) {ck(cudaGraphExecDestroy(exec[q]));ck(cudaGraphDestroy(graph[q]));}
    ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
    for(void* p:{(void*)db,(void*)dx,(void*)out,(void*)scratch,(void*)ptr,(void*)starts,(void*)ng,(void*)dst,(void*)tok})ck(cudaFree(p));
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("usage: nvfp4_gpu_reuse_bench checkpoint-export-directory (six expert-*.bin files)");
        const char* f32=std::getenv("STRATA_NVFP4_F32");const char* tc=std::getenv("STRATA_NVFP4_TC");
        if(!f32 || std::strcmp(f32,"1") || (tc && std::strcmp(tc,"0")))
            throw std::runtime_error("requires STRATA_NVFP4_F32=1 and STRATA_NVFP4_TC=0 (or unset)");
        const auto L=k::native_expert_layout(40,40,2560,640);
        std::vector<uint8_t> blobs(6*L.bytes);int index=0;
        for(const char* name:{"0-0","0-7","12-100","24-300","40-5","47-511"}) {
            const auto path=std::filesystem::path(argv[1])/(std::string("expert-")+name+".bin");
            if(std::filesystem::file_size(path)!=L.bytes)throw std::runtime_error("expert size mismatch");
            std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(blobs.data()+index++*L.bytes),L.bytes);
            if(!f)throw std::runtime_error("expert read failed");
        }
        cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
        for(int groups:{1,16,64})for(int nt=1;nt<=8;++nt)run(blobs,groups,nt,s);
        ck(cudaStreamDestroy(s));return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
