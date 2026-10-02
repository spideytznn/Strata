
#include "strata/core/weights.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <vector>
static void check(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static void cu(cudaError_t e){check(e==cudaSuccess,cudaGetErrorString(e));}
int main(){
 namespace fs=std::filesystem;using namespace strata::core;
 const auto dir=fs::temp_directory_path()/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());fs::create_directory(dir);
 struct Cleanup{fs::path p;~Cleanup(){std::error_code ec;fs::remove_all(p,ec);}}cleanup{dir};
 try{
  int devices=0;cu(cudaGetDeviceCount(&devices));if(devices<1)return 77;cu(cudaSetDevice(0));
  const std::vector<std::string> names={"output_hc_norm.weight","blk.0.foo.weight","blk.1.ple_value.weight","blk.23.foo.weight","blk.24.foo.weight","blk.47.foo.weight","blk.0.attn_q.weight","blk.24.attn_q.weight"};
  std::ofstream idx(dir/"index.txt"),data(dir/"dense.bin",std::ios::binary);idx<<"# align 64 pool 512 tensors 8\n";
  for(size_t i=0;i<names.size();++i){uint32_t v[4]={uint32_t(i),0x3f800000u,0xc0800000u,0x7f123456u};data.write((char*)v,sizeof v);idx<<names[i]<<" 0 2 "<<i*16<<" 16 "<<i*64<<" 16 4 0 0 0 1 0 0 0 0 0 0 0\n";}idx.close();data.close();
  std::string err;void* full=nullptr;cu(cudaMalloc(&full,512));WeightTable reference;check(reference.load(dir.string(),full,512,err),err);
  std::set<std::string> native={names[6],names[7]};
  for(int split=1;split<48;++split)for(const auto range:{WeightStage{0,split},WeightStage{split,48}}){
   uint64_t n=0;check(WeightTable::pool_bytes(dir.string(),n,err,&native,&range),err);
   void* compact=nullptr;cu(cudaMalloc(&compact,n+32));cu(cudaMemset(compact,0xa5,n+32));WeightTable table;check(table.load(dir.string(),compact,n,err,&native,&range),err);
   for(const auto& name:names){auto*w=table.find(name);auto*ref=reference.find(name);check(w&&ref,"all metadata retained");const bool own=range.owns(name),uploaded=own&&!native.count(name);
    check(w->stage_resident==own&&w->resident==uploaded,"native skip distinguished from unowned layer");check(w->ne0==ref->ne0&&w->src_off==ref->src_off&&w->bytes==ref->bytes,"metadata exact");
    if(uploaded){std::vector<uint8_t>a(w->bytes),b(ref->bytes);cu(cudaMemcpy(a.data(),w->data,a.size(),cudaMemcpyDeviceToHost));cu(cudaMemcpy(b.data(),ref->data,b.size(),cudaMemcpyDeviceToHost));check(a==b,"uploaded bytes exact");}else check(!w->data,"unowned pointer null");}
   uint8_t canary[32];cu(cudaMemcpy(canary,(uint8_t*)compact+n,32,cudaMemcpyDeviceToHost));for(auto b:canary)check(b==0xa5,"arena overrun");cu(cudaFree(compact));
  }
  cu(cudaFree(full));std::cout<<"PASS stage loaders: 47 split points, byte-exact tensors, metadata and arena canaries\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
