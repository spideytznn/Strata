
#include "strata/core/resident_swap.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <algorithm>
using namespace strata::core;
using namespace strata::kernels::cpu;
static void check(bool ok,const std::string& why) {if(!ok)throw std::runtime_error(why);}
static void cu(cudaError_t e){check(e==cudaSuccess,cudaGetErrorString(e));}
struct Swap {int32_t layer,in,out;};
int main(){
 namespace fs=std::filesystem;
 const auto dir=fs::temp_directory_path()/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
 fs::create_directory(dir);
 struct Cleanup{fs::path p;~Cleanup(){std::error_code ec;fs::remove_all(p,ec);}}cleanup{dir};
 try{
  int n=0;cu(cudaGetDeviceCount(&n));if(n<2)return 77;
  std::string err;check(expert_layout_load(dir.string(),2,3,err),err);
  std::vector<std::vector<uint8_t>> truth(6,std::vector<uint8_t>(BLOB));
  std::ofstream file(dir/"experts.bin",std::ios::binary);
  for(int i=0;i<6;++i){for(size_t j=0;j<BLOB;++j)truth[i][j]=uint8_t(i*79+j*37+j/257);file.write((const char*)truth[i].data(),BLOB);}file.close();
  for(bool pin:{false,true}){
   ExpertCache caches[2];cudaStream_t streams[2]{};
   for(int dev=0;dev<2;++dev){cu(cudaSetDevice(dev));check(caches[dev].open(1,2,3,BLOB,err),err);
    const int slot=caches[dev].admit(dev,0);check(slot==0,"slot");check(caches[dev].fill_slot_blocking(slot,truth[dev*3].data(),err),err);
    cu(cudaStreamCreateWithFlags(&streams[dev],cudaStreamNonBlocking));}
   cu(cudaSetDevice(0));FileExpertSource source;check(source.open(dir.string(),2,3,err),err);
   check(source.pin_cache_complement(caches[0],err,pin,{{1,0}},-1,1ull<<30),err);
   check(source.resident_bytes()==4ull*BLOB,"both GPU residents excluded from RAM");
   check(source.reserve_exchanges(2,err),err);
   std::vector<int32_t> host_res(6,kNotResident);host_res[0]=host_res[3]=0;
   std::vector<uint8_t> got(BLOB);int victim[2]={0,0};
   for(int turn=0;turn<128;++turn){
    std::vector<Swap> swaps;
    for(int l=0;l<2;++l)swaps.push_back({l,(victim[l]+1)%3,victim[l]});
    check(resident_stage_swaps(source,host_res,3,swaps,[&](int64_t l){return ResidentSwapTarget{&caches[l],int(l),streams[l]};}),"stage victim on owning GPU");
    check(swaps.size()==2,"both swaps retained");
    // The CPU must already be able to read both evicted experts while H2D is in flight.
    for(auto s:swaps){auto p=source.blob(s.layer,s.out);check(p&&std::equal(truth[s.layer*3+s.out].begin(),truth[s.layer*3+s.out].end(),p),"in-flight victim bytes");
     cu(cudaSetDevice(s.layer));cu(cudaMemcpyAsync(caches[s.layer].device_slot(0),source.blob(s.layer,s.in),BLOB,cudaMemcpyHostToDevice,streams[s.layer]));}
    for(int dev=0;dev<2;++dev){cu(cudaSetDevice(dev));cu(cudaStreamSynchronize(streams[dev]));}
    check(source.commit_exchanges()==2,"commit both RAM ownership transfers");
    for(auto s:swaps){cu(cudaSetDevice(s.layer));cu(cudaMemcpy(got.data(),caches[s.layer].device_slot(0),BLOB,cudaMemcpyDeviceToHost));
     check(got==truth[s.layer*3+s.in],"GPU incoming bytes exact");
     auto p=source.blob(s.layer,s.out);check(p&&std::equal(truth[s.layer*3+s.out].begin(),truth[s.layer*3+s.out].end(),p),"committed RAM bytes exact");
     check(source.has_resident(s.layer,s.out)&&!source.has_resident(s.layer,s.in),"exclusive RAM/GPU ownership");
     host_res[s.layer*3+s.out]=kNotResident;host_res[s.layer*3+s.in]=0;victim[s.layer]=s.in;}
   }
   check(source.exchanges()==256,"exchange count");check(source.file_reads()==0,"no mmap fallback after allocation");
   for(int dev=0;dev<2;++dev){cu(cudaSetDevice(dev));cu(cudaStreamDestroy(streams[dev]));caches[dev].close();}
   cu(cudaSetDevice(0));source.close();
   std::cout<<"PASS dual GPU resident swaps pinned="<<pin<<" exchanges=256 byte-exact, no file fallback\n";
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
