// Real checkpoint expert validation and CUDA-graph microbenchmark for SM120 FP4.
#include "strata/kernels/nvfp4_tensor.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
namespace k=strata::kernels;
void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(1);} }
template<class T> T* alloc(size_t n){T* p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> void upload(T* p,const std::vector<T>& v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> std::vector<T> download(T* p,size_t n){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
size_t align256(size_t n){return (n+255)&~size_t(255);}
float fp8(unsigned c){return c<8 ? std::ldexp(float(c),-9) : std::ldexp(1.0f+float(c&7)/8,int(c>>3)-7);}
float fp4(unsigned c){static const float v[8]={0,.5,1,1.5,2,3,4,6};return (c&8?-1:1)*v[c&7];}
unsigned nearest(float x,bool four){unsigned best=0;float d=1e30f;for(unsigned i=0;i<(four?8:127);++i){float e=std::fabs(x-(four?fp4(i):fp8(i)));if(e<d || (e==d && !(i&1))){best=i;d=e;}}return best;}
std::vector<float> quant(const std::vector<float>& x,int cols,int terms,float gs){
    std::vector<float> out(x.size(),0);
    const float checkpoint_scale=gs;
    for(size_t i=0;i<x.size();i+=16){
        if(i%cols==0) {gs=checkpoint_scale;if(terms>1){float mx=0;for(int c=0;c<cols;++c)mx=std::max(mx,std::fabs(x[i+c]));while(gs*2688.0f<mx)gs*=2;}}
        float v[16];for(int j=0;j<16;++j)v[j]=x[i+j]/gs;
        for(int t=0;t<terms;++t){float mx=0;for(float a:v)mx=std::max(mx,std::fabs(a));float s=fp8(nearest(mx/6,false)),inv=s>0?1/s:0;
            for(int j=0;j<16;++j){float a=fp4(nearest(std::fabs(v[j]*inv),true)|(std::signbit(v[j])?8:0))*s;out[i+j]+=a*gs;v[j]-=a;}
        }
    }return out;
}
struct QB{unsigned char s[4];unsigned q[8];};
std::vector<float> unquant(const std::vector<QB>& q,int count,int cols,int terms,const std::vector<float>& gs){
    std::vector<float> out(size_t(count)*cols,0);
    for(int e=0;e<count;++e)for(int t=0;t<terms;++t)for(int c=0;c<cols;++c){const QB& b=q[(size_t(e)*terms+t)*(cols/64)+c/64];int j=c%64;out[size_t(e)*cols+c]+=fp4((b.q[j/8]>>(4*(j%8)))&15)*fp8(b.s[j/16])*gs[e];}
    return out;
}
double rel(const std::vector<float>& a,const std::vector<float>& b){double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::pow(double(a[i])-b[i],2);d+=double(b[i])*b[i];}return std::sqrt(n/std::max(d,1e-30));}
std::vector<float> linear(const std::vector<float>& w,const std::vector<float>& x,int rows,int cols,float scale){
    std::vector<float> out(x.size()/cols*rows);for(size_t t=0;t<x.size()/cols;++t)for(int r=0;r<rows;++r){double a=0;for(int c=0;c<cols;++c)a+=double(w[size_t(r)*cols+c])*x[t*cols+c];out[t*rows+r]=float(a*scale);}return out;
}
std::vector<float> hidden(const std::vector<float>& gu,int ff){std::vector<float> h(gu.size()/2);for(size_t t=0;t<gu.size()/(2*ff);++t)for(int r=0;r<ff;++r){float g=gu[t*2*ff+r];h[t*ff+r]=g/(1+std::exp(-g))*gu[t*2*ff+ff+r];}return h;}
int main(int argc,char** argv){
    if(argc<3){std::puts("usage: nvfp4_tensor_test experts.bin input-scales.bin [layer expert]");return 2;}
    constexpr int N=2560,FF=640,MAX=80;
    int layer=argc>3?std::atoi(argv[3]):0,expert=argc>4?std::atoi(argv[4]):0;
    k::NativeExpertLayout L=k::native_expert_layout(40,40,N,FF);L.layer=layer;
    std::ifstream sf(argv[2],std::ios::binary);sf.seekg(layer*8);sf.read(reinterpret_cast<char*>(&L.input_scale_gu),4);sf.read(reinterpret_cast<char*>(&L.input_scale_down),4);if(!sf)return 2;
    std::vector<unsigned char> blob(L.bytes);std::ifstream f(argv[1],std::ios::binary);f.seekg((size_t(layer)*512+expert)*L.bytes);f.read(reinterpret_cast<char*>(blob.data()),blob.size());if(!f)return 2;
    float tail[4];std::memcpy(tail,blob.data()+L.tail_off,16);
    const auto* tt=ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> wg(2*FF*N),wd(N*FF);tt->to_float(blob.data(),wg.data(),2*FF*N);tt->to_float(blob.data()+L.down_off,wd.data(),N*FF);
    for(int r=0;r<2*FF;++r)for(int c=0;c<N;++c)wg[size_t(r)*N+c]*=tail[r>=FF];
    std::mt19937 rng(19);std::normal_distribution<float> nd(0,argc>5?float(std::atof(argv[5])):0.5f);std::vector<float> x(8*N);for(auto& a:x)a=nd(rng);
    auto* db=alloc<unsigned char>(blob.size());upload(db,blob);auto* dx=alloc<float>(x.size());upload(dx,x);
    auto* ptr=alloc<unsigned long long>(MAX);auto* starts=alloc<int32_t>(MAX+1);auto* ng=alloc<int32_t>(1);auto* dst=alloc<int32_t>(MAX);auto* tok=alloc<int32_t>(MAX);
    auto* scr=alloc<unsigned char>(MAX*20000);auto* out=alloc<float>(MAX*N);auto* xq=alloc<unsigned char>(8*N/32*36);
    upload(ptr,std::vector<unsigned long long>(MAX,reinterpret_cast<unsigned long long>(db)));
    upload(starts,std::vector<int32_t>{0,8});upload(ng,std::vector<int32_t>{1});upload(dst,std::vector<int32_t>{0,1,2,3,4,5,6,7});upload(tok,std::vector<int32_t>{0,1,2,3,4,5,6,7});
    _putenv_s("STRATA_NVFP4_TC","0");_putenv_s("STRATA_NVFP4_F32","1");
    k::quantize_q8_1_rows(dx,8,N,xq,nullptr);
    k::native_expert_grouped(L,ptr,starts,ng,dst,tok,1,8,xq,scr,out,nullptr,0,dx);ck(cudaDeviceSynchronize());auto baseline=download(out,8*N);
    bool ok=true;
    for(int terms:{1,2,3}){
        k::nvfp4_tensor_grouped(L,ptr,starts,ng,dst,tok,1,8,dx,scr,out,nullptr,terms);ck(cudaDeviceSynchronize());auto got=download(out,8*N);
        auto* scales=reinterpret_cast<float*>(scr+align256(8*terms*N/64*sizeof(QB))+align256(8*2*FF*4)+align256(8*terms*FF/64*sizeof(QB)));
        auto qx=unquant(download(reinterpret_cast<QB*>(scr),8*terms*N/64),8,N,terms,download(scales,8));
        size_t qbytes=align256(8*terms*N/64*sizeof(QB));auto gu=download(reinterpret_cast<float*>(scr+qbytes),8*2*FF);
        auto qh=unquant(download(reinterpret_cast<QB*>(scr+qbytes+align256(8*2*FF*4)),8*terms*FF/64),8,FF,terms,download(scales+8,8));
        auto cpqx=quant(x,N,terms,L.input_scale_gu),cpgu=linear(wg,qx,2*FF,N,1),cpqh=quant(hidden(gu,FF),FF,terms,L.input_scale_down),cpout=linear(wd,qh,N,FF,tail[2]);
        double a=rel(qx,cpqx),b=rel(gu,cpgu),c=rel(qh,cpqh),d=rel(got,cpout),quality=rel(got,baseline);
        std::printf("layer=%d expert=%d terms=%d quantX=%.8g GU=%.8g quantH=%.8g down=%.8g vsF32=%.6f%%\n",layer,expert,terms,a,b,c,d,100*quality);
        ok &= std::isfinite(a+b+c+d+quality) && a<2e-5 && b<2e-5 && c<.001 && d<2e-5;
        // Uneven groups, token/destination permutation, and capacity larger than actual count.
        std::vector<int32_t> ids={7,2,5,0,3,6,1,4};upload(starts,std::vector<int32_t>{0,1,4,8});upload(ng,std::vector<int32_t>{3});upload(dst,ids);upload(tok,ids);
        k::nvfp4_tensor_grouped(L,ptr,starts,ng,dst,tok,1,12,dx,scr,out,nullptr,terms);ck(cudaDeviceSynchronize());double perm=rel(download(out,8*N),got);std::printf("  uneven/permuted/strided rel=%.8g\n",perm);ok &= perm<2e-5;
        upload(ng,std::vector<int32_t>{0});ck(cudaMemset(out,0,8*N*4));k::nvfp4_tensor_grouped(L,ptr,starts,ng,dst,tok,5,12,dx,scr,out,nullptr,terms);ck(cudaDeviceSynchronize());for(float z:download(out,8*N))ok &= z==0;
        upload(starts,std::vector<int32_t>{0,8});upload(ng,std::vector<int32_t>{1});upload(dst,std::vector<int32_t>{0,1,2,3,4,5,6,7});upload(tok,std::vector<int32_t>{0,1,2,3,4,5,6,7});
    }
    if(!ok){std::puts("FAIL numerical validation");return 1;}
    // Distinct actual checkpoint weights: 221 MB at 80 experts, exceeding L2.
    std::vector<unsigned char> bench_blob(MAX*L.bytes);f.clear();f.seekg(size_t(layer)*512*L.bytes);f.read(reinterpret_cast<char*>(bench_blob.data()),bench_blob.size());if(!f)return 2;
    auto* bench_db=alloc<unsigned char>(bench_blob.size());upload(bench_db,bench_blob);
    std::vector<unsigned long long> bench_ptr(MAX);for(int i=0;i<MAX;++i)bench_ptr[i]=reinterpret_cast<unsigned long long>(bench_db+i*L.bytes);upload(ptr,bench_ptr);
    cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    for(int T:{1,2,4,8})for(bool shared:{true,false}){
        int groups=shared?10:10*T,entries=10*T;std::vector<int32_t> s(groups+1),ids(entries),ts(entries);
        for(int g=0;g<=groups;++g)s[g]=g*(shared?T:1);for(int e=0;e<entries;++e){ids[e]=e;ts[e]=e%T;}
        upload(starts,s);upload(ng,std::vector<int32_t>{groups});upload(dst,ids);upload(tok,ts);
        for(int terms:{0,1,2,3}){
            auto run=[&]{if(terms)k::nvfp4_tensor_grouped(L,ptr,starts,ng,dst,tok,groups,entries,dx,scr,out,stream,terms);else k::native_expert_grouped(L,ptr,starts,ng,dst,tok,groups,entries,xq,scr,out,stream,0,dx);};
            run();ck(cudaStreamSynchronize(stream));cudaGraph_t graph;cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
            for(int i=0;i<20;++i)ck(cudaGraphLaunch(exec,stream));ck(cudaEventRecord(begin,stream));for(int i=0;i<200;++i)ck(cudaGraphLaunch(exec,stream));ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));
            std::printf("BENCH tokens=%d groups=%d mode=%s us=%.3f\n",T,groups,terms==0?"F32":terms==1?"FP4":terms==2?"FP4x2":"FP4x3",ms*1000/200);ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
        }
    }
    std::puts("PASS (compute-only, distinct checkpoint weights; excludes PCIe and routing)");return 0;
}
