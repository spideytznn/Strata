// Model-free parity and timings for the existing BF16 MMVF output-row layout.
#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__host__ __device__ uint16_t weight(size_t i) {
    unsigned h=unsigned(i)*1664525u+1013904223u;
    return uint16_t(0x3b00u | ((h>>12)&127u) | ((h&1u)<<15));
}
__global__ void fill(uint16_t* w,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) w[i]=weight(i);
}
float wide(uint16_t b) { unsigned u=unsigned(b)<<16;float v;std::memcpy(&v,&u,4);return v; }
void run(int k,int n,cudaStream_t s) {
    uint16_t* w=nullptr;float *x=nullptr,*y=nullptr;
    ck(cudaMalloc(&w,size_t(k)*n*2));ck(cudaMalloc(&x,k*4));ck(cudaMalloc(&y,(n+8)*4));
    fill<<<1024,256,0,s>>>(w,size_t(k)*n);
    std::vector<float> hx(k),a(n+8),b(n+8);
    for(int i=0;i<k;++i) hx[i]=std::sin(float(i)*0.17f)*0.91371f+float(i%7)*0.0000137f;
    ck(cudaMemcpyAsync(x,hx.data(),k*4,cudaMemcpyHostToDevice,s));
    for(int layout : {1,4}) {
        ck(cudaMemsetAsync(y,0xA5,(n+8)*4,s));
        strata::kernels::bf16_gemv_fp32_mmvf_layout(x,w,y,k,n,layout,s);
        auto& out=layout==1?a:b;
        ck(cudaMemcpyAsync(out.data(),y,out.size()*4,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));
    }
    if(std::memcmp(a.data(),b.data(),a.size()*4)) throw std::runtime_error("row layout changed output bits or tail");
    for(int r : {0,n/2,n-1}) {
        double ref=0,power=0;
        for(int c=0;c<k;++c) {double p=double(wide(weight(size_t(r)*k+c)))*hx[c];ref+=p;power+=std::abs(p);}
        if(!std::isfinite(a[r]) || std::abs(a[r]-ref)>3e-6*std::max(power,1e-20))
            throw std::runtime_error("FP64 sampled reference mismatch");
    }
    // Graph replay mirrors the runtime's captured projections and removes CPU launch gaps.
    cudaGraph_t graph[2]{};cudaGraphExec_t exec[2]{};
    for(int q=0;q<2;++q) {
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<20;++i) strata::kernels::bf16_gemv_fp32_mmvf_layout(x,w,y,k,n,q?4:1,s);
        ck(cudaStreamEndCapture(s,&graph[q]));ck(cudaGraphInstantiate(&exec[q],graph[q],nullptr,nullptr,0));
        ck(cudaGraphLaunch(exec[q],s));ck(cudaStreamSynchronize(s));
    }
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    for(int round=0;round<5;++round) for(int order=0;order<2;++order) {
        int q=(round%2)?1-order:order;
        ck(cudaEventRecord(start,s));ck(cudaGraphLaunch(exec[q],s));ck(cudaEventRecord(end,s));ck(cudaEventSynchronize(end));
        float ms;ck(cudaEventElapsedTime(&ms,start,end));
        std::printf("{\"K\":%d,\"N\":%d,\"round\":%d,\"layout\":%d,\"us\":%.6f,\"bit_equal\":true}\n",k,n,round,q?4:1,ms*1000/20);
        std::fflush(stdout);
    }
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
    for(int q=0;q<2;++q) {ck(cudaGraphExecDestroy(exec[q]));ck(cudaGraphDestroy(graph[q]));}
    ck(cudaFree(w));ck(cudaFree(x));ck(cudaFree(y));
}
}
int main() {
    try {
        cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
        for(int k : {2,66,640,2560,6144}) for(int n : {1,63,129}) run(k,n,s);
        for(const auto shape : {std::pair<int,int>{2560,2560},{2560,6144},{6144,2560},{2560,248320}}) run(shape.first,shape.second,s);
        ck(cudaStreamDestroy(s));return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
