#include "strata/kernels/mtp_fp8.hpp"
#include "strata/weights/safetensors.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata;
void ck(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
double decode(uint8_t v) {
    const int e=(v>>3)&15,m=v&7;
    const double a=e?std::ldexp(1.0+m/8.0,e-7):std::ldexp(double(m),-9);
    return v&128?-a:a;
}
double dot(const uint8_t* w,const float* x,int r,int cols) {
    double y=0;
    for(int c=0;c<cols;++c) {
        uint16_t bits;
        std::memcpy(&bits,w+kernels::kMtpFp8Matrix+2*((r/128)*(cols/128)+c/128),2);
        y+=decode(w[r*cols+c])*std::bit_cast<float>(uint32_t(bits)<<16)*x[c];
    }
    return y;
}
int run(const std::filesystem::path& path) {
    try {
        std::unique_ptr<weights::SafetensorsSource> source;
        if(!path.empty()) source=std::make_unique<weights::SafetensorsSource>(path);
        constexpr int E=2,T=2,K=2;
        std::vector<uint8_t> packed(E*kernels::kMtpFp8Expert);
        std::mt19937 rng(120);
        const char* names[]={"gate_proj","up_proj","down_proj"};
        for(int e=0;e<E;++e) for(int p=0;p<3;++p) {
            auto* dst=packed.data()+e*kernels::kMtpFp8Expert+p*kernels::kMtpFp8Projection;
            if(source) {
                const std::string stem="mtp.layers.0.mlp.experts."+std::to_string(e?511:0)+"."+names[p];
                auto w=source->read(source->tensor(stem+".weight")),s=source->read(source->tensor(stem+".weight_scale_inv"));
                std::memcpy(dst,w.data(),w.size()); std::memcpy(dst+kernels::kMtpFp8Matrix,s.data(),s.size());
            } else {
                for(uint64_t i=0;i<kernels::kMtpFp8Matrix;++i) { auto v=uint8_t(rng()); if((v&127)==127) v=0; dst[i]=v; }
                for(int i=0;i<100;++i) { uint16_t s=uint16_t(0x3600+(rng()%384)); std::memcpy(dst+kernels::kMtpFp8Matrix+i*2,&s,2); }
            }
        }
        std::vector<float> x(T*2560),h(T*K*640),y(T*K*2560),ref(y.size());
        for(auto& v:x) v=(int(rng()%2001)-1000)*0.001f;
        const int32_t ids[]={0,1,1,0};
        uint8_t* dw=nullptr; float *dx=nullptr,*dh=nullptr,*dy=nullptr; int32_t* di=nullptr;
        ck(cudaMalloc(&dw,packed.size())); ck(cudaMalloc(&dx,x.size()*4)); ck(cudaMalloc(&dh,h.size()*4)); ck(cudaMalloc(&dy,y.size()*4)); ck(cudaMalloc(&di,sizeof(ids)));
        ck(cudaMemcpy(dw,packed.data(),packed.size(),cudaMemcpyHostToDevice)); ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice)); ck(cudaMemcpy(di,ids,sizeof(ids),cudaMemcpyHostToDevice));
        kernels::mtp_fp8_experts(dw,di,dx,dh,dy,T,K,nullptr);
        ck(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<T*K;++i) {
            const auto* w=packed.data()+ids[i]*kernels::kMtpFp8Expert;
            // FP64 dot-products, with the ABI's FP32 intermediate storage.
            for(int r=0;r<640;++r) {
                const double g=dot(w,x.data()+(i/K)*2560,r,2560),u=dot(w+kernels::kMtpFp8Projection,x.data()+(i/K)*2560,r,2560);
                h[i*640+r]=float(g/(1+std::exp(-g))*u);
            }
            for(int r=0;r<2560;++r) ref[i*2560+r]=float(dot(w+2*kernels::kMtpFp8Projection,h.data()+i*640,r,640));
        }
        double se=0,sr=0,mx=0,peak=0;
        for(size_t i=0;i<y.size();++i) {
            if(!std::isfinite(y[i])) throw std::runtime_error("nonfinite output");
            const double d=y[i]-ref[i]; se+=d*d; sr+=double(ref[i])*ref[i]; mx=std::max(mx,std::abs(d)); peak=std::max(peak,std::abs(double(ref[i])));
        }
        const double rel=std::sqrt(se/std::max(sr,1e-30));
        std::printf("{\"source\":\"%s\",\"outputs\":%zu,\"relative_l2\":%.9g,\"max_abs\":%.9g,\"scaled_max\":%.9g}\n",source?"checkpoint":"synthetic",y.size(),rel,mx,mx/std::max(peak,1e-30));
        cudaFree(dw);cudaFree(dx);cudaFree(dh);cudaFree(dy);cudaFree(di);
        return rel<5e-5 && mx/std::max(peak,1e-30)<1e-4?0:1;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
#if defined(_WIN32)
int wmain(int argc,wchar_t** argv) { return run(argc==2?std::filesystem::path(argv[1]):std::filesystem::path()); }
#else
int main(int argc,char** argv) { return run(argc==2?std::filesystem::path(argv[1]):std::filesystem::path()); }
#endif
