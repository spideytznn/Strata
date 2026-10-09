// Numerical regression for the new float paths; FP64 dot products are independent references.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static float f32(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f,&u,4); return f; }
static void gdn_float_regression(cudaStream_t s) {
    constexpr int S=128, HV=48, C=10240;
    for (int T : {1,17,128}) {
        const int LD=HV*S+32;
        std::vector<float> h(size_t(T)*C), gate(T*HV,-0.3f), beta(T*HV,0.6f), z(size_t(T)*HV*S), gamma(S);
        for(size_t i=0;i<h.size();++i) h[i]=std::sin(float(i)*0.013f)*0.09f;
        for(size_t i=0;i<z.size();++i) z[i]=std::cos(float(i)*0.019f);
        for(int i=0;i<S;++i) gamma[i]=0.7f+i*0.003f;
        std::vector<void*> allocations;
        auto upload=[&](const std::vector<float>& v) {
            float* p; ck(cudaMalloc(&p,v.size()*4)); allocations.push_back(p);
            ck(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice)); return p;
        };
        float *dh=upload(h), *dg=upload(gate), *db=upload(beta), *dz=upload(z), *dgm=upload(gamma);
        float *state,*dy; uint16_t* dy16;
        ck(cudaMalloc(&state,size_t(S)*HV*S*4)); allocations.push_back(state);
        ck(cudaMalloc(&dy,z.size()*4)); allocations.push_back(dy);
        ck(cudaMalloc(&dy16,size_t(T)*LD*2)); allocations.push_back(dy16);
        std::vector<float> raw(z.size()), normalized(z.size());
        std::vector<uint16_t> half0(size_t(T)*LD), half1(half0.size());
        for(int mode=0;mode<2;++mode) {
            ck(cudaMemsetAsync(state,0,size_t(S)*HV*S*4,s));
            ck(cudaMemsetAsync(dy16,0,half0.size()*2,s));
            strata::prefill::gdn_recurrence_variant(0,state,dh,dg,db,dz,dgm,1e-6f,dy,dy16,T,s,LD,mode!=0);
            auto& out=mode?normalized:raw; auto& half=mode?half1:half0;
            ck(cudaMemcpyAsync(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost,s));
            ck(cudaMemcpyAsync(half.data(),dy16,half.size()*2,cudaMemcpyDeviceToHost,s));
            ck(cudaStreamSynchronize(s));
        }
        if(half0!=half1) throw std::runtime_error("GDN opt-in changes default FP16 output");
        double error=0, power=0;
        for(int row=0;row<T*HV;++row) {
            double sum=0; for(int c=0;c<S;++c) {double v=raw[row*S+c];sum+=v*v;}
            for(int c=0;c<S;++c) {
                const size_t at=size_t(row)*S+c;
                const double ref=raw[at]/std::sqrt(sum/S+1e-6)*gamma[c]/(1+std::exp(-double(z[at])));
                const double d=normalized[at]-ref; error+=d*d;power+=ref*ref;
            }
        }
        double rms=std::sqrt(error/std::max(power,1e-30));
        std::printf("GDN normalized FP32 T=%d rel_rms=%.3e (default half identical)\n",T,rms);
        if(!std::isfinite(rms)||rms>3e-6) throw std::runtime_error("GDN FP32 output is not normalized/gated");
        for(void* p:allocations) ck(cudaFree(p));
    }
}
int main() {
    try {
        cudaStream_t s; ck(cudaStreamCreate(&s));
        for (int K : {640,2560,6144}) for (int T : {1,4,8,17}) {
            const int N=129, LDY=N+7;
            std::vector<uint16_t> w(size_t(K)*N);
            std::vector<float> x(size_t(K)*T), ref(size_t(LDY)*T,0), got(ref.size(),0);
            for (size_t i=0;i<w.size();++i) w[i]=uint16_t((i%19==0?0x3381:0x3c01)+(i*13%128)) | (i%3==0?0x8000:0);
            for (size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*0.17f)*0.91371f + float(i%7)*0.0000137f;
            for (int t=0;t<T;++t) for (int r=0;r<N;++r) {
                double sum=0; for (int c=0;c<K;++c) sum+=double(x[t*K+c])*f32(w[r*K+c]);
                ref[t*LDY+r]=float(sum);
            }
            uint16_t* dw; float *dx,*dy;
            ck(cudaMalloc(&dw,w.size()*2)); ck(cudaMalloc(&dx,x.size()*4)); ck(cudaMalloc(&dy,got.size()*4));
            ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
            ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
            if (T<=8) {
                strata::kernels::native_projection_f32(30,dw,dx,nullptr,dy,K,N,T,s);
                std::vector<float> compact(size_t(N)*T);
                ck(cudaMemcpyAsync(compact.data(),dy,compact.size()*4,cudaMemcpyDeviceToHost,s)); ck(cudaStreamSynchronize(s));
                double e=0,v=0; for(int t=0;t<T;++t)for(int r=0;r<N;++r){double d=compact[t*N+r]-ref[t*LDY+r];e+=d*d;v+=double(ref[t*LDY+r])*ref[t*LDY+r];}
                const double rms=std::sqrt(e/std::max(v,1e-30));
                std::printf("decode K=%d T=%d rel_rms=%.3e\n",K,T,rms);
                if(rms>3e-6) throw std::runtime_error("decode exceeds FP64 reference tolerance");
            }
            strata::prefill::Gemm gm; std::string err;
            // Deliberately small scratch: exercises multiple token tiles as well as output stride.
            if(!gm.init(s,2*K*3,err)) throw std::runtime_error(err);
            gm.bf16_f32(dx,dw,dy,T,N,K,LDY);
            ck(cudaMemcpyAsync(got.data(),dy,got.size()*4,cudaMemcpyDeviceToHost,s)); ck(cudaStreamSynchronize(s));
            double e=0,v=0; for(int t=0;t<T;++t)for(int r=0;r<N;++r){double d=got[t*LDY+r]-ref[t*LDY+r];e+=d*d;v+=double(ref[t*LDY+r])*ref[t*LDY+r];}
            const double rms=std::sqrt(e/std::max(v,1e-30));
            std::printf("prefill K=%d T=%d rel_rms=%.3e\n",K,T,rms);
            if(rms>3e-6) throw std::runtime_error("prefill exceeds FP64 reference tolerance");
            ck(cudaFree(dw));ck(cudaFree(dx));ck(cudaFree(dy));
        }
        gdn_float_regression(s);
        ck(cudaStreamDestroy(s)); std::puts("PASS: float projection numerical references and GDN producer");
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
