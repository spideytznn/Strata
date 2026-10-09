// SM120 block-scaled FP4 MMA for small grouped expert windows.
// MMA register mapping/PTX follows llama.cpp ggml-cuda/mma.cuh (MIT,
// third_party/ggml/LICENSE) and NVIDIA's warp-level block-scaling specification.
#include "strata/kernels/nvfp4_tensor.hpp"
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <string>
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

namespace strata::kernels {
namespace {
struct QBlock { unsigned char scale[4]; unsigned int q[8]; };
static_assert(sizeof(QBlock)==36);
size_t aligned(size_t n) { return (n+255)&~size_t(255); }
__device__ unsigned pack4(unsigned v) {
    return (v&15)|((v>>4)&240)|((v>>8)&3840)|((v>>12)&61440);
}
__device__ unsigned weight_word(const block_nvfp4* b, int word) {
    const int sub=word/2, shift=(word%2)*4;
    const unsigned* p=reinterpret_cast<const unsigned*>(b->qs+sub*8);
    return pack4(p[0]>>shift)|(pack4(p[1]>>shift)<<16);
}
__device__ float scale_value(unsigned char c) {
    __nv_fp8_e4m3 v; v.__x=c; return float(v);
}
__device__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2], unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
#endif
}

template<bool DOWN, int TERMS>
__global__ void quantize_rows(const float* x, QBlock* q, const int32_t* starts,
    const int32_t* ng, const int32_t* tok, int cols, int ff, float global_scale, float* row_scales) {
    if (*ng==0) return;
    const int entry=blockIdx.x;
    if (entry>=starts[*ng]) return;
    const int source=DOWN ? entry : tok[entry], blocks=cols/64;
    // Residual modes preserve outliers outside calibration's finite range.
    // A power-of-two extension leaves the checkpoint scale unchanged otherwise.
    if constexpr(TERMS>1) {
        __shared__ float maxima[128];
        float mx=0;
        for(int c=threadIdx.x;c<cols;c+=blockDim.x) {
            float a;
            if constexpr(DOWN) {
                float g=x[size_t(source)*2*ff+c];
                a=g/(1.0f+expf(-g))*x[size_t(source)*2*ff+ff+c];
            } else a=x[size_t(source)*cols+c];
            mx=fmaxf(mx,fabsf(a));
        }
        maxima[threadIdx.x]=mx; __syncthreads();
        for(int stride=64;stride;stride/=2) {
            if(threadIdx.x<stride) maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+stride]);
            __syncthreads();
        }
        while(global_scale*2688.0f<maxima[0]) global_scale*=2.0f;
    }
    if(threadIdx.x==0) row_scales[entry]=global_scale;
    for (int sub=threadIdx.x; sub<cols/16; sub+=blockDim.x) {
        float v[16];
#pragma unroll
        for(int j=0;j<16;++j) {
            const int k=sub*16+j;
            float a;
            if constexpr(DOWN) {
                const float g=x[size_t(source)*2*ff+k], u=x[size_t(source)*2*ff+ff+k];
                a=(g/(1.0f+expf(-g)))*u;
            } else a=x[size_t(source)*cols+k];
            v[j]=__fdiv_rn(a,global_scale);
        }
#pragma unroll
        for(int term=0;term<TERMS;++term) {
            float amax=0;
#pragma unroll
            for(int j=0;j<16;++j) amax=fmaxf(amax,fabsf(v[j]));
            const unsigned char sc=__nv_cvt_float_to_fp8(__fdiv_rn(amax,6.0f),__NV_SATFINITE,__NV_E4M3);
            const float scale=scale_value(sc), inv=scale>0 ? __fdiv_rn(1.0f,scale) : 0.0f;
            unsigned words[2]={0,0};
#pragma unroll
            for(int j=0;j<16;++j) {
                const unsigned char code=__nv_cvt_float_to_fp4(v[j]*inv,__NV_E2M1,cudaRoundNearest);
                words[j/8]|=unsigned(code&15)<<(4*(j%8));
                if constexpr(TERMS>1) v[j]-=__half2float(__nv_cvt_fp4_to_halfraw(code,__NV_E2M1))*scale;
            }
            QBlock& dest=q[(size_t(entry)*TERMS+term)*blocks+sub/4];
            dest.scale[sub%4]=sc;
            dest.q[2*(sub%4)]=words[0]; dest.q[2*(sub%4)+1]=words[1];
        }
    }
}

template<bool DOWN,int TERMS>
__global__ void product(NativeExpertLayout L, const unsigned long long* ptr,
    const int32_t* starts,const int32_t* ng,const int32_t* dst,const QBlock* q,
    float* out,const float* row_scales) {
    const int lane=threadIdx.x, row0=(blockIdx.x*blockDim.y+threadIdx.y)*16;
    const int rows=DOWN ? L.n_embd : 2*L.n_ff, cols=DOWN ? L.n_ff : L.n_embd;
    if(row0>=rows) return;
    const int blocks=cols/64;
    for(int group=blockIdx.y;group<*ng;group+=gridDim.y) {
        const auto* blob=reinterpret_cast<const unsigned char*>(ptr[group]);
        const auto* weights=reinterpret_cast<const block_nvfp4*>(blob+(DOWN?L.down_off:0));
        const float* tails=reinterpret_cast<const float*>(blob+L.tail_off);
        for(int first=starts[group];first<starts[group+1];first+=8) {
            float acc[4]={0,0,0,0};
            for(int k=0;k<blocks;++k) {
                unsigned a[4];
#pragma unroll
                for(int z=0;z<4;++z) {
                    const int row=row0+(z%2)*8+lane/4;
                    a[z]=weight_word(weights+size_t(row)*blocks+k,lane%4+(z/2)*4);
                }
                const int scale_row=row0+lane/4+(lane%2)*8;
                const unsigned sa=*reinterpret_cast<const unsigned*>((weights+size_t(scale_row)*blocks+k)->d);
                const int entry=first+lane/4;
#pragma unroll
                for(int term=0;term<TERMS;++term) {
                    unsigned b[2]={0,0}, sb=0;
                    if(entry<starts[group+1]) {
                        const QBlock& qb=q[(size_t(entry)*TERMS+term)*blocks+k];
                        b[0]=qb.q[lane%4]; b[1]=qb.q[4+lane%4];
                        sb=*reinterpret_cast<const unsigned*>(qb.scale);
                    }
                    mma(acc,a,b,sa,sb);
                }
            }
#pragma unroll
            for(int z=0;z<4;++z) {
                const int row=row0+(z/2)*8+lane/4;
                const int entry=first+(lane%4)*2+z%2;
                if(entry<starts[group+1]) {
                    const float ws=DOWN ? tails[2] : tails[row>=L.n_ff ? 1 : 0];
                    const int output_entry=DOWN ? dst[entry] : entry;
                    out[size_t(output_entry)*rows+row]=acc[z]*row_scales[entry]*ws;
                }
            }
        }
    }
}

template<int TERMS>
void launch(const NativeExpertLayout& L,const unsigned long long* ptr,const int32_t* starts,
    const int32_t* ng,const int32_t* dst,const int32_t* tok,int64_t groups,int64_t entries,
    const float* x,void* scratch,float* out,cudaStream_t stream,float gu_scale,float down_scale) {
    auto* qx=static_cast<QBlock*>(scratch);
    const size_t qx_bytes=aligned(size_t(entries)*TERMS*(L.n_embd/64)*sizeof(QBlock));
    auto* gu=reinterpret_cast<float*>(static_cast<unsigned char*>(scratch)+qx_bytes);
    auto* qh=reinterpret_cast<QBlock*>(reinterpret_cast<unsigned char*>(gu)+aligned(size_t(entries)*2*L.n_ff*4));
    auto* gx=reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(qh)+aligned(size_t(entries)*TERMS*(L.n_ff/64)*sizeof(QBlock)));
    float* gh=gx+entries;
    quantize_rows<false,TERMS><<<unsigned(entries),128,0,stream>>>(x,qx,starts,ng,tok,int(L.n_embd),int(L.n_ff),gu_scale,gx);
    product<false,TERMS><<<dim3(unsigned((2*L.n_ff+63)/64),unsigned(groups)),dim3(32,4),0,stream>>>(L,ptr,starts,ng,dst,qx,gu,gx);
    quantize_rows<true,TERMS><<<unsigned(entries),128,0,stream>>>(gu,qh,starts,ng,tok,int(L.n_ff),int(L.n_ff),down_scale,gh);
    product<true,TERMS><<<dim3(unsigned((L.n_embd+63)/64),unsigned(groups)),dim3(32,4),0,stream>>>(L,ptr,starts,ng,dst,qh,out,gh);
}
}

void nvfp4_tensor_grouped(const NativeExpertLayout& L,const unsigned long long* ptr,const int32_t* starts,
    const int32_t* ng,const int32_t* dst,const int32_t* tok,int64_t groups,int64_t entries,
    const float* x,void* scratch,float* out,void* stream,int terms) {
    if(!x || L.gu_type!=40 || L.d_type!=40 || L.n_embd!=2560 || L.n_ff!=640 || !L.tail_off) throw std::invalid_argument("NVFP4 Tensor decode requires FP32 input and the supported 2560/640 expert geometry");
    if(terms<1 || terms>3) throw std::invalid_argument("NVFP4 Tensor decode terms must be 1, 2 or 3");
    if(groups<=0 || entries<=0) return;
    static const bool supported=[] {
        int dev=0; cudaDeviceProp prop{};
        if(cudaGetDevice(&dev)!=cudaSuccess || cudaGetDeviceProperties(&prop,dev)!=cudaSuccess || prop.major!=12 || prop.minor!=0)
            throw std::runtime_error("This NVFP4 Tensor decode build requires SM120");
        return true;
    }();
    (void)supported;
    static const std::vector<float> scales=[] {
        std::vector<float> values;
        const char* path=std::getenv("STRATA_NVFP4_INPUT_SCALES");
        if(path) { std::ifstream f(path,std::ios::binary); values.resize(96); if(!f.read(reinterpret_cast<char*>(values.data()),96*4) || f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("expected exactly 48 pairs of NVFP4 activation scales"); }
        return values;
    }();
    float gs=L.input_scale_gu, ds=L.input_scale_down;
    if(gs<=0 || ds<=0) {
        if(L.layer<0 || L.layer>=48 || scales.size()!=96) throw std::runtime_error("NVFP4 Tensor decode needs checkpoint activation scales and a layer id");
        gs=scales[L.layer*2]; ds=scales[L.layer*2+1];
    }
    if(!std::isfinite(gs) || !std::isfinite(ds) || gs<=0 || ds<=0) throw std::runtime_error("invalid NVFP4 checkpoint activation scales");
    static const bool announced=[&] {
        std::fprintf(stderr,"[nvfp4-tensor] SM120 native FP4 MMA enabled, %d activation terms, checkpoint scales, FP32 accumulation; weights unchanged\n",terms);
        return true;
    }();
    (void)announced;
    if(terms==1) launch<1>(L,ptr,starts,ng,dst,tok,groups,entries,x,scratch,out,static_cast<cudaStream_t>(stream),gs,ds);
    else if(terms==2) launch<2>(L,ptr,starts,ng,dst,tok,groups,entries,x,scratch,out,static_cast<cudaStream_t>(stream),gs,ds);
    else launch<3>(L,ptr,starts,ng,dst,tok,groups,entries,x,scratch,out,static_cast<cudaStream_t>(stream),gs,ds);
    const cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(std::string("NVFP4 Tensor decode: ")+cudaGetErrorString(error));
}
}
