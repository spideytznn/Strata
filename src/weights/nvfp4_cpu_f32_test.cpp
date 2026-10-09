#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace c=strata::kernels::cpu;
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("usage: nvfp4_cpu_f32_test experts.bin");
        c::NativeFmt fmt; std::string err;
        if(!c::native_fmt(40,40,2560,640,fmt,err)) throw std::runtime_error(err);
        fmt.fp32_activations=true; fmt.act_bytes=2560*4; fmt.h_bytes=640*4;
        constexpr int T=4,N=2560,F=640;
        std::ifstream input(argv[1],std::ios::binary);
        std::mt19937 rng(9950); std::normal_distribution<float> random;
        double worst=0;
        for(int expert:{0,47*512+511}) {
            std::vector<uint8_t> b(fmt.bytes);
            input.seekg(uint64_t(expert)*fmt.bytes); input.read(reinterpret_cast<char*>(b.data()),b.size());
            if(!input) throw std::runtime_error("expert read failed");
            std::vector<float> x(T*N),h(T*F),y(T*N),one(T*N),wg(F*N),wu(F*N),wd(N*F);
            for(auto& v:x) v=random(rng);
            const auto* tr=ggml_get_type_traits(GGML_TYPE_NVFP4);
            tr->to_float(b.data(),wg.data(),F*N); tr->to_float(b.data()+fmt.up_off,wu.data(),F*N);
            tr->to_float(b.data()+fmt.down_off,wd.data(),N*F);
            const void* a[T]; const void* hp[T]; float* hs[T]; float* ys[T];
            for(int t=0;t<T;++t) { a[t]=x.data()+t*N; hp[t]=hs[t]=h.data()+t*F; ys[t]=y.data()+t*N; }
            std::vector<uint8_t> copy(c::kNativeActBytes); c::native_quant_act(fmt,x.data(),copy.data());
            if(std::memcmp(copy.data(),x.data(),N*4)) throw std::runtime_error("input rounded");
            c::native_gu_rows(fmt,b.data(),a,T,hs,0,F); c::native_down_rows(fmt,b.data(),hp,T,ys,0,N);
            for(int t=0;t<T;++t) {
                float* dest=one.data()+t*N;
                c::native_gu_rows(fmt,b.data(),a+t,1,hs+t,0,F); c::native_down_rows(fmt,b.data(),hp+t,1,&dest,0,N);
            }
            if(std::memcmp(y.data(),one.data(),y.size()*4)) throw std::runtime_error("token grouping changed output");
            float scales[4]; std::memcpy(scales,b.data()+fmt.tail_off,16);
            double se=0,sr=0;
            for(int t=0;t<T;++t) {
                std::vector<double> hidden(F);
                for(int r=0;r<F;++r) {
                    double g=0,u=0;
                    for(int col=0;col<N;++col) { g+=double(wg[r*N+col])*x[t*N+col]; u+=double(wu[r*N+col])*x[t*N+col]; }
                    g*=scales[0]; hidden[r]=g/(1+std::exp(-g))*(u*scales[1]);
                }
                for(int r=0;r<N;++r) {
                    double sum=0; for(int col=0;col<F;++col) sum+=double(wd[r*F+col])*hidden[col];
                    sum*=scales[2]; const double d=y[t*N+r]-sum;
                    if(!std::isfinite(d)) throw std::runtime_error("nonfinite output");
                    se+=d*d; sr+=sum*sum;
                }
            }
            const double relative=std::sqrt(se/sr); worst=std::max(worst,relative);
            std::printf("expert %d relative_l2 %.9g; T=1/4 bit-identical\n",expert,relative);
        }
        return worst<1e-4?0:1;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
