#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

// Manual AVX-512 microbenchmark, not an inference-throughput claim. Run only on
// a CPU already validated by the engine's runtime ISA check.
int main() {
    try {
        if(!strata::kernels::cpu::cpu_avx512_ok()) throw std::runtime_error("AVX-512 runtime ISA check failed");
        std::mt19937 rng(9950);
        std::uniform_real_distribution<float> values(-1.f,1.f);
        for(int n : {640,2560}) {
            const int rows=n==640?2560:1280;
            const size_t stride=ggml_row_size(GGML_TYPE_NVFP4,n);
            std::vector<uint8_t> w(stride*rows);
            // ggml NVFP4 blocks: four UE4M3 scales, 32 bytes of paired codes.
            for(size_t b=0;b<w.size();b+=36) {
                for(int s=0;s<4;++s) w[b+s]=(b/36+s)%29==0?0x7f:uint8_t(24+rng()%64);
                for(int j=4;j<36;++j) w[b+j]=uint8_t(rng());
            }
            std::vector<float> decoded(size_t(n)*rows);
            ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(w.data(),decoded.data(),int64_t(n)*rows);
            for(int nt=1;nt<=8;++nt) {
                std::vector<float> x(size_t(nt)*n),old(size_t(nt)*(rows+2),123.f),next=old;
                for(auto& v:x) v=values(rng);
                const void* acts[8];float* a[8];float* b[8];
                for(int t=0;t<nt;++t) { acts[t]=x.data()+t*n; a[t]=old.data()+t*(rows+2)+1;b[t]=next.data()+t*(rows+2)+1; }
                auto run=[&](bool unroll) {
                    strata::kernels::cpu::nvfp4_512_f32_rows_layout(w.data(),stride,n,acts,nt,unroll?b:a,0,rows,.375f,unroll);
                };
                run(false);run(true);
                if(std::memcmp(old.data(),next.data(),old.size()*sizeof(float))) throw std::runtime_error("layout bits differ");
                double se=0,sr=0;
                for(int t=0;t<nt;++t) {
                    if(a[t][-1]!=123.f || a[t][rows]!=123.f) throw std::runtime_error("row canary overwritten");
                    for(int r=0;r<rows;r+=std::max(1,rows/37)) {
                        double sum=0;for(int j=0;j<n;++j) sum+=double(decoded[size_t(r)*n+j])*x[t*n+j];
                        sum*=.375;const double delta=a[t][r]-sum;
                        if(!std::isfinite(delta)) throw std::runtime_error("nonfinite output");
                        se+=delta*delta;sr+=sum*sum;
                    }
                }
                const double relative=std::sqrt(se/std::max(sr,1e-300));
                if(relative>1e-5) throw std::runtime_error("FP64 oracle mismatch");
                std::vector<double> timing[2];
                for(int round=0;round<7;++round) for(int step=0;step<2;++step) {
                    const int kind=(round+step)%2;
                    const auto start=std::chrono::steady_clock::now();
                    for(int repeat=0;repeat<3;++repeat)run(kind!=0);
                    timing[kind].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/3);
                }
                for(auto& v:timing)std::sort(v.begin(),v.end());
                std::printf("{\"n\":%d,\"rows\":%d,\"tokens\":%d,\"rolled_ms\":%.6f,\"unrolled_ms\":%.6f,\"relative_l2\":%.9g,\"bit_equal\":true}\n",
                            n,rows,nt,timing[0][3],timing[1][3],relative);
            }
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
