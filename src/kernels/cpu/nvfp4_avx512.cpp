// src/kernels/cpu/nvfp4_avx512.cpp - see include/strata/kernels/cpu/nvfp4_avx512.hpp.
//
// Per 64-value block, once: the 64 E2M1 codes in element order, their magnitudes from |kvalues_fp4| (0..12, the
// doubled E2M1 values), a sign mask (code bit 3), and the four sub-block scales. Per token: the sign moves onto the
// activation (a masked subtract - Q8_0 is quantized to [-127, 127], so negation never overflows), one vpdpbusd gives
// 16 int32 lanes = four per 16-value sub-block, and one FMA applies ue4m3(d[s]) * d_q8. The UE4M3 decode is ggml's
// CPU one (ggml-impl.h ggml_ue4m3_to_fp32): halved for the doubled table, 0x7F read as 0.
//
// Needs AVX512F/BW/VL and VNNI - what cpu_avx512_ok() checks before anything here is called.
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

struct Ue4m3Table {
    float v[256];
    Ue4m3Table() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) { v[x] = 0.0f; continue; }
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
const Ue4m3Table kUe4m3;

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off): the rows stream from DRAM through 4 KB
// pages, where the hardware prefetchers stop at every page boundary; the IQ kernel's E-2 knob, same default
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_NVFP4_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// The block's 64 codes in element order. Sub-block s keeps its codes in qs[8 s .. 8 s + 7]: byte j holds value j
// in its low nibble and value j + 8 in its high one, so a sub-block is [low nibbles, high nibbles] of its 8 bytes.
inline __m512i codes64(const uint8_t* qs) {
    const __m256i q = _mm256_loadu_si256((const __m256i*) qs);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(q, m4);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
    const __m256i ul = _mm256_unpacklo_epi64(lo, hi);                 // [lo 0-7, hi 0-7 | lo 16-23, hi 16-23]
    const __m256i uh = _mm256_unpackhi_epi64(lo, hi);                 // [lo 8-15, hi 8-15 | lo 24-31, hi 24-31]
    const __m256i s01 = _mm256_permute2x128_si256(ul, uh, 0x20);      // sub-blocks 0, 1
    const __m256i s23 = _mm256_permute2x128_si256(ul, uh, 0x31);      // sub-blocks 2, 3
    return _mm512_inserti64x4(_mm512_castsi256_si512(s01), s23, 1);
}

// lanes 4 s .. 4 s + 3 = ue4m3(d[s])
inline __m512 sub_scales(const uint8_t* d) {
    const __m128 s4 = _mm_setr_ps(kUe4m3.v[d[0]], kUe4m3.v[d[1]], kUe4m3.v[d[2]], kUe4m3.v[d[3]]);
    const __m512i idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    return _mm512_permutexvar_ps(idx, _mm512_castps128_ps512(s4));
}

// Per token and block, the activation's scales as lanes: 0-7 -> the first Q8_0 block's d (sub-blocks 0, 1),
// 8-15 -> the second's (sub-blocks 2, 3). Built once per call, reused by every row.
constexpr int kMaxBlocks = 64;                                        // rows of up to 4096 values (n_embd 2560)
struct Acts {
    int nt = 0, nb = 0;
    const block_q8_0* y[7] = {};
    __m512 dy[7 * kMaxBlocks];                                         // [t * nb + ib]; on the stack, no allocation
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < nb; ++ib) {
                const block_q8_0* yb = y[t] + 2 * ib;
                dy[(size_t) t * nb + ib] = _mm512_mask_blend_ps((__mmask16) 0xFF00, _mm512_set1_ps(h2f(yb[0].d)),
                                                                 _mm512_set1_ps(h2f(yb[1].d)));
            }
        }
    }
};

template <int NT>
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) _mm_prefetch((const char*) (x + ib) + prefetch_ahead, _MM_HINT_T0);
        const __m512i codes = codes64(x[ib].qs);
        const __m512i g = _mm512_shuffle_epi8(mag, codes);
        const __mmask64 neg = _mm512_test_epi8_mask(codes, eight);
        const __m512 sw = sub_scales(x[ib].d);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m512i yv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) yb[0].qs)),
                                                  _mm256_loadu_si256((const __m256i*) yb[1].qs), 1);
            const __m512i ys = _mm512_mask_sub_epi8(yv, neg, zero, yv);
            const __m512i p = _mm512_dpbusd_epi32(zero, g, ys);
            acc[t] = _mm512_fmadd_ps(_mm512_mul_ps(sw, a.dy[(size_t) t * a.nb + ib]), _mm512_cvtepi32_ps(p), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff, int r0, int r1,
             float sg, float su) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, a, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, a, u);
        for (int t = 0; t < NT; ++t) {
            const float gs = g[t] * sg;
            ff[t][r] = (gs / (1.f + std::exp(-gs))) * (u[t] * su);
        }
    }
}

template <int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, const Acts& a, float* const* out, int r0, int r1, float scale) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, a, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t] * scale;
    }
}

}  // namespace

bool nvfp4_512_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

// Windows wider than 7 run in balanced slices of at most 7: at 8 the accumulators no longer fit beside the loads and
// MSVC spills (measured 0.32 ms for 8 tokens against 2 x 0.13 for 4 + 4).
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + 6) / 7;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            default: gu_rows<7>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, a, o, r0, r1, scale); break;
            case 2: dot_rows<2>(w, row_bytes, a, o, r0, r1, scale); break;
            case 3: dot_rows<3>(w, row_bytes, a, o, r0, r1, scale); break;
            case 4: dot_rows<4>(w, row_bytes, a, o, r0, r1, scale); break;
            case 5: dot_rows<5>(w, row_bytes, a, o, r0, r1, scale); break;
            case 6: dot_rows<6>(w, row_bytes, a, o, r0, r1, scale); break;
            default: dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

namespace {
// Same original blocks, with unquantized activations. Decode each weight tile
// once for all tokens; each token retains an independent FP32 accumulator.
void f32_dot_rolled(const uint8_t* row,int n,const void* const* acts,int nt,float* out) {
    const __m512 lut=_mm512_setr_ps(0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6);
    __m512 acc[8];
    for(int t=0;t<nt;++t) acc[t]=_mm512_setzero_ps();
    const auto* w=reinterpret_cast<const block_nvfp4*>(row);
    for(int b=0;b<n/64;++b) {
        alignas(64) uint8_t codes[64];
        _mm512_store_si512(codes,codes64(w[b].qs));
        for(int sub=0;sub<4;++sub) {
            const __m512i idx=_mm512_cvtepu8_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(codes+16*sub)));
            const __m512 v=_mm512_mul_ps(_mm512_permutexvar_ps(idx,lut),_mm512_set1_ps(kUe4m3.v[w[b].d[sub]]*2.f));
            for(int t=0;t<nt;++t)
                acc[t]=_mm512_fmadd_ps(v,_mm512_loadu_ps(static_cast<const float*>(acts[t])+b*64+sub*16),acc[t]);
        }
    }
    for(int t=0;t<nt;++t) out[t]=_mm512_reduce_add_ps(acc[t]);
}

// Preserve each lane's FMA order. Compile-time token/sub-block counts keep the
// accumulators in registers and extract codes without a stack round trip.
template<int NT,int SUB>
inline void f32_sub(__m512i codes,const block_nvfp4& w,const void* const* acts,int b,
                    __m512 lut,__m512* acc) {
    const __m512i idx=_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(codes,SUB));
    const __m512 v=_mm512_mul_ps(_mm512_permutexvar_ps(idx,lut),_mm512_set1_ps(kUe4m3.v[w.d[SUB]]*2.f));
    for(int t=0;t<NT;++t)
        acc[t]=_mm512_fmadd_ps(v,_mm512_loadu_ps(static_cast<const float*>(acts[t])+b*64+SUB*16),acc[t]);
}
// Match the CUDA warp's 32 lanes and XOR 16/8/4/2/1 reduction explicitly.
// The normal 16-lane CPU tree remains available unchanged.
float gpu_order_reduce(__m512 lo,__m512 hi) {
    const __m512 both=_mm512_add_ps(lo,hi);
    const __m256 a=_mm256_add_ps(_mm512_castps512_ps256(both),
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(both),1)));
    const __m128 b=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    const __m128 c=_mm_add_ps(b,_mm_shuffle_ps(b,b,_MM_SHUFFLE(1,0,3,2)));
    return _mm_cvtss_f32(_mm_add_ps(c,_mm_shuffle_ps(c,c,_MM_SHUFFLE(2,3,0,1))));
}
template<int NT,bool GPU_ORDER=false>
void f32_dot_unrolled(const uint8_t* row,int n,const void* const* acts,float* out) {
    const __m512 lut=_mm512_setr_ps(0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6);
    __m512 acc[NT];
    __m512 high[NT];
    for(int t=0;t<NT;++t) acc[t]=_mm512_setzero_ps();
    if constexpr(GPU_ORDER) for(int t=0;t<NT;++t) high[t]=_mm512_setzero_ps();
    const auto* w=reinterpret_cast<const block_nvfp4*>(row);
    for(int b=0;b<n/64;++b) {
        const __m512i codes=codes64(w[b].qs);
        f32_sub<NT,0>(codes,w[b],acts,b,lut,acc);
        f32_sub<NT,1>(codes,w[b],acts,b,lut,GPU_ORDER?high:acc);
        f32_sub<NT,2>(codes,w[b],acts,b,lut,acc);
        f32_sub<NT,3>(codes,w[b],acts,b,lut,GPU_ORDER?high:acc);
    }
    for(int t=0;t<NT;++t) {
        if constexpr(GPU_ORDER) out[t]=gpu_order_reduce(acc[t],high[t]);
        else out[t]=_mm512_reduce_add_ps(acc[t]);
    }
}
bool f32_unroll_on() {
    static const bool enabled=[] { const char* v=std::getenv("STRATA_NVFP4_F32_UNROLL"); return v && std::strcmp(v,"1")==0; }();
    return enabled;
}
bool f32_gpu_order_on() {
    static const bool enabled=[] { const char* v=std::getenv("STRATA_NVFP4_F32_GPU_ORDER");
        const char* c=std::getenv("STRATA_NVFP4_F32_CANONICAL");
        return (v && std::strcmp(v,"1")==0) || (c && std::strcmp(c,"1")==0); }();
    return enabled;
}
bool f32_canonical_on() {
    static const bool enabled=[] { const char* v=std::getenv("STRATA_NVFP4_F32_CANONICAL"); return v && std::strcmp(v,"1")==0; }();
    return enabled;
}
void f32_dot(const uint8_t* row,int n,const void* const* acts,int nt,float* out,bool unroll,bool gpu_order=false) {
    if(gpu_order) {
        switch(nt) {
            case 1:f32_dot_unrolled<1,true>(row,n,acts,out);return;
            case 2:f32_dot_unrolled<2,true>(row,n,acts,out);return;
            case 3:f32_dot_unrolled<3,true>(row,n,acts,out);return;
            case 4:f32_dot_unrolled<4,true>(row,n,acts,out);return;
            case 5:f32_dot_unrolled<5,true>(row,n,acts,out);return;
            case 6:f32_dot_unrolled<6,true>(row,n,acts,out);return;
            case 7:f32_dot_unrolled<7,true>(row,n,acts,out);return;
            case 8:f32_dot_unrolled<8,true>(row,n,acts,out);return;
        }
    }
    if(!unroll) { f32_dot_rolled(row,n,acts,nt,out); return; }
    switch(nt) {
        case 1: f32_dot_unrolled<1>(row,n,acts,out); break;
        case 2: f32_dot_unrolled<2>(row,n,acts,out); break;
        case 3: f32_dot_unrolled<3>(row,n,acts,out); break;
        case 4: f32_dot_unrolled<4>(row,n,acts,out); break;
        case 5: f32_dot_unrolled<5>(row,n,acts,out); break;
        case 6: f32_dot_unrolled<6>(row,n,acts,out); break;
        case 7: f32_dot_unrolled<7>(row,n,acts,out); break;
        case 8: f32_dot_unrolled<8>(row,n,acts,out); break;
        default: f32_dot_rolled(row,n,acts,nt,out); break;
    }
}
}
void nvfp4_512_f32_gu_rows(const uint8_t* blob,size_t stride,size_t up,int n,const void* const* acts,int nt,
                          float* const* ff,int r0,int r1,float sg,float su) {
    float g[8],u[8];
    const bool unroll=f32_unroll_on();
    const bool gpu_order=f32_gpu_order_on();
    const bool canonical=f32_canonical_on();
    for(int r=r0;r<r1;++r) {
        f32_dot(blob+r*stride,n,acts,nt,g,unroll,gpu_order); f32_dot(blob+up+r*stride,n,acts,nt,u,unroll,gpu_order);
        for(int t=0;t<nt;++t) {
            const float v=g[t]*sg;
            const float e=canonical?static_cast<float>(std::exp(-static_cast<double>(v))):std::exp(-v);
            ff[t][r]=(v/(1.f+e))*(u[t]*su);
        }
    }
}
void nvfp4_512_f32_rows(const uint8_t* w,size_t stride,int n,const void* const* acts,int nt,
                       float* const* out,int r0,int r1,float scale) {
    if(f32_gpu_order_on()) {
        float y[8];
        for(int r=r0;r<r1;++r) {
            f32_dot(w+r*stride,n,acts,nt,y,true,true);
            for(int t=0;t<nt;++t) out[t][r]=y[t]*scale;
        }
        return;
    }
    nvfp4_512_f32_rows_layout(w,stride,n,acts,nt,out,r0,r1,scale,f32_unroll_on());
}
void nvfp4_512_f32_rows_layout(const uint8_t* w,size_t stride,int n,const void* const* acts,int nt,
                              float* const* out,int r0,int r1,float scale,bool unroll) {
    float y[8];
    for(int r=r0;r<r1;++r) {
        f32_dot(w+r*stride,n,acts,nt,y,unroll);
        for(int t=0;t<nt;++t) out[t][r]=y[t]*scale;
    }
}
}  // namespace strata::kernels::cpu
