#include "strata/core/session.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/ngram.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;
static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
static uint64_t aligned(uint64_t n) { return (n + 255) / 256 * 256; }

// Independent expression of the pre-change full-session byte contract.
static uint64_t legacy_raw_bytes(const ModelGeometry& g, int64_t cells) {
    const uint64_t gdn = (uint64_t)g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                         g.ssm_conv_channels * (g.ssm_d_conv - 1);
    uint64_t n = gdn_buffers_bytes(g) + g.n_gdn_layers() * gdn * 4;
    if (g.n_qsa_layers())
        n += qsa_state_bytes(g, cells, true) + (g.n_qsa_layers()-1) * qsa_state_bytes(g, cells, false);
    n += qsa_buffers_bytes(g, cells) + moe_buffers_bytes(g, 10) + block_buffers_bytes(g);
    n += (uint64_t)strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * sizeof(float);
    return n;
}
static uint64_t legacy_bytes(const ModelGeometry& g, int64_t cells) {
    return aligned(legacy_raw_bytes(g, cells)) + 4096;
}

static void layout_test() {
    ModelGeometry g;
    for (const bool int8 : {false, true}) {
        qsa_set_kv_int8(int8);
        for (const int64_t cells : {32, 4096, 32768, 262144}) {
            const auto full = session_bytes(g, cells, 10);
            check(full == legacy_bytes(g, cells), "default layout changed");
            check(full == session_bytes(g, cells, 10, 0, 12), "explicit full range differs");
            for (int64_t begin = 0; begin < 12; ++begin)
                for (int64_t end = begin+1; end <= 12; ++end) {
                    const auto bytes = session_bytes(g, cells, 10, begin, end);
                    const auto removed = (12 - (end-begin)) * qsa_state_bytes(g, cells, false);
                    check(bytes == aligned(legacy_raw_bytes(g,cells)-removed)+4096,
                          "partial layout savings inconsistent");
                    check(bytes > 0 && bytes <= full, "invalid partial size");
                }
        }
    }
    check(session_bytes(g, 32, 10, -1, 6) == 0, "negative begin accepted");
    check(session_bytes(g, 32, 10, 7, 6) == 0, "reversed range accepted");
    check(session_bytes(g, 32, 10, 0, 13) == 0, "out-of-bounds range accepted");
    check(session_bytes(g, 32, 10, 6, 6) == 0, "empty stage accepted");
    qsa_set_kv_int8(true);
    std::printf("262K int8 saved: %.2f MiB per 24-layer stage\n",
                double(session_bytes(g,262144,10)-session_bytes(g,262144,10,6,12))/1048576);
    std::puts("CPU layout checks PASS (no CUDA context created)");
}

static void gpu_test(int device) {
    check(cudaSetDevice(device) == cudaSuccess, "cudaSetDevice");
    ModelGeometry g;
    // Keep GPU tests small; QSA shape is real, GDN is only initialized/zeroed here.
    g.ssm_state_size=4; g.ssm_v_heads=2; g.ssm_k_heads=2;
    g.ssm_conv_channels=64; g.ssm_value_dim=8;
    qsa_set_kv_int8(true);
    for (const int begin : {0, 6}) {
        const auto bytes = session_bytes(g, 64, 10, begin, begin+6);
        void* arena=nullptr;
        check(cudaMalloc(&arena, bytes+256) == cudaSuccess, "test arena allocation");
        check(cudaMemset(arena,0xa5,bytes+256)==cudaSuccess,"fill canary");
        SessionState ss;
        const auto used=session_init(g,64,10,arena,ss,begin,begin+6);
        check(used>0 && used<=bytes,"arena carve exceeds allocation");
        session_zero(ss,g,nullptr,nullptr);
        check(cudaDeviceSynchronize()==cudaSuccess,"session zero failed");
        for (int i=0;i<12;++i) {
            const auto& q=ss.qsa_states[i];
            if (!ss.owns_qsa(i)) {
                check(q.k_q==nullptr && q.idx_tail==nullptr && q.cos_tab==nullptr,
                      "unowned QSA has an allocation");
                continue;
            }
            check(q.max_cells==64 && q.k_q && q.v_q,"owned QSA missing");
            check(q.cos_tab==ss.first_qsa().cos_tab && q.sin_tab==ss.first_qsa().sin_tab,
                  "RoPE not shared within owning device");
            std::vector<int8_t> kv(64*g.n_head_kv*g.head_dim);
            check(cudaMemcpy(kv.data(),q.k_q,kv.size(),cudaMemcpyDeviceToHost)==cudaSuccess,"read K");
            for(auto x:kv) check(x==0,"K not zero");
            std::vector<int32_t> pages(q.n_pages);
            check(cudaMemcpy(pages.data(),q.page_table,pages.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess,"read pages");
            for(size_t p=0;p<pages.size();++p) check(pages[p]==(int)p,"page identity changed");
        }
        std::vector<unsigned char> canary(256);
        check(cudaMemcpy(canary.data(),(char*)arena+bytes,256,cudaMemcpyDeviceToHost)==cudaSuccess,"read canary");
        for(auto x:canary) check(x==0xa5,"arena overflow");
        // A whole-model execution entry must reject a partial session before launching work.
        WeightTable wt; SessionGraphs graphs; std::string err;
        check(!session_capture(wt,g,ss,nullptr,graphs,err,false),"full capture accepted partial state");
        cudaFree(arena);
        delete[] ss.qsa_states;
        // The few pinned QSA metadata blocks are process-lifetime allocations, as in existing session tests.
    }
    std::printf("GPU %d staged init/zero/ownership/RoPE/page table/guard checks PASS\n",device);
}

int main(int argc,char** argv) {
    try {
        layout_test();
        if(argc==3 && std::string(argv[1])=="--gpu") gpu_test(std::atoi(argv[2]));
        else check(argc==1,"usage: stage_kv_test [--gpu DEVICE]");
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
}
