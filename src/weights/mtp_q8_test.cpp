#include "strata/weights/mtp_q8.hpp"
#include <algorithm>
#include <cstdio>

using namespace strata::weights;
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct MemorySource : WeightSource {
    TensorDesc desc{"test",DType::BF16,Family::Mtp,{5,64}};
    std::vector<uint8_t> bytes = std::vector<uint8_t>(5*64*2);
    IoStats stats;
    const TensorDesc& tensor(const std::string&) const override { return desc; }
    const IoStats& io_stats() const override { return stats; }
    void read_many(std::span<const ReadRequest> requests) override {
        for (const auto& r : requests) {
            require(r.relative_offset<=bytes.size() && r.destination.size()<=bytes.size()-r.relative_offset,"test read extent");
            std::memcpy(r.destination.data(),bytes.data()+r.relative_offset,r.destination.size());
            ++stats.data_calls;
        }
    }
    void set(size_t i, float value) {
        const uint16_t bits=uint16_t(std::bit_cast<uint32_t>(value)>>16);
        std::memcpy(bytes.data()+i*2,&bits,2);
    }
};
void synthetic() {
    MemorySource s;
    for(size_t row=0;row<5;++row) for(size_t block=0;block<2;++block) {
        const size_t at=row*64+block*32;
        const float scale=float(1u<<row); // powers of two keep 127*scale exact in BF16
        s.set(at,127*scale); s.set(at+1,0.5f*scale); s.set(at+2,-0.5f*scale);
        for(size_t i=3;i<32;++i) s.set(at+i,float(int(i)-16)*scale);
    }
    DenseBinding b;
    b.name="fc_hidden.weight";b.source=&s.desc;b.first_row=1;b.rows=3;b.columns=64;
    std::vector<uint8_t> reference(3*68), tiled(3*68+16,0xa5);
    size_t calls=0;
    stream_mtp_q8(s,b,[&](uint64_t off,std::span<const uint8_t> data) {
        require(off==calls*68 && data.size()==68,"Q8 tile offset or tail");
        std::copy(data.begin(),data.end(),tiled.begin()+off);++calls;
    },256); // one row per tile, with a first-row slice
    stream_mtp_q8(s,b,[&](uint64_t off,std::span<const uint8_t> data) {
        std::copy(data.begin(),data.end(),reference.begin()+off);
    });
    require(calls==3 && std::equal(reference.begin(),reference.end(),tiled.begin()),"Q8 tile-dependent rounding");
    for(size_t i=reference.size();i<tiled.size();++i) require(tiled[i]==0xa5,"Q8 overrun");
    for(size_t block=0;block<6;++block) {
        const size_t at=block*34;
        uint16_t bits;std::memcpy(&bits,reference.data()+at,2);
        require(ggml_fp16_to_fp32(bits)==float(1u<<(block/2+1)),"wrong Q8 scale");
        require(int8_t(reference[at+2])==127 && int8_t(reference[at+3])==1 && int8_t(reference[at+4])==-1,"Q8 tie rounding");
    }
    auto refused=[&] {
        try { stream_mtp_q8(s,b,[](uint64_t,std::span<const uint8_t>){}); }
        catch(const std::runtime_error&) { return true; }
        return false;
    };
    b.columns=63;require(refused(),"unaligned columns accepted");b.columns=64;
    b.output=DType::F32;require(refused(),"non-BF16 binding accepted");b.output=DType::BF16;
    s.set(64,std::bit_cast<float>(0x7fc00000u));require(refused(),"NaN input accepted");
    s.set(64,std::bit_cast<float>(0x7f7f0000u));require(refused(),"overflowing Q8 scale accepted");
    s.bytes.assign(s.bytes.size(),0);
    stream_mtp_q8(s,b,[&](uint64_t,std::span<const uint8_t> data) {
        require(std::all_of(data.begin(),data.end(),[](uint8_t v){return v==0;}),"zero Q8 block");
    });
    require(mtp_q8_projection("self_attn.indexer.index_qk_proj.weight") &&
            !mtp_q8_projection("mlp.gate.weight") && !mtp_q8_projection("attn_hyper_connection.input_mix_weight_down.weight"),"Q8 projection inventory");
}
int run(const std::filesystem::path& model) {
    try {
        synthetic();
        if(!model.empty()) {
            SafetensorsSource source(model);
            size_t count=0;uint64_t input=0,output=0;
            for(const auto& [name,t]:source.tensors()) {
                if(t.family!=Family::Mtp) continue;
                std::string local=name.substr(4);
                if(local.starts_with("layers.0.")) local.erase(0,9);
                if(!mtp_q8_projection(local)) continue;
                require(t.dtype==DType::BF16 && t.physical_shape.size()==2,"real Q8 projection geometry");
                DenseBinding b;b.source=&t;b.name=local;b.rows=t.physical_shape[0];b.columns=t.physical_shape[1];
                uint64_t next=0;
                stream_mtp_q8(source,b,[&](uint64_t off,std::span<const uint8_t> data) {
                    require(off==next,"real Q8 output gap");next+=data.size();
                });
                require(next==b.rows*mtp_q8_row_bytes(b.columns),"real Q8 output extent");
                output+=next;input+=b.bytes();++count;
            }
            require(count==10,"real Q8 projection inventory");
            require(source.io_stats().data_bytes[size_t(Family::Expert)]==0 &&
                    source.io_stats().data_bytes[size_t(Family::Ngram)]==0,"unrelated model reads");
            std::printf("real MTP projections=%zu BF16_bytes=%llu Q8_bytes=%llu GPU_inference=0\n",count,
                        (unsigned long long)input,(unsigned long long)output);
        }
        std::puts("MTP Q8 streaming tests passed");return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) { return run(argc>1?std::filesystem::path(argv[1]):std::filesystem::path{}); }
#else
int main(int argc,char** argv) { return run(argc>1?std::filesystem::path(argv[1]):std::filesystem::path{}); }
#endif
