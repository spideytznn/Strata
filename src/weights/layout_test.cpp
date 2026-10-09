#include "strata/weights/nvfp4.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace w = strata::weights;
namespace {
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct MockSource : w::WeightSource {
    std::map<std::string,w::TensorDesc> descriptors;
    std::map<std::string,std::vector<uint8_t>> data;
    w::IoStats stats;
    const w::TensorDesc& tensor(const std::string& name) const override { return descriptors.at(name); }
    const w::IoStats& io_stats() const override { return stats; }
    void read_many(std::span<const w::ReadRequest> reads) override {
        for (const auto& r : reads) {
            const auto& b=data.at(r.tensor->name);
            check(r.relative_offset<=b.size() && r.destination.size()<=b.size()-r.relative_offset,"mock bounds");
            std::memcpy(r.destination.data(),b.data()+r.relative_offset,r.destination.size());
        }
    }
    void add(const std::string& name,w::DType dtype,std::vector<uint64_t> shape,std::vector<uint8_t> bytes) {
        descriptors[name]={name,dtype,w::Family::Expert,std::move(shape),0,0,bytes.size()};
        data[name]=std::move(bytes);
    }
    void scale(const std::string& name,float value) {
        std::vector<uint8_t> b(4); std::memcpy(b.data(),&value,4);
        add(name,w::DType::F32,{},std::move(b));
    }
};
void projection_test() {
    MockSource source;
    std::array<w::Nvfp4Projection,3> projections;
    for (size_t i=0;i<3;++i) {
        const auto prefix=std::to_string(i);
        const uint64_t rows=i==2?2560:640, cols=i==2?640:2560;
        source.add(prefix+".weight",w::DType::U8,{rows,cols/2},std::vector<uint8_t>(rows*cols/2,0x98));
        source.add(prefix+".weight_scale",w::DType::F8_E4M3,{rows,cols/16},std::vector<uint8_t>(rows*cols/16,1));
        source.scale(prefix+".weight_scale_2",std::ldexp(1.0f,-130)); // cannot be retained in FP16
        source.scale(prefix+".input_scale",static_cast<float>(i+1)); // deliberately unequal gate/up
        projections[i]=w::nvfp4_projection(source,prefix,prefix,rows,cols);
    }
    const auto loaded=w::load_nvfp4_expert(source,projections,7,true);
    check(loaded.layout.bytes==2764816 && loaded.layout.tail_off==2764800 && loaded.layout.layer==7,"layout ABI");
    check(loaded.input_scales==std::array<float,3>{1,2,3} && loaded.layout.input_scale_gu==0,"unequal input scales collapsed");
    check(loaded.roundtrip_bytes==2764824,"roundtrip coverage");
    for (const auto& suffix : {"weight_scale_2","input_scale"}) {
        for (float bad : {0.f,-1.f,INFINITY,NAN}) {
            source.scale(std::string("0.")+suffix,bad);
            bool refused=false;
            try { w::load_nvfp4_expert(source,projections,0,false); } catch(const std::runtime_error&) { refused=true; }
            check(refused,"invalid global/input scale accepted");
        }
        source.scale(std::string("0.")+suffix,1.f);
    }
    bool refused=false;
    source.descriptors.at("0.weight").physical_shape={1280,640};
    try { w::nvfp4_projection(source,"0","bad",640,2560); } catch(const std::runtime_error&) { refused=true; }
    check(refused,"transposed HF shape accepted");
}
double fp4(unsigned code) {
    constexpr double magnitude[] = {0,0.5,1,1.5,2,3,4,6};
    return (code & 8 ? -1 : 1) * magnitude[code & 7];
}
double fp8(uint8_t b) {
    const auto e = (b >> 3) & 15, m = b & 7;
    return e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(static_cast<double>(m), -9);
}
// Independent direct weight indexing. No use of the inverse permutation for
// the numerical check, including negative zero and all valid micro scales.
double native_weight(const std::vector<uint8_t>& p, size_t index) {
    const auto b = index / 64, sub = index % 64 / 16, j = index % 16;
    const auto q = p[b*36+4+sub*8+j%8];
    return fp4((q >> (j/8*4)) & 15) * fp8(p[b*36+sub]);
}
void test(uint64_t rows, uint64_t cols) {
    const size_t n = static_cast<size_t>(rows*cols);
    std::vector<uint8_t> weights(n/2), scales(n/16), packed(n/64*36), back(n/2), sb(n/16);
    for (size_t i=0;i<weights.size();++i) weights[i] = static_cast<uint8_t>((i*73+19)&255);
    for (size_t i=0;i<scales.size();++i) scales[i] = static_cast<uint8_t>(i%127);
    w::pack_nvfp4(weights,scales,rows,cols,packed);
    w::unpack_nvfp4(packed,rows,cols,back,sb);
    check(weights==back && scales==sb,"bytes not restored");
    for (size_t i=0;i<n;++i) {
        const double source=fp4((weights[i/2] >> (i%2*4))&15)*fp8(scales[i/16]);
        const double dest=native_weight(packed,i);
        check(source==dest && std::signbit(source)==std::signbit(dest),"decoded value/sign changed");
    }
    for (uint8_t bad : {uint8_t(0x7f),uint8_t(0x80),uint8_t(0xff)}) {
        scales[0]=bad;
        bool refused=false;
        try { w::pack_nvfp4(weights,scales,rows,cols,packed); } catch(const std::runtime_error&) { refused=true; }
        check(refused,"invalid UE4M3 scale accepted");
    }
}
}
int main() {
    try {
        test(7,128); test(640,2560); test(2560,640); projection_test();
        std::cout << "PASS: byte roundtrip, independent FP64 decode, signed zero, all finite micro scales, invalid scales\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
