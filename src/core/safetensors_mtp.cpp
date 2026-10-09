#include "strata/core/mtp.hpp"
#include "strata/weights/dense.hpp"
#include "strata/kernels/mtp_fp8.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <map>

namespace strata::core {
bool MtpDrafter::load_safetensors(weights::SafetensorsSource& source,std::string& err) {
    try {
        using namespace weights;
        if(q4_ || q4_head_) throw std::runtime_error("native MTP does not requantize to Q4");
        auto check=[](cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); };
        // Validate names and dimensions before uploading: the inherited forward
        // uses fixed geometry, so a valid safetensors extent alone is insufficient.
        std::map<std::string,std::vector<uint64_t>> expected={
            {"fc_embedding.weight",{2560,2560}}, {"fc_hidden.weight",{2560,2560}},
            {"pre_fc_norm_embedding.weight",{2560}}, {"pre_fc_norm_hidden.weight",{10240}},
            {"mlp.gate.weight",{512,2560}}, {"mlp.shared_expert_gate.weight",{1,2560}},
            {"mlp.shared_expert.gate_proj.weight",{640,2560}},
            {"mlp.shared_expert.up_proj.weight",{640,2560}},
            {"mlp.shared_expert.down_proj.weight",{2560,640}},
            {"self_attn.q_proj.weight",{12288,2560}}, {"self_attn.k_proj.weight",{512,2560}},
            {"self_attn.v_proj.weight",{512,2560}}, {"self_attn.o_proj.weight",{2560,6144}},
            {"self_attn.q_norm.weight",{256}}, {"self_attn.k_norm.weight",{256}},
            {"self_attn.indexer.index_qk_proj.weight",{640,2560}},
            {"self_attn.indexer.q_layernorm.weight",{128}},
            {"self_attn.indexer.k_layernorm.weight",{128}}};
        for(const std::string prefix:{"hyper_connection_mixer.","attn_hyper_connection.","mlp_hyper_connection."}) {
            expected.emplace(prefix+"hc_norm.weight",std::vector<uint64_t>{10240});
            expected.emplace(prefix+"input_mix_weight_down.weight",std::vector<uint64_t>{320,10240});
            expected.emplace(prefix+"input_mix_weight_up.weight",std::vector<uint64_t>{10240,320});
            if(prefix!="hyper_connection_mixer.")
                expected.emplace(prefix+"block_inject_weight.weight",std::vector<uint64_t>{4,10240});
        }
        std::vector<DenseBinding> bindings;
        uint64_t bytes=0;
        for(const auto& [name,t]:source.tensors()) {
            if(t.family!=Family::Mtp || name.find(".experts.")!=std::string::npos) continue;
            if(t.dtype!=DType::BF16 || t.physical_shape.empty() || t.physical_shape.size()>2)
                throw std::runtime_error("unsupported MTP dense tensor: "+name);
            DenseBinding b;
            b.source=&t; b.name=name.substr(4);
            if(b.name.starts_with("layers.0.")) b.name.erase(0,9);
            const auto shape=expected.find(b.name);
            if(shape==expected.end() || shape->second!=t.physical_shape)
                throw std::runtime_error("MTP dense geometry mismatch: "+name);
            expected.erase(shape);
            b.rows=t.physical_shape[0]; b.columns=t.physical_shape.size()==2?t.physical_shape[1]:1;
            const bool norm=t.physical_shape.size()==1;
            b.output=norm?DType::F32:DType::BF16;
            b.math=norm?DenseMath::AddOne:DenseMath::Identity;
            tensors_.push_back({b.name,norm?"f32":"bf16",int64_t(b.rows),int64_t(b.columns),bytes,b.bytes()});
            bytes=(bytes+b.bytes()+255)&~255ull;
            bindings.push_back(b);
        }
        if(bindings.size()!=29 || !expected.empty()) throw std::runtime_error("MTP dense tensor inventory mismatch");
        check(cudaMalloc(reinterpret_cast<void**>(&dense_),bytes)); vram_+=bytes;
        for(size_t i=0;i<bindings.size();++i) stream_dense(source,bindings[i],[&](uint64_t off,std::span<const uint8_t> data) {
            check(cudaMemcpy(dense_+tensors_[i].off+off,data.data(),data.size(),cudaMemcpyHostToDevice));
        });
        const uint64_t total=512*kernels::kMtpFp8Expert;
        check(cudaMalloc(reinterpret_cast<void**>(&experts_),total)); vram_+=total;
        const char* projections[]={"gate_proj","up_proj","down_proj"};
        for(int e=0;e<512;++e) for(int p=0;p<3;++p) {
            const std::string stem="mtp.layers.0.mlp.experts."+std::to_string(e)+"."+projections[p];
            const auto& w=source.tensor(stem+".weight"); const auto& s=source.tensor(stem+".weight_scale_inv");
            const std::vector<uint64_t> ws=p==2?std::vector<uint64_t>{2560,640}:std::vector<uint64_t>{640,2560};
            const std::vector<uint64_t> ss=p==2?std::vector<uint64_t>{20,5}:std::vector<uint64_t>{5,20};
            if(w.dtype!=DType::F8_E4M3 || w.physical_shape!=ws || s.dtype!=DType::BF16 || s.physical_shape!=ss)
                throw std::runtime_error("MTP FP8 layout mismatch: "+stem);
            auto wb=source.read(w),sb=source.read(s);
            for(auto v:wb) if((v&127)==127) throw std::runtime_error("nonfinite MTP FP8 code");
            for(size_t j=0;j<sb.size();j+=2) {
                uint16_t bits; std::memcpy(&bits,sb.data()+j,2);
                const auto v=std::bit_cast<float>(uint32_t(bits)<<16);
                if(!std::isfinite(v)||v<=0) throw std::runtime_error("invalid MTP block scale");
            }
            auto* dst=experts_+e*kernels::kMtpFp8Expert+p*kernels::kMtpFp8Projection;
            check(cudaMemcpy(dst,wb.data(),wb.size(),cudaMemcpyHostToDevice));
            check(cudaMemcpy(dst+kernels::kMtpFp8Matrix,sb.data(),sb.size(),cudaMemcpyHostToDevice));
        }
        native_fp8_=true;
        std::fprintf(stderr,"safetensors MTP: 29 dense tensors, 512 original FP8 experts; %.1f MiB, no requantization\n",double(bytes+total)/1048576.0);
        return true;
    } catch(const std::exception& e) { err=e.what(); return false; }
}
}
