#include "strata/core/safetensors_model.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/platform/memory.hpp"
#include <cuda_runtime.h>
#include <bit>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <stdexcept>

namespace strata::core {
namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
std::filesystem::path u8path(const std::string& p) {
    return std::filesystem::path(std::u8string(p.begin(),p.end()));
}
uint64_t align(uint64_t n) { return (n+255)/256*256; }
void upload(weights::WeightSource& source,const weights::DenseBinding& b,void* dest) {
    weights::stream_dense(source,b,[&](uint64_t off,std::span<const uint8_t> data) {
        cuda_check(cudaMemcpy(static_cast<uint8_t*>(dest)+off,data.data(),data.size(),cudaMemcpyHostToDevice));
    });
}
}
SafetensorsModel::SafetensorsModel(const std::string& path) : source_(u8path(path)),plan_(source_),dense_(source_) {
    plan_.validate_ngram_constants(source_);
    std::vector<SessionModelFile> files;
    for (const auto& shard:source_.shards()) {
        const auto p=shard.path.u8string(),name=shard.path.filename().u8string();
        files.push_back({std::string(name.begin(),name.end()),std::string(p.begin(),p.end())});
    }
    for (const auto* name:{"config.json","hf_quant_config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json"}) {
        const auto p=(u8path(path)/name).u8string(); files.push_back({name,std::string(p.begin(),p.end())});
    }
    std::string err;
    // Existing sampled identity contract, computed ONCE during startup. Session
    // save/load must not reopen expert shards after the residency barrier.
    if (!session_model_fingerprint(files,session_identity_,err)) throw std::runtime_error(err);
}
SafetensorsModel::~SafetensorsModel() {
    if (sealed_expert_bytes_) std::fprintf(stderr,"safetensors: post-residency expert source bytes=%llu\n",
        static_cast<unsigned long long>(source_.io_stats().data_bytes[static_cast<size_t>(weights::Family::Expert)]-sealed_expert_bytes_));
    if (sealed_expert_bytes_) std::fprintf(stderr,"safetensors: post-residency MTP source bytes=%llu\n",
        static_cast<unsigned long long>(source_.io_stats().data_bytes[static_cast<size_t>(weights::Family::Mtp)]-sealed_mtp_bytes_));
    if (scratch_) cudaFree(scratch_);
}
uint64_t SafetensorsModel::dense_bytes() const {
    uint64_t n = 0;
    for (const auto& b : dense_.tensors)
        if (b.name != "token_embd.weight" && b.name != "output.weight") n += align(b.bytes());
    return n;
}
bool SafetensorsModel::load_dense(WeightTable& table,void* arena,uint64_t capacity,std::string& err) {
    try {
        if (!table.table_.empty() || !arena || capacity < dense_bytes()) throw std::runtime_error("invalid dense arena/table");
        cuda_check(cudaMalloc(&scratch_,65536)); // shared ordered-stream native ABI scratch
        uint64_t off = 0;
        for (const auto& b : dense_.tensors) {
            WeightRef r;
            const bool vector = b.columns == 1;
            r.ne0 = vector ? b.rows : b.columns; r.ne1 = vector ? 0 : b.rows;
            r.elements = b.rows*b.columns; r.bytes=b.bytes();
            r.kind=b.output == weights::DType::BF16 ? WeightKind::Bf16InF32 : WeightKind::F32;
            r.src_off=b.source->offset; r.src_bytes=b.source->bytes; r.file_id=static_cast<int>(b.source->file_id);
            if (b.name == "token_embd.weight" || b.name == "output.weight") r.resident=false;
            else {
                auto* dst=static_cast<uint8_t*>(arena)+off;
                upload(source_,b,dst); r.data=dst;
                if (b.native_projection) { r.native_data=dst; r.native_type=30; r.native_q8_1=scratch_; }
                off+=align(b.bytes());
            }
            table.table_.emplace(b.name,r);
        }
        table.report_.tensors=table.table_.size(); table.report_.arena_bytes=off;
        return true;
    } catch (const std::exception& e) { err=e.what(); return false; }
}
bool SafetensorsModel::load_embed(NativeEmbed& e,std::string& err) {
    try {
        if (e.host_ || e.dev_) throw std::runtime_error("embedding already loaded");
        const auto& b=dense_.find("token_embd.weight");
        e.bytes_=b.bytes(); e.row_=b.columns*2; e.n_embd_=b.columns; e.n_vocab_=b.rows; e.type_=30;
        // A failed pin is an error here, not an unreported pageable expert/table fallback.
        cuda_check(cudaHostAlloc(&e.host_,e.bytes_,cudaHostAllocMapped|cudaHostAllocPortable));
        weights::stream_dense(source_,b,[&](uint64_t off,std::span<const uint8_t> data) {
            std::memcpy(static_cast<uint8_t*>(e.host_)+off,data.data(),data.size());
        });
        void* alias=nullptr; cuda_check(cudaHostGetDevicePointer(&alias,e.host_,0)); e.dev_=alias;
        return true;
    } catch (const std::exception& e) { err=e.what(); return false; }
}
bool SafetensorsModel::load_head(NativeHead& h,std::string& err) {
    try {
        if (h.loaded()) throw std::runtime_error("head already loaded");
        const auto& b=dense_.find("output.weight");
        cuda_check(cudaMalloc(&h.weights_,b.bytes()));
        upload(source_,b,h.weights_);
        h.bytes_=b.bytes(); h.n_in_=static_cast<int>(b.columns); h.n_out_=static_cast<int>(b.rows); h.type_=30;
        return true;
    } catch (const std::exception& e) { err=e.what(); return false; }
}
strata::kernels::PleConsts SafetensorsModel::ple_constants() {
    strata::kernels::PleConsts c{};
    const auto m=source_.read(*plan_.ngram_multipliers),v=source_.read(*plan_.ngram_vocab_sizes),o=source_.read(*plan_.ngram_offsets);
    std::memcpy(c.mult,m.data(),m.size()); std::memcpy(c.vocab,v.data(),v.size()); std::memcpy(c.offset,o.data(),o.size());
    return c;
}
bool SafetensorsModel::open_ple(strata::kernels::PleTable& table,const strata::kernels::PleIoOptions& io,std::string& err) {
    try {
        std::vector<strata::ngram::PleReader::Segment> spans;
        const auto id=plan_.ngram.front().tensor->file_id;
        for (const auto& s : plan_.ngram) {
            if (s.tensor->file_id != id) throw std::runtime_error("native PLE spans multiple files; reader extension required");
            spans.push_back({s.first_row,s.row_count,s.tensor->offset});
        }
        const auto raw=source_.read(*plan_.ngram_scale);
        uint16_t bits; std::memcpy(&bits,raw.data(),2);
        const auto path=source_.shards().at(id).path.u8string();
        return table.open_fp8_segments(std::string(path.begin(),path.end()),spans,
            std::bit_cast<float>(uint32_t(bits)<<16),err,io);
    } catch (const std::exception& e) { err=e.what(); return false; }
}
bool SafetensorsModel::load_experts(FileExpertSource& s,std::string& err) {
    try {
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        auto elapsed = [&] { return std::chrono::duration<double>(Clock::now()-start).count(); };
        auto setting = [](const char* key, long fallback, long upper) {
            const char* value = std::getenv(key);
            if (!value) return fallback;
            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!*value || *end || parsed < 1 || parsed > upper)
                throw std::runtime_error(std::string(key) + " must be an integer in [1," + std::to_string(upper) + "]");
            return parsed;
        };
        const int batch = static_cast<int>(setting("STRATA_NATIVE_LOAD_BATCH", 1, 128));
        const unsigned workers = static_cast<unsigned>(setting("STRATA_NATIVE_LOAD_WORKERS", 1, 16));
        double read_ms = 0, pack_ms = 0;
        uint64_t scratch_peak = 0;
        weights::Nvfp4LoadScratch load_scratch;
        s.close();
        const auto& l=strata::kernels::cpu::expert_layout();
        if (!l.native || l.total != plan_.expert_arena_bytes) throw std::runtime_error("native expert layout mismatch");
        s.n_layers_=48; s.n_expert_=512; s.blobs_=48*512;
        s.layer_offsets_=l.offset; s.layer_blob_bytes_=l.bytes;
        s.complement_bytes_=l.total;
        bool allocate_pinned=false;
        if (const char* value=std::getenv("STRATA_NATIVE_ALLOC_PINNED")) {
            if (std::strcmp(value,"0") && std::strcmp(value,"1"))
                throw std::runtime_error("STRATA_NATIVE_ALLOC_PINNED must be 0 or 1");
            allocate_pinned=std::strcmp(value,"1")==0;
        }
        long register_gib=0;
        if (const char* value=std::getenv("STRATA_NATIVE_REGISTER_GIB")) {
            char* end=nullptr; register_gib=std::strtol(value,&end,10);
            if(!*value || *end || register_gib<0 || register_gib>44)
                throw std::runtime_error("STRATA_NATIVE_REGISTER_GIB must be an integer in [0,44]");
        }
        if (allocate_pinned && register_gib)
            throw std::runtime_error("native pinned allocation and partial registration are mutually exclusive");
        std::fprintf(stderr,"safetensors: allocating %.3f GiB expert RAM (%s); startup batch=%d workers=%u\n",
                     double(l.total)/(1ull<<30), allocate_pinned ? "CUDA pinned" : "OS locked", batch, workers);
        // Both choices own every expert, including RAM mirrors of hot GPU slots.
        // Windows cudaHostAlloc and cudaHostRegister have different measured
        // admission behavior; do not infer one's limit from the other.
        if (allocate_pinned) {
            cuda_check(cudaHostAlloc(&s.complement_arena_,static_cast<size_t>(l.total),cudaHostAllocMapped));
            s.complement_pinned_=true; s.complement_pin_limit_=l.total;
            void* alias=nullptr; cuda_check(cudaHostGetDevicePointer(&alias,s.complement_arena_,0));
            s.complement_device_=static_cast<uint8_t*>(alias);
        } else {
            s.complement_arena_=std::malloc(static_cast<size_t>(l.total));
            if (!s.complement_arena_) throw std::runtime_error("cannot allocate native expert RAM arena");
            const auto lock=strata::platform::lock_resident(s.complement_arena_,l.total);
            s.complement_locked_=lock.locked_bytes;
            if (!lock.ok || lock.locked_bytes != l.total) throw std::runtime_error("cannot physically lock all experts: "+lock.note);
        }
        s.complement_host_=static_cast<uint8_t*>(s.complement_arena_);
        if (!allocate_pinned) {
            s.native_stage_stride_=l.max_blob;
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&s.native_stage_host_),128*l.max_blob,cudaHostAllocMapped));
            cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.native_stage_device_),s.native_stage_host_,0));
        }
        std::fprintf(stderr,"safetensors: expert RAM allocated and locked in %.2f s\n",elapsed());
        s.complement_offsets_.resize(static_cast<size_t>(s.blobs_));
        input_scales_.resize(static_cast<size_t>(s.blobs_));
        for (int layer=0;layer<48;++layer) {
            for (int first=0;first<512;first+=batch) {
                const int count = std::min(batch, 512-first);
                if (batch > 1) {
                    std::vector<weights::Nvfp4ExpertBuffer> targets;
                    for (int id=first;id<first+count;++id) {
                        const auto off = l.blob_offset(layer,id);
                        const auto index = layer*512+id;
                        s.complement_offsets_[index] = off;
                        targets.push_back({&plan_.expert(layer,id),
                            {static_cast<uint8_t*>(s.complement_arena_)+off,static_cast<size_t>(l.blob_bytes(layer))},&input_scales_[index]});
                    }
                    const auto stats = weights::load_nvfp4_expert_batch(source_,targets,workers,&load_scratch);
                    read_ms += stats.read_ms; pack_ms += stats.pack_ms;
                    scratch_peak = std::max(scratch_peak,stats.scratch_bytes);
                    continue;
                }
                const int id = first;
                auto expert=weights::load_nvfp4_expert(source_,plan_.expert(layer,id),layer,false);
                const auto off=l.blob_offset(layer,id);
                const auto index=layer*512+id;
                if (expert.bytes.size() != l.blob_bytes(layer)) throw std::runtime_error("expert blob size mismatch");
                std::memcpy(static_cast<uint8_t*>(s.complement_arena_)+off,expert.bytes.data(),expert.bytes.size());
                s.complement_offsets_[index]=off; input_scales_[index]=expert.input_scales;
            }
            const auto& scales=input_scales_[layer*512];
            bool uniform=scales[0] == scales[1];
            for (int id=1;id<512;++id) uniform=uniform && input_scales_[layer*512+id] == scales;
            if (!uniform) throw std::runtime_error("per-expert activation scales need a grouped ABI extension");
            strata::kernels::cpu::expert_layout_nvfp4_scales(layer,scales[0],scales[2]);
            std::fprintf(stderr,"safetensors: resident experts %d/48 layers (elapsed %.2f s, read %.2f s, pack %.2f s)\n",
                         layer+1,elapsed(),batch>1 ? read_ms/1000 : -1,batch>1 ? pack_ms/1000 : -1);
        }
        if(register_gib>0) {
                // Keep the full owned/locked copy. Register only a bounded prefix
                // within WDDM's shared-memory budget; remaining experts use staging.
                const uint64_t bytes=std::min<uint64_t>(l.total,uint64_t(register_gib)<<30);
                cuda_check(cudaHostRegister(s.complement_arena_,bytes,cudaHostRegisterMapped));
                s.complement_registered_=true;
                void* alias=nullptr; cuda_check(cudaHostGetDevicePointer(&alias,s.complement_arena_,0));
                s.complement_device_=static_cast<uint8_t*>(alias);
                s.complement_pinned_=true; s.complement_partial_=bytes<l.total; s.complement_pin_limit_=bytes;
                std::fprintf(stderr,"safetensors: %.1f GiB of the locked expert arena also registered for direct GPU reads\n",
                             double(bytes)/(1ull<<30));
        }
        // base_ is the existing source's opened marker; no file or mapping exists.
        // unmapped_ prevents both fallback reads and unmapping the owned arena.
        s.base_=s.complement_host_; s.unmapped_=true; s.complement_ready_=true;
        std::fprintf(stderr,"safetensors: %.3f GiB physically %s, %.1f MiB GPU staging\n",
            static_cast<double>(allocate_pinned ? s.complement_pin_limit_ : s.complement_locked_)/(1ull<<30),
            allocate_pinned ? "page-locked and GPU-mapped via cudaHostAlloc" : "locked",
            s.native_stage_host_ ? 128.0*l.max_blob/(1ull<<20) : 0.0);
        source_.seal_expert_reads();
        const char* close_files = std::getenv("STRATA_NATIVE_CLOSE_FILES");
        if (close_files && close_files[0] == '1' && close_files[1] == '\0') {
            source_.seal_resident_reads();
            std::fprintf(stderr,"safetensors: startup weight handles closed; all resident payload reads sealed\n");
        }
        sealed_mtp_bytes_=source_.io_stats().data_bytes[static_cast<size_t>(weights::Family::Mtp)];
        sealed_expert_bytes_=source_.io_stats().data_bytes[static_cast<size_t>(weights::Family::Expert)];
        std::fprintf(stderr,"safetensors: expert startup complete: %.2f s, batch=%d workers=%u scratch_peak_mib=%.2f "
                            "read_s=%.2f pack_s=%.2f source_read_calls=%llu\n",elapsed(),batch,workers,
                     double(scratch_peak)/(1ull<<20),batch>1 ? read_ms/1000 : -1,batch>1 ? pack_ms/1000 : -1,
                     static_cast<unsigned long long>(source_.io_stats().data_calls));
        return true;
    } catch (const std::exception& e) { err=e.what(); s.close(); return false; }
}
}
