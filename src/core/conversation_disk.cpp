#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_memory.hpp"
#include <bit>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {
constexpr uint64_t alignment = 4096, chunk_bytes = 8 * 1024 * 1024;
constexpr uint64_t disk_floor = 2ull * 1024 * 1024 * 1024;
constexpr uint64_t state_limit = 1024ull * 1024 * 1024;
uint64_t padded(uint64_t n) {
    if (n > UINT64_MAX - alignment + 1) throw std::runtime_error("SSD extent overflow");
    return (n + alignment - 1) & ~(alignment - 1);
}
// Fast 64-bit streaming integrity check, not a cryptographic identity. The
// checksum covers logical bytes; deterministic zero padding is not inference state.
struct Hash {
    uint64_t h = 0x9e3779b185ebca87ull, tail = 0, length = 0;
    unsigned used = 0;
    void word(uint64_t x) { h = std::rotl(h ^ (x * 0xc2b2ae3d27d4eb4full), 27) * 0x165667b19e3779f9ull; }
    void update(const void* bytes, size_t n) {
        const auto* p = static_cast<const uint8_t*>(bytes); length += n;
        while (used && n) { tail |= uint64_t(*p++) << (8 * used++); --n; if (used == 8) { word(tail); tail = 0; used = 0; } }
        while (n >= 8) { uint64_t x; std::memcpy(&x, p, 8); word(x); p += 8; n -= 8; }
        while (n) { tail |= uint64_t(*p++) << (8 * used++); --n; }
    }
    uint64_t finish() const {
        uint64_t x = h ^ std::rotl(tail, 13) ^ (length * 0x9e3779b185ebca87ull);
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
        return x ^ (x >> 33);
    }
};
struct Ref { uint64_t offset = 0, size = 0, hash = 0; };

class File {
public:
    std::filesystem::path path;
    uint64_t extent = 0, limit = 0;
    bool writable = false;
    uint8_t* scratch = nullptr;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int handle = -1;
    bool direct = false;
#endif
    File(std::filesystem::path p, bool create, uint64_t max) : path(std::move(p)), limit(max), writable(create) {
#ifdef _WIN32
        scratch = static_cast<uint8_t*>(_aligned_malloc(chunk_bytes, alignment));
        handle = CreateFileW(path.c_str(), create ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, create ? CREATE_NEW : OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | (create ? FILE_FLAG_WRITE_THROUGH : 0), nullptr);
        if (handle == INVALID_HANDLE_VALUE) { if (scratch) _aligned_free(scratch); scratch = nullptr;
            throw std::runtime_error("SSD file open failed (Windows " + std::to_string(GetLastError()) + ")"); }
#else
        void* memory = nullptr;
        if (posix_memalign(&memory, alignment, chunk_bytes) == 0) scratch = static_cast<uint8_t*>(memory);
        const int flags = create ? O_CREAT | O_EXCL | O_RDWR : O_RDONLY;
#ifdef O_DIRECT
        handle = ::open(path.c_str(), flags | O_DIRECT, 0600); direct = handle >= 0;
#endif
        if (handle < 0) handle = ::open(path.c_str(), flags, 0600);
        if (handle < 0) { std::free(scratch); scratch = nullptr; throw std::runtime_error("SSD file open failed"); }
#endif
        if (!scratch) { close(); throw std::bad_alloc(); }
        if (!create) extent = std::filesystem::file_size(path);
    }
    void close() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); handle = INVALID_HANDLE_VALUE; }
        if (scratch) _aligned_free(scratch);
#else
        if (handle >= 0) { ::close(handle); handle = -1; }
        std::free(scratch);
#endif
        scratch = nullptr;
    }
    ~File() { close(); }
    bool io(uint64_t at, size_t n, bool write) {
        if ((at % alignment) || (n % alignment) || n > chunk_bytes || at > extent || n > extent - at) return false;
#ifdef _WIN32
        LARGE_INTEGER where; where.QuadPart = static_cast<LONGLONG>(at);
        if (!SetFilePointerEx(handle, where, nullptr, FILE_BEGIN)) return false;
        DWORD done = 0;
        const BOOL ok = write ? WriteFile(handle, scratch, static_cast<DWORD>(n), &done, nullptr)
                              : ReadFile(handle, scratch, static_cast<DWORD>(n), &done, nullptr);
        return ok && done == n;
#else
        size_t done = 0;
        while (done < n) {
            const auto count = write ? pwrite(handle, scratch + done, n - done, at + done)
                                     : pread(handle, scratch + done, n - done, at + done);
            if (count <= 0) return false;
            done += static_cast<size_t>(count);
        }
        if (!direct) posix_fadvise(handle, at, n, POSIX_FADV_DONTNEED);
        return true;
#endif
    }
    Ref reserve(uint64_t size) {
        const uint64_t n = padded(size);
        if (!writable || extent > limit || n > limit - extent) throw std::runtime_error("SSD snapshot exceeds reserved budget");
        Ref ref{extent, size, 0}; extent += n;
#ifdef _WIN32
        LARGE_INTEGER end; end.QuadPart = static_cast<LONGLONG>(extent);
        if (!SetFilePointerEx(handle, end, nullptr, FILE_BEGIN) || !SetEndOfFile(handle))
            throw std::runtime_error("SSD file allocation failed");
#else
        if (ftruncate(handle, extent)) throw std::runtime_error("SSD file allocation failed");
#endif
        return ref;
    }
    bool flush() {
#ifdef _WIN32
        return FlushFileBuffers(handle) != 0;
#else
        return fsync(handle) == 0;
#endif
    }
};

class Slice final : public ConversationStorage {
public:
    std::shared_ptr<File> file;
    Ref ref;
    bool complete = false;
    Slice(std::shared_ptr<File> f, Ref r, bool ready) : file(std::move(f)), ref(r), complete(ready) {}
    bool visit(size_t offset, size_t count, bool write, const Visitor& fn) override {
        if (offset > ref.size || count > ref.size - offset || (write && (!file->writable || complete))) return false;
        const bool whole = offset == 0 && count == ref.size;
        Hash hash;
        while (count) {
            const size_t skip = offset % alignment;
            const size_t n = std::min<size_t>(count, chunk_bytes - skip);
            const size_t physical = static_cast<size_t>(padded(skip + n));
            const uint64_t at = ref.offset + offset - skip;
            if (write) {
                if (skip || offset + n != ref.size && n % alignment) {
                    if (!file->io(at, physical, false)) return false;
                } else std::memset(file->scratch, 0, physical);
                if (!fn(file->scratch + skip, n, offset)) return false;
                if (whole) hash.update(file->scratch + skip, n);
                if (!file->io(at, physical, true)) return false;
            } else {
                if (!file->io(at, physical, false)) return false;
                if (whole) hash.update(file->scratch + skip, n);
                if (!fn(file->scratch + skip, n, offset)) return false;
            }
            offset += n; count -= n;
        }
        if (whole) {
            if (write) { ref.hash = hash.finish(); complete = true; }
            else if (!complete || hash.finish() != ref.hash) return false;
        }
        return true;
    }
};

struct Writer {
    std::vector<uint8_t> bytes;
    void u64(uint64_t n) { for (int j = 0; j < 8; ++j) { bytes.push_back(uint8_t(n)); n >>= 8; } }
    void ids(const std::vector<int32_t>& ids) {
        u64(ids.size()); for (int32_t n : ids) for (int j = 0; j < 4; ++j) bytes.push_back(uint8_t(uint32_t(n) >> (j * 8)));
    }
    void images(const std::vector<ConversationImageKey>& images) {
        u64(images.size()); for (const auto& i : images) { u64(i.start); u64(i.hash); }
    }
    void region(const Ref& ref) { u64(ref.offset); u64(ref.size); u64(ref.hash); }
};
struct Reader {
    const std::vector<uint8_t>& bytes;
    size_t at = 0;
    uint64_t u64() {
        if (at > bytes.size() || bytes.size() - at < 8) throw std::runtime_error("SSD index truncated");
        uint64_t n = 0; for (int j = 0; j < 8; ++j) n |= uint64_t(bytes[at++]) << (j * 8); return n;
    }
    size_t count(size_t per, size_t cap = SIZE_MAX) {
        const uint64_t n = u64();
        if (n > cap || n > (bytes.size() - at) / per) throw std::runtime_error("SSD index count out of range");
        return static_cast<size_t>(n);
    }
    std::vector<int32_t> ids() {
        std::vector<int32_t> out(count(4));
        for (auto& value : out) {
            uint32_t n = 0; for (int j = 0; j < 4; ++j) n |= uint32_t(bytes[at++]) << (j * 8);
            value = static_cast<int32_t>(n); if (value < 0) throw std::runtime_error("SSD invalid token ID");
        }
        return out;
    }
    std::vector<ConversationImageKey> images() {
        std::vector<ConversationImageKey> out(count(16));
        for (auto& image : out) { image.start = static_cast<int64_t>(u64()); image.hash = u64(); } return out;
    }
    Ref region(uint64_t extent) {
        Ref r{u64(), u64(), u64()};
        if (r.offset % alignment || r.offset > extent || r.size > extent - r.offset || padded(r.size) > extent - r.offset)
            throw std::runtime_error("SSD region outside payload");
        return r;
    }
};
Ref append(const std::shared_ptr<File>& file, const std::vector<uint8_t>& bytes) {
    Slice slice(file, file->reserve(bytes.size()), false);
    if (!slice.visit(0, bytes.size(), true, [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, bytes.data() + at, n); return true;
        })) throw std::runtime_error("SSD state write failed");
    return slice.ref;
}
void checkpoint(Writer& w, const ConversationCheckpoint& cp, const std::shared_ptr<File>& file) {
    if (!cp.stage_parts.empty()) throw std::runtime_error("SSD checkpoint stage parts must be split first");
    w.ids(cp.ids); w.images(cp.imgs); w.u64(cp.used);
    for (const auto* b : {&cp.gdn, &cp.ple, &cp.tails, &cp.dead, &cp.block_pos}) w.region(append(file, *b));
}
void write_image(Writer& w, const SavedConversation& image, const std::shared_ptr<File>& file,
           const std::vector<const std::vector<ConversationCheckpoint>*>& overrides, size_t& ordinal) {
    for (auto n : image.geometry) w.u64(n);
    w.u64(image.layer_lo); w.u64(image.layer_hi); w.u64(image.cvec);
    checkpoint(w, image.live, file);
    const auto& checks = ordinal < overrides.size() && overrides[ordinal] ? *overrides[ordinal] : image.checkpoints;
    ++ordinal;
    w.u64(checks.size()); for (const auto& cp : checks) checkpoint(w, cp, file);
    w.u64(image.kv.size());
    for (const auto& kv : image.kv) {
        w.u64(kv.format); w.u64(kv.cells); w.u64(kv.heads); w.u64(kv.head_dim);
        w.u64(kv.page_size); w.u64(kv.pooled_rows); w.u64(kv.idx_dim);
        for (const auto* b : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) {
            if (b->empty()) { w.region(Ref{0, 0, Hash{}.finish()}); continue; }
            const auto slice = std::dynamic_pointer_cast<Slice>(b->backing());
            if (!slice || slice->file != file || !slice->complete) throw std::runtime_error("SSD K/V capture incomplete");
            w.region(slice->ref);
        }
    }
    w.u64(image.stage_images.size());
    for (const auto& stage : image.stage_images) write_image(w, stage, file, overrides, ordinal);
}

struct Decoder {
    Reader r;
    std::shared_ptr<File> file;
    bool materialize;
    uint64_t state_bytes = 0;
    std::vector<Ref> refs;
    void state(std::vector<uint8_t>& out) {
        Ref ref = r.region(file->extent); refs.push_back(ref);
        if (ref.size > state_limit || state_bytes > state_limit - ref.size) throw std::runtime_error("SSD checkpoint state exceeds 1 GiB limit");
        state_bytes += ref.size;
        if (materialize) {
            out.resize(static_cast<size_t>(ref.size));
            Slice slice(file, ref, true);
            if (!slice.visit(0, out.size(), false, [&](uint8_t* p, size_t n, size_t at) {
                    std::memcpy(out.data() + at, p, n); return true;
                })) throw std::runtime_error("SSD running-state checksum/read failed");
        }
    }
    ConversationCheckpoint checkpoint() {
        ConversationCheckpoint cp; cp.ids = r.ids(); cp.imgs = r.images(); cp.used = r.u64();
        for (auto* b : {&cp.gdn, &cp.ple, &cp.tails, &cp.dead, &cp.block_pos}) state(*b);
        return cp;
    }
    SavedConversation image(unsigned depth = 0) {
        if (depth > 1) throw std::runtime_error("SSD nested layer split unsupported");
        SavedConversation image;
        for (auto& n : image.geometry) n = static_cast<int64_t>(r.u64());
        image.layer_lo = static_cast<int64_t>(r.u64()); image.layer_hi = static_cast<int64_t>(r.u64());
        const auto cvec = r.u64(); if (cvec > 1) throw std::runtime_error("SSD invalid steering mode"); image.cvec = cvec;
        image.live = checkpoint();
        const auto checks = r.count(24, 128); for (size_t i = 0; i < checks; ++i) image.checkpoints.push_back(checkpoint());
        const auto layers = r.count(56 + 120, 256);
        for (size_t i = 0; i < layers; ++i) {
            ConversationKv kv; kv.format = static_cast<int>(r.u64()); kv.cells = static_cast<int64_t>(r.u64());
            kv.heads = static_cast<int64_t>(r.u64()); kv.head_dim = static_cast<int64_t>(r.u64());
            kv.page_size = static_cast<int64_t>(r.u64()); kv.pooled_rows = static_cast<int64_t>(r.u64()); kv.idx_dim = static_cast<int64_t>(r.u64());
            for (auto* b : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) {
                Ref ref = r.region(file->extent); refs.push_back(ref);
                if (materialize && ref.size) *b = ConversationBuffer::backed(static_cast<size_t>(ref.size), std::make_shared<Slice>(file, ref, true));
            }
            if (materialize) image.kv.push_back(std::move(kv));
        }
        const auto stages = r.count(18 * 8 + 24, 16);
        for (size_t i = 0; i < stages; ++i) image.stage_images.push_back(this->image(depth + 1));
        return image;
    }
};
std::vector<uint8_t> index_read(const std::filesystem::path& p) {
    const auto n = std::filesystem::file_size(p);
    if (n < 24 || n > 64ull * 1024 * 1024) throw std::runtime_error("SSD invalid index extent");
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    std::ifstream f(p, std::ios::binary); f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("SSD index read failed");
    Reader trailer{bytes, bytes.size() - 8}; const auto checksum = trailer.u64();
    Hash h; h.update(bytes.data(), bytes.size() - 8);
    if (checksum != h.finish()) throw std::runtime_error("SSD index checksum mismatch");
    bytes.resize(bytes.size() - 8); return bytes;
}
bool owned_name(const std::string& name) {
    if (name.size() < 20 || name.substr(0, 3) != "cc-") return false;
    for (size_t j = 3; j < 19; ++j) if (!std::isxdigit(static_cast<unsigned char>(name[j]))) return false;
    const auto ext = name.substr(19);
    return ext == ".payload" || ext == ".index" || ext == ".payload.tmp" || ext == ".index.tmp";
}
} // namespace

struct ConversationDiskCapture::Impl {
    std::shared_ptr<File> file;
    std::filesystem::path payload, index;
    size_t id = 0;
    bool committed = false;
};
ConversationDiskCapture::~ConversationDiskCapture() {
    if (!impl || impl->committed) return;
    impl->file.reset(); std::error_code ec;
    std::filesystem::remove(impl->payload, ec); std::filesystem::remove(impl->index, ec);
}
ConversationStorageFactory ConversationDiskCapture::factory() {
    const auto file = impl->file;
    return [file](size_t n) -> std::shared_ptr<ConversationStorage> {
        return std::make_shared<Slice>(file, file->reserve(n), false);
    };
}

struct ConversationDiskCache::Impl {
    struct Entry {
        size_t id;
        SavedConversation metadata;
        std::filesystem::path payload, index;
        uint64_t bytes;
        bool valid = true;
    };
    std::filesystem::path directory;
    uint64_t budget = 0, bytes = 0;
    size_t slots = 0, next = 0, pinned = 0, evictions = 0;
    std::deque<Entry> entries;
    std::string error;
    bool ready = false;
#ifdef _WIN32
    HANDLE lock = INVALID_HANDLE_VALUE;
#else
    int lock = -1;
#endif
    bool erase(size_t i) {
        auto& entry = entries.at(i); std::error_code ec;
        std::filesystem::remove(entry.payload, ec);
        if (ec) { entry.valid = false; return false; }
        std::filesystem::remove(entry.index, ec);
        if (ec) { entry.valid = false; return false; }
        bytes -= entry.bytes; entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(i));
        return true;
    }
    Entry& find(size_t id) {
        for (auto& e : entries) if (e.id == id) return e;
        throw std::runtime_error("SSD entry was evicted");
    }
    bool room(uint64_t n) {
        if (n > budget) return false;
        while (!entries.empty() && (bytes > budget - n || entries.size() >= slots)) {
            size_t i = 0; while (i < entries.size() && entries[i].id == pinned) ++i;
            if (i == entries.size()) return false;
            if (!erase(i)) return false; // Undeletable files keep their quota charge.
            ++evictions;
        }
        return bytes <= budget - n && entries.size() < slots;
    }
    SavedConversation decode(Entry& e, bool materialize, bool verify) {
        const auto index_bytes = index_read(e.index);
        auto file = std::make_shared<File>(e.payload, false, 0);
        Decoder decoder{Reader{index_bytes}, file, false};
        if (decoder.r.u64() != 0x3130304453534353ull || decoder.r.u64() != file->extent)
            throw std::runtime_error("SSD schema or payload extent mismatch");
        auto metadata = decoder.image();
        if (decoder.r.at != index_bytes.size()) throw std::runtime_error("SSD trailing index fields");
        if (verify) for (Ref ref : decoder.refs) {
            Slice slice(file, ref, true);
            if (!slice.visit(0, static_cast<size_t>(ref.size), false, [](uint8_t*, size_t, size_t) { return true; }))
                throw std::runtime_error("SSD payload checksum/read failed");
        }
        if (!materialize) return metadata;
        if (!conversation_memory_admit(conversation_available_memory(), decoder.state_bytes + index_bytes.size(), 128ull * 1024 * 1024))
            throw std::runtime_error("SSD restore has insufficient physical RAM for CPU checkpoints");
        Decoder full{Reader{index_bytes}, file, true}; full.r.u64(); full.r.u64();
        return full.image();
    }
};

ConversationDiskCache::ConversationDiskCache(const std::string& directory, uint64_t budget, size_t slots)
    : impl(std::make_unique<Impl>()) {
    impl->budget = budget; impl->slots = slots;
    if (directory.empty() || !budget || !slots) return;
    try {
        impl->directory = std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(directory.data()), directory.size()));
        if (std::filesystem::is_symlink(impl->directory)) throw std::runtime_error("SSD directory must not be a symlink");
        std::filesystem::create_directories(impl->directory);
        const auto lock_path = impl->directory / ".strata-ssd.lock";
#ifdef _WIN32
        impl->lock = CreateFileW(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (impl->lock == INVALID_HANDLE_VALUE) throw std::runtime_error("SSD directory is already in use or not writable");
#else
        impl->lock = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0600);
        if (impl->lock < 0 || flock(impl->lock, LOCK_EX | LOCK_NB)) throw std::runtime_error("SSD directory is already in use or not writable");
#endif
        // We hold the exclusive directory lease. Clean only our precisely named
        // orphan files; never import another process/model's inference state.
        for (const auto& e : std::filesystem::directory_iterator(impl->directory))
            if (e.is_regular_file() && !e.is_symlink() && owned_name(e.path().filename().string())) std::filesystem::remove(e.path());
        impl->ready = true;
    } catch (const std::exception& e) { impl->error = e.what(); }
}
ConversationDiskCache::~ConversationDiskCache() {
    if (!impl) return;
    while (!impl->entries.empty()) if (!impl->erase(0)) impl->entries.pop_front();
#ifdef _WIN32
    if (impl->lock != INVALID_HANDLE_VALUE) CloseHandle(impl->lock);
#else
    if (impl->lock >= 0) ::close(impl->lock);
#endif
}
bool ConversationDiskCache::enabled() const { return impl->ready; }
uint64_t ConversationDiskCache::bytes() const { return impl->bytes; }
size_t ConversationDiskCache::size() const { return impl->entries.size(); }
size_t ConversationDiskCache::evictions() const { return impl->evictions; }
const std::string& ConversationDiskCache::error() const { return impl->error; }
template<class Token> ConversationDiskCache::Match disk_best(const ConversationDiskCache::Impl& cache,
        const std::vector<Token>& ids, const std::vector<ConversationImageKey>& images, bool cvec) {
    ConversationDiskCache::Match best;
    for (size_t j = cache.entries.size(); j-- > 0;) {
        const auto& e = cache.entries[j]; if (!e.valid || e.metadata.cvec != cvec) continue;
        auto consider = [&](const ConversationCheckpoint& cp, bool live) {
            const auto n = conversation_prefix(cp, ids, images);
            if (n > best.tokens) best = {e.id, n, live};
        };
        consider(e.metadata.live, true);
        for (const auto& cp : e.metadata.checkpoints) consider(cp, false);
    }
    return best;
}
ConversationDiskCache::Match ConversationDiskCache::best(const std::vector<int32_t>& ids,
        const std::vector<ConversationImageKey>& images, bool cvec) const { return disk_best(*impl, ids, images, cvec); }
ConversationDiskCache::Match ConversationDiskCache::best(const std::vector<int64_t>& ids,
        const std::vector<ConversationImageKey>& images, bool cvec) const { return disk_best(*impl, ids, images, cvec); }
void ConversationDiskCache::pin(size_t id) { impl->pinned = id; }
void ConversationDiskCache::unpin() { impl->pinned = 0; }
void ConversationDiskCache::discard(size_t id) {
    for (size_t j = 0; j < impl->entries.size(); ++j) if (impl->entries[j].id == id) { impl->erase(j); break; }
    if (impl->pinned == id) impl->pinned = 0;
}
std::unique_ptr<ConversationDiskCapture> ConversationDiskCache::begin(uint64_t estimate, std::string& error) {
    try {
        if (!enabled()) return {};
        if (estimate > UINT64_MAX - chunk_bytes || !impl->room(estimate + chunk_bytes))
            throw std::runtime_error("SSD snapshot exceeds remaining cache budget");
        const auto free = std::filesystem::space(impl->directory).available;
        if (free < disk_floor || estimate + chunk_bytes > free - disk_floor) throw std::runtime_error("SSD cache requires 2 GiB disk headroom");
        auto capture = std::make_unique<ConversationDiskCapture>(); capture->impl = std::make_unique<ConversationDiskCapture::Impl>();
        auto& c = *capture->impl; c.id = ++impl->next;
        std::ostringstream name; name << "cc-" << std::hex << std::setw(16) << std::setfill('0') << c.id;
        c.payload = impl->directory / (name.str() + ".payload.tmp"); c.index = impl->directory / (name.str() + ".index.tmp");
        c.file = std::make_shared<File>(c.payload, true, impl->budget - impl->bytes - chunk_bytes);
        return capture;
    } catch (const std::exception& e) { error = e.what(); return {}; }
}
bool ConversationDiskCache::put(ConversationDiskCapture& capture, const SavedConversation& snapshot,
        const std::vector<const std::vector<ConversationCheckpoint>*>& checks, std::string& error) {
    try {
        auto& c = *capture.impl;
        Writer w; w.u64(0x3130304453534353ull); w.u64(0);
        size_t ordinal = 0; write_image(w, snapshot, c.file, checks, ordinal);
        const auto extent = c.file->extent;
        for (int j = 0; j < 8; ++j) w.bytes[8 + j] = uint8_t(extent >> (j * 8));
        Hash hash; hash.update(w.bytes.data(), w.bytes.size()); w.u64(hash.finish());
        if (w.bytes.size() > 64ull * 1024 * 1024 || impl->bytes > impl->budget || extent > impl->budget - impl->bytes ||
            w.bytes.size() > impl->budget - impl->bytes - extent)
            throw std::runtime_error("SSD index exceeds cache reservation");
        { std::ofstream out(c.index, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<const char*>(w.bytes.data()), w.bytes.size());
          out.flush(); if (!out) throw std::runtime_error("SSD index write failed"); }
        if (!c.file->flush()) throw std::runtime_error("SSD payload flush failed");
        auto payload = c.payload; payload.replace_extension(); // remove .tmp
        auto index = c.index; index.replace_extension();
        std::filesystem::rename(c.payload, payload);
        try { std::filesystem::rename(c.index, index); }
        catch (...) { std::error_code ec; std::filesystem::remove(payload, ec); throw; }
        Impl::Entry entry{c.id, {}, payload, index, extent + w.bytes.size()};
        try { entry.metadata = impl->decode(entry, false, false); }
        catch (...) { std::error_code ec; std::filesystem::remove(payload, ec); std::filesystem::remove(index, ec); throw; }
        try { impl->entries.push_back(std::move(entry)); }
        catch (...) { std::error_code ec; std::filesystem::remove(payload, ec); std::filesystem::remove(index, ec); throw; }
        impl->bytes += extent + w.bytes.size(); c.committed = true;
        // Superseded branches disappear only after their replacement committed.
        auto held = [&](const ConversationCheckpoint& cp) {
            const auto& metadata = impl->entries.back().metadata;
            if (cp.ids == metadata.live.ids && cp.imgs == metadata.live.imgs) return true;
            for (const auto& k : metadata.checkpoints) if (cp.ids == k.ids && cp.imgs == k.imgs) return true;
            return false;
        };
        for (size_t j = 0; j + 1 < impl->entries.size();) {
            const auto& old = impl->entries[j]; const ConversationCheckpoint* deepest = nullptr;
            for (const auto& cp : old.metadata.checkpoints) if (!deepest || cp.ids.size() > deepest->ids.size()) deepest = &cp;
            if (old.id != impl->pinned && old.metadata.cvec == impl->entries.back().metadata.cvec && deepest && !deepest->ids.empty() && held(*deepest)) {
                if (!impl->erase(j)) ++j;
            }
            else ++j;
        }
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
bool ConversationDiskCache::verify(size_t id, std::string& error) {
    try { impl->decode(impl->find(id), false, true); return true; }
    catch (const std::exception& e) { error = e.what(); return false; }
}
bool ConversationDiskCache::load(size_t id, SavedConversation& snapshot, std::string& error) {
    try { snapshot = impl->decode(impl->find(id), true, false); return true; }
    catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace strata::core
