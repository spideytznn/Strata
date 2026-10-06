#include "strata/core/conversation_disk.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <chrono>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool good, const char* message) {
    ++checks; if (!good) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
constexpr size_t mib = 1024 * 1024;
uint8_t value(size_t at, int salt) { return uint8_t(at * 7 + at / 257 + salt); }
SavedConversation make(ConversationDiskCapture& capture, int salt, size_t bytes) {
    auto factory = capture.factory(); ConversationStorageScope scope(factory);
    SavedConversation image; image.geometry.fill(17); image.layer_hi = 4;
    image.live.ids = {salt, 2, 3, 4}; image.live.imgs = {{1, 789}};
    image.live.gdn.assign(997, uint8_t(salt)); image.live.ple.assign(501, 72);
    image.kv.resize(1); auto& kv = image.kv[0]; kv.cells = 4; kv.heads = 1; kv.head_dim = 64; kv.page_size = 4;
    kv.k.resize(bytes);
    check(kv.k.external() && kv.k.bytes() < 1024, "capture allocates no payload-sized RAM buffer");
    check(kv.k.visit(0, bytes, [&](uint8_t* p, size_t n, size_t at) {
        check(n <= 8 * mib, "disk transfer stays within 8 MiB");
        for (size_t i = 0; i < n; ++i) p[i] = value(at + i, salt); return true;
    }), "write non-aligned payload with multiple chunks");
    return image;
}
size_t store(ConversationDiskCache& cache, int salt, size_t bytes) {
    std::string error; auto capture = cache.begin(bytes + mib, error);
    check(bool(capture), error.c_str());
    auto image = make(*capture, salt, bytes);
    ConversationCheckpoint cp; cp.ids = {salt, 2}; cp.imgs = {{1, 789}}; cp.used = 42; cp.gdn.assign(12345, 51);
    std::vector<ConversationCheckpoint> checkpoints; checkpoints.push_back(std::move(cp));
    check(cache.put(*capture, image, {&checkpoints}, error), error.c_str());
    return cache.best(std::vector<int32_t>{salt, 2, 3, 4, 5}, {{1,789}}, true).index;
}
}
int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--benchmark") {
        const auto root = std::filesystem::u8path(argv[2]);
        {
            ConversationDiskCache cache(root.string(), 2048ull * mib, 4);
            check(cache.enabled(), cache.error().c_str());
            const auto start = std::chrono::steady_clock::now();
            const auto id = store(cache, 31, 512 * mib + 37);
            const auto saved = std::chrono::steady_clock::now();
            std::string error; check(cache.verify(id,error),error.c_str());
            SavedConversation image; check(cache.load(id,image,error),error.c_str());
            const auto loaded = std::chrono::steady_clock::now();
            const auto& buffer = image.kv[0].k;
            check(buffer.visit(0,buffer.size(),[](const uint8_t* p,size_t n,size_t at) {
                return n == 0 || (p[0]==value(at,31) && p[n-1]==value(at+n-1,31));
            }),"benchmark readback integrity");
            const auto read = std::chrono::steady_clock::now();
            auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
            size_t peak=0;
#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS memory{};
            if(GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))) peak=memory.PeakWorkingSetSize;
#endif
            std::printf("SSD_BENCH payload_mib=512 write_ms=%.1f verify_and_metadata_ms=%.1f second_read_ms=%.1f peak_working_set_mib=%.1f\n",
                ms(start,saved),ms(saved,loaded),ms(loaded,read),double(peak)/mib);
        }
        std::filesystem::remove(root/".strata-ssd.lock"); std::filesystem::remove(root);
        return 0;
    }
    const auto root = std::filesystem::temp_directory_path() / ("strata-disk-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    { std::ofstream keep(root / "user-file.txt"); keep << "never delete"; }
    { std::ofstream stale(root / "cc-0000000000000088.payload.tmp"); stale << "orphan"; }
    {
        ConversationDiskCache cache(root.string(), 96 * mib, 4);
        check(cache.enabled(), cache.error().c_str());
        check(!std::filesystem::exists(root / "cc-0000000000000088.payload.tmp"), "clean exact orphan names under exclusive lease");
        check(std::filesystem::exists(root / "user-file.txt"), "unrelated user files survive cleanup");
        ConversationDiskCache second(root.string(), 96 * mib, 4);
        check(!second.enabled(), "another engine cannot share model state through same directory");
        std::string error;
        { auto failed = cache.begin(mib, error); check(bool(failed), "temporary capture admitted");
          auto image = make(*failed, 9, 511); }
        check(cache.size() == 0 && cache.bytes() == 0, "uncommitted capture not published or charged");
        const auto a = store(cache, 11, 20 * mib + 37);
        check(a != 0 && cache.size() == 1, "metadata-only index selects live prefix");
        check(cache.best(std::vector<int32_t>{11,2,3,4,5}, {{1,790}}, true).tokens == 0, "image mismatch cannot reuse");
        check(cache.best(std::vector<int32_t>{11,2,3,4,5}, {{1,789}}, false).tokens == 0, "steering mismatch cannot reuse");
        auto cp = cache.best(std::vector<int32_t>{11,2,8}, {{1,789}}, true);
        check(cp.tokens == 2 && !cp.live, "rewritten answer can select saved turn checkpoint");
        check(cache.verify(a, error), error.c_str());
        SavedConversation restored; check(cache.load(a, restored, error), error.c_str());
        check(restored.live.gdn == std::vector<uint8_t>(997, 11) && restored.live.ple == std::vector<uint8_t>(501,72), "running state round trip");
        check(restored.checkpoints[0].used == 42 && restored.checkpoints[0].gdn == std::vector<uint8_t>(12345,51), "borrowed checkpoint state and LRU round trip");
        const auto& buffer = restored.kv[0].k;
        check(buffer.external() && buffer.bytes() < 1024, "restore does not load entire K/V into RAM");
        check(buffer.visit(0, buffer.size(), [&](const uint8_t* p, size_t n, size_t at) {
            for (size_t i=0;i<n;++i) if (p[i]!=value(at+i,11)) return false; return true;
        }), "all K/V bytes round trip including final partial sector");
        std::vector<uint8_t> range(10019);
        check(buffer.read(range.data(), 4091, range.size()), "unaligned bounded read");
        bool exact=true; for (size_t i=0;i<range.size();++i) exact &= range[i]==value(i+4091,11);
        check(exact,"unaligned bytes match");
        restored = {};
        std::filesystem::path payload;
        for (const auto& p : std::filesystem::directory_iterator(root)) if (p.path().extension()==".payload") payload=p.path();
        { std::fstream corrupt(payload, std::ios::binary|std::ios::in|std::ios::out); char x=0; corrupt.read(&x,1); x^=1; corrupt.seekp(0); corrupt.write(&x,1); }
        check(!cache.verify(a,error), "corrupt K/V rejected before GPU writes");
        cache.discard(a); check(cache.size()==0 && cache.bytes()==0,"invalid entry removed");
        auto oversized=cache.begin(96*mib,error); check(!oversized,"oversized snapshot refused before capture");
        const auto b=store(cache,12,mib+7);
        std::filesystem::path index;
        for (const auto& p : std::filesystem::directory_iterator(root)) if (p.path().extension()==".index") index=p.path();
        std::filesystem::resize_file(index,17);
        check(!cache.verify(b,error),"truncated index rejected");
        cache.discard(b);
    }
    {
        ConversationDiskCache cache(root.string(),40*mib,2); check(cache.enabled(),"lease released after shutdown");
        const auto a=store(cache,21,9*mib); cache.pin(a);
        const auto b=store(cache,22,9*mib); (void)b;
        const auto c=store(cache,23,9*mib); (void)c;
        check(cache.size()==2 && cache.evictions()==1 && cache.bytes()<=40*mib,"bounded oldest-first eviction");
        check(cache.best(std::vector<int32_t>{21,2,3,4,5},{{1,789}},true).tokens==4,"incoming snapshot is protected during outgoing park");
        check(cache.best(std::vector<int32_t>{22,2,3,4,5},{{1,789}},true).tokens==0,"oldest unpinned entry evicted");
        cache.unpin();
    }
    check(std::filesystem::exists(root/"user-file.txt"),"shutdown preserves unrelated file");
    std::filesystem::remove(root/"user-file.txt"); std::filesystem::remove(root/".strata-ssd.lock"); std::filesystem::remove(root);
    std::printf("conversation_disk_test: %d checks passed\n",checks);
}
