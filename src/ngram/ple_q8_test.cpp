// Compare every decoded value with a ggml-generated oracle, through all table APIs.
#include "strata/kernels/ngram.hpp"
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <stdexcept>
using namespace strata::kernels;
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
    if (!f || f.tellg() <= 0 || (size_t) f.tellg() % (160 * sizeof(float))) return 2;
    std::vector<float> expected((size_t) f.tellg() / sizeof(float));
    f.seekg(0); f.read((char*) expected.data(), expected.size() * sizeof(float));
    const uint32_t n = (uint32_t) (expected.size() / 160);
    auto check = [&](uint32_t row, const float* got) {
        float zeros[160] = {};
        const float* want = row < n ? expected.data() + (size_t) row * 160 : zeros;
        if (std::memcmp(got, want, 160 * sizeof(float)))
            throw std::runtime_error("oracle mismatch at row " + std::to_string(row));
    };
    try {
        for (auto mode : {PleIo::Mmap, PleIo::Direct})
        for (bool worker : {false, true})
        for (uint64_t cache : {0ull, 256ull}) {
            PleTable table; PleIoOptions opts; std::string err;
            opts.mode = mode; opts.io_thread = worker; opts.cache_rows = cache;
            if (!table.open(argv[1], err, opts)) throw std::runtime_error(err);
            if (table.rows() != n || table.row_bytes() != 170 || std::strcmp(table.type_name(), "Q8_0"))
                throw std::runtime_error("wrong geometry/type");
            float out[2560];
            for (uint32_t row = 0; row < n; ++row) { table.read_row(row, out); check(row, out); }
            table.read_row(n, out); check(n, out);
            uint32_t rows[16] = {0,1,23,24,n-1,n,0xffffffffu,1,48,49,72,73,96,97,0,1};
            for (int pass = 0; pass < 3; ++pass) {
                if (!table.issue(rows) || !table.collect(out, err)) throw std::runtime_error(err);
                for (int i = 0; i < 16; ++i) check(rows[i], out + 160*i);
                table.gather(rows,out);
                for (int i = 0; i < 16; ++i) check(rows[i], out + 160*i);
            }
            std::vector<uint32_t> batch(16*1024);
            for (size_t i=0; i<batch.size(); ++i) batch[i]=(uint32_t) ((i*499)%(n+1));
            std::vector<float> result(batch.size()*160);
            if (!table.gather_batch(batch.data(),batch.size()/16,result.data(),err)) throw std::runtime_error(err);
            for (size_t i=0; i<batch.size(); ++i) check(batch[i],result.data()+i*160);
            table.close();
            if (!table.open(argv[1],err,opts)) throw std::runtime_error(err);
            table.read_row(n-1,out); check(n-1,out);
        }
        std::puts("Q8_0 PleTable: ggml oracle bit-identical for mmap/direct, sync/worker, cached/uncached, row/split/gather/batch/reopen.");
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
    return 0;
}
