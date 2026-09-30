#pragma once
// Only for the CPU loader test. Production include paths never contain this
// directory. It exercises the real loader using host allocations and copies.
#include <cstdlib>
#include <cstring>
constexpr int cudaSuccess=0, cudaHostAllocDefault=0, cudaMemcpyHostToDevice=1;
inline int cudaHostAlloc(void** p,size_t n,unsigned) { *p=std::malloc(n);return *p?0:1; }
inline int cudaFreeHost(void* p) { std::free(p);return 0; }
inline int cudaMemcpy(void* dst,const void* src,size_t n,int) { std::memcpy(dst,src,n);return 0; }
inline int cudaDeviceSynchronize() { return 0; }
