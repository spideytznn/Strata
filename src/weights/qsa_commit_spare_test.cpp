#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <cstdio>
#include <stdexcept>
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
 try {
    constexpr int D=128,C=64,ROWS=C/4+1;
    float *pooled=nullptr,*dead=nullptr;int32_t *pos=nullptr,*commit=nullptr;
    check(cudaMalloc(&pooled,ROWS*D*4));check(cudaMalloc(&dead,D*4));
    check(cudaMalloc(&pos,4));check(cudaMalloc(&commit,10*4));
    std::vector<float> input(ROWS*D),output(input.size());std::array<float,D> spare;
    for(int d=0;d<D;++d)spare[d]=float(-d-1);
    check(cudaMemcpy(dead,spare.data(),D*4,cudaMemcpyHostToDevice));
    int cases=0;
    for(int start=0;start<C;++start)for(int keep=1;keep<=8 && start+keep<=C;++keep) {
       for(size_t i=0;i<input.size();++i)input[i]=float(i+100);
       std::array<int32_t,10> c; c.fill(-1);c[0]=keep;c[1]=keep-1;
       for(int j=0;j<keep;++j)c[j+2]=start+j;
       check(cudaMemcpy(pooled,input.data(),input.size()*4,cudaMemcpyHostToDevice));
       check(cudaMemcpy(commit,c.data(),c.size()*4,cudaMemcpyHostToDevice));
       strata::kernels::qsa_commit_spare(pooled,dead,pos,commit,D,4,C,nullptr);
       check(cudaMemcpy(output.data(),pooled,output.size()*4,cudaMemcpyDeviceToHost));
       const int row=(start+keep)/4;int32_t block;
       check(cudaMemcpy(&block,pos,4,cudaMemcpyDeviceToHost));
       if(block!=(row?row-1:0)*4)throw std::runtime_error("stale completed block position");
       for(int r=0;r<ROWS;++r)for(int d=0;d<D;++d)
          if(output[r*D+d]!=(r==row?spare[d]:input[r*D+d]))throw std::runtime_error("spare restoration modified another row");
       ++cases;
    }
    cudaFree(pooled);cudaFree(dead);cudaFree(pos);cudaFree(commit);
    std::printf("PASS: %d commit boundaries; spare restored, completed/future rows unchanged\n",cases);
    return 0;
 } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
