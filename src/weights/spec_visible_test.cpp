#include "strata/core/spec_visible.hpp"
#include <cstdio>
int main() {
    const int32_t tokens[]={10,20,30,40,50,60,70,80};
    for(int accepted=0;accepted<8;++accepted) for(int remaining=1;remaining<12;++remaining)
        for(int eos=-1;eos<8;++eos) {
            std::vector<int64_t> stops;
            if(eos>=0) stops.push_back(tokens[eos]);
            int emitted=0;
            while(emitted<=accepted && emitted<remaining) {
                ++emitted;
                if(eos==emitted-1) break;
            }
            if(strata::core::visible_accepted(accepted,tokens,remaining,stops,true)+1!=emitted) return 1;
        }
    if(strata::core::visible_accepted(7,tokens,8,{10},false)!=7) return 1;
    std::puts("PASS: 792 max_tokens/EOS acceptance boundaries");
}
