#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {
// A verify window commits its inputs, including the last previously emitted
// token. Its final output remains the next input. Trim accepted lookahead before
// committing so max_tokens / EOS cannot leave invisible tokens in live state.
inline int visible_accepted(int accepted,const int32_t* outputs,int64_t remaining,
                            const std::vector<int64_t>& eos,bool stop_eos) {
    accepted=static_cast<int>(std::max<int64_t>(0,std::min<int64_t>(accepted,remaining-1)));
    if(stop_eos) for(int i=0;i<accepted;++i)
        if(std::find(eos.begin(),eos.end(),int64_t(outputs[i]))!=eos.end()) return i;
    return accepted;
}
}
