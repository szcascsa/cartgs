#include "strict_topk.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace mycartgs {
namespace {
bool Better(const ScoreKey& a, const ScoreKey& b) {
    return a.score > b.score || (a.score == b.score && (a.uid < b.uid || (a.uid == b.uid && a.row < b.row)));
}
std::vector<ScoreKey> Keys(const std::vector<float>& score, const std::vector<uint64_t>& uid,
                           const std::vector<uint8_t>& eligible) {
    if (score.size() != uid.size() || score.size() != eligible.size()) throw std::invalid_argument("TopK shape mismatch");
    std::vector<ScoreKey> keys; keys.reserve(score.size());
    for (uint32_t i=0;i<score.size();++i) {
        if (!std::isfinite(score[i])) throw std::invalid_argument("Nonfinite Selector score");
        if (eligible[i]) keys.push_back({score[i],uid[i],i});
    }
    return keys;
}
}

std::vector<uint32_t> StrictTopK(const std::vector<float>& score, const std::vector<uint64_t>& uid,
                                 const std::vector<uint8_t>& eligible, uint32_t k) {
    auto keys=Keys(score,uid,eligible); k=std::min<uint32_t>(k,uint32_t(keys.size()));
    if (k && k<keys.size()) std::nth_element(keys.begin(),keys.begin()+k,keys.end(),Better);
    keys.resize(k); std::vector<uint32_t> rows; rows.reserve(k);
    for (auto& key:keys) rows.push_back(key.row);
    std::sort(rows.begin(),rows.end()); // Renderer/master row order is preserved after selection.
    return rows;
}

void StrictTopKInto(const std::vector<float>& score,const std::vector<uint64_t>& uid,
                   const std::vector<uint8_t>& eligible,uint32_t k,
                   std::vector<ScoreKey>& keys,std::vector<uint32_t>& rows) {
    if(score.size()!=uid.size()||score.size()!=eligible.size())throw std::invalid_argument("TopK shape mismatch");
    keys.clear();
    for(uint32_t i=0;i<score.size();++i){
        if(!std::isfinite(score[i]))throw std::invalid_argument("Nonfinite Selector score");
        if(eligible[i])keys.push_back({score[i],uid[i],i});
    }
    k=std::min<uint32_t>(k,uint32_t(keys.size()));
    if(k&&k<keys.size())std::nth_element(keys.begin(),keys.begin()+k,keys.end(),Better);
    rows.resize(k);for(uint32_t i=0;i<k;++i)rows[i]=keys[i].row;
    std::sort(rows.begin(),rows.end());
}

void StrictTopKHistogram(const std::vector<float>& score,const std::vector<uint64_t>& uid,
                        const std::vector<uint8_t>& eligible,uint32_t k,
                        std::vector<ScoreKey>& keys,std::vector<uint32_t>& rows) {
    if(score.size()!=uid.size()||score.size()!=eligible.size())throw std::invalid_argument("TopK shape mismatch");
    if(score.size()<2048){StrictTopKInto(score,uid,eligible,k,keys,rows);return;}
    std::array<uint32_t,256> histogram{};uint32_t available=0;bool probabilities=true;
    const auto bucket=[](float value){return std::min(255u,uint32_t(value*256));};
    for(uint32_t i=0;i<score.size();++i){
        if(!std::isfinite(score[i]))throw std::invalid_argument("Nonfinite Selector score");
        if(score[i]<0||score[i]>1){probabilities=false;continue;}
        if(eligible[i]){++histogram[bucket(score[i])];++available;}
    }
    if(!probabilities){StrictTopKInto(score,uid,eligible,k,keys,rows);return;}
    k=std::min(k,available);rows.clear();if(!k)return;
    if(k==available){for(uint32_t i=0;i<score.size();++i)if(eligible[i])rows.push_back(i);return;}
    uint32_t above=0,boundary=255;
    while(above+histogram[boundary]<k){above+=histogram[boundary];--boundary;}
    // Buckets identify only the boundary workload. Original FP32 scores and full
    // score/UID/row comparisons still determine the exact selected identities.
    if(histogram[boundary]>available/4){StrictTopKInto(score,uid,eligible,k,keys,rows);return;}
    keys.clear();for(uint32_t i=0;i<score.size();++i)if(eligible[i]&&bucket(score[i])==boundary)keys.push_back({score[i],uid[i],i});
    const uint32_t rank=k-above-1;
    std::nth_element(keys.begin(),keys.begin()+rank,keys.end(),Better);const auto cutoff=keys[rank];
    for(uint32_t i=0;i<score.size();++i)if(eligible[i]&&!Better(cutoff,{score[i],uid[i],i}))rows.push_back(i);
    if(rows.size()!=k)throw std::runtime_error("Histogram strict K violated");
}

std::vector<uint32_t> GlobalRatioSelect(const std::vector<float>& score, const std::vector<uint64_t>& uid,
                                        const std::vector<uint8_t>& eligible, float ratio) {
    if (!(ratio>0 && ratio<=1) || !std::isfinite(ratio)) throw std::invalid_argument("Invalid global ratio");
    auto keys=Keys(score,uid,eligible); if (keys.empty()) return {};
    const uint32_t rank=uint32_t((1-ratio)*(keys.size()-1));
    std::nth_element(keys.begin(),keys.begin()+rank,keys.end(),[](const ScoreKey&a,const ScoreKey&b){return a.score<b.score;});
    const float threshold=keys[rank].score; std::vector<uint32_t> rows;
    for (uint32_t i=0;i<score.size();++i) if (eligible[i] && score[i]>=threshold) rows.push_back(i);
    return rows; // FlexGS threshold ties are retained, so this is intentionally not strict K.
}
}
