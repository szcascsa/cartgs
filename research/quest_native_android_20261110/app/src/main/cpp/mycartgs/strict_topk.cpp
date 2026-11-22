#include "strict_topk.h"
#include <algorithm>
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
