#pragma once
#include <cstdint>
#include <vector>

namespace mycartgs {
struct ScoreKey { float score; uint64_t uid; uint32_t row; };
std::vector<uint32_t> StrictTopK(const std::vector<float>& scores, const std::vector<uint64_t>& uids,
                                 const std::vector<uint8_t>& eligible, uint32_t k);
std::vector<uint32_t> GlobalRatioSelect(const std::vector<float>& scores, const std::vector<uint64_t>& uids,
                                        const std::vector<uint8_t>& eligible, float ratio);
}
