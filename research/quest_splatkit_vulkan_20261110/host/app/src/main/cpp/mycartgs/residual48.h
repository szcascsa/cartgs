#pragma once

#include <array>
#include <cstdint>

namespace mycartgs {

std::array<float, 3> ContinuousRatioCode(float ratio, float minimumRatio, float maximumRatio);
std::array<float, 6> Residual48Weights(float ratio, const std::array<float, 6>& knots);
void GatherResidual48(const float* const tables[6], uint32_t anchor, const std::array<float, 6>& weights,
                      float output[48]);

}  // namespace mycartgs
