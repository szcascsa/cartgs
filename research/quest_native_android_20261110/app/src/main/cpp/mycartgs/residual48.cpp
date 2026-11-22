#include "residual48.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mycartgs {

std::array<float, 3> ContinuousRatioCode(float ratio, float minimumRatio, float maximumRatio) {
    if (!std::isfinite(ratio) || ratio <= 0 || ratio > 1 || minimumRatio <= 0 || maximumRatio <= minimumRatio)
        throw std::invalid_argument("Invalid continuous ratio encoding range");
    const float t = 2.f * (std::log(ratio) - std::log(minimumRatio)) /
                        (std::log(maximumRatio) - std::log(minimumRatio)) - 1.f;
    constexpr float kHalfPi = 1.57079632679489661923f;
    return {t, std::sin(kHalfPi * t), std::cos(kHalfPi * t)};
}

std::array<float, 6> Residual48Weights(float ratio, const std::array<float, 6>& knots) {
    if (!std::isfinite(ratio) || ratio <= 0 || ratio > 1) throw std::invalid_argument("Invalid Residual48 ratio");
    std::array<float, 6> weights{};
    weights[0] = 1.f;
    for (size_t level = 1; level < knots.size(); ++level) {
        if (ratio <= knots[level - 1] + 1e-8f) break;
        weights[level] = std::min(1.f, (ratio - knots[level - 1]) / (knots[level] - knots[level - 1]));
    }
    return weights;
}

void GatherResidual48(const float* const tables[6], uint32_t anchor, const std::array<float, 6>& weights,
                      float output[48]) {
    if (tables[0] == nullptr || output == nullptr) throw std::invalid_argument("Null Residual48 buffer");
    const size_t begin = size_t(anchor) * 48;
    for (size_t channel = 0; channel < 48; ++channel) output[channel] = tables[0][begin + channel];
    for (size_t level = 1; level < weights.size(); ++level) {
        if (weights[level] == 0) break;
        if (tables[level] == nullptr) throw std::invalid_argument("Null active Residual48 table");
        for (size_t channel = 0; channel < 48; ++channel)
            output[channel] += weights[level] * tables[level][begin + channel];
    }
}

}  // namespace mycartgs
