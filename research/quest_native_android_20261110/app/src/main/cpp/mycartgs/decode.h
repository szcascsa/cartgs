#pragma once
#include <array>

namespace mycartgs {
struct MasterAttribute {
    std::array<float,3> xyz, scale, dc;
    std::array<float,4> rotation;
    float opacity_logit=0, reference=1;
};
struct DecodedAttribute {
    std::array<float,3> xyz, scale, dc;
    std::array<float,4> rotation;
    float opacity=0;
};
DecodedAttribute DecodeResidual(const MasterAttribute& master, const std::array<float,14>& residual, bool softplusScale);
}
