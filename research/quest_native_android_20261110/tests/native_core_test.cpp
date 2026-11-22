#include "decode.h"
#include "sha256.h"
#include "strict_topk.h"
#include "residual48.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void Near(float actual, float expected, float tolerance, const char* message) {
    if (std::abs(actual - expected) > tolerance) throw std::runtime_error(message);
}
}

int main() {
    const char* abc = "abc";
    Check(mycartgs::Sha256Hex(reinterpret_cast<const uint8_t*>(abc), 3) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA256 known vector failed");

    const std::vector<float> scores{0.4f, 0.9f, 0.9f, 0.1f};
    const std::vector<uint64_t> uids{20, 30, 10, 5};
    const std::vector<uint8_t> eligible{1, 1, 1, 1};
    Check(mycartgs::StrictTopK(scores, uids, eligible, 2) == std::vector<uint32_t>({1, 2}),
          "StrictTopK tie policy failed");
    Check(mycartgs::StrictTopK(scores, uids, eligible, 1) == std::vector<uint32_t>({2}),
          "StrictTopK UID tie breaker failed");
    Check(mycartgs::StrictTopK(scores, uids, eligible, 0).empty(), "StrictTopK K=0 failed");
    Check(mycartgs::StrictTopK(scores, uids, eligible, 99).size() == 4, "StrictTopK clamp failed");

    const std::array<float, 6> knots{.01f, .05f, .1f, .3f, .5f, .7f};
    const auto weights = mycartgs::Residual48Weights(.2f, knots);
    Near(weights[0], 1.f, 1e-7f, "Residual48 base failed");
    Near(weights[1], 1.f, 1e-7f, "Residual48 level1 failed");
    Near(weights[2], 1.f, 1e-7f, "Residual48 level2 failed");
    Near(weights[3], .5f, 2e-7f, "Residual48 interpolation failed");
    Near(weights[4], 0.f, 1e-7f, "Residual48 inactive level failed");
    const auto ratioCode = mycartgs::ContinuousRatioCode(.01f, .01f, .7f);
    Near(ratioCode[0], -1.f, 1e-6f, "Continuous ratio code failed");

    mycartgs::MasterAttribute master;
    master.xyz = {1.f, -2.f, .5f}; master.scale = {.2f, .4f, .8f}; master.rotation = {1.f, 0.f, 0.f, 0.f};
    master.opacity_logit = -.4f; master.dc = {.1f, -.2f, .3f};
    const std::array<float, 14> residual{.01f, -.02f, .03f, .1f, -.2f, .3f, .02f, .03f, -.04f, .01f,
                                         .15f, -.01f, .02f, -.03f};
    const auto decoded = mycartgs::DecodeResidual(master, residual, true);
    Near(decoded.xyz[2], .53f, 1e-7f, "XYZ decode failed");
    Near(decoded.scale[0], .2128728323f, 2e-7f, "Scale decode failed");
    Near(decoded.scale[2], .9598363595f, 2e-7f, "Scale decode failed");
    Near(decoded.rotation[0], .9987528177f, 2e-7f, "Rotation decode failed");
    Near(decoded.opacity, .4378234991f, 2e-7f, "Opacity decode failed");
    Near(decoded.dc[1], -.18f, 1e-7f, "DC decode failed");
    auto extreme=residual;extreme[3]=100.f;extreme[4]=-200.f;
    const auto stable=mycartgs::DecodeResidual(master,extreme,true);
    Near(stable.scale[0],master.scale[0]*(100.f+.5413248546129181f),1e-6f,"Softplus linear threshold failed");
    Check(stable.scale[1]>0 && std::isfinite(stable.scale[1]),"Softplus underflow clamp failed");
    bool invalidResidual=false;
    try {extreme[0]=NAN;(void)mycartgs::DecodeResidual(master,extreme,true);}
    catch(const std::runtime_error&){invalidResidual=true;}
    Check(invalidResidual,"Nonfinite compensation residual was accepted");

    bool rejected = false;
    try {
        std::vector<float> bad = scores; bad[0] = NAN;
        (void)mycartgs::StrictTopK(bad, uids, eligible, 1);
    } catch (const std::invalid_argument&) { rejected = true; }
    Check(rejected, "Nonfinite Selector score was not rejected");
    std::cout << "native_core_test: PASS\n";
}
