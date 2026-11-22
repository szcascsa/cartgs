#pragma once

#include <torch/torch.h>
#include <cmath>
#include <cstdint>
#include "quaternion_utils.h"

enum class GaussianAttrFlag : std::uint32_t {
  NONE = 0, XYZ = 1u << 0, SCALE = 1u << 1, ROTATION = 1u << 2
};
constexpr std::uint32_t toBits(GaussianAttrFlag flag) {
  return static_cast<std::uint32_t>(flag);
}

// Scale is logarithmic; rotation is a normalized wxyz quaternion.
struct GaussianAttrState {
  torch::Tensor xyz;
  torch::Tensor log_scaling;
  torch::Tensor rotation;
  GaussianAttrState select(const torch::Tensor& indices) const {
    return {xyz.index_select(0, indices), log_scaling.index_select(0, indices),
            rotation.index_select(0, indices)};
  }
  GaussianAttrState detach() const {
    return {xyz.detach(), log_scaling.detach(), rotation.detach()};
  }
};

struct GaussianTransformResult {
  torch::Tensor indices;
  // When true, the correction applies to the complete population at the
  // moment it was produced. The indices field may remain empty to avoid an
  // unnecessary arange allocation; the coordinator resolves the snapshot
  // range when assembling an event.
  bool affects_all = false;
  int64_t population_count = 0;
  // Transposed SE3 matrix, matching the rasterizer row-vector convention.
  torch::Tensor transform_4x4;
  torch::Tensor rotation_wxyz;
  float scale = 1.0f;
  std::uint32_t changed_attributes = 0;
};

inline GaussianAttrState applyGaussianTransform(
    const GaussianAttrState& state, const GaussianTransformResult& correction) {
  TORCH_CHECK(std::isfinite(correction.scale) && correction.scale > 0.0f,
              "Gaussian correction requires a positive finite scale");
  auto result = state;
  if (correction.changed_attributes & toBits(GaussianAttrFlag::XYZ)) {
    const auto& matrix = correction.transform_4x4;
    result.xyz = torch::matmul(state.xyz * correction.scale,
                               matrix.slice(0, 0, 3).slice(1, 0, 3)) +
                 matrix.select(0, 3).slice(0, 0, 3);
  }
  if (correction.changed_attributes & toBits(GaussianAttrFlag::SCALE))
    result.log_scaling = state.log_scaling + std::log(correction.scale);
  if (correction.changed_attributes & toBits(GaussianAttrFlag::ROTATION))
    result.rotation = quaternion_utils::normalize(quaternion_utils::multiply(
        correction.rotation_wxyz.reshape({1, 4}).expand({state.xyz.size(0), 4}),
        state.rotation));
  return result;
}
