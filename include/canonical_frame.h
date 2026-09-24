#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace canonical_frame {

inline torch::Tensor normalizeQuaternion(const torch::Tensor& quaternion) {
  auto norm = torch::sqrt((quaternion * quaternion).sum(-1, true));
  return quaternion / torch::clamp_min(norm, 1e-12);
}

inline torch::Tensor conjugateQuaternion(const torch::Tensor& quaternion) {
  auto conjugated = quaternion.clone();
  conjugated.index_put_({torch::indexing::Slice(),
                         torch::indexing::Slice(1, 4)},
                        -quaternion.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(1, 4)}));
  return conjugated;
}

inline torch::Tensor multiplyQuaternion(const torch::Tensor& lhs,
                                        const torch::Tensor& rhs) {
  auto lhs_w = lhs.index({torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)});
  auto lhs_v = lhs.index({torch::indexing::Slice(),
                          torch::indexing::Slice(1, 4)});
  auto rhs_w = rhs.index({torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)});
  auto rhs_v = rhs.index({torch::indexing::Slice(),
                          torch::indexing::Slice(1, 4)});
  auto scalar = lhs_w * rhs_w - (lhs_v * rhs_v).sum(-1, true);
  auto vector = lhs_w * rhs_v + rhs_w * lhs_v +
                torch::cross(lhs_v, rhs_v, /*dim=*/-1);
  return torch::cat({scalar, vector}, /*dim=*/-1);
}

inline torch::Tensor rotate(const torch::Tensor& quaternion,
                            const torch::Tensor& vectors) {
  auto normalized = normalizeQuaternion(quaternion);
  auto scalar = normalized.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(0, 1)});
  auto vector = normalized.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(1, 4)});
  auto twice_cross = 2.0 * torch::cross(vector, vectors, /*dim=*/-1);
  return vectors + scalar * twice_cross +
         torch::cross(vector, twice_cross, /*dim=*/-1);
}

inline torch::Tensor worldToCanonical(
    const torch::Tensor& world_xyz,
    const torch::Tensor& frame_scale,
    const torch::Tensor& frame_rotation,
    const torch::Tensor& frame_translation) {
  auto centered = world_xyz - frame_translation;
  return rotate(conjugateQuaternion(normalizeQuaternion(frame_rotation)),
                centered) /
         torch::clamp_min(frame_scale, 1e-12);
}

inline torch::Tensor canonicalToWorld(
    const torch::Tensor& canonical_xyz,
    const torch::Tensor& frame_scale,
    const torch::Tensor& frame_rotation,
    const torch::Tensor& frame_translation) {
  return frame_scale * rotate(frame_rotation, canonical_xyz) +
         frame_translation;
}

inline torch::Tensor worldToCanonicalRotation(
    const torch::Tensor& world_rotation,
    const torch::Tensor& frame_rotation) {
  return normalizeQuaternion(multiplyQuaternion(
      conjugateQuaternion(normalizeQuaternion(frame_rotation)),
      normalizeQuaternion(world_rotation)));
}

inline torch::Tensor canonicalToWorldRotation(
    const torch::Tensor& canonical_rotation,
    const torch::Tensor& frame_rotation) {
  return normalizeQuaternion(multiplyQuaternion(
      normalizeQuaternion(frame_rotation),
      normalizeQuaternion(canonical_rotation)));
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compose(
    const torch::Tensor& frame_scale,
    const torch::Tensor& frame_rotation,
    const torch::Tensor& frame_translation,
    const torch::Tensor& delta_scale,
    const torch::Tensor& delta_rotation,
    const torch::Tensor& delta_translation) {
  auto normalized_delta_rotation = normalizeQuaternion(delta_rotation);
  auto composed_scale = delta_scale * frame_scale;
  auto composed_rotation = normalizeQuaternion(
      multiplyQuaternion(normalized_delta_rotation, frame_rotation));
  auto composed_translation =
      delta_scale * rotate(normalized_delta_rotation, frame_translation) +
      delta_translation;
  return {composed_scale, composed_rotation, composed_translation};
}

inline torch::Tensor identityScale(std::int64_t count,
                                   const torch::TensorOptions& options) {
  return torch::ones({count, 1}, options.requires_grad(false));
}

inline torch::Tensor identityRotation(std::int64_t count,
                                      const torch::TensorOptions& options) {
  auto rotation = torch::zeros({count, 4}, options.requires_grad(false));
  if (count > 0)
    rotation.index_put_({torch::indexing::Slice(), 0}, 1.0f);
  return rotation;
}

inline torch::Tensor identityTranslation(std::int64_t count,
                                         const torch::TensorOptions& options) {
  return torch::zeros({count, 3}, options.requires_grad(false));
}

}  // namespace canonical_frame
