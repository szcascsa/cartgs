#pragma once

#include <torch/torch.h>

namespace quaternion_utils {

inline torch::Tensor normalize(const torch::Tensor& quaternion) {
  auto norm = torch::sqrt((quaternion * quaternion).sum(
      /*dim=*/1, /*keepdim=*/true));
  return quaternion / torch::clamp_min(norm, 1e-12);
}

inline torch::Tensor multiply(const torch::Tensor& lhs,
                              const torch::Tensor& rhs) {
  auto scalar = lhs.slice(1, 0, 1) * rhs.slice(1, 0, 1) -
                (lhs.slice(1, 1, 4) * rhs.slice(1, 1, 4))
                    .sum(/*dim=*/1, /*keepdim=*/true);
  auto vector = lhs.slice(1, 0, 1) * rhs.slice(1, 1, 4) +
                rhs.slice(1, 0, 1) * lhs.slice(1, 1, 4) +
                torch::cross(lhs.slice(1, 1, 4), rhs.slice(1, 1, 4), 1);
  return torch::cat({scalar, vector}, /*dim=*/1);
}

}  // namespace quaternion_utils
