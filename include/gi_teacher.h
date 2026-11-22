#pragma once
#include <torch/torch.h>
#include <cstdint>

struct GITeacher {
  torch::Tensor label, supervised_mask, weight;
  std::int64_t target_k = 0, supervised_k = 0;
  float ratio = 0;
  std::uint64_t gi_version = 0, topology_version = 0;
  bool valid() const { return label.defined() && weight.defined(); }
  torch::Tensor loss(const torch::Tensor& prediction, bool smooth_l1) const {
    const auto difference = (prediction - label).abs();
    const auto per_row = smooth_l1
        ? torch::where(difference < 1, .5 * difference.square(), difference - .5)
        : difference;
    return (weight * per_row).sum() / (weight.sum() + 1e-8);
  }
};

class GITeacherBuilder {
 public:
  GITeacher build(const torch::Tensor& importance, const torch::Tensor& valid_mask,
      const torch::Tensor& observed_mask, const torch::Tensor& maturity, float ratio,
      std::uint64_t gi_version, std::uint64_t topology_version = 0, float entry = 0) const;
};
