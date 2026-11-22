#pragma once
#include <torch/torch.h>
#include <cmath>
#include <algorithm>

inline torch::Tensor mixSelectorProbability(const torch::Tensor& probability,
                                           const torch::Tensor& maturity, float ratio) {
  TORCH_CHECK(std::isfinite(ratio) && ratio >= 0 && ratio <= 1, "Invalid global LoD ratio");
  TORCH_CHECK(probability.dim() == 1 && probability.sizes() == maturity.sizes() &&
      probability.device() == maturity.device(), "Selector maturity is not row-aligned");
  const auto m = maturity.detach();
  return (1 - m) * ratio + m * probability;
}
inline torch::Tensor sampleSelectorST(const torch::Tensor& probability,
                                     const torch::Tensor& uniform) {
  TORCH_CHECK(probability.sizes() == uniform.sizes() && probability.device() == uniform.device(),
      "Selector uniforms are not row-aligned");
  const auto hard = (uniform < probability).to(probability.scalar_type());
  // Parenthesize the zero-valued surrogate to keep forward values exactly 0/1.
  return hard + (probability - probability.detach());
}
inline torch::Tensor selectorRatioLoss(const torch::Tensor& probability, float ratio) {
  const double target = static_cast<double>(ratio) * probability.numel();
  return torch::abs(probability.sum() - target) / std::max(1.0, target);
}
