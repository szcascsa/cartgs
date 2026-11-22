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

inline torch::Tensor sampleSelectorGumbelST(const torch::Tensor& logits, double tau,
                                           const torch::Tensor& noise = {}) {
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) == 2 && std::isfinite(tau) && tau > 0,
              "Gumbel selection requires Nx2 logits and a positive finite temperature");
  auto gumbel = noise.defined() ? noise : -torch::empty_like(logits).exponential_().log();
  TORCH_CHECK(gumbel.sizes() == logits.sizes() && gumbel.device() == logits.device(),
              "Gumbel noise must match Selector logits");
  auto soft = ((logits + gumbel) / tau).softmax(-1);
  auto indices = std::get<1>(soft.max(-1, true));
  auto hard = torch::zeros_like(logits).scatter_(-1, indices, 1.0);
  return (hard - soft.detach() + soft).select(1, 1);
}

inline torch::Tensor selectorGumbelRatioLoss(const torch::Tensor& mask, float ratio,
                                           bool normalize_target = false) {
  // Keep the noisy ST mask; normalize by target K to retain low-ratio budget pressure.
  if (normalize_target) return selectorRatioLoss(mask, ratio);
  return mask.numel() ? torch::abs(mask.mean() - ratio) : mask.sum();
}

inline torch::Tensor selectorGumbelBudgetLoss(const torch::Tensor& mask,int64_t target,
                                             bool normalize_target) {
  TORCH_CHECK(target>=0&&target<=mask.numel(),"Invalid explicit Gumbel budget");
  return torch::abs(mask.sum()-double(target))/
      (normalize_target?std::max(1.,double(target)):std::max(1.,double(mask.numel())));
}
