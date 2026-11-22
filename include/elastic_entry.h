#pragma once
#include "selector_handoff.h"

inline torch::Tensor elasticMaturity(const torch::Tensor& maturity, float entry) {
  TORCH_CHECK(std::isfinite(entry) && entry >= 0 && entry < 1, "Elastic.entry_maturity must be in [0,1)");
  // Preserve the original numerical path when disabled.
  if (entry == 0) return maturity.detach();
  return ((maturity.detach() - entry) / (1 - entry)).clamp(0, 1);
}

inline torch::Tensor elasticDomainMask(const torch::Tensor& xyz,
    const torch::Tensor& maturity, float threshold) {
  auto finite = torch::isfinite(xyz).all(1);
  return threshold == 0 ? finite : finite & (maturity.detach() >= threshold);
}

// A callback receives undefined indices on the unchanged all-row path.
// Otherwise only eligible compact rows may be forwarded through the MLP.
template <typename Forward>
torch::Tensor elasticSelectorProbability(const torch::Tensor& maturity,
    const torch::Tensor& legacy_selection_maturity, float entry, float ratio,
    const torch::Tensor& eligible_indices, Forward&& forward) {
  if (entry == 0)
    return mixSelectorProbability(forward(torch::Tensor{}), legacy_selection_maturity, ratio);
  auto probability = torch::full_like(maturity, ratio, maturity.options().requires_grad(false));
  if (!eligible_indices.numel()) return probability;
  const auto weight = elasticMaturity(maturity.index_select(0, eligible_indices), entry);
  return probability.index_copy(0, eligible_indices,
      mixSelectorProbability(forward(eligible_indices), weight, ratio));
}
