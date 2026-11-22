#pragma once

#include <torch/torch.h>
#include <vector>

// Both paths return ascending global indices. Dense unions avoid sorting
// repeated view populations; sparse unions avoid allocating a full-map mask.
inline torch::Tensor historicalPopulationUnion(
    const std::vector<torch::Tensor>& populations, int64_t population_count) {
  TORCH_CHECK(!populations.empty(), "Historical union needs a population");
  int64_t joined_count = 0;
  for (const auto& ids : populations) joined_count += ids.numel();
  if (joined_count >= population_count) {
    auto mask = torch::zeros({population_count},
        populations.front().options().dtype(torch::kBool));
    for (const auto& ids : populations) mask.index_fill_(0, ids, true);
    return torch::nonzero(mask).reshape({-1});
  }
  auto joined = std::get<0>(torch::cat(populations).to(torch::kLong).sort());
  if (joined.numel() <= 1) return joined;
  auto keep = torch::ones({joined.numel()}, joined.options().dtype(torch::kBool));
  keep.slice(0, 1).copy_(
      joined.slice(0, 1) != joined.slice(0, 0, joined.numel() - 1));
  return joined.index({keep});
}
