/*
 * Global selector supervision derived from persistent Online GI statistics.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>

struct GITeacher {
  // [N] float labels. Protected and otherwise unsupervised entries are zero,
  // and must be ignored through supervised_mask.
  torch::Tensor label;
  // [N] bool mask of mature, non-protected Gaussians used by GI loss.
  torch::Tensor supervised_mask;
  // Protection protocol budget: floor(ratio * mature_count), independent of
  // the number of protected Gaussians.
  std::int64_t target_k = 0;
  std::int64_t supervised_k = 0;
  float ratio = 0.0f;
  std::uint64_t gi_version = 0;
  std::uint64_t topology_version = 0;

  bool valid() const {
    return label.defined() && supervised_mask.defined();
  }
};

class GITeacherBuilder {
 public:
  GITeacher build(const torch::Tensor& combined_gi,
                  const torch::Tensor& mature_mask,
                  const torch::Tensor& protected_mask, float ratio,
                  std::uint64_t gi_version,
                  std::uint64_t topology_version = 0) const;
};
