/*
 * Selector supervision from GI, restricted to observed rows for recent-window GI.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>

struct GITeacher {
  // [N] float labels. Protected and otherwise unsupervised entries are zero,
  // and must be ignored through supervised_mask.
  torch::Tensor label;
  // [N] bool mask of observed mature, non-protected Gaussians used by GI loss.
  torch::Tensor supervised_mask;
  // floor(effective_mature_ratio * mature_count). The caller derives this
  // ratio from the remaining total budget after actual Protection admission.
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
                  std::uint64_t topology_version = 0,
                  std::int64_t known_mature_count = -1,
                  const torch::Tensor& observed_mask = {}) const;
};
