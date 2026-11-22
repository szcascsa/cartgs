#pragma once

#include <torch/torch.h>

// One cache belongs to one frozen teacher, ratio and Gaussian snapshot.
// Keep it local to the distillation call so input/teacher changes cannot reuse it.
class FrozenTeacherTargetCache {
 public:
  FrozenTeacherTargetCache(int64_t population_count, torch::TensorOptions options)
      : ready_(torch::zeros({population_count},
                           options.dtype(torch::kBool).requires_grad(false))) {}

  template <typename Evaluate>
  torch::Tensor get(const torch::Tensor& indices, Evaluate evaluate, bool fully_cached = false) {
    // The caller must have populated this complete set (or a superset) through
    // get() earlier in this event. Unknown/resampled coverage stays lazy.
    if (fully_cached) return getCached(indices);
    torch::NoGradGuard no_grad;
    auto missing = indices.index({~ready_.index_select(0, indices)});
    if (missing.numel() > 0) {
      auto target = evaluate(missing).detach();
      if (!values_.defined()) {
        auto shape = target.sizes().vec();
        shape[0] = ready_.size(0);
        values_ = torch::empty(shape, target.options().requires_grad(false));
      }
      values_.index_copy_(0, missing, target);
      ready_.index_fill_(0, missing, true);
    }
    return values_.index_select(0, indices);
  }

  torch::Tensor getCached(const torch::Tensor& indices) const {
    torch::NoGradGuard no_grad;
    TORCH_CHECK(values_.defined(), "Teacher cache has not been populated");
    return values_.index_select(0, indices);
  }

 private:
  torch::Tensor ready_;
  torch::Tensor values_;
};
