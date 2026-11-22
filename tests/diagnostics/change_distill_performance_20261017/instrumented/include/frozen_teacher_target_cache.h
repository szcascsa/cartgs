#include "perf_scope.h"
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
  torch::Tensor get(const torch::Tensor& indices, Evaluate evaluate) {
    cdperf::Scope cache_scope("target_cache");
    torch::NoGradGuard no_grad;
    auto missing = CD_OP("cache_missing_indices", indices.index({~ready_.index_select(0, indices)}));
    if (missing.numel() > 0) {
      auto target = CD_OP("cache_miss_evaluate", evaluate(missing).detach());
      if (!values_.defined()) {
        auto shape = target.sizes().vec();
        shape[0] = ready_.size(0);
        values_ = CD_OP("cache_storage_allocate", torch::empty(shape, target.options().requires_grad(false)));
      }
      CD_OP("cache_write", values_.index_copy_(0, missing, target));
      CD_OP("cache_ready_update", ready_.index_fill_(0, missing, true));
    }
    cdperf::tensor(cdperf::branch + "/cache_values", values_);
    return CD_OP("cache_target_gather", values_.index_select(0, indices));
  }

 private:
  torch::Tensor ready_;
  torch::Tensor values_;
};
