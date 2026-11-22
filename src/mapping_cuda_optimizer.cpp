#include "include/mapping_cuda_optimizer.h"
#include "include/transform_field_cuda_optimizer.h"
#include <numeric>
#include <cstdint>
#include <unordered_set>

namespace mapping_cuda {
bool tryFusedDenseAdam(torch::optim::Adam& optimizer) {
  std::unordered_set<void*> seen;
  c10::optional<torch::Device> device;
  std::vector<std::pair<std::uintptr_t,std::uintptr_t>> ranges;
  for (const auto& group : optimizer.param_groups()) {
    const auto& options = static_cast<const torch::optim::AdamOptions&>(group.options());
    if (options.amsgrad()) return false;
    for (const auto& p : group.params()) {
      if (!seen.insert(p.unsafeGetTensorImpl()).second) return false;
      if (!p.grad().defined()) continue;
      if (!p.is_cuda() || p.scalar_type()!=torch::kFloat32 || !p.is_contiguous() ||
          p.grad().is_sparse() || p.grad().scalar_type()!=p.scalar_type() ||
          p.grad().device()!=p.device() || p.grad().sizes()!=p.sizes() || !p.grad().is_contiguous()) return false;
      if (device && p.device()!=*device) return false;
      device=p.device();
      const auto begin=reinterpret_cast<std::uintptr_t>(p.data_ptr<float>());
      const auto end=begin+p.numel()*sizeof(float);
      for(const auto& range:ranges)if(begin<range.second && range.first<end)return false;
      ranges.emplace_back(begin,end);
      auto found=optimizer.state().find(p.unsafeGetTensorImpl());
      if (found!=optimizer.state().end()) {
        const auto& s=static_cast<const torch::optim::AdamParamState&>(*found->second);
        for (const auto& moment : {s.exp_avg(),s.exp_avg_sq()})
          if (!moment.defined() || moment.device()!=p.device() || moment.scalar_type()!=p.scalar_type() ||
              moment.sizes()!=p.sizes() || !moment.is_contiguous()) return false;
      }
    }
  }
  std::vector<std::size_t> groups(optimizer.param_groups().size());
  std::iota(groups.begin(),groups.end(),0);
  // Reuse the FP32 multi-tensor CUDA formula, ordinary Adam state/leaf clocks,
  // and version bookkeeping already shared with the existing TF implementation.
  tf_cuda::tfCudaAdamDirect(optimizer,groups);
  return true;
}
}
