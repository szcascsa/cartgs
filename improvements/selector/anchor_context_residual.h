#pragma once
#include <torch/torch.h>
#include <limits>
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/anchor_context_residual_cuda.h"
#endif

namespace improvements::selector {

// Reuses the first eight channels of the original 11D Selector's first ReLU.
// One bias-free projection per Anchor; no separate per-member encoder.
class AnchorContextResidualImpl : public torch::nn::Module {
 public:
  AnchorContextResidualImpl() {
    projection=register_module("projection",torch::nn::Linear(torch::nn::LinearOptions(17,32).bias(false)));
    torch::NoGradGuard guard;projection->weight.zero_();
  }
  torch::Tensor context(const torch::Tensor& hidden,const torch::Tensor& owners,
      const torch::Tensor& eligible,int64_t anchors) const {
    const auto n=hidden.size(0);
    TORCH_CHECK(hidden.sizes()==torch::IntArrayRef({n,32}) && owners.sizes()==torch::IntArrayRef({n}) &&
        owners.scalar_type()==torch::kLong && eligible.sizes()==owners.sizes() && eligible.scalar_type()==torch::kBool,
        "Context residual hidden/owners/eligibility must align");
    const auto count=torch::zeros({anchors,1},hidden.options()).index_add(0,owners,eligible.to(hidden.scalar_type()).unsqueeze(1));
    const auto value=hidden.slice(1,0,8).masked_fill(~eligible.unsqueeze(1),0);
    const auto mean=torch::zeros({anchors,8},hidden.options()).index_add(0,owners,value)/count.clamp_min(1);
    const auto max_input=value.masked_fill(~eligible.unsqueeze(1),-std::numeric_limits<float>::infinity());
    auto maximum=torch::zeros_like(mean).scatter_reduce(0,owners.unsqueeze(1).expand({n,8}),max_input,"amax",false);
    maximum=torch::where(count>0,maximum,torch::zeros_like(maximum));
    return torch::cat({mean,maximum,count/10.},1);
  }
  torch::Tensor forward(const torch::Tensor& hidden,const torch::Tensor& owners,
      const torch::Tensor& eligible,int64_t anchors) {
    return projection->forward(context(hidden,owners,eligible,anchors)).index_select(0,owners);
  }
  torch::nn::Linear projection{nullptr};
};
TORCH_MODULE(AnchorContextResidual);
}
