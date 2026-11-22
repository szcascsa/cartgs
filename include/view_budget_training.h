#pragma once
#include <torch/torch.h>
#include "include/selector_handoff.h"
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/flexgs_selector_cuda.h"
#endif

// Same Gumbel-ST sampler as global LoD; the caller supplies a private RNG stream.
// Budget is a loss on the whole candidate population, never a Top-K projection.
struct ViewBudgetSelection {
  torch::Tensor probability, mask, chosen;
};

inline ViewBudgetSelection viewBudgetSelection(const torch::Tensor& logits,
    const torch::Tensor& eligible, const torch::Tensor& exponentials, double temperature) {
  TORCH_CHECK(logits.dim()==2&&logits.size(1)==2&&eligible.sizes()==logits.select(1,0).sizes()&&
      eligible.scalar_type()==torch::kBool&&exponentials.sizes()==logits.sizes()&&
      std::isfinite(temperature)&&temperature>0,
      "Invalid view budget selection inputs");
#ifdef CARTGS_FUSED_MAPPING_CUDA
  if(logits.is_cuda()&&logits.size(0)) {
    auto out=mapping_cuda::flexgsSampleCompact(logits,exponentials.detach(),eligible,temperature);
    return {out[0],out[1],out[2]};
  }
#endif
  auto probability=(logits/temperature).softmax(-1).select(1,1)*eligible;
  auto mask=sampleSelectorGumbelST(logits,temperature,-exponentials.detach().log())*eligible;
  return {probability,mask,torch::nonzero(mask.detach()>0).reshape({-1})};
}
