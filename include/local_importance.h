#pragma once
#include "include/flexgs_importance.h"

// Teacher sorting never projects the actual Gumbel-ST sample to exact K.
inline torch::Tensor localImportanceTarget(const torch::Tensor& contribution,
    const torch::Tensor& global,const torch::Tensor& ids,const torch::Tensor& eligible,double ratio) {
  TORCH_CHECK(contribution.dim()==1&&!contribution.requires_grad(),"LI must be detached Full contribution");
  TORCH_CHECK(torch::isfinite(contribution).all().item<bool>()&&(contribution>=0).all().item<bool>(),
      "Nonfinite or negative Full LI");
  if(!contribution.numel()||!eligible.any().item<bool>()||
      !(contribution.index({eligible})>0).any().item<bool>())return {};
  return budgetAlignedImportanceTarget(contribution,global.detach(),ids,eligible,ratio);
}

inline torch::Tensor localImportanceLoss(const torch::Tensor& mask,const torch::Tensor& labels) {
  TORCH_CHECK(mask.sizes()==labels.sizes()&&!labels.requires_grad(),"LI label/mask mismatch");
  return (mask-labels).abs().sum()/std::max<int64_t>(1,mask.numel());
}

// autograd::grad leaves all existing .grad tensors intact, including Master,
// Compensation and Anchor Feature. Only explicitly supplied MLP leaves receive LI.
inline std::vector<torch::Tensor> accumulateLocalImportanceGradients(const torch::Tensor& loss,
    const std::vector<torch::Tensor>& selector_parameters) {
  auto gradients=torch::autograd::grad({loss},selector_parameters,{},true,false,false);
  torch::NoGradGuard guard;
  for(size_t i=0;i<selector_parameters.size();++i) {
    auto parameter=selector_parameters[i];
    if(parameter.grad().defined())parameter.mutable_grad().add_(gradients[i]);
    else parameter.mutable_grad()=gradients[i].contiguous().clone();
  }
  return gradients;
}
