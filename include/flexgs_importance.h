#pragma once
#include <torch/torch.h>
#include <cmath>
#include <algorithm>

// Official FlexGS train.py: sort all scores, floor((1-r)*(N-1)), strict >.
// Keep zero-observation rows and ties; no volume/visibility/maturity weighting.
inline torch::Tensor flexgsImportanceTarget(const torch::Tensor& importance, double ratio) {
  TORCH_CHECK(importance.dim()==1 && std::isfinite(ratio) && ratio>0 && ratio<=1,
              "Invalid FlexGS importance target");
  torch::NoGradGuard guard;
  if (!importance.numel()) return torch::zeros_like(importance);
  auto sorted=std::get<0>(importance.detach().sort());
  auto index=static_cast<int64_t>((1.-ratio)*(importance.numel()-1));
  return (importance.detach()>sorted[index]).to(importance.scalar_type());
}

// Fixed-topology Fine uses persistent source-row IDs for the final tie-break.
// Stable sorts preserve the full lexicographic order: primary GI, global GI,
// then ascending UID. Zero-contribution points still fill the explicit budget.
inline torch::Tensor budgetAlignedImportanceTargetK(const torch::Tensor& primary,
    const torch::Tensor& global,const torch::Tensor& uids,const torch::Tensor& eligible,int64_t requested) {
  TORCH_CHECK(primary.dim()==1&&global.sizes()==primary.sizes()&&uids.sizes()==primary.sizes()&&
      eligible.sizes()==primary.sizes()&&uids.scalar_type()==torch::kLong&&eligible.scalar_type()==torch::kBool&&
      primary.device()==global.device()&&primary.device()==uids.device()&&primary.device()==eligible.device()&&
      requested>=0&&requested<=primary.numel(),"Invalid budget-aligned GI inputs");
  torch::NoGradGuard guard;
  auto labels=torch::zeros_like(primary);
  auto order=torch::nonzero(eligible).reshape({-1});
  const auto k=std::min(requested,order.numel());
  if(!k)return labels;
  order=order.index_select(0,at::argsort(uids.index_select(0,order),true,0,false));
  order=order.index_select(0,at::argsort(global.index_select(0,order),true,0,true));
  order=order.index_select(0,at::argsort(primary.index_select(0,order),true,0,true));
  labels.index_fill_(0,order.slice(0,0,k),1);
  return labels;
}

inline torch::Tensor budgetAlignedImportanceTarget(const torch::Tensor& primary,
    const torch::Tensor& global,const torch::Tensor& uids,const torch::Tensor& eligible,double ratio) {
  TORCH_CHECK(std::isfinite(ratio)&&ratio>=0&&ratio<=1,"Invalid budget-aligned GI ratio");
  return budgetAlignedImportanceTargetK(primary,global,uids,eligible,
      static_cast<int64_t>(std::floor(ratio*primary.numel())));
}

// BCE(sigmoid(keep-drop), label), evaluated in logit space to retain finite
// gradients even when FP32 softmax saturates. Inactive slots are not supervised.
inline torch::Tensor budgetAlignedImportanceLoss(const torch::Tensor& logits,
    const torch::Tensor& labels,const torch::Tensor& eligible,double temperature) {
  TORCH_CHECK(logits.dim()==2&&logits.size(1)==2&&labels.numel()==logits.size(0)&&
      eligible.sizes()==labels.sizes()&&std::isfinite(temperature)&&temperature>0,"Invalid GI BCE inputs");
  auto keep=(logits.select(1,1)-logits.select(1,0))/temperature;
  auto losses=torch::nn::functional::binary_cross_entropy_with_logits(keep,labels.detach(),
      torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
  auto mask=eligible.detach().to(logits.scalar_type());
  return (losses*mask).sum()/mask.sum().clamp_min(1);
}
