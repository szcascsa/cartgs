#pragma once
#include <torch/torch.h>
#include "quaternion_utils.h"

inline torch::Tensor reliabilityAlpha(const torch::Tensor& maturity,float low,float high) {
  auto t=((maturity.detach()-low)/(high-low)).clamp(0,1);
  return t*t*(3-2*t);
}

// Interpolate the decoded unit correction on SO(3), then re-encode for decode14.
inline torch::Tensor reliabilityGate(const torch::Tensor& residual,
    const torch::Tensor& alpha,bool appearance) {
  auto a=alpha.detach().reshape({-1,1});
  auto identity=torch::zeros_like(residual.slice(1,6,10));identity.select(1,0).fill_(1);
  auto raw=identity+residual.slice(1,6,10);
  auto length=raw.norm(2,1,true);
  auto q=torch::where(length>1e-8,raw/length.clamp_min(1e-8),identity);
  q=q*torch::where(q.slice(1,0,1)<0,-torch::ones_like(a),torch::ones_like(a));
  auto v=q.slice(1,1,4);auto vn=v.norm(2,1,true).clamp_min(1e-8);
  auto angle=torch::atan2(vn,q.slice(1,0,1));
  auto qa=torch::cat({(a*angle).cos(),v*((a*angle).sin()/vn)},1);
  return torch::cat({residual.slice(1,0,6)*a,qa-identity,
      appearance?residual.slice(1,10)*a:residual.slice(1,10)},1);
}

inline torch::Tensor reliabilitySelectionLoss(const torch::Tensor& noisy_weights,
    const torch::Tensor& maturity,const torch::Tensor& eligible,float delta) {
  auto allowed=eligible.detach().to(maturity.scalar_type());
  auto m=maturity.detach();auto weights=noisy_weights*allowed;
  auto pool=(m*allowed).sum()/allowed.sum().clamp_min(1);
  auto selected=(weights*m).sum()/weights.sum().clamp_min(1e-8);
  return torch::relu((pool+delta).clamp_max(1)-selected).square();
}
