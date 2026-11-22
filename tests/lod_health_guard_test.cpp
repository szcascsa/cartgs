#include "include/lod_health_guard.h"
#include <iostream>

int main() {
  auto options=torch::TensorOptions().dtype(torch::kFloat32);
  auto eligible=torch::ones({10},options.dtype(torch::kBool));
  auto low=torch::full({10},1e-20,options);
  LodHealthGuard zero;zero.zero_patience=2;zero.saturation_patience=20;
  TORCH_CHECK(zero.selection(0,1,.1,100,0,low,eligible).empty());
  TORCH_CHECK(zero.selection(0,2,.1,100,0,low,eligible)=="consecutive_zero_selection_with_candidates");
  LodHealthGuard no_candidates;
  for(int i=0;i<20;++i)TORCH_CHECK(no_candidates.selection(0,i,.01,100,0,low,~eligible).empty());
  LodHealthGuard cold;cold.zero_patience=20;cold.saturation_patience=2;
  TORCH_CHECK(cold.selection(0,1,.1,100,1,low,eligible).empty());
  TORCH_CHECK(cold.selection(0,2,.1,100,1,low,eligible)=="selector_all_drop_saturation");
  LodHealthGuard hot;hot.saturation_patience=2;
  auto high=torch::ones_like(low);
  TORCH_CHECK(hot.selection(0,1,.01,100,10,high,eligible).empty());
  TORCH_CHECK(hot.selection(0,2,.01,100,10,high,eligible)=="selector_all_keep_saturation");
  LodHealthGuard feasible;
  for(int i=0;i<20;++i)TORCH_CHECK(feasible.selection(0,i,.4,100,10,high,eligible).empty());
  LodHealthGuard alternating;alternating.zero_patience=2;
  TORCH_CHECK(alternating.selection(0,1,.1,100,0,low,eligible).empty());
  TORCH_CHECK(alternating.selection(1,2,.5,100,1,high,eligible).empty());
  TORCH_CHECK(alternating.selection(0,3,.1,100,0,low,eligible)=="consecutive_zero_selection_with_candidates");
  LodHealthGuard nonfinite;auto bad=low.clone();bad[0]=NAN;
  TORCH_CHECK(nonfinite.selection(0,1,.1,100,1,bad,eligible)=="nonfinite_selector_probability");
  LodHealthGuard healthy;auto p=torch::zeros({1000},options);p.slice(0,0,10).fill_(1);
  for(int i=0;i<20;++i)TORCH_CHECK(healthy.selection(0,i,.01,1000,10,p,torch::ones_like(p,options.dtype(torch::kBool))).empty());
  auto xyz=torch::zeros({100,3},options),scale=torch::ones_like(xyz),op=torch::ones({100,1},options),ref=torch::ones({100,1},options);
  TORCH_CHECK(healthy.compensation(xyz,scale,op,xyz,scale,ref).empty());
  auto large=scale.clone();large[0].fill_(1000);
  TORCH_CHECK(healthy.compensation(xyz,large,op,xyz,scale,ref)=="compensation_scale_tail");
  auto displaced=xyz.clone();displaced[0][0]=1000;
  TORCH_CHECK(healthy.compensation(displaced,scale,op,xyz,scale,ref)=="compensation_xyz_tail");
  TORCH_CHECK(healthy.compensation(xyz,scale,op,xyz,torch::zeros_like(scale),ref)=="nonfinite_compensation");
  auto differentiable=torch::ones({100},options.requires_grad(true));
  auto probability=differentiable.sigmoid();auto before=probability.detach().clone();
  TORCH_CHECK(healthy.selection(0,1,.4,100,40,probability,torch::ones({100},options.dtype(torch::kBool))).empty());
  TORCH_CHECK(torch::equal(before,probability));
  TORCH_CHECK(!differentiable.grad().defined());
  probability.sum().backward();
  TORCH_CHECK(torch::equal(differentiable.grad(),before*(1-before)));
  std::cout<<"PASS: zero-selection, ratio rotation, no candidates, cold/hot saturation, feasible budget, finite checks, sparse probabilities, 1% scale/XYZ tails and gradient isolation\n";
}
