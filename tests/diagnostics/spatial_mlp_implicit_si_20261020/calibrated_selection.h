#pragma once
#include <torch/torch.h>
#include <utility>

// 保持原42次二分和forward概率；补回sum(p)=K的隐式一阶导数。
// 仅用于本实验的一阶Adam训练，不提供校准器的高阶导数。
inline std::pair<torch::Tensor, double> calibratedSelectionProbability(
    const torch::Tensor& score, int64_t k) {
  TORCH_CHECK(score.dim()==1 && score.scalar_type()==torch::kFloat64 &&
      torch::isfinite(score).all().item<bool>(), "Invalid FP64 selection score");
  TORCH_CHECK(k>0 && k<score.numel(), "Expected-K calibration requires 0<K<N");
  auto z=score.detach();
  double low=z.min().item<double>()-40, high=z.max().item<double>()+40;
  for(int it=0;it<42;++it){
    double mid=.5*(low+high);
    if((z-mid).sigmoid().sum().item<double>()>k)low=mid;else high=mid;
  }
  const double offset=.5*(low+high);
  auto p=(z-offset).sigmoid();
  auto d=p*(1-p), mass=d.sum();
  TORCH_CHECK(torch::isfinite(mass).item<bool>() && mass.item<double>()>0,
      "Degenerate expected-K calibration derivative");
  // 此项forward严格为0，backward为d offset/d score_j = d_j/sum(d)。
  auto offset_gradient=((score-score.detach())*(d/mass)).sum();
  return {(score-offset-offset_gradient).sigmoid(), offset};
}
