#include "include/gi_teacher.h"
#include <cmath>
#include "include/elastic_entry.h"

GITeacher GITeacherBuilder::build(const torch::Tensor& importance,
    const torch::Tensor& valid_mask, const torch::Tensor& observed_mask,
    const torch::Tensor& maturity, float ratio, std::uint64_t gi_version,
    std::uint64_t topology_version, float entry) const {
  if (!importance.defined() || !valid_mask.defined() || !observed_mask.defined() || !maturity.defined()) return {};
  for (const auto& tensor : {valid_mask, observed_mask, maturity})
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == importance.size(0) &&
        tensor.device() == importance.device(), "GI teacher inputs must match the population");
  TORCH_CHECK(valid_mask.scalar_type() == torch::kBool && observed_mask.scalar_type() == torch::kBool,
      "GI validity and observation must be bool");
  TORCH_CHECK(std::isfinite(ratio) && ratio >= 0 && ratio <= 1, "GI ratio must be in [0,1]");
  torch::NoGradGuard guard;
  GITeacher teacher;
  teacher.supervised_mask = valid_mask & observed_mask;
  if (entry > 0) teacher.supervised_mask &= maturity.detach() >= entry;
  teacher.weight = elasticMaturity(maturity, entry) * teacher.supervised_mask.to(maturity.scalar_type());
  const auto indices = torch::nonzero(teacher.supervised_mask).reshape({-1});
  teacher.supervised_k = indices.numel();
  teacher.target_k = std::floor(double(ratio) * teacher.supervised_k);
  teacher.ratio = ratio;
  teacher.gi_version = gi_version;
  teacher.topology_version = topology_version;
  teacher.label = torch::zeros_like(importance);
  if (!teacher.supervised_k) return teacher;
  const auto scores = importance.detach().index_select(0, indices);
  const auto percentile = static_cast<int64_t>(std::floor((1.0 - double(ratio)) * (teacher.supervised_k - 1)));
  const auto threshold = std::get<0>(torch::kthvalue(scores, percentile + 1));
  teacher.label.index_copy_(0, indices, (scores > threshold).to(importance.scalar_type()));
  return teacher;
}
