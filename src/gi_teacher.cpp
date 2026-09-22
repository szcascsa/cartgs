#include "include/gi_teacher.h"

#include <cmath>
#include <stdexcept>

GITeacher GITeacherBuilder::build(const torch::Tensor& combined_gi,
                                  const torch::Tensor& mature_mask,
                                  const torch::Tensor& protected_mask,
                                  float ratio, std::uint64_t gi_version,
                                  std::uint64_t topology_version) const {
  if (!combined_gi.defined() || !mature_mask.defined() ||
      !protected_mask.defined())
    return {};
  if (combined_gi.dim() != 1 || mature_mask.dim() != 1 ||
      protected_mask.dim() != 1 ||
      mature_mask.size(0) != combined_gi.size(0) ||
      protected_mask.size(0) != combined_gi.size(0)) {
    throw std::invalid_argument(
        "GI teacher inputs must be matching one-dimensional tensors");
  }
  if (!std::isfinite(ratio) || ratio < 0.0f || ratio > 1.0f)
    throw std::invalid_argument("GI teacher ratio must be in [0, 1]");

  torch::NoGradGuard no_grad;
  const auto count = combined_gi.size(0);
  auto mature = mature_mask.to(combined_gi.device(), torch::kBool);
  auto protected_points =
      protected_mask.to(combined_gi.device(), torch::kBool);
  auto supervised = torch::logical_and(mature, ~protected_points);

  GITeacher teacher;
  teacher.supervised_mask = supervised;
  const auto mature_count = mature.sum().item<std::int64_t>();
  teacher.supervised_k = supervised.sum().item<std::int64_t>();
  teacher.target_k = static_cast<std::int64_t>(std::floor(
      static_cast<double>(ratio) * static_cast<double>(mature_count)));
  teacher.ratio = ratio;
  teacher.gi_version = gi_version;
  teacher.topology_version = topology_version;
  teacher.label = torch::zeros(
      {count}, combined_gi.options().dtype(torch::kFloat32));

  if (teacher.supervised_k == 0) return teacher;

  // Match FlexGS: compute the percentile on the eligible global scores and
  // use a strict threshold. Equal scores therefore intentionally produce
  // fewer labels than target_k.
  auto supervised_indices = torch::nonzero(supervised).reshape({-1});
  auto supervised_scores = combined_gi.detach().index({supervised_indices});
  auto sorted_scores = std::get<0>(torch::sort(supervised_scores, 0));
  const auto percentile_index = static_cast<std::int64_t>(std::floor(
      (1.0 - static_cast<double>(ratio)) *
      static_cast<double>(teacher.supervised_k - 1)));
  auto threshold = sorted_scores.index({percentile_index});
  auto labels = (supervised_scores > threshold).to(torch::kFloat32);
  teacher.label.index_put_({supervised_indices}, labels);
  return teacher;
}
