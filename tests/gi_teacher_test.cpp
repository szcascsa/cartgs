#include "include/gi_teacher.h"

#include <torch/torch.h>

#include <stdexcept>

int main() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto combined_gi = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}, options);
  const auto mature = torch::ones({4}, torch::TensorOptions().dtype(torch::kBool));
  auto protected_mask = torch::zeros_like(mature);
  protected_mask.index_put_({1}, true);

  const auto teacher = GITeacherBuilder().build(
      combined_gi, mature, protected_mask, 0.5f, 7, 11);
  if (!teacher.valid() || teacher.supervised_k != 3 || teacher.target_k != 2 ||
      teacher.gi_version != 7 || teacher.topology_version != 11)
    throw std::runtime_error("GI teacher metadata is incorrect");

  const auto expected_mask = torch::tensor({false, true, false, false},
                                           torch::TensorOptions().dtype(torch::kBool));
  if (!teacher.supervised_mask.equal(expected_mask))
    throw std::runtime_error("GI teacher protection mask is incorrect");

  // FlexGS uses a strict percentile threshold. With eligible scores
  // {0.1, 0.3, 0.4} and ratio 0.5, the threshold is 0.3, so only 0.4 is
  // labelled positive.
  const auto expected_label = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}, options);
  if (!teacher.label.equal(expected_label))
    throw std::runtime_error("GI teacher strict threshold is incorrect");
  return 0;
}
