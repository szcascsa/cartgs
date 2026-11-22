#pragma once

#include <torch/torch.h>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include "transform_field_cuda_optimizer.h"

// Event-local transaction over the CURRENT optimizer population. Construct only
// after append/prune/reset; neither preview nor discarded candidates mutate Adam.
class AdamStepPreview {
 public:
  explicit AdamStepPreview(torch::optim::Adam& optimizer,
                           const std::vector<std::size_t>& cuda_tf_groups = {});
  const std::vector<torch::Tensor>& parameters() const { return parameters_; }
  const std::vector<torch::Tensor>& delta() const { return delta_; }
  const std::vector<std::size_t>& groups() const { return groups_; }
  const std::vector<bool>& movable() const { return movable_; }
  bool finite() const;
  bool finiteCandidate(const std::vector<torch::Tensor>& candidate) const;
  bool changed(const std::vector<torch::Tensor>& candidate) const;
  std::vector<double> updateRatios(const std::vector<torch::Tensor>& candidate) const;
  std::vector<torch::Tensor> candidate(
      const std::vector<torch::Tensor>& direction, double alpha) const;
  torch::Tensor value(const torch::Tensor& parameter,
                      const std::vector<torch::Tensor>& candidate) const;
  void commit(const std::vector<torch::Tensor>& candidate);

 private:
  torch::optim::Adam& optimizer_;
  std::vector<torch::Tensor> parameters_, delta_;
  std::vector<std::size_t> groups_;
  std::vector<bool> movable_;
  std::vector<std::unique_ptr<torch::optim::AdamParamState>> states_;
  std::unordered_map<void*, std::size_t> indices_;
  tf_cuda::AdamPreparation cuda_prepared_;
  std::vector<std::size_t> cuda_slots_;
  const torch::optim::AdamParamState& stateAt(std::size_t i) const;
  bool committed_ = false;
};
