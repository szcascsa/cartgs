#include "include/constrained_adam.h"

#include <algorithm>
#include <cmath>

AdamStepPreview::AdamStepPreview(torch::optim::Adam& optimizer,
                                 const std::vector<std::size_t>& cuda_tf_groups)
    : optimizer_(optimizer) {
  torch::NoGradGuard no_grad;
  cuda_prepared_ = tf_cuda::tfCudaAdamPrepare(optimizer, cuda_tf_groups);
  std::unordered_map<void*, std::size_t> cuda_indices;
  for (std::size_t i = 0; i < cuda_prepared_.size(); ++i)
    cuda_indices.emplace(cuda_prepared_[i].parameter.unsafeGetTensorImpl(), i);
  for (std::size_t g = 0; g < optimizer.param_groups().size(); ++g) {
    const auto& group = optimizer.param_groups()[g];
    const auto& options = static_cast<const torch::optim::AdamOptions&>(group.options());
    for (const auto& p : group.params()) {
      if (!p.grad().defined()) continue;
      TORCH_CHECK(!p.grad().is_sparse(), "Adam preview does not support sparse gradients");
      TORCH_CHECK(indices_.count(p.unsafeGetTensorImpl()) == 0,
                  "Adam preview requires unique parameter groups");
      const auto cuda = cuda_indices.find(p.unsafeGetTensorImpl());
      if (cuda != cuda_indices.end()) {
        indices_.emplace(p.unsafeGetTensorImpl(), parameters_.size());
        parameters_.push_back(p); delta_.push_back(cuda_prepared_[cuda->second].delta);
        groups_.push_back(g); movable_.push_back(options.lr() != 0.0);
        states_.push_back(nullptr); cuda_slots_.push_back(cuda->second);
        continue;
      }
      auto state = std::make_unique<torch::optim::AdamParamState>();
      const auto found = optimizer.state().find(p.unsafeGetTensorImpl());
      if (found != optimizer.state().end()) {
        const auto& old = static_cast<const torch::optim::AdamParamState&>(*found->second);
        state->step(old.step());
        state->exp_avg(old.exp_avg().clone());
        state->exp_avg_sq(old.exp_avg_sq().clone());
        if (options.amsgrad()) state->max_exp_avg_sq(old.max_exp_avg_sq().clone());
      } else {
        state->exp_avg(torch::zeros_like(p));
        state->exp_avg_sq(torch::zeros_like(p));
        if (options.amsgrad()) state->max_exp_avg_sq(torch::zeros_like(p));
      }
      state->step(state->step() + 1);
      const double b1 = std::get<0>(options.betas()), b2 = std::get<1>(options.betas());
      const double bc1 = 1.0 - std::pow(b1, state->step());
      const double bc2 = 1.0 - std::pow(b2, state->step());
      auto grad = p.grad();
      if (options.weight_decay() != 0.0) grad = grad.add(p, options.weight_decay());
      // Match LibTorch 2.3.1 Adam's operation ordering, without writing .grad.
      state->exp_avg().mul_(b1).add_(grad, 1.0 - b1);
      state->exp_avg_sq().mul_(b2).addcmul_(grad, grad, 1.0 - b2);
      auto variance = state->exp_avg_sq();
      if (options.amsgrad()) {
        state->max_exp_avg_sq().copy_(torch::maximum(state->max_exp_avg_sq(), variance));
        variance = state->max_exp_avg_sq();
      }
      auto denom = (variance.sqrt() / std::sqrt(bc2)).add_(options.eps());
      // addcdiv on the original value preserves Adam's float rounding. delta
      // is the actual representable parameter increment, not an ideal double step.
      auto proposed = p.detach().clone();
      proposed.addcdiv_(state->exp_avg(), denom, -options.lr() / bc1);
      indices_.emplace(p.unsafeGetTensorImpl(), parameters_.size());
      parameters_.push_back(p);
      delta_.push_back(proposed - p.detach());
      groups_.push_back(g);
      movable_.push_back(options.lr() != 0.0);
      states_.push_back(std::move(state));
      cuda_slots_.push_back(static_cast<std::size_t>(-1));
    }
  }
}

const torch::optim::AdamParamState& AdamStepPreview::stateAt(std::size_t i) const {
  return states_[i] ? *states_[i] : *cuda_prepared_[cuda_slots_[i]].state;
}

bool AdamStepPreview::finite() const {
  torch::NoGradGuard no_grad;
  std::vector<torch::Tensor> checks;
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    checks.push_back(torch::isfinite(parameters_[i]).all());
    checks.push_back(torch::isfinite(parameters_[i].grad()).all());
    checks.push_back(torch::isfinite(delta_[i]).all());
    checks.push_back(torch::isfinite(stateAt(i).exp_avg()).all());
    checks.push_back(torch::isfinite(stateAt(i).exp_avg_sq()).all());
    if (stateAt(i).max_exp_avg_sq().defined())
      checks.push_back(torch::isfinite(stateAt(i).max_exp_avg_sq()).all());
  }
  return checks.empty() || torch::stack(checks).all().item<bool>();
}

bool AdamStepPreview::finiteCandidate(const std::vector<torch::Tensor>& candidate) const {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(candidate.size() == parameters_.size(), "Candidate block count mismatch");
  std::vector<torch::Tensor> checks;
  for (const auto& value : candidate) checks.push_back(torch::isfinite(value).all());
  return checks.empty() || torch::stack(checks).all().item<bool>();
}

bool AdamStepPreview::changed(const std::vector<torch::Tensor>& candidate) const {
  torch::NoGradGuard no_grad;
  std::vector<torch::Tensor> checks;
  TORCH_CHECK(candidate.size() == parameters_.size(), "Candidate block count mismatch");
  for (std::size_t i = 0; i < candidate.size(); ++i)
    checks.push_back((candidate[i] != parameters_[i].detach()).any());
  return !checks.empty() && torch::stack(checks).any().item<bool>();
}

std::vector<double> AdamStepPreview::updateRatios(
    const std::vector<torch::Tensor>& candidate) const {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(candidate.size() == parameters_.size(), "Candidate block count mismatch");
  std::vector<double> result(optimizer_.param_groups().size(), 0.0);
  if (parameters_.empty()) return result;
  std::vector<torch::Tensor> statistics;
  const auto opts = parameters_.front().options().requires_grad(false).dtype(torch::kFloat64);
  for (std::size_t g = 0; g < result.size(); ++g) {
    auto old_norm = torch::zeros({}, opts), new_norm = torch::zeros({}, opts);
    for (std::size_t i = 0; i < parameters_.size(); ++i) if (groups_[i] == g) {
      old_norm = old_norm + delta_[i].to(torch::kFloat64).square().sum();
      new_norm = new_norm + (candidate[i] - parameters_[i].detach()).to(torch::kFloat64).square().sum();
    }
    statistics.push_back(old_norm); statistics.push_back(new_norm);
  }
  const auto packed = torch::stack(statistics).to(torch::kCPU).contiguous();
  const auto* norms = packed.data_ptr<double>();
  for (std::size_t g = 0; g < result.size(); ++g)
    result[g] = std::sqrt(norms[2 * g + 1]) / std::max(1e-20, std::sqrt(norms[2 * g]));
  return result;
}

std::vector<torch::Tensor> AdamStepPreview::candidate(
    const std::vector<torch::Tensor>& direction, double alpha) const {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(direction.size() == parameters_.size(), "Candidate block count mismatch");
  std::vector<torch::Tensor> result(parameters_.size()), cuda_parameters, cuda_direction;
  std::vector<std::size_t> slots;
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    if (states_[i]) result[i] = parameters_[i].detach().add(direction[i], alpha);
    else {
      slots.push_back(i); cuda_parameters.push_back(parameters_[i]);
      cuda_direction.push_back(direction[i].contiguous());
    }
  }
  auto values = tf_cuda::adamCandidate(cuda_parameters, cuda_direction, alpha);
  for (std::size_t i = 0; i < slots.size(); ++i) result[slots[i]] = values[i];
  return result;
}

torch::Tensor AdamStepPreview::value(const torch::Tensor& parameter,
                                   const std::vector<torch::Tensor>& candidate) const {
  const auto found = indices_.find(parameter.unsafeGetTensorImpl());
  return found == indices_.end() ? parameter.detach() : candidate.at(found->second);
}

void AdamStepPreview::commit(const std::vector<torch::Tensor>& candidate) {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(!committed_ && candidate.size() == parameters_.size(),
              "Adam transaction must commit exactly once");
  // Check the complete transaction before any live write.
  std::vector<torch::Tensor> cuda_candidate(cuda_prepared_.size());
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    TORCH_CHECK(candidate[i].sizes() == parameters_[i].sizes() &&
                    candidate[i].device() == parameters_[i].device() &&
                    candidate[i].scalar_type() == parameters_[i].scalar_type(),
                "Candidate parameter mismatch");
    TORCH_CHECK(indices_.at(parameters_[i].unsafeGetTensorImpl()) == i,
                "Stale Adam transaction");
    if (!states_[i]) cuda_candidate[cuda_slots_[i]] = candidate[i];
  }
  tf_cuda::tfCudaAdamValidate(optimizer_, cuda_prepared_, cuda_candidate);
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    if (!states_[i]) continue;
    parameters_[i].copy_(candidate[i]);
    optimizer_.state()[parameters_[i].unsafeGetTensorImpl()] = std::move(states_[i]);
  }
  tf_cuda::tfCudaAdamCommit(optimizer_, cuda_prepared_, cuda_candidate);
  committed_ = true;
}
