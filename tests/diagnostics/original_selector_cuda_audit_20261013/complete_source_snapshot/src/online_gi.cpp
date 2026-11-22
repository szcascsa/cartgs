#include "include/online_gi.h"

#include <algorithm>
#include <stdexcept>

OnlineGIManager::OnlineGIManager(torch::DeviceType device_type,
                                 OnlineGIConfig config)
    : device_type_(device_type), config_(config) {
  config_.fast_ema_decay = std::clamp(config_.fast_ema_decay, 0.0f, 1.0f);
  config_.fast_weight = std::clamp(config_.fast_weight, 0.0f, 1.0f);
  config_.eps = std::max(config_.eps, 1e-12f);
}

FrameGIScore OnlineGIManager::normalizeFrame(
    const torch::Tensor& raw_importance,
    const torch::Tensor& contribution_count) const {
  if (!raw_importance.defined() || !contribution_count.defined())
    return {};
  if (raw_importance.dim() != 1 || contribution_count.dim() != 1 ||
      raw_importance.size(0) != contribution_count.size(0)) {
    throw std::invalid_argument(
        "OnlineGI frame statistics must be matching one-dimensional tensors");
  }

  FrameGIScore result;
  result.observed_mask = contribution_count > 0;
  result.observed_indices = torch::nonzero(result.observed_mask).reshape({-1});
  result.num_observed = result.observed_indices.numel();
  result.score = torch::zeros_like(raw_importance);
  if (result.num_observed == 0) return result;

  auto observed_raw = raw_importance.index({result.observed_indices});
  auto total = observed_raw.sum();
  // Do not amplify an empty frame into an Online GI update.
  if (!torch::isfinite(total).item<bool>() ||
      total.abs().item<float>() <= config_.eps) {
    result.observed_mask.fill_(false);
    result.observed_indices = torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(
                 raw_importance.device()));
    result.num_observed = 0;
    return result;
  }
  auto normalized = observed_raw * static_cast<float>(result.num_observed) /
                    (total + config_.eps);
  result.score.index_put_({result.observed_indices}, normalized);
  return result;
}

void OnlineGIManager::ensureStateInitialized(std::int64_t count) {
  if (count < 0) throw std::invalid_argument("Gaussian count cannot be negative");
  if (!gi_fast_.defined()) {
    resetForGaussianCount(count);
    return;
  }
  assertStateInvariants();
  if (size() != count)
    throw std::runtime_error(
        "OnlineGI state is not aligned with the Gaussian model");
}

void OnlineGIManager::assertStateInvariants() const {
  if (!gi_fast_.defined() || !gi_slow_.defined() ||
      !gi_slow_count_.defined() || !gi_last_seen_.defined()) {
    throw std::runtime_error("OnlineGI state is only partially initialized");
  }

  if (gi_fast_.dim() != 1 || gi_slow_.dim() != 1 ||
      gi_slow_count_.dim() != 1 || gi_last_seen_.dim() != 1) {
    throw std::runtime_error(
        "OnlineGI tensors must be aligned one-dimensional state");
  }
  const auto count = gi_fast_.size(0);
  if (gi_slow_.size(0) != count || gi_slow_count_.size(0) != count ||
      gi_last_seen_.size(0) != count)
    throw std::runtime_error("OnlineGI tensor length invariant violated");
  if (gi_fast_.scalar_type() != torch::kFloat32 ||
      gi_slow_.scalar_type() != torch::kFloat32 ||
      gi_slow_count_.scalar_type() != torch::kInt32 ||
      gi_last_seen_.scalar_type() != torch::kInt64) {
    throw std::runtime_error("OnlineGI tensor dtype invariant violated");
  }
  if (gi_fast_.device().type() != device_type_ ||
      gi_slow_.device() != gi_fast_.device() ||
      gi_slow_count_.device() != gi_fast_.device() ||
      gi_last_seen_.device() != gi_fast_.device()) {
    throw std::runtime_error("OnlineGI tensor device invariant violated");
  }
  if (gi_fast_.requires_grad() || gi_slow_.requires_grad() ||
      gi_slow_count_.requires_grad() || gi_last_seen_.requires_grad()) {
    throw std::runtime_error("OnlineGI state must not require gradients");
  }
}

void OnlineGIManager::assertAligned(
    const torch::Tensor& gaussian_xyz) const {
  if (!gaussian_xyz.defined() || gaussian_xyz.dim() < 1)
    throw std::invalid_argument(
        "Gaussian tensor must be defined for OnlineGI alignment");
  assertStateInvariants();
  if (size() != gaussian_xyz.size(0))
    throw std::runtime_error(
        "OnlineGI state is not aligned with the Gaussian model");
  if (gi_fast_.device() != gaussian_xyz.device())
    throw std::runtime_error(
        "OnlineGI state is not on the Gaussian model device");
}

void OnlineGIManager::resetForGaussianCount(std::int64_t count) {
  if (count < 0) throw std::invalid_argument("Gaussian count cannot be negative");
  auto float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);
  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_);
  auto last_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device_type_);
  gi_fast_ = torch::zeros({count}, float_options);
  gi_slow_ = torch::zeros({count}, float_options);
  gi_slow_count_ = torch::zeros({count}, int_options);
  gi_last_seen_ = torch::full({count}, -1, last_options);
  assertStateInvariants();
  ++gi_version_;
  topology_version_ = 0;
}

void OnlineGIManager::appendGaussians(std::int64_t count) {
  if (count < 0)
    throw std::invalid_argument("Gaussian count cannot be negative");
  if (count == 0) return;
  if (!gi_fast_.defined()) {
    resetForGaussianCount(count);
    return;
  }
  assertStateInvariants();
  auto float_options = gi_fast_.options();
  auto int_options = gi_slow_count_.options();
  auto last_options = gi_last_seen_.options();
  gi_fast_ = torch::cat({gi_fast_, torch::zeros({count}, float_options)});
  gi_slow_ = torch::cat({gi_slow_, torch::zeros({count}, float_options)});
  gi_slow_count_ =
      torch::cat({gi_slow_count_, torch::zeros({count}, int_options)});
  gi_last_seen_ = torch::cat(
      {gi_last_seen_, torch::full({count}, -1, last_options)});
  assertStateInvariants();
  ++gi_version_;
  ++topology_version_;
}

void OnlineGIManager::pruneGaussians(const torch::Tensor& survivor_mask) {
  if (!survivor_mask.defined()) return;
  assertStateInvariants();
  if (survivor_mask.dim() != 1 ||
      survivor_mask.size(0) != gi_fast_.size(0)) {
    throw std::invalid_argument(
        "OnlineGI survivor mask does not match Gaussian state");
  }
  if (survivor_mask.scalar_type() != torch::kBool)
    throw std::invalid_argument("OnlineGI survivor mask must be boolean");
  if (survivor_mask.device() != gi_fast_.device())
    throw std::invalid_argument(
        "OnlineGI survivor mask is not on the Gaussian state device");
  gi_fast_ = gi_fast_.index({survivor_mask});
  gi_slow_ = gi_slow_.index({survivor_mask});
  gi_slow_count_ = gi_slow_count_.index({survivor_mask});
  gi_last_seen_ = gi_last_seen_.index({survivor_mask});
  assertStateInvariants();
  ++gi_version_;
  ++topology_version_;
}

void OnlineGIManager::updateFast(const FrameGIScore& frame,
                                  std::int64_t mapping_iter) {
  if (!frame.score.defined()) {
    if (frame.num_observed != 0)
      throw std::invalid_argument("OnlineGI frame score is undefined");
    return;
  }
  ensureStateInitialized(frame.score.size(0));
  if (frame.num_observed == 0) return;
  torch::NoGradGuard no_grad;
  auto idx = frame.observed_indices;
  auto x = frame.score.index({idx});
  auto old = gi_fast_.index({idx});
  auto first = gi_last_seen_.index({idx}) < 0;
  auto updated = torch::where(
      first, x,
      config_.fast_ema_decay * old + (1.0f - config_.fast_ema_decay) * x);
  gi_fast_.index_put_({idx}, updated);
  gi_last_seen_.index_put_({idx}, mapping_iter);
}

void OnlineGIManager::updateSlow(const FrameGIScore& frame) {
  if (!frame.score.defined()) {
    if (frame.num_observed != 0)
      throw std::invalid_argument("OnlineGI frame score is undefined");
    return;
  }
  ensureStateInitialized(frame.score.size(0));
  if (frame.num_observed == 0) return;
  torch::NoGradGuard no_grad;
  auto idx = frame.observed_indices;
  auto old_count = gi_slow_count_.index({idx});
  auto new_count = old_count + 1;
  auto old_mean = gi_slow_.index({idx});
  auto x = frame.score.index({idx});
  auto new_mean = old_mean + (x - old_mean) / new_count.to(torch::kFloat32);
  gi_slow_.index_put_({idx}, new_mean);
  gi_slow_count_.index_put_({idx}, new_count);
}

torch::Tensor OnlineGIManager::getCombinedGI() const {
  if (!gi_fast_.defined()) return torch::Tensor();
  assertStateInvariants();
  auto fast_valid = gi_last_seen_ >= 0;
  auto slow_valid = gi_slow_count_ > 0;
  auto both = fast_valid & slow_valid;
  auto fast_only = fast_valid & ~slow_valid;
  auto slow_only = ~fast_valid & slow_valid;
  auto combined = config_.fast_weight * gi_fast_ +
                  (1.0f - config_.fast_weight) * gi_slow_;
  return torch::where(
      both, combined,
      torch::where(fast_only, gi_fast_,
                   torch::where(slow_only, gi_slow_,
                                torch::zeros_like(gi_fast_))));
}

std::int64_t OnlineGIManager::size() const {
  return gi_fast_.defined() ? gi_fast_.size(0) : 0;
}
