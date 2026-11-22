#include "include/recent_window_importance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "include/gaussian_keyframe.h"

RecentWindowImportanceManager::RecentWindowImportanceManager(
    torch::DeviceType device_type, RecentWindowConfig config)
    : device_type_(device_type), config_(std::move(config)) {
  if (config_.size <= 0 || !std::isfinite(config_.ema_decay) ||
      config_.ema_decay < 0.0f || config_.ema_decay > 1.0f)
    throw std::invalid_argument("Invalid RecentWindow configuration");
}

bool RecentWindowImportanceManager::contains(std::size_t keyframe_id) const {
  return std::any_of(entries_.begin(), entries_.end(),
                     [keyframe_id](const auto& entry) {
                       return entry.keyframe_id == keyframe_id;
                     });
}

void RecentWindowImportanceManager::pushKeyframe(
    const std::shared_ptr<GaussianKeyframe>& keyframe,
    int64_t gaussian_count) {
  if (!config_.enabled || !keyframe) return;
  if (!config_.collect_importance) {
    auto existing = std::find_if(entries_.begin(), entries_.end(), [&](const auto& entry) {
      return entry.keyframe_id == keyframe->fid_;
    });
    if (existing != entries_.end()) { existing->keyframe = keyframe; return; }
    if (static_cast<int64_t>(entries_.size()) == config_.size) entries_.pop_front();
    entries_.push_back({keyframe->fid_, keyframe, {}, false, {}});
    ++version_;
    return;
  }
  if (!weighted_importance_sum_.defined()) resetForGaussianCount(gaussian_count);
  if (weighted_importance_sum_.size(0) != gaussian_count)
    throw std::runtime_error("RecentWindow is not aligned with Gaussian map");

  auto existing = std::find_if(entries_.begin(), entries_.end(),
                               [&](const auto& entry) {
                                 return entry.keyframe_id == keyframe->fid_;
                               });
  if (existing != entries_.end()) {
    existing->keyframe = keyframe;
    return;
  }

  if (static_cast<int64_t>(entries_.size()) == config_.size) {
    const auto beta = static_cast<double>(config_.ema_decay);
    const auto expired_weight = std::pow(beta, config_.size);
    weighted_importance_sum_.mul_(beta).add_(entries_.front().importance, -expired_weight);
    observation_count_.sub_(entries_.front().observed.to(torch::kInt32));
    entries_.pop_front();
  } else {
    weighted_importance_sum_.mul_(config_.ema_decay);
  }
  entries_.push_back({keyframe->fid_, keyframe,
                      torch::zeros_like(weighted_importance_sum_), false,
                      torch::zeros_like(weighted_importance_sum_, torch::kBool)});
  // At beta=0 only the newest keyframe belongs to the effective window.
  if (config_.ema_decay == 0.0f) {
    observation_count_.zero_();
    for (auto& entry : entries_) entry.observed.zero_();
  }
  ++version_;
  // Incremental updates are O(N). Rebase infrequently to bound roundoff drift.
  if (++window_updates_ % 256 == 0) rebuildWeightedSum();
}

torch::Tensor RecentWindowImportanceManager::normalize(
    const torch::Tensor& raw_importance,
    const torch::Tensor& contribution_count) const {
  if (!raw_importance.defined() || !contribution_count.defined() ||
      raw_importance.dim() != 1 || contribution_count.dim() != 1 ||
      raw_importance.size(0) != contribution_count.size(0))
    throw std::invalid_argument("Invalid RecentWindow contribution tensors");
  auto observed = contribution_count > 0;
  auto zeros = torch::zeros_like(raw_importance);
  auto values = torch::where(observed, raw_importance, zeros);
  auto total = values.sum();
  auto valid = torch::isfinite(total) & (total.abs() >= 1e-8);
  auto denominator = torch::where(valid, total + 1e-8, torch::ones_like(total));
  auto factor = observed.sum().to(raw_importance.scalar_type()) / denominator;
  // Keep the finite/empty decision on device; only importance_valid is read back.
  return torch::where(observed & valid, values * factor, zeros).detach();
}

void RecentWindowImportanceManager::updateContribution(
    std::size_t keyframe_id, const torch::Tensor& raw_importance,
    const torch::Tensor& contribution_count) {
  if (!config_.enabled || !config_.collect_importance) return;
  auto found = std::find_if(entries_.begin(), entries_.end(),
                            [keyframe_id](const auto& entry) {
                              return entry.keyframe_id == keyframe_id;
                            });
  if (found == entries_.end()) return;
  auto contribution = normalize(raw_importance, contribution_count);
  if (contribution.size(0) != weighted_importance_sum_.size(0) ||
      contribution.device() != weighted_importance_sum_.device())
    throw std::runtime_error("RecentWindow contribution has stale topology");
  const bool has_signal = contribution.numel() > 0 &&
      contribution.abs().sum().item<double>() > 1e-8;
  const auto index = static_cast<int64_t>(std::distance(entries_.begin(), found));
  const auto weight = std::pow(static_cast<double>(config_.ema_decay),
                                static_cast<double>(entries_.size() - 1 - index));
  auto observed = (contribution_count > 0) & torch::isfinite(raw_importance);
  if (!has_signal || weight == 0.0) observed.zero_();
  observation_count_.sub_(found->observed.to(torch::kInt32));
  observation_count_.add_(observed.to(torch::kInt32));
  found->observed = std::move(observed);
  if (found->importance_valid)
    weighted_importance_sum_ += weight * (contribution - found->importance);
  else if (has_signal)
    weighted_importance_sum_ += weight * contribution;
  found->importance = std::move(contribution);
  found->importance_valid = has_signal;
  ++version_;
}

void RecentWindowImportanceManager::invalidateContributions() {
  if (!weighted_importance_sum_.defined()) return;
  for (auto& entry : entries_) {
    entry.importance.zero_();
    entry.importance_valid = false;
    entry.observed.zero_();
  }
  weighted_importance_sum_.zero_();
  observation_count_.zero_();
  ++version_;
}

void RecentWindowImportanceManager::resetForGaussianCount(int64_t count) {
  if (!config_.collect_importance) return;
  if (count < 0) throw std::invalid_argument("Gaussian count cannot be negative");
  weighted_importance_sum_ = torch::zeros(
      {count}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
  observation_count_ = torch::zeros_like(weighted_importance_sum_, torch::kInt32);
  for (auto& entry : entries_) {
    entry.importance = torch::zeros_like(weighted_importance_sum_);
    entry.importance_valid = false;
    entry.observed = torch::zeros_like(weighted_importance_sum_, torch::kBool);
  }
  ++version_;
  topology_version_ = 0;
  window_updates_ = 0;
}

void RecentWindowImportanceManager::appendGaussians(int64_t count) {
  if (!config_.collect_importance) return;
  if (count < 0) throw std::invalid_argument("Gaussian count cannot be negative");
  if (count == 0) return;
  if (!weighted_importance_sum_.defined()) {
    resetForGaussianCount(count);
    return;
  }
  weighted_importance_sum_ = torch::cat(
      {weighted_importance_sum_, torch::zeros({count}, weighted_importance_sum_.options())});
  observation_count_ = torch::cat(
      {observation_count_, torch::zeros({count}, observation_count_.options())});
  for (auto& entry : entries_) {
    entry.importance = torch::cat(
        {entry.importance, torch::zeros({count}, entry.importance.options())});
    entry.observed = torch::cat(
        {entry.observed, torch::zeros({count}, entry.observed.options())});
  }
  ++topology_version_;
  ++version_;
}

void RecentWindowImportanceManager::pruneGaussians(
    const torch::Tensor& survivor_mask) {
  if (!config_.collect_importance) return;
  if (!survivor_mask.defined()) return;
  if (!weighted_importance_sum_.defined() || survivor_mask.dim() != 1 ||
      survivor_mask.size(0) != weighted_importance_sum_.size(0) ||
      survivor_mask.scalar_type() != torch::kBool ||
      survivor_mask.device() != weighted_importance_sum_.device())
    throw std::invalid_argument("Invalid RecentWindow survivor mask");
  weighted_importance_sum_ = weighted_importance_sum_.index({survivor_mask});
  observation_count_ = observation_count_.index({survivor_mask});
  for (auto& entry : entries_) {
    entry.importance = entry.importance.index({survivor_mask});
    entry.observed = entry.observed.index({survivor_mask});
  }
  ++topology_version_;
  ++version_;
}

void RecentWindowImportanceManager::assertAligned(
    const torch::Tensor& gaussian_xyz) const {
  if (!config_.collect_importance) return;
  if (!gaussian_xyz.defined() || !weighted_importance_sum_.defined() ||
      weighted_importance_sum_.size(0) != gaussian_xyz.size(0) ||
      weighted_importance_sum_.device() != gaussian_xyz.device() ||
      !observation_count_.defined() || observation_count_.size(0) != gaussian_xyz.size(0))
    throw std::runtime_error("RecentWindow state is not aligned with Gaussian map");
  for (const auto& entry : entries_)
    if (!entry.importance.defined() ||
        entry.importance.size(0) != gaussian_xyz.size(0) ||
        entry.importance.device() != gaussian_xyz.device() ||
        !entry.observed.defined() || entry.observed.size(0) != gaussian_xyz.size(0))
      throw std::runtime_error("RecentWindow entry is not aligned with Gaussian map");
}

void RecentWindowImportanceManager::rebuildWeightedSum() {
  if (entries_.empty()) {
    weighted_importance_sum_.zero_();
    return;
  }
  weighted_importance_sum_.zero_();
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    const auto weight = std::pow(static_cast<double>(config_.ema_decay),
                                 static_cast<double>(entries_.size() - 1 - i));
    weighted_importance_sum_ += weight * entries_[i].importance;
  }
}

torch::Tensor RecentWindowImportanceManager::observedMask() const {
  return observation_count_.defined() ? observation_count_ > 0 : torch::Tensor{};
}

torch::Tensor RecentWindowImportanceManager::getImportance() const {
  if (!config_.enabled || !weighted_importance_sum_.defined()) return {};
  if (normalized_importance_.defined() && normalized_version_ == version_)
    return normalized_importance_;
  double weight_sum = 0.0;
  for (std::size_t i = 0; i < entries_.size(); ++i)
    weight_sum += std::pow(static_cast<double>(config_.ema_decay),
                           static_cast<double>(entries_.size() - 1 - i));
  normalized_importance_ = weight_sum <= 0.0 ? torch::zeros_like(weighted_importance_sum_)
                                            : weighted_importance_sum_ / weight_sum;
  normalized_version_ = version_;
  return normalized_importance_;
}

std::vector<RecentKeyframeEntry> RecentWindowImportanceManager::entries() const {
  return std::vector<RecentKeyframeEntry>(entries_.begin(), entries_.end());
}

std::vector<std::size_t> RecentWindowImportanceManager::keyframeIds() const {
  std::vector<std::size_t> ids;
  ids.reserve(entries_.size());
  for (const auto& entry : entries_) ids.push_back(entry.keyframe_id);
  return ids;
}

bool RecentWindowImportanceManager::hasValidContribution() const {
  return std::any_of(entries_.begin(), entries_.end(),
                     [](const auto& entry) {
                       return entry.importance_valid;
                     });
}

int64_t RecentWindowImportanceManager::size() const {
  return weighted_importance_sum_.defined() ? weighted_importance_sum_.size(0) : 0;
}
