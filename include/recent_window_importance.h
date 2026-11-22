#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <cstddef>
#include <vector>

class GaussianKeyframe;

struct RecentWindowConfig {
  bool enabled = false;
  int64_t size = 16;
  float ema_decay = 0.9f;
  // Keep temporal membership for Guard/Replay without allocating a second GI cache.
  bool collect_importance = true;
};

struct RecentKeyframeEntry {
  std::size_t keyframe_id = 0;
  std::shared_ptr<GaussianKeyframe> keyframe;
  torch::Tensor importance;
  bool importance_valid = false;
  torch::Tensor observed;
};

class RecentWindowImportanceManager {
 public:
  RecentWindowImportanceManager(torch::DeviceType device_type,
                                RecentWindowConfig config);

  bool enabled() const { return config_.enabled; }
  bool contains(std::size_t keyframe_id) const;
  void pushKeyframe(const std::shared_ptr<GaussianKeyframe>& keyframe,
                    int64_t gaussian_count);
  void updateContribution(std::size_t keyframe_id,
                          const torch::Tensor& raw_importance,
                          const torch::Tensor& contribution_count);
  void invalidateContributions();
  void resetForGaussianCount(int64_t count);
  void appendGaussians(int64_t count);
  void pruneGaussians(const torch::Tensor& survivor_mask);
  void assertAligned(const torch::Tensor& gaussian_xyz) const;

  torch::Tensor getImportance() const;
  torch::Tensor observedMask() const;
  std::vector<RecentKeyframeEntry> entries() const;
  std::vector<std::size_t> keyframeIds() const;
  bool hasValidContribution() const;
  int64_t size() const;
  uint64_t version() const { return version_; }
  uint64_t topologyVersion() const { return topology_version_; }

 private:
  torch::Tensor normalize(const torch::Tensor& raw_importance,
                          const torch::Tensor& contribution_count) const;
  void rebuildWeightedSum();

  torch::DeviceType device_type_;
  RecentWindowConfig config_;
  std::deque<RecentKeyframeEntry> entries_;
  torch::Tensor weighted_importance_sum_;
  torch::Tensor observation_count_;
  uint64_t version_ = 0;
  uint64_t topology_version_ = 0;
  uint64_t window_updates_ = 0;
  mutable torch::Tensor normalized_importance_;
  mutable uint64_t normalized_version_ = 0;
};
