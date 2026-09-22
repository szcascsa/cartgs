/*
 * Online Gaussian importance management for the selector path.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>

struct OnlineGIConfig {
  bool enabled = true;
  float fast_ema_decay = 0.9f;
  float fast_weight = 0.7f;
  float eps = 1e-8f;
};

struct FrameGIScore {
  torch::Tensor score;
  torch::Tensor observed_mask;
  torch::Tensor observed_indices;
  std::int64_t num_observed = 0;
};

class OnlineGIManager {
 public:
  OnlineGIManager(torch::DeviceType device_type, OnlineGIConfig config);

  bool enabled() const { return config_.enabled; }

  FrameGIScore normalizeFrame(const torch::Tensor& raw_importance,
                              const torch::Tensor& contribution_count) const;

  void updateFast(const FrameGIScore& frame, std::int64_t mapping_iter);
  void updateSlow(const FrameGIScore& frame);
  // Mark one completed normal-frame Fast/Slow update. Replay never calls it.
  void commitVersion() { ++gi_version_; }

  torch::Tensor getCombinedGI() const;
  std::uint64_t version() const { return gi_version_; }
  std::uint64_t topologyVersion() const { return topology_version_; }

  // Gaussian lifecycle hooks. New points start with no GI evidence; pruning
  // receives the same survivor mask used by GaussianModel.
  void resetForGaussianCount(std::int64_t count);
  void appendGaussians(std::int64_t count);
  void pruneGaussians(const torch::Tensor& survivor_mask);
  void assertAligned(const torch::Tensor& gaussian_xyz) const;

  std::int64_t size() const;

 private:
  void ensureStateInitialized(std::int64_t count);
  void assertStateInvariants() const;

  torch::DeviceType device_type_;
  OnlineGIConfig config_;
  torch::Tensor gi_fast_;
  torch::Tensor gi_slow_;
  torch::Tensor gi_slow_count_;
  torch::Tensor gi_last_seen_;
  std::uint64_t gi_version_ = 0;
  std::uint64_t topology_version_ = 0;
};
