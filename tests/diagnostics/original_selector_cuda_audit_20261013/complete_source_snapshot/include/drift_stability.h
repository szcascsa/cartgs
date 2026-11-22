#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "change_management.h"
#include "historical_probe_cache.h"

class FrozenTeacherTargetCache;

struct DriftStabilityConfig {
  // Optional post-degradation recovery; not the online prevention switch.
  bool enabled = false;
  int64_t probe_interval = 1000;
  int64_t min_bad_views = 2;
  int64_t max_gaussians = 32768;
  int64_t batch_size = 8192;
  int64_t consecutive_trigger_count = 2;
  int64_t reference_horizon = 5000;
  int64_t reference_min_age = 3000;
  int64_t reference_max_age = 10000;
  float reference_ema_decay = 0.5f;
  int64_t non_regression_refresh_count = 3;
  float function_threshold = 0.02f;
  float psnr_regression_threshold = 0.1f;
  float budget_relative_tolerance = 0.25f;
  float position_weight = 1.0f;
  float scale_weight = 1.0f;
  float rotation_weight = 1.0f;
  float selector_learning_rate = 5.0e-5f;
  float tf_grid_learning_rate = 8.0e-5f;
  float tf_mlp_learning_rate = 8.0e-6f;
  int64_t epochs = 1;
  uint64_t seed = 0;
};

struct StabilityReference {
  improvements::selector::GumbelNetwork selector_teacher = nullptr;
  TransformField tf_teacher = nullptr;
  uint64_t selector_reference_iter = 0;
  uint64_t tf_reference_iter = 0;
  uint64_t selector_full_reset_iter = 0;
  uint64_t tf_full_reset_iter = 0;
  uint64_t change_version = 0;
};

struct HistoricalViewObservation {
  std::size_t keyframe_id = 0;
  double outlier_score = 0.0;
  torch::Tensor visible_mature_indices;
  bool guard_only = false;  // Recent view: validation only, never a DSD target.
};

struct HistoricalRatioReport {
  float ratio = 0.0f;
  double selector_drift = 0.0;
  double tf_drift = 0.0;
  double weighted_psnr_regression = 0.0;
  int64_t bad_views = 0;
  int64_t selector_bad_views = 0, tf_bad_views = 0;
  bool selector_budget_qualified = false;
  bool quality_evaluated = false;
  bool selector_triggered = false;
  bool tf_triggered = false;
  int64_t selector_steps = 0;
  int64_t tf_steps = 0;
  double selector_post_drift = 0.0;
  double tf_post_drift = 0.0;
};

struct DriftProbeReport {
  bool ran = false;
  bool reference_refreshed = false;
  bool selector_recovery_passed = false;
  bool tf_recovery_passed = false;
  std::string selector_outcome = "SKIP", tf_outcome = "SKIP";
  int64_t view_count = 0;
  int64_t gaussian_count = 0;
  std::vector<HistoricalRatioReport> ratios;
};

class DriftStabilityManager {
 public:
  // Renders the SAME current Gaussian snapshot with the supplied networks.
  using QualityEvaluator = std::function<HistoricalRenderQuality(
      std::size_t, float, const improvements::selector::GumbelNetwork&,
      const TransformField&)>;
  explicit DriftStabilityManager(DriftStabilityConfig config = {});

  bool shouldProbe(uint64_t mapping_iter) const;
  void refreshReference(
      const improvements::selector::GumbelNetwork& selector,
      const TransformField& tf,
      uint64_t mapping_iter,
      uint64_t change_version,
      uint64_t skip_probe_iter = 0,
      bool refresh_selector = true,
      bool refresh_tf = true,
      const char* reason = "structural_change");
  bool prepareProbe(const improvements::selector::GumbelNetwork& selector,
      const TransformField& tf, uint64_t mapping_iter, uint64_t change_version,
      bool deferred = false);
  void clearProbeState();

  DriftProbeReport maybeProbeAndDistill(
      const GaussianAttrState& current,
      const improvements::selector::GumbelNetwork& selector,
      const TransformField& tf,
      const std::vector<float>& ratios,
      double temperature,
      const AABB3D& aabb,
      const std::vector<HistoricalViewObservation>& views,
      uint64_t mapping_iter,
      uint64_t change_version,
      const QualityEvaluator& evaluate_quality,
      HistoricalProbeCache* probe_cache = nullptr,
      bool deferred = false);

  const DriftStabilityConfig& config() const { return config_; }
  const StabilityReference& reference() const { return reference_; }

 private:
  // Test-only optimizer access; normal mapping's trigger/acceptance guards
  // and public recovery interface remain unchanged.
  friend struct DistillRecoveryTestAccess;
  torch::Tensor shuffle(const torch::Tensor& indices);
  torch::Tensor selectorLoss(
      const torch::Tensor& indices,
      const GaussianAttrState& current,
      const improvements::selector::GumbelNetwork& student,
      const improvements::selector::GumbelNetwork& teacher,
      float ratio,
      double temperature,
      FrozenTeacherTargetCache* targets = nullptr) const;
  torch::Tensor tfLossTensor(
      const torch::Tensor& indices,
      const GaussianAttrState& current,
      const TransformField& student,
      const TransformField& teacher,
      float ratio,
      double diagonal,
      FrozenTeacherTargetCache* targets = nullptr) const;
  std::array<double, 3> tfLoss(
      const torch::Tensor& indices,
      const GaussianAttrState& current,
      const TransformField& student,
      const TransformField& teacher,
      float ratio,
      double diagonal,
      FrozenTeacherTargetCache* targets = nullptr) const;
  std::vector<int64_t> distillModule(bool selector_module,
      const std::vector<torch::Tensor>& populations, const GaussianAttrState& current,
      const improvements::selector::GumbelNetwork& selector, const TransformField& tf,
      const std::vector<float>& ratios, double temperature, double diagonal,
      uint64_t mapping_iter, HistoricalProbeCache* probe_cache);

  DriftStabilityConfig config_;
  StabilityReference reference_;
  std::mt19937_64 rng_;
  std::vector<int64_t> selector_over_count_;
  std::vector<int64_t> tf_over_count_;
  int64_t selector_stable_count_ = 0;
  int64_t tf_stable_count_ = 0;
  uint64_t skip_probe_iter_ = 0;
};
