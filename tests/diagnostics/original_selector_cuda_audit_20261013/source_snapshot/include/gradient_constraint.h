#pragma once

#include "constrained_adam.h"
#include <functional>
#include <cstdint>
#include <string>
#include <vector>

struct GradientConstraintConfig {
  bool enabled = false;
  int check_interval = 1;
  int memory_views_per_check = 1;
  int memory_ratios_per_check = 1;
  int max_candidate_checks = 3;
  int tf_warmup_steps = 1000;
  int log_interval = 100;
  bool cancel_fallback_enable = false;
  double cancel_fallback_scale = 0.2;
  void validate() const;
};

struct GuardObjective {
  std::string name;
  std::size_t keyframe_id = 0;
  bool full = true;
  float ratio = 1.0f;
};

enum class GuardFailureKind { None, QualityConflict, FiniteProjectionFailure, NonFinite, InvalidState };

struct GuardEvaluation {
  std::vector<GuardObjective> objectives;
  // nullptr means LIVE baseline. Use inference when GradMode is disabled;
  // only rejected raw candidates need fresh historical autograd graphs.
  std::function<torch::Tensor(std::size_t, const AdamStepPreview&,
                             const std::vector<torch::Tensor>*)> evaluate;
  std::function<void(const AdamStepPreview&, const std::vector<torch::Tensor>&)> prepare_candidate;
};

struct GradientConstraintReport {
  std::string status = "CANCEL", reason;
  double alpha = 0.0, milliseconds = 0.0;
  double preview_ms = 0.0, baseline_ms = 0.0, history_ms = 0.0;
  double projection_ms = 0.0, validation_ms = 0.0;
  int candidates = 0, conflicts = 0;
  int fallback_checks = 0;
  bool would_cancel = false, fallback_attempted = false, fallback_committed = false;
  GuardFailureKind failure_kind = GuardFailureKind::None;
  bool committed = false;
  bool projection_converged = false;
  double projection_violation = 0.0;
  std::vector<double> before, after, update_ratio_by_group;
  std::vector<double> raw_after, last_checked_after, committed_after;
};

class GradientConstraintManager {
 public:
  explicit GradientConstraintManager(GradientConstraintConfig config = {});
  GradientConstraintReport step(torch::optim::Adam& optimizer,
                                const GuardEvaluation& evaluation,
                                const std::vector<std::size_t>& cuda_tf_groups = {}) const;
 private:
  GradientConstraintConfig config_;
};
