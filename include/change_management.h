#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include "aabb.h"
#include "gaussian_transform_ops.h"
#include "improvements/selector/gumbel_network.h"
#include "transform_field.h"

struct AABBPressure {
  int64_t mature_count = 0, outside_total = 0;
  int64_t x_low_count = 0, x_high_count = 0;
  int64_t y_low_count = 0, y_high_count = 0;
  int64_t z_low_count = 0, z_high_count = 0;
  float outside_ratio = 0.0f;
  torch::Tensor outside_mask;
};
struct AABBUpdatePlan {
  bool should_expand = false;
  AABB3D old_aabb, new_aabb;
  bool x_min_changed = false, x_max_changed = false;
  bool y_min_changed = false, y_max_changed = false;
  bool z_min_changed = false, z_max_changed = false;
  PlaneChangeMask affected_planes;
  AABBPressure pressure;
};
struct DynamicAABBConfig {
  bool enabled = true;
  int64_t check_interval = 1000, outside_count_threshold = 100;
  float initial_margin = 1.3f, expansion_margin = 0.2f;
  float bound_quantile_low = 0.001f, bound_quantile_high = 0.999f;
  bool fixed_resolution = true, outside_identity = true;
};
struct ChangeDistillConfig {
  bool enabled = false;
  bool distill_selector_on_geometry_change = true;
  bool distill_tf_on_geometry_change = true;
  bool distill_tf_on_aabb_change = true;
  float position_weight = 1.0f, scale_weight = 1.0f;
  float rotation_weight = 1.0f, selector_weight = 1.0f;
  float selector_learning_rate = 1.0e-4f;
  float tf_grid_learning_rate = 1.6e-4f;
  float tf_mlp_learning_rate = 1.6e-5f;
  int64_t batch_size = 8192, epochs = 1;
  // Unset branch counts preserve the legacy shared epoch count.
  std::optional<int64_t> selector_epochs, tf_epochs;
  int64_t selectorEpochs() const { return selector_epochs.value_or(epochs); }
  int64_t tfEpochs() const { return tf_epochs.value_or(epochs); }
  uint64_t seed = 0;
  bool enable_unaffected_anchors = true;
  float anchor_ratio_to_corrected = 1.0f;
  int64_t anchor_max_count = 32768;
  float anchor_loss_weight = 1.0f;
  bool post_change_replay = false;
};
enum class ChangeFlag : uint32_t {
  NONE = 0, LOCAL_GEOMETRY = 1u << 0,
  GLOBAL_GEOMETRY = 1u << 1, FIELD_DOMAIN = 1u << 2
};
constexpr uint32_t toBits(ChangeFlag flag) { return static_cast<uint32_t>(flag); }
enum class DistillScope { NONE, CORRECTED_MATURE, ALL_SNAPSHOT_MATURE };
std::string changeEventTypeName(uint32_t flags);
const char* distillScopeName(DistillScope scope);
struct CorrectionGroup : GaussianTransformResult { unsigned long keyframe_id = 0; };
struct ChangeEvent {
  uint64_t id = 0, mapping_iter = 0;
  uint32_t flags = 0;
  int64_t snapshot_count = 0;
  torch::Tensor mature_indices_before, correction_group_id;
  std::vector<CorrectionGroup> correction_groups;
  bool aabb_changed = false;
  AABB3D old_aabb, new_aabb;
  PlaneChangeMask affected_planes;
  DistillScope selector_scope = DistillScope::NONE;
  DistillScope tf_scope = DistillScope::NONE;
};
// Full [0,N0) storage. Append is allowed; prune/reset/reorder invalidates capture.
struct ChangeCapture {
  uint32_t flags = 0;
  // Actual mutation, including corrections when no CD snapshot is available.
  bool has_geometry_mutation = false;
  int64_t snapshot_count = 0;
  torch::Tensor mature_indices_before, mature_mask_before;
  GaussianAttrState old_state;
  // Optional view-independent Selector inputs captured before geometry changes.
  torch::Tensor old_selector_sh_dc, old_selector_opacity;
  torch::Tensor old_selector_opacity_logits;
  AABB3D old_aabb;
  uint64_t topology_version = 0, correspondence_version = 0, mapping_iter = 0;
  TransformField teacher_tf = nullptr;
  improvements::selector::GumbelNetwork teacher_selector = nullptr;
  std::vector<CorrectionGroup> corrections;
};
// Resolved at execution time because TF is initialized lazily. Caller holds
// mutex_render_ from finalization through resampling and the complete CD pass.
struct ChangeTrainingContext {
  GaussianAttrState current;
  torch::Tensor selector_sh_dc, selector_opacity;
  torch::Tensor selector_opacity_logits;
  improvements::selector::GumbelNetwork selector = nullptr;
  bool selector_compensation_active = false;
  TransformField tf = nullptr;
  torch::optim::Adam* normal_selector_optimizer = nullptr;
  torch::optim::Adam* normal_tf_optimizer = nullptr;
  // Normal Mapping uses the Gaussian optimizer as the shared optimizer for
  // Selector/TF parameters. This pointer is only used to reset resampled TF
  // state; Change Distill still uses the independent optimizer above.
  torch::optim::Adam* shared_mapping_optimizer = nullptr;
  std::vector<float> ratios;
  double temperature = 1.0;
};
struct ChangeDistillReport {
  uint64_t event_id = 0;
  bool aabb_changed = false;
  int64_t selector_count = 0, tf_count = 0, skipped_outside = 0;
  int64_t anchor_count = 0, selector_steps = 0, tf_steps = 0;
  int64_t compensation_steps = 0;
  double selector_pre = 0, selector_post = 0, anchor_pre = 0, anchor_post = 0;
  double tf_pre = 0, tf_post = 0;
};
class DynamicAABBManager {
 public:
  explicit DynamicAABBManager(DynamicAABBConfig config = {});
  bool shouldPeriodicCheck(uint64_t mapping_iter) const;
  AABBPressure evaluate(const torch::Tensor& mature_xyz, const AABB3D& aabb) const;
  bool shouldExpand(const AABBPressure& pressure) const;
  AABBUpdatePlan makeExpansionPlan(const torch::Tensor& mature_xyz,
      const AABB3D& aabb, const AABBPressure& pressure) const;
  const DynamicAABBConfig& config() const { return config_; }
 private:
  DynamicAABBConfig config_;
};
class ChangeDistillManager {
 public:
  explicit ChangeDistillManager(ChangeDistillConfig config = {});
  ChangeDistillReport execute(const ChangeEvent& event, const ChangeCapture& capture,
                              ChangeTrainingContext& context);
  const ChangeDistillConfig& config() const { return config_; }
  uint64_t nextSeed() { return rng_(); }
  torch::Tensor sampleAnchors(const torch::Tensor& unaffected, int64_t corrected_count);
 private:
  torch::Tensor shuffle(const torch::Tensor& indices);
  ChangeDistillConfig config_;
  std::mt19937_64 rng_;
};
class ChangeCoordinator {
 public:
  ChangeCoordinator(DynamicAABBManager* aabb_manager, ChangeDistillManager* distill_manager);
  ChangeCapture captureBeforeGeometryChange(ChangeFlag flag,
      const torch::Tensor& mature_mask, const GaussianAttrState& state,
      const AABB3D& old_aabb, uint64_t topology_version,
      uint64_t correspondence_version, uint64_t mapping_iter) const;
  void addCorrection(ChangeCapture& capture, const GaussianTransformResult& correction,
                     unsigned long keyframe_id = 0) const;
  ChangeDistillReport finalizeGeometryChange(ChangeCapture&& capture,
      ChangeTrainingContext& context, uint64_t correspondence_version_after,
      const AABBUpdatePlan* prepared_plan = nullptr);
 private:
  ChangeEvent buildEvent(const ChangeCapture& capture,
      const ChangeTrainingContext& context, const AABBUpdatePlan* prepared_plan);
  DynamicAABBManager* aabb_manager_ = nullptr;
  ChangeDistillManager* distill_manager_ = nullptr;
  uint64_t next_event_id_ = 1;
};
