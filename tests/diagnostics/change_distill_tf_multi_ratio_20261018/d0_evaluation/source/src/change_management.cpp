#include "include/change_management.h"
#include "include/model_snapshot.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
torch::Tensor detachedClone(const torch::Tensor& value) {
  return value.defined() ? value.detach().clone() : torch::Tensor();
}

bool sameMaskSize(const torch::Tensor& mask, const torch::Tensor& xyz) {
  return mask.defined() && mask.dim() == 1 && xyz.defined() &&
         xyz.dim() == 2 && mask.size(0) == xyz.size(0);
}
}  // namespace

std::string changeEventTypeName(uint32_t flags) {
  std::string name;
  const auto append = [&](ChangeFlag flag, const char* label) {
    if (!(flags & toBits(flag))) return;
    if (!name.empty()) name += "|";
    name += label;
  };
  append(ChangeFlag::LOCAL_GEOMETRY, "LOCAL_GEOMETRY");
  append(ChangeFlag::GLOBAL_GEOMETRY, "GLOBAL_GEOMETRY");
  append(ChangeFlag::FIELD_DOMAIN, "FIELD_DOMAIN");
  return name.empty() ? "NONE" : name;
}

const char* distillScopeName(DistillScope scope) {
  switch (scope) {
    case DistillScope::NONE: return "NONE";
    case DistillScope::CORRECTED_MATURE: return "CORRECTED_MATURE";
    case DistillScope::ALL_SNAPSHOT_MATURE: return "ALL_SNAPSHOT_MATURE";
  }
  return "UNKNOWN";
}

DynamicAABBManager::DynamicAABBManager(DynamicAABBConfig config)
    : config_(std::move(config)) {}

bool DynamicAABBManager::shouldPeriodicCheck(
    std::uint64_t mapping_iter) const {
  return config_.enabled && config_.check_interval > 0 &&
         mapping_iter > 0 &&
         mapping_iter % static_cast<std::uint64_t>(config_.check_interval) == 0;
}

AABBPressure DynamicAABBManager::evaluate(
    const torch::Tensor& mature_xyz, const AABB3D& current_aabb) const {
  torch::NoGradGuard no_grad;
  if (!current_aabb.defined())
    throw std::invalid_argument("Dynamic AABB evaluation requires a valid AABB");
  if (!mature_xyz.defined() || mature_xyz.numel() == 0) return {};
  if (mature_xyz.dim() != 2 || mature_xyz.size(1) != 3)
    throw std::invalid_argument("Mature Gaussian positions must have shape [N,3]");

  AABBPressure pressure;
  pressure.mature_count = mature_xyz.size(0);
  const auto x = mature_xyz.index({torch::indexing::Slice(), 0});
  const auto y = mature_xyz.index({torch::indexing::Slice(), 1});
  const auto z = mature_xyz.index({torch::indexing::Slice(), 2});
  const auto x_low = x < current_aabb.min_xyz[0];
  const auto x_high = x > current_aabb.max_xyz[0];
  const auto y_low = y < current_aabb.min_xyz[1];
  const auto y_high = y > current_aabb.max_xyz[1];
  const auto z_low = z < current_aabb.min_xyz[2];
  const auto z_high = z > current_aabb.max_xyz[2];
  pressure.outside_mask = torch::logical_or(
      torch::logical_or(torch::logical_or(x_low, x_high),
                        torch::logical_or(y_low, y_high)),
      torch::logical_or(z_low, z_high));
  pressure.outside_total = pressure.outside_mask.sum().item<std::int64_t>();
  pressure.x_low_count = x_low.sum().item<std::int64_t>();
  pressure.x_high_count = x_high.sum().item<std::int64_t>();
  pressure.y_low_count = y_low.sum().item<std::int64_t>();
  pressure.y_high_count = y_high.sum().item<std::int64_t>();
  pressure.z_low_count = z_low.sum().item<std::int64_t>();
  pressure.z_high_count = z_high.sum().item<std::int64_t>();
  pressure.outside_ratio = static_cast<float>(pressure.outside_total) /
                           static_cast<float>(pressure.mature_count);
  return pressure;
}

bool DynamicAABBManager::shouldExpand(const AABBPressure& pressure) const {
  return config_.enabled && config_.outside_count_threshold > 0 &&
         pressure.outside_total >= config_.outside_count_threshold;
}

AABBUpdatePlan DynamicAABBManager::makeExpansionPlan(
    const torch::Tensor& mature_xyz,
    const AABB3D& current_aabb,
    const AABBPressure& pressure) const {
  torch::NoGradGuard no_grad;
  if (!current_aabb.defined())
    throw std::invalid_argument("AABB expansion requires a valid AABB");
  if (!mature_xyz.defined() || mature_xyz.numel() == 0)
    return {};
  if (mature_xyz.dim() != 2 || mature_xyz.size(1) != 3)
    throw std::invalid_argument("Mature Gaussian positions must have shape [N,3]");
  AABBUpdatePlan plan;
  plan.old_aabb.min_xyz = detachedClone(current_aabb.min_xyz);
  plan.old_aabb.max_xyz = detachedClone(current_aabb.max_xyz);
  plan.new_aabb = plan.old_aabb;
  plan.pressure = pressure;
  if (!shouldExpand(pressure)) return plan;

  const auto sorted_xyz = std::get<0>(torch::sort(mature_xyz, /*dim=*/0));
  const auto sample_count = mature_xyz.size(0);
  const auto low_q = std::clamp(config_.bound_quantile_low, 0.0f, 1.0f);
  const auto high_q = std::clamp(config_.bound_quantile_high, 0.0f, 1.0f);
  TORCH_CHECK(low_q <= high_q,
              "Dynamic AABB lower quantile must not exceed upper quantile");
  // Round inward at both tails so negative and positive outliers receive
  // the same protection. When no rank lies inside the interval (e.g. N=2),
  // retain the adjacent ranks bracketing it instead of inverting the bounds.
  const auto low_rank = static_cast<int64_t>(std::ceil(
      low_q * static_cast<float>(sample_count - 1)));
  const auto high_rank = static_cast<int64_t>(std::floor(
      high_q * static_cast<float>(sample_count - 1)));
  const auto low_index = std::min(low_rank, high_rank);
  const auto high_index = std::max(low_rank, high_rank);
  const auto lower = sorted_xyz.index({low_index});
  const auto upper = sorted_xyz.index({high_index});
  const auto extent = current_aabb.extent();
  const auto margin = extent * config_.expansion_margin;
  auto new_min = current_aabb.min_xyz.clone();
  auto new_max = current_aabb.max_xyz.clone();
  for (int axis = 0; axis < 3; ++axis) {
    if (lower[axis].item<float>() < current_aabb.min_xyz[axis].item<float>())
      new_min[axis] = lower[axis] - margin[axis];
    if (upper[axis].item<float>() > current_aabb.max_xyz[axis].item<float>())
      new_max[axis] = upper[axis] + margin[axis];
  }
  TORCH_CHECK(
      torch::all(new_min <= current_aabb.min_xyz).item<bool>(),
      "Dynamic AABB expansion cannot shrink lower bounds");
  TORCH_CHECK(torch::all(new_max >= current_aabb.max_xyz).item<bool>(),
              "Dynamic AABB expansion cannot shrink upper bounds");
  plan.new_aabb.min_xyz = new_min.detach();
  plan.new_aabb.max_xyz = new_max.detach();
  plan.x_min_changed = new_min[0].item<float>() < current_aabb.min_xyz[0].item<float>();
  plan.x_max_changed = new_max[0].item<float>() > current_aabb.max_xyz[0].item<float>();
  plan.y_min_changed = new_min[1].item<float>() < current_aabb.min_xyz[1].item<float>();
  plan.y_max_changed = new_max[1].item<float>() > current_aabb.max_xyz[1].item<float>();
  plan.z_min_changed = new_min[2].item<float>() < current_aabb.min_xyz[2].item<float>();
  plan.z_max_changed = new_max[2].item<float>() > current_aabb.max_xyz[2].item<float>();
  plan.affected_planes.xy = plan.x_min_changed || plan.x_max_changed ||
                            plan.y_min_changed || plan.y_max_changed;
  plan.affected_planes.xz = plan.x_min_changed || plan.x_max_changed ||
                            plan.z_min_changed || plan.z_max_changed;
  plan.affected_planes.yz = plan.y_min_changed || plan.y_max_changed ||
                            plan.z_min_changed || plan.z_max_changed;
  plan.affected_planes.xe = plan.x_min_changed || plan.x_max_changed;
  plan.affected_planes.ye = plan.y_min_changed || plan.y_max_changed;
  plan.affected_planes.ze = plan.z_min_changed || plan.z_max_changed;
  plan.should_expand = plan.x_min_changed || plan.x_max_changed ||
                       plan.y_min_changed || plan.y_max_changed ||
                       plan.z_min_changed || plan.z_max_changed;
  return plan;
}

ChangeCoordinator::ChangeCoordinator(
    DynamicAABBManager* aabb_manager, ChangeDistillManager* distill_manager)
    : aabb_manager_(aabb_manager), distill_manager_(distill_manager) {}

ChangeCapture ChangeCoordinator::captureBeforeGeometryChange(
    ChangeFlag flag, const torch::Tensor& mature_mask, const GaussianAttrState& state,
    const AABB3D& old_aabb, uint64_t topology_version,
    uint64_t correspondence_version, uint64_t mapping_iter) const {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(sameMaskSize(mature_mask, state.xyz) && mature_mask.scalar_type() == torch::kBool,
              "Change mature mask does not match Gaussian topology");
  TORCH_CHECK(state.log_scaling.sizes() == state.xyz.sizes() &&
                  state.rotation.size(0) == state.xyz.size(0), "Invalid snapshot attributes");
  ChangeCapture capture;
  capture.flags = toBits(flag);
  capture.snapshot_count = state.xyz.size(0);
  capture.mature_mask_before = detachedClone(mature_mask);
  capture.mature_indices_before = torch::nonzero(mature_mask).reshape({-1});
  capture.old_state = {detachedClone(state.xyz), detachedClone(state.log_scaling),
                       detachedClone(state.rotation)};
  capture.old_aabb = old_aabb.clone();
  capture.topology_version = topology_version;
  capture.correspondence_version = correspondence_version;
  capture.mapping_iter = mapping_iter;
  return capture;
}

void ChangeCoordinator::addCorrection(
    ChangeCapture& capture, const GaussianTransformResult& correction,
    unsigned long keyframe_id) const {
  torch::NoGradGuard no_grad;
  torch::Tensor indices = correction.indices;
  if (correction.affects_all) {
    TORCH_CHECK(correction.population_count >= 0,
                "Global correction has an invalid population count");
    const auto count = std::min(correction.population_count,
                                capture.snapshot_count);
    if (count <= 0) return;
    indices = torch::arange(
        count, capture.mature_indices_before.options().dtype(torch::kLong));
  }
  if (!indices.defined() || indices.numel() == 0 ||
      correction.changed_attributes == 0)
    return;
  TORCH_CHECK((indices >= 0).all().item<bool>(), "Negative correction index");
  TORCH_CHECK(correction.transform_4x4.dim() == 2 &&
                  correction.transform_4x4.size(0) == 4 &&
                  correction.transform_4x4.size(1) == 4 &&
                  correction.rotation_wxyz.numel() == 4, "Invalid correction payload");
  indices = indices.index({indices < capture.snapshot_count});
  if (indices.numel() == 0) return;
  CorrectionGroup group;
  static_cast<GaussianTransformResult&>(group) = correction;
  group.indices = indices.detach().clone();
  group.transform_4x4 = correction.transform_4x4.detach().clone();
  group.rotation_wxyz = correction.rotation_wxyz.detach().clone();
  group.keyframe_id = keyframe_id;
  capture.corrections.push_back(std::move(group));
}

ChangeEvent ChangeCoordinator::buildEvent(
    const ChangeCapture& capture, const ChangeTrainingContext& context,
    const AABBUpdatePlan* prepared_plan) {
  torch::NoGradGuard no_grad;
  ChangeEvent event;
  event.flags = capture.flags;
  event.snapshot_count = capture.snapshot_count;
  event.mapping_iter = capture.mapping_iter;
  event.old_aabb = capture.old_aabb;
  event.mature_indices_before = capture.mature_indices_before;
  event.correction_groups = capture.corrections;
  event.correction_group_id = torch::full({capture.snapshot_count}, -1,
      capture.mature_indices_before.options().dtype(torch::kLong));
  for (std::size_t g = 0; g < event.correction_groups.size(); ++g) {
    const auto& indices = event.correction_groups[g].indices;
    TORCH_CHECK((event.correction_group_id.index_select(0, indices) == -1).all().item<bool>(),
                "Overlapping correction groups destroy teacher correspondence");
    auto sorted = std::get<0>(indices.sort());
    TORCH_CHECK(sorted.numel() < 2 ||
        (sorted.slice(0, 1) != sorted.slice(0, 0, -1)).all().item<bool>(),
        "Duplicate indices in correction group");
    event.correction_group_id.index_put_({indices}, static_cast<int64_t>(g));
  }
  const auto* cfg = distill_manager_ ? &distill_manager_->config() : nullptr;
  const bool distill_enabled = cfg && cfg->enabled;
  const bool geometry = !capture.corrections.empty();
  const bool global = (capture.flags & toBits(ChangeFlag::GLOBAL_GEOMETRY)) != 0;
  const auto geometry_scope = global ? DistillScope::ALL_SNAPSHOT_MATURE
                                     : DistillScope::CORRECTED_MATURE;
  if (distill_enabled && geometry && cfg->distill_selector_on_geometry_change && context.selector &&
      capture.teacher_selector && context.normal_selector_optimizer)
    event.selector_scope = geometry_scope;
  if (distill_enabled && geometry && cfg->distill_tf_on_geometry_change && context.tf &&
      capture.teacher_tf && context.normal_tf_optimizer)
    event.tf_scope = geometry_scope;

  if (context.tf && aabb_manager_ && aabb_manager_->config().enabled &&
      (capture.mature_indices_before.numel() > 0 || prepared_plan)) {
    AABBUpdatePlan plan;
    if (prepared_plan) plan = *prepared_plan;
    else {
      auto xyz = context.current.xyz.index_select(0, capture.mature_indices_before);
      AABB3D domain{context.tf->aabbMin(), context.tf->aabbMax()};
      plan = aabb_manager_->makeExpansionPlan(xyz, domain, aabb_manager_->evaluate(xyz, domain));
    }
    if (plan.should_expand) {
      event.aabb_changed = true;
      event.flags |= toBits(ChangeFlag::FIELD_DOMAIN);
      event.old_aabb = plan.old_aabb;
      event.new_aabb = plan.new_aabb;
      event.affected_planes = plan.affected_planes;
      if (distill_enabled && cfg->distill_tf_on_aabb_change && capture.teacher_tf &&
          context.normal_tf_optimizer)
        event.tf_scope = DistillScope::ALL_SNAPSHOT_MATURE;
    }
  }
  return event;
}

ChangeDistillReport ChangeCoordinator::finalizeGeometryChange(
    ChangeCapture&& capture, ChangeTrainingContext& context,
    uint64_t correspondence_version_after, const AABBUpdatePlan* prepared_plan) {
  if (capture.snapshot_count == 0) {
    std::cout << "[ChangeCoordinator] skip type="
              << changeEventTypeName(capture.flags)
              << " reason=no_snapshot" << std::endl;
    return {};
  }
  TORCH_CHECK(capture.correspondence_version == correspondence_version_after &&
                  context.current.xyz.size(0) >= capture.snapshot_count,
              "Prune/reset/reorder occurred during Change capture; only append is allowed");
  auto event = buildEvent(capture, context, prepared_plan);
  if (event.correction_groups.empty() && !event.aabb_changed) {
    std::cout << "[ChangeCoordinator] skip type="
              << changeEventTypeName(event.flags)
              << " mapping_iter=" << event.mapping_iter
              << " snapshot_count=" << event.snapshot_count
              << " corrected_groups=" << event.correction_groups.size()
              << " aabb_changed=" << event.aabb_changed
              << " reason=no_correction_and_no_aabb" << std::endl;
    return {};
  }
  event.id = next_event_id_++;
  std::cout << "[ChangeCoordinator] event=" << event.id
            << " type=" << changeEventTypeName(event.flags) << " flags=" << event.flags
            << " mapping_iter=" << event.mapping_iter
            << " snapshot_count=" << event.snapshot_count
            << " corrected_groups=" << event.correction_groups.size()
            << " selector_scope=" << distillScopeName(event.selector_scope)
            << " tf_scope=" << distillScopeName(event.tf_scope)
            << " ratios=" << context.ratios.size()
            << " aabb_changed=" << event.aabb_changed << std::endl;
  if (event.aabb_changed) {
    ScopedTorchRNG rng(context.current.xyz.device(),
        distill_manager_ ? std::optional<uint64_t>(distill_manager_->nextSeed())
                         : std::nullopt);
    context.tf->resampleDomain(event.old_aabb, event.new_aabb, event.affected_planes);
    context.tf->setAABB(event.new_aabb);
    const auto affected_parameters =
        context.tf->affectedParameters(event.affected_planes);
    for (const auto& parameter : affected_parameters) {
      if (context.normal_tf_optimizer)
        context.normal_tf_optimizer->state().erase(parameter.unsafeGetTensorImpl());
      if (context.shared_mapping_optimizer)
        context.shared_mapping_optimizer->state().erase(
            parameter.unsafeGetTensorImpl());
    }
  }
  ChangeDistillReport report;
  if (distill_manager_ && distill_manager_->config().enabled &&
      (event.selector_scope != DistillScope::NONE || event.tf_scope != DistillScope::NONE))
    report = distill_manager_->execute(event, capture, context);
  report.event_id = event.id;
  report.aabb_changed = event.aabb_changed;
  return report;
}
