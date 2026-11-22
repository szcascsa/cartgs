#include "include/change_management.h"
#include "tf_backend_test_options.h"

#include <cmath>
#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void close(const torch::Tensor& a, const torch::Tensor& b) {
  check(torch::allclose(a, b, 1e-5, 1e-6), "Tensor mismatch");
}
class StdoutCapture {
 public:
  StdoutCapture() : previous_(std::cout.rdbuf(buffer_.rdbuf())) {}
  ~StdoutCapture() { std::cout.rdbuf(previous_); }
  std::string str() const { return buffer_.str(); }
 private:
  std::ostringstream buffer_;
  std::streambuf* previous_;
};

double logNumber(const std::string& line, const std::string& key) {
  const auto token = " " + key + "=";
  const auto position = line.find(token);
  check(position != std::string::npos, "Missing numeric CD log field");
  return std::stod(line.substr(position + token.size()));
}

void checkEpochLogs(const std::string& log, const std::string& prefix,
                    int64_t epochs, double pre, double post, int64_t steps) {
  // Fixed four-decimal logs round each metric by at most half a display unit.
  constexpr double rounding_error = 0.00005 + 1e-10;
  std::istringstream lines(log);
  std::string line;
  int64_t count = 0, total_steps = 0;
  double previous = pre;
  while (std::getline(lines, line)) {
    if (line.find(prefix) != 0) continue;
    check(line.find(" epoch=" + std::to_string(count) + "/" + std::to_string(epochs) + " ") !=
              std::string::npos, "Missing or unordered CD epoch log");
    const double loss = logNumber(line, "loss");
    const double delta = logNumber(line, "delta_prev");
    check(std::isfinite(loss) && std::isfinite(delta), "Nonfinite CD epoch metrics");
    check(std::abs(delta - (loss - previous)) <=
              3 * rounding_error,
          "CD epoch delta does not compare consecutive fixed-set losses");
    if (count == 0)
      check(std::abs(loss - pre) <= rounding_error, "CD baseline mismatch");
    total_steps += static_cast<int64_t>(logNumber(line, "steps"));
    previous = loss;
    ++count;
  }
  check(count == epochs + 1 && total_steps == steps, "Incorrect CD epoch or step count");
  check(std::abs(previous - post) <= rounding_error, "CD final loss mismatch");
}

void eventNames() {
  const auto local = toBits(ChangeFlag::LOCAL_GEOMETRY);
  const auto global = toBits(ChangeFlag::GLOBAL_GEOMETRY);
  const auto domain = toBits(ChangeFlag::FIELD_DOMAIN);
  check(changeEventTypeName(0) == "NONE", "Incorrect empty event label");
  check(changeEventTypeName(local) == "LOCAL_GEOMETRY", "Incorrect local event label");
  check(changeEventTypeName(global) == "GLOBAL_GEOMETRY", "Incorrect global event label");
  check(changeEventTypeName(domain) == "FIELD_DOMAIN", "Incorrect domain event label");
  check(changeEventTypeName(local | domain) == "LOCAL_GEOMETRY|FIELD_DOMAIN", "Incorrect combined local label");
  check(changeEventTypeName(global | domain) == "GLOBAL_GEOMETRY|FIELD_DOMAIN", "Incorrect combined global label");
  check(std::string(distillScopeName(DistillScope::CORRECTED_MATURE)) == "CORRECTED_MATURE" &&
        std::string(distillScopeName(DistillScope::ALL_SNAPSHOT_MATURE)) == "ALL_SNAPSHOT_MATURE" &&
        std::string(distillScopeName(DistillScope::NONE)) == "NONE", "Incorrect scope labels");
}
const auto floats = torch::TensorOptions().device(tf_test_device).dtype(torch::kFloat32);
const auto longs = torch::TensorOptions().device(tf_test_device).dtype(torch::kLong);
const auto bools = torch::TensorOptions().device(tf_test_device).dtype(torch::kBool);
improvements::selector::GumbelNetwork makeSelector() {
  improvements::selector::GumbelNetwork result;result->to(tf_test_device);
#ifdef CARTGS_TEST_SELECTOR_CUDA
  result->setBackend("cuda");
#endif
  return result;
}
using AdamSnapshot = std::map<void*, std::tuple<int64_t, torch::Tensor, torch::Tensor>>;
AdamSnapshot snapshot(torch::optim::Adam& optimizer) {
  AdamSnapshot result;
  for (const auto& item : optimizer.state()) {
    const auto& value = static_cast<const torch::optim::AdamParamState&>(*item.second);
    result.emplace(item.first, std::make_tuple(value.step(), value.exp_avg().clone(),
                                              value.exp_avg_sq().clone()));
  }
  return result;
}
void checkState(torch::optim::Adam& optimizer, const AdamSnapshot& before) {
  check(optimizer.state().size() == before.size(), "Normal Adam state count changed");
  for (const auto& item : before) {
    const auto& value = static_cast<const torch::optim::AdamParamState&>(
        *optimizer.state().at(item.first));
    check(value.step() == std::get<0>(item.second), "Normal Adam clock advanced");
    check(torch::equal(value.exp_avg(), std::get<1>(item.second)), "Normal Adam momentum changed");
    check(torch::equal(value.exp_avg_sq(), std::get<2>(item.second)), "Normal Adam variance changed");
  }
}
void prime(torch::nn::Module& module, torch::optim::Adam& optimizer) {
  auto loss = torch::zeros({}, floats);
  for (const auto& p : module.parameters()) loss = loss + p.square().sum();
  loss.backward();
  optimizer.step();
  optimizer.zero_grad();
}

GaussianTransformResult correction(const torch::Tensor& indices) {
  GaussianTransformResult result;
  result.indices = indices;
  result.transform_4x4 = torch::eye(4, floats);
  result.transform_4x4.index_put_({3, 0}, 0.5f);
  result.rotation_wxyz = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}, floats);
  result.changed_attributes = toBits(GaussianAttrFlag::XYZ) |
      toBits(GaussianAttrFlag::SCALE) | toBits(GaussianAttrFlag::ROTATION);
  return result;
}
GaussianAttrState master() {
  return {torch::tensor({{0.0f, 0.0f, 0.0f}, {0.2f, 0.1f, 0.0f},
                         {0.3f, 0.1f, 0.1f}, {0.4f, 0.1f, 0.2f},
                         {0.1f, 0.2f, 0.3f}, {0.2f, 0.2f, 0.1f}}, floats),
          torch::zeros({6, 3}, floats),
          torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, floats).repeat({6, 1})};
}
TransformField makeTF() {
  TransformFieldConfig config;
  config.backend = tf_test_backend;
  config.hidden_dim = 8;
  config.hexplane.spatial_resolution = 3;
  config.hexplane.ratio_resolution = 3;
  config.hexplane.feature_dim = 2;
  config.hexplane.multires = {1};
  TransformField result(torch::full({3}, -1.0f, floats), torch::ones({3}, floats), config);
  result->to(tf_test_device);return result;
}

void transport() {
  auto state = master().select(torch::tensor({0, 1}, longs));
  state.xyz = torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, floats);
  auto move = correction(torch::tensor({0, 1}, longs));
  move.scale = 2.0f;
  move.transform_4x4 = torch::tensor({{0.0f, 1.0f, 0.0f, 0.0f},
                                     {-1.0f, 0.0f, 0.0f, 0.0f},
                                     {0.0f, 0.0f, 1.0f, 0.0f},
                                     {3.0f, 4.0f, 5.0f, 1.0f}}, floats);
  const float c = std::sqrt(0.5f);
  move.rotation_wxyz = torch::tensor({c, 0.0f, 0.0f, c}, floats);
  auto result = applyGaussianTransform(state, move);
  close(result.xyz, torch::tensor({{3.0f, 6.0f, 5.0f}, {1.0f, 4.0f, 5.0f}}, floats));
  close(result.log_scaling, torch::full({2, 3}, std::log(2.0f), floats));
  close(result.rotation, move.rotation_wxyz.reshape({1, 4}).repeat({2, 1}));
  move.changed_attributes = toBits(GaussianAttrFlag::XYZ);
  auto position_only = applyGaussianTransform(state, move);
  close(position_only.rotation, state.rotation);
  close(position_only.log_scaling, state.log_scaling);
}

void anchorsAndSelector() {
  ChangeDistillConfig config;
  config.enabled = true;
  config.seed = 123;
  config.epochs = 2;
  config.batch_size = 1;
  config.anchor_max_count = 2;
  ChangeDistillManager distiller(config), identical(config);
  auto unaffected = torch::tensor({1, 3, 4}, longs);
  auto anchors = distiller.sampleAnchors(unaffected, 2);
  check(torch::equal(anchors, identical.sampleAnchors(unaffected, 2)), "Anchor seed is not deterministic");
  check(anchors.numel() == 2, "Incorrect anchor count");
  auto sorted = std::get<0>(anchors.sort());
  check(sorted[0].item<int64_t>() != sorted[1].item<int64_t>(), "Duplicate anchor");
  check(((anchors.unsqueeze(1) == unaffected.unsqueeze(0)).any(1)).all().item<bool>(),
        "Anchor escaped unaffected population");
  check(distiller.sampleAnchors(unaffected, 0).numel() == 0, "Empty corrected set produced anchors");
  check(distiller.sampleAnchors(unaffected.slice(0, 0, 0), 2).numel() == 0, "Empty unaffected set failed");

  ChangeCoordinator coordinator(nullptr, &distiller);
  auto state = master();
  auto mask = torch::tensor(
      {true, true, true, true, true, false},
      bools);
  auto capture = coordinator.captureBeforeGeometryChange(
      ChangeFlag::LOCAL_GEOMETRY, mask, state, {}, 10, 3, 20);
  check(capture.old_state.xyz.size(0) == 6, "Snapshot must contain full population");
  auto selector = makeSelector();
  {
    torch::NoGradGuard no_grad;
    for (const auto& p : selector->parameters()) p.fill_(0.02);
    auto weights = selector->named_parameters()["soft_net.weight"];
    weights.select(0, 0).fill_(-0.05);
    weights.select(0, 1).fill_(0.05);
  }
  torch::optim::Adam optimizer(selector->parameters(), torch::optim::AdamOptions(1e-4));
  prime(*selector, optimizer);
  capture.teacher_selector = improvements::selector::GumbelNetwork(selector->cloneFrozen());
  auto optimizer_before = snapshot(optimizer);
  auto move = correction(torch::tensor({0, 2, 6}, longs));
  coordinator.addCorrection(capture, move);
  check(capture.corrections.front().indices.numel() == 2, "Appended Gaussian entered teacher set");
  state.xyz.index_put_({0, 0}, 0.5f);
  state.xyz.index_put_({2, 0}, 0.8f);
  state.xyz = torch::cat({state.xyz, torch::zeros({1, 3}, floats)}).requires_grad_(true);
  state.log_scaling = torch::cat({state.log_scaling, torch::zeros({1, 3}, floats)}).requires_grad_(true);
  state.rotation = torch::cat({state.rotation, state.rotation.slice(0, 0, 1)}).requires_grad_(true);
  (state.xyz.sum() + state.log_scaling.sum() + state.rotation.sum()).backward();
  auto original_xyz = state.xyz.detach().clone();
  auto original_scale = state.log_scaling.detach().clone();
  auto original_rotation = state.rotation.detach().clone();
  auto xyz_grad = state.xyz.grad().clone();
  auto scale_grad = state.log_scaling.grad().clone();
  auto rotation_grad = state.rotation.grad().clone();
  ChangeTrainingContext context;
  context.current = state;
  context.selector = selector;
  context.normal_selector_optimizer = &optimizer;
  context.ratios = {0.1f, 0.5f};
  auto report = coordinator.finalizeGeometryChange(std::move(capture), context, 3);
  check(report.selector_count == 2 && report.anchor_count == 2, "Selector populations are wrong");
  check(report.selector_steps > 0 && report.selector_steps <= 4, "Unexpected Selector CD step count");
  check(report.tf_steps == 0, "Selector-only event changed TF");
  check(report.anchor_pre == 0.0, "Unaffected inputs changed before CD");
  checkState(optimizer, optimizer_before);
  close(state.xyz, original_xyz); close(state.log_scaling, original_scale);
  close(state.rotation, original_rotation);
  close(state.xyz.grad(), xyz_grad); close(state.log_scaling.grad(), scale_grad);
  close(state.rotation.grad(), rotation_grad);
}

void aabbOnlyAndOutside() {
  auto field = makeTF();
  auto selector = makeSelector();
  torch::optim::Adam tf_optimizer(field->mlpParameters(), torch::optim::AdamOptions(1e-4));
  tf_optimizer.add_param_group(field->gridParameters());
  torch::optim::Adam selector_optimizer(selector->parameters(), torch::optim::AdamOptions(1e-4));
  prime(*field, tf_optimizer);
  prime(*selector, selector_optimizer);
  {
    torch::NoGradGuard no_grad;
    field->named_parameters()["position_out.bias"].fill_(0.05);
  }
  auto state = master();
  state.xyz.index_put_({2, 0}, 1.5f); // Old-outside, new-inside.
  state.xyz.index_put_({3, 0}, 5.0f); // Remains outside the new domain.
  auto mask = torch::tensor(
      {true, true, true, true, true, false},
      bools);
  ChangeDistillConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 2;
  ChangeDistillManager distiller(cfg);
  DynamicAABBManager aabb_manager;
  ChangeCoordinator coordinator(&aabb_manager, &distiller);
  AABB3D domain{field->aabbMin(), field->aabbMax()};
  auto capture = coordinator.captureBeforeGeometryChange(
      ChangeFlag::FIELD_DOMAIN, mask, state, domain, 1, 1, 1000);
  capture.teacher_tf = TransformField(field->cloneFrozen());
  auto old_teacher = capture.teacher_tf;
  auto identity = old_teacher->applyLogResidual(capture.old_state.select(torch::tensor({2}, longs)), 0.1f);
  close(identity.xyz, state.xyz.slice(0, 2, 3));
  auto selector_before = snapshot(selector_optimizer);
  auto tf_before = snapshot(tf_optimizer);
  std::vector<torch::Tensor> selector_parameters, selector_gradients;
  auto pending_loss = torch::zeros({}, floats);
  for (const auto& p : selector->parameters()) pending_loss = pending_loss + p.sum();
  pending_loss.backward();
  for (const auto& p : selector->parameters()) {
    selector_parameters.push_back(p.detach().clone());
    selector_gradients.push_back(p.grad().clone());
  }
  AABBUpdatePlan plan;
  plan.should_expand = true;
  plan.old_aabb = domain.clone();
  plan.new_aabb = domain.clone();
  plan.new_aabb.max_xyz[0] = 3.0f;
  plan.affected_planes.xy = plan.affected_planes.xz = plan.affected_planes.xe = true;
  for (const auto& p : field->affectedParameters(plan.affected_planes))
    tf_before.erase(p.unsafeGetTensorImpl());
  ChangeTrainingContext context;
  context.current = state;
  context.selector = selector;
  context.tf = field;
  context.normal_selector_optimizer = &selector_optimizer;
  context.normal_tf_optimizer = &tf_optimizer;
  context.ratios = {0.1f, 0.5f};
  auto generator = at::globalContext().defaultGenerator(torch::Device(torch::kCPU));
  auto rng = generator.get_state();
  auto report = coordinator.finalizeGeometryChange(std::move(capture), context, 1, &plan);
  check(report.selector_steps == 0 && report.selector_count == 0, "AABB-only updated Selector");
  check(report.tf_count == 4 && report.skipped_outside == 1, "TF domain routing is wrong");
  check(report.tf_steps > 0, "TF CD did not optimize nonzero targets");
  check(torch::equal(rng, generator.get_state()), "CD consumed normal RNG");
  checkState(tf_optimizer, tf_before);
  checkState(selector_optimizer, selector_before);
  auto parameters = selector->parameters();
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    check(torch::equal(parameters[i], selector_parameters[i]), "AABB-only modified Selector parameters");
    check(torch::equal(parameters[i].grad(), selector_gradients[i]), "AABB-only cleared Selector gradients");
  }
  close(old_teacher->aabbMax(), plan.old_aabb.max_xyz);
  close(field->aabbMax(), plan.new_aabb.max_xyz);
}

void invalidCorrespondence() {
  ChangeDistillManager distiller;
  check(!distiller.config().enabled, "ChangeDistill must default to disabled");
  ChangeCoordinator coordinator(nullptr, &distiller);
  auto state = master();
  auto mask = torch::ones({6}, bools);
  ChangeTrainingContext context;
  context.current = state;
  context.ratios = {0.1f};
  for (int scenario = 0; scenario < 4; ++scenario) {
    auto capture = coordinator.captureBeforeGeometryChange(
        ChangeFlag::LOCAL_GEOMETRY, mask, state, {}, 1, 4, 20);
    bool rejected = false;
    try {
      auto move = correction(torch::tensor({0, 1}, longs));
      coordinator.addCorrection(capture, move);
      if (scenario == 0) coordinator.addCorrection(capture, move); // Overlap.
      if (scenario == 1) {
        capture.corrections.clear();
        move.indices = torch::tensor({0, 0}, longs);
        coordinator.addCorrection(capture, move); // Duplicate within a group.
      }
      context.current = scenario == 3 ? state.select(torch::arange(5, longs)) : state;
      coordinator.finalizeGeometryChange(std::move(capture), context, scenario == 2 ? 5 : 4);
    } catch (const c10::Error&) { rejected = true; }
    check(rejected, "Invalid teacher correspondence was accepted");
  }
}

void pressurePlanning() {
  DynamicAABBConfig cfg;
  cfg.outside_count_threshold = 2;
  cfg.bound_quantile_low = 0.0f;
  cfg.bound_quantile_high = 1.0f;
  DynamicAABBManager manager(cfg);
  AABB3D box{torch::full({3}, -1.0f, floats), torch::ones({3}, floats)};
  check(!manager.shouldPeriodicCheck(0) && manager.shouldPeriodicCheck(1000) &&
        !manager.shouldPeriodicCheck(1001), "Periodic mapping clock is wrong");
  auto xyz = torch::tensor({{1.5f, 0.0f, 0.0f}, {1.2f, 0.0f, 0.0f},
                            {0.0f, 0.0f, 0.0f}}, floats);
  check(!manager.shouldExpand(manager.evaluate(xyz.slice(0, 0, 1), box)), "Below-threshold trigger");
  auto pressure = manager.evaluate(xyz, box);
  check(pressure.outside_total == 2 && pressure.x_high_count == 2, "Outside count is wrong");
  auto plan = manager.makeExpansionPlan(xyz, box, pressure);
  check(plan.should_expand && plan.x_max_changed && !plan.x_min_changed, "Directional plan failed");
  close(plan.new_aabb.min_xyz, box.min_xyz);
  close(plan.new_aabb.max_xyz, torch::tensor({1.9f, 1.0f, 1.0f}, floats));
  check(plan.affected_planes.xy && plan.affected_planes.xz && plan.affected_planes.xe &&
        !plan.affected_planes.yz && !plan.affected_planes.ye && !plan.affected_planes.ze,
        "Affected planes are wrong");
  check(plan.new_aabb.containsMask(xyz).all().item<bool>(), "Planned domain misses data");
  check(!manager.shouldExpand(manager.evaluate(torch::zeros({2, 3}, floats), box)), "Inside points triggered change");
}

void robustPressurePlanning() {
  DynamicAABBManager manager; // Exercise the production quantiles and threshold.
  AABB3D box{torch::full({3}, -1.0f, floats), torch::ones({3}, floats)};
  for (int64_t count : {999, 1000, 1001, 10000}) {
    for (float direction : {-1.0f, 1.0f}) {
      auto xyz = torch::zeros({count, 3}, floats);
      xyz.index_put_({0, 0}, -direction * 1000.0f);
      xyz.index_put_({torch::indexing::Slice(1, 101), 0}, direction * 2.0f);
      auto pressure = manager.evaluate(xyz, box);
      check(pressure.outside_total == 101, "Robust test must trigger expansion");
      auto plan = manager.makeExpansionPlan(xyz, box, pressure);
      auto expected_min = box.min_xyz.clone();
      auto expected_max = box.max_xyz.clone();
      if (direction > 0) expected_max[0] = 2.4f;
      else expected_min[0] = -2.4f;
      check(plan.should_expand, "Robust bound lost the non-outlier population");
      close(plan.new_aabb.min_xyz, expected_min);
      close(plan.new_aabb.max_xyz, expected_max);
      check(plan.new_aabb.containsMask(xyz.slice(0, 1)).all().item<bool>(),
            "Robust plan excluded non-outlier points");
      check(!plan.new_aabb.containsMask(xyz.slice(0, 0, 1)).item<bool>(),
            "An extreme outlier enlarged the robust domain");
      check(plan.affected_planes.xy && plan.affected_planes.xz &&
                plan.affected_planes.xe && !plan.affected_planes.yz &&
                !plan.affected_planes.ye && !plan.affected_planes.ze,
            "Robust expansion changed an unrelated axis");
    }
  }

  DynamicAABBConfig small_config;
  small_config.outside_count_threshold = 1;
  DynamicAABBManager small_manager(small_config);
  const auto pair = torch::tensor({{-2.0f, 0.0f, 0.0f},
                                    {2.0f, 0.0f, 0.0f}}, floats);
  for (int64_t count : {1, 2}) {
    auto xyz = pair.slice(0, 0, count);
    auto plan = small_manager.makeExpansionPlan(
        xyz, box, small_manager.evaluate(xyz, box));
    check(plan.should_expand && plan.new_aabb.containsMask(xyz).all().item<bool>(),
          "Small-sample bounds crossed or lost their bracketing samples");
    close(plan.new_aabb.min_xyz, torch::tensor({-2.4f, -1.0f, -1.0f}, floats));
    close(plan.new_aabb.max_xyz, torch::tensor(
        {count == 2 ? 2.4f : 1.0f, 1.0f, 1.0f}, floats));
  }
}

void geometryScopes() {
  for (int scenario = 0; scenario < 3; ++scenario) {
    auto field = makeTF();
    auto selector = makeSelector();
    torch::optim::Adam tf_optimizer(field->mlpParameters(), torch::optim::AdamOptions(1e-4));
    tf_optimizer.add_param_group(field->gridParameters());
    torch::optim::Adam selector_optimizer(selector->parameters(), torch::optim::AdamOptions(1e-4));
    {
      torch::NoGradGuard no_grad;
      field->named_parameters()["position_out.bias"].fill_(0.05);
    }
    ChangeDistillConfig config;
    config.enabled = true;
    config.epochs = 2;
    ChangeDistillManager distiller(config);
    DynamicAABBManager aabb_manager;
    ChangeCoordinator coordinator(&aabb_manager, &distiller);
    auto state = master();
    auto mask = torch::tensor(
        {true, true, true, true, true, false},
        bools);
    AABB3D domain{field->aabbMin(), field->aabbMax()};
    const auto flag = scenario == 1 ? ChangeFlag::GLOBAL_GEOMETRY : ChangeFlag::LOCAL_GEOMETRY;
    auto capture = coordinator.captureBeforeGeometryChange(flag, mask, state, domain, 1, 1, 1000);
    capture.teacher_tf = TransformField(field->cloneFrozen());
    capture.teacher_selector = improvements::selector::GumbelNetwork(selector->cloneFrozen());
    auto move = correction(scenario == 1 ? torch::arange(6, longs) : torch::tensor({0, 2, 5}, longs));
    move.transform_4x4.index_put_({3, 0}, 0.2f);
    move.scale = 1.5f;
    auto transported = applyGaussianTransform(state.select(move.indices), move);
    state.xyz.index_copy_(0, move.indices, transported.xyz);
    state.log_scaling.index_copy_(0, move.indices, transported.log_scaling);
    state.rotation.index_copy_(0, move.indices, transported.rotation);
    if (scenario == 1) {
      move.affects_all = true;
      move.population_count = state.xyz.size(0);
      move.indices = torch::Tensor();
    }
    coordinator.addCorrection(capture, move);
    ChangeTrainingContext context;
    context.current = state;
    context.tf = field;
    context.selector = selector;
    context.normal_tf_optimizer = &tf_optimizer;
    context.normal_selector_optimizer = &selector_optimizer;
    context.ratios = {0.1f, 0.5f};
    AABBUpdatePlan plan;
    plan.should_expand = true;
    plan.old_aabb = domain.clone();
    plan.new_aabb = domain.clone();
    plan.new_aabb.max_xyz[0] = 3.0f;
    plan.affected_planes.xy = plan.affected_planes.xz = plan.affected_planes.xe = true;
    ChangeDistillReport report;
    std::string log;
    const auto stream_flags = std::cout.flags();
    const auto stream_precision = std::cout.precision();
    {
      StdoutCapture logging;
      report = coordinator.finalizeGeometryChange(
          std::move(capture), context, 1, scenario == 2 ? &plan : nullptr);
      log = logging.str();
    }
    const auto event_type = changeEventTypeName(toBits(flag) |
        (scenario == 2 ? toBits(ChangeFlag::FIELD_DOMAIN) : 0));
    const auto identity = "event=" + std::to_string(report.event_id) + " type=" + event_type;
    check(log.find("[ChangeCoordinator] " + identity) != std::string::npos &&
          log.find("[CD BEGIN] " + identity) != std::string::npos &&
          log.find("[CD END] " + identity) != std::string::npos,
          "CD lifecycle log lost event identity");
    checkEpochLogs(log, "[SELECTOR CD EPOCH] " + identity, config.epochs,
                   report.selector_pre + report.anchor_pre,
                   report.selector_post + report.anchor_post, report.selector_steps);
    checkEpochLogs(log, "[TF CD EPOCH] " + identity, config.epochs,
                   report.tf_pre, report.tf_post, report.tf_steps);
    check(std::cout.flags() == stream_flags && std::cout.precision() == stream_precision,
          "CD logging changed the global stream format");
    check(report.selector_count == (scenario == 1 ? 5 : 2), "Geometry Selector scope mismatch");
    check(report.tf_count == (scenario == 0 ? 2 : 5), "Geometry TF scope mismatch");
    check(report.anchor_count == (scenario == 1 ? 0 : 2), "Geometry anchor scope mismatch");
    check(report.skipped_outside == 0, "Unexpected outside exclusion");
    check(report.tf_pre > 0.0 && report.tf_steps > 0, "Transported TF target was not trained");
    check(tf_optimizer.state().empty() && selector_optimizer.state().empty(),
          "CD populated normal optimizer state");
  }
}

void splitEpochCompatibility() {
  struct Result {
    ChangeDistillReport report;
    std::vector<torch::Tensor> selector, tf;
  };
  const auto run = [&](ChangeDistillConfig config) {
    torch::manual_seed(37);
    auto selector = makeSelector();
    auto field = makeTF();
    {
      torch::NoGradGuard no_grad;
      for (const auto& p : selector->parameters()) p.fill_(0.02);
      auto weights = selector->named_parameters()["soft_net.weight"];
      weights.select(0, 0).fill_(-0.05);
      weights.select(0, 1).fill_(0.05);
      field->named_parameters()["position_out.bias"].fill_(0.05);
    }
    torch::optim::Adam selector_optimizer(selector->parameters(), torch::optim::AdamOptions(1e-4));
    torch::optim::Adam tf_optimizer(field->mlpParameters(), torch::optim::AdamOptions(1e-4));
    tf_optimizer.add_param_group(field->gridParameters());
    ChangeDistillManager distiller(config);
    ChangeCoordinator coordinator(nullptr, &distiller);
    auto state = master();
    const auto mask = torch::tensor({true, true, true, true, true, false}, bools);
    const AABB3D domain{field->aabbMin(), field->aabbMax()};
    auto capture = coordinator.captureBeforeGeometryChange(
        ChangeFlag::LOCAL_GEOMETRY, mask, state, domain, 1, 1, 1000);
    capture.teacher_selector = improvements::selector::GumbelNetwork(selector->cloneFrozen());
    capture.teacher_tf = TransformField(field->cloneFrozen());
    const auto move = correction(torch::tensor({0, 2}, longs));
    coordinator.addCorrection(capture, move);
    state.xyz = state.xyz.clone();
    state.xyz.index_put_({0, 0}, 0.5f);
    state.xyz.index_put_({2, 0}, 0.8f);
    ChangeTrainingContext context;
    context.current = state;
    context.selector = selector;
    context.tf = field;
    context.normal_selector_optimizer = &selector_optimizer;
    context.normal_tf_optimizer = &tf_optimizer;
    context.ratios = {0.1f, 0.5f};
    Result result;
    std::string log;
    {
      StdoutCapture output;
      result.report = coordinator.finalizeGeometryChange(std::move(capture), context, 1);
      log = output.str();
    }
    checkEpochLogs(log, "[SELECTOR CD EPOCH] event=1 type=LOCAL_GEOMETRY",
                   config.selectorEpochs(), result.report.selector_pre + result.report.anchor_pre,
                   result.report.selector_post + result.report.anchor_post, result.report.selector_steps);
    checkEpochLogs(log, "[TF CD EPOCH] event=1 type=LOCAL_GEOMETRY",
                   config.tfEpochs(), result.report.tf_pre, result.report.tf_post, result.report.tf_steps);
    for (const auto& p : selector->parameters()) result.selector.push_back(p.detach().clone());
    for (const auto& p : field->parameters()) result.tf.push_back(p.detach().clone());
    return result;
  };
  ChangeDistillConfig legacy;
  legacy.enabled = true;
  legacy.epochs = 5;
  legacy.batch_size = 1;
  legacy.seed = 123;
  check(legacy.selectorEpochs() == 5 && legacy.tfEpochs() == 5, "Legacy epoch fallback changed");
  auto explicit_counts = legacy;
  explicit_counts.selector_epochs = 5;
  explicit_counts.tf_epochs = 5;
  const auto before = run(legacy), after = run(explicit_counts);
  check(before.report.selector_steps == after.report.selector_steps &&
        before.report.tf_steps == after.report.tf_steps, "Explicit S5T5 changed optimizer steps");
  for (std::size_t i = 0; i < before.selector.size(); ++i) close(before.selector[i], after.selector[i]);
  for (std::size_t i = 0; i < before.tf.size(); ++i) close(before.tf[i], after.tf[i]);
  explicit_counts.selector_epochs = 2;
  explicit_counts.tf_epochs = 3;
  const auto asymmetric = run(explicit_counts);
  check(asymmetric.report.selector_steps == 4 && asymmetric.report.tf_steps == 6,
        "Branch epochs did not control their own optimizer steps");
  auto partial = legacy;
  partial.selector_epochs = 20;
  check(partial.selectorEpochs() == 20 && partial.tfEpochs() == 5, "Per-branch fallback changed");
  for (bool selector_branch : {false, true}) {
    auto invalid = legacy;
    if (selector_branch) invalid.selector_epochs = 0;
    else invalid.tf_epochs = -1;
    bool rejected = false;
    try { ChangeDistillManager manager(invalid); }
    catch (const c10::Error&) { rejected = true; }
    check(rejected, "Invalid branch epoch count accepted");
  }
}

void skippedDistillLogging() {
  for (bool outside : {false, true}) {
    auto field = makeTF();
    auto state = master();
    ChangeDistillConfig config;
    config.enabled = true;
    ChangeDistillManager distiller(config);
    ChangeCoordinator coordinator(nullptr, &distiller);
    auto capture = coordinator.captureBeforeGeometryChange(
        ChangeFlag::FIELD_DOMAIN,
        torch::full({6}, outside, bools),
        state, {field->aabbMin(), field->aabbMax()}, 1, 1, 1000);
    capture.teacher_tf = TransformField(field->cloneFrozen());
    torch::optim::Adam optimizer(field->mlpParameters(), torch::optim::AdamOptions(1e-4));
    optimizer.add_param_group(field->gridParameters());
    ChangeEvent event;
    event.id = 7;
    event.flags = toBits(ChangeFlag::FIELD_DOMAIN);
    event.tf_scope = DistillScope::ALL_SNAPSHOT_MATURE;
    event.correction_group_id = torch::full({6}, -1, longs);
    ChangeTrainingContext context;
    context.current = state;
    if (outside) context.current.xyz = state.xyz + 5.0f;
    context.tf = field;
    context.normal_tf_optimizer = &optimizer;
    context.ratios = {0.5f};
    StdoutCapture logging;
    auto report = distiller.execute(event, capture, context);
    const auto log = logging.str();
    const std::string reason = outside ? "no_in_domain_gaussians" : "empty_population";
    check(log.find("[TF CD SKIP] event=7 type=FIELD_DOMAIN reason=" + reason) != std::string::npos &&
          log.find("[TF CD EPOCH]") == std::string::npos &&
          log.find("[SELECTOR CD EPOCH]") == std::string::npos,
          "Skipped distillation was logged as a successful zero-loss epoch");
    check(report.tf_steps == 0 && report.skipped_outside == (outside ? 6 : 0),
          "Skip logging changed the distillation report");
  }
}

void globalCorrectionPopulation() {
  ChangeCoordinator coordinator(nullptr, nullptr);
  auto state = master();
  auto capture = coordinator.captureBeforeGeometryChange(
      ChangeFlag::GLOBAL_GEOMETRY,
      torch::ones({6}, bools),
      state, {}, 1, 1, 1000);
  auto move = correction(torch::Tensor());
  move.affects_all = true;
  move.population_count = 9; // Includes appended Gaussians, outside the snapshot.
  coordinator.addCorrection(capture, move);
  check(!move.indices.defined(), "Global source result allocated indices");
  check(capture.corrections.size() == 1, "Global correction was dropped");
  close(capture.corrections.front().indices, torch::arange(6, longs));
  check(capture.corrections.front().population_count == 9, "Source population was rewritten");
  capture.corrections.clear();
  move.population_count = 0;
  coordinator.addCorrection(capture, move);
  check(capture.corrections.empty(), "Empty global correction was recorded");
  move.population_count = 9;
  move.changed_attributes = toBits(GaussianAttrFlag::NONE);
  coordinator.addCorrection(capture, move);
  check(capture.corrections.empty(), "No-change correction was recorded");
}

void expansionWithoutDistill() {
  // Periodic plan, immediate geometry check, and no distiller at all.
  for (int scenario = 0; scenario < 3; ++scenario) {
    auto field = makeTF();
    torch::optim::Adam optimizer(field->mlpParameters(), torch::optim::AdamOptions(1e-4));
    optimizer.add_param_group(field->gridParameters());
    prime(*field, optimizer);
    auto before_state = snapshot(optimizer);
    auto parameters = field->parameters();
    std::vector<torch::Tensor> before_parameters;
    for (const auto& p : parameters) before_parameters.push_back(p.detach().clone());
    ChangeDistillConfig cd_config;
    cd_config.enabled = false;
    ChangeDistillManager distiller(cd_config);
    DynamicAABBConfig aabb_config;
    aabb_config.outside_count_threshold = 1;
    aabb_config.bound_quantile_low = 0.0f;
    aabb_config.bound_quantile_high = 1.0f;
    DynamicAABBManager manager(aabb_config);
    ChangeCoordinator coordinator(&manager, scenario == 2 ? nullptr : &distiller);
    auto state = master();
    auto mask = torch::tensor({true, true, true, true, true, false},
                             bools);
    if (scenario != 1) state.xyz.index_put_({0, 0}, 1.5f);
    state.xyz.index_put_({5, 0}, 1000.0f); // Protected, never used to plan.
    AABB3D domain{field->aabbMin(), field->aabbMax()};
    auto capture = coordinator.captureBeforeGeometryChange(
        scenario == 1 ? ChangeFlag::GLOBAL_GEOMETRY : ChangeFlag::FIELD_DOMAIN,
        mask, state, domain, 1, 1, 1000);
    if (scenario == 1) {
      auto move = correction(torch::Tensor());
      move.affects_all = true;
      move.population_count = 6;
      move.transform_4x4.index_put_({3, 0}, 1.5f);
      state = applyGaussianTransform(state, move);
      coordinator.addCorrection(capture, move);
      state.xyz = torch::cat({state.xyz, torch::full({1, 3}, 5000.0f, floats)});
      state.log_scaling = torch::cat({state.log_scaling, torch::zeros({1, 3}, floats)});
      state.rotation = torch::cat({state.rotation, state.rotation.slice(0, 0, 1)});
    }
    auto mature_xyz = state.xyz.index_select(0, capture.mature_indices_before);
    auto plan = manager.makeExpansionPlan(mature_xyz, domain, manager.evaluate(mature_xyz, domain));
    check(plan.should_expand, "Test did not create AABB pressure");
    const auto affected = field->affectedParameters(plan.affected_planes);
    for (const auto& p : affected) before_state.erase(p.unsafeGetTensorImpl());
    ChangeTrainingContext context;
    context.current = state;
    context.tf = field;
    context.normal_tf_optimizer = &optimizer;
    // No teachers or ratios: a disabled CD path must not need either.
    auto generator = at::globalContext().defaultGenerator(torch::Device(torch::kCPU));
    const auto rng = generator.get_state();
    auto report = coordinator.finalizeGeometryChange(
        std::move(capture), context, 1, scenario == 1 ? nullptr : &plan);
    check(report.event_id != 0 && report.selector_steps == 0 && report.tf_steps == 0,
          "AABB change did not remain independent of CD");
    close(field->aabbMin(), plan.new_aabb.min_xyz);
    close(field->aabbMax(), plan.new_aabb.max_xyz);
    check(field->aabbMax()[0].item<float>() < 3.0f, "Protected/appended points enlarged the domain");
    check(torch::equal(rng, generator.get_state()), "AABB without CD consumed normal RNG");
    checkState(optimizer, before_state);
    bool changed = false;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      bool resampled = false;
      for (const auto& p : affected)
        resampled |= p.unsafeGetTensorImpl() == parameters[i].unsafeGetTensorImpl();
      if (resampled) changed |= !torch::equal(parameters[i], before_parameters[i]);
      else check(torch::equal(parameters[i], before_parameters[i]), "Unresampled parameters changed");
    }
    check(changed, "CD-disabled expansion did not resample planes");
  }
}

void independentLearningRates() {
  std::array<std::vector<torch::Tensor>, 3> reference;
  for (int scenario = 0; scenario < 5; ++scenario) {
    torch::manual_seed(117);
    auto field = makeTF();
    auto selector = makeSelector();
    {
      torch::NoGradGuard no_grad;
      // Positive activations guarantee nonzero grid gradients in the LR test.
      for (const auto& p : field->mlpParameters()) p.fill_(0.02);
    }
    const double normal_lr = scenario == 1 ? 0.4 : 1e-8;
    torch::optim::Adam selector_optimizer(selector->parameters(),
        torch::optim::AdamOptions(normal_lr).weight_decay(0.1));
    torch::optim::Adam tf_optimizer(field->mlpParameters(),
        torch::optim::AdamOptions(normal_lr).weight_decay(0.1));
    tf_optimizer.add_param_group(field->gridParameters());
    tf_optimizer.param_groups()[1].options().set_lr(normal_lr * 2);
    ChangeDistillConfig cfg;
    cfg.enabled = true;
    if (scenario == 2) cfg.selector_learning_rate *= 2;
    if (scenario == 3) cfg.tf_mlp_learning_rate *= 2;
    if (scenario == 4) cfg.tf_grid_learning_rate *= 2;
    ChangeDistillManager distiller(cfg);
    ChangeCoordinator coordinator(nullptr, &distiller);
    auto state = master();
    auto capture = coordinator.captureBeforeGeometryChange(
        ChangeFlag::GLOBAL_GEOMETRY,
        torch::ones({6}, bools),
        state, {field->aabbMin(), field->aabbMax()}, 1, 1, 1000);
    capture.teacher_tf = TransformField(field->cloneFrozen());
    capture.teacher_selector = improvements::selector::GumbelNetwork(selector->cloneFrozen());
    {
      torch::NoGradGuard no_grad;
      selector->named_parameters()["soft_net.bias"][1].add_(0.1);
      field->named_parameters()["position_out.bias"].add_(0.05);
    }
    ChangeEvent event;
    event.selector_scope = event.tf_scope = DistillScope::ALL_SNAPSHOT_MATURE;
    event.correction_group_id = torch::full({6}, -1, longs);
    ChangeTrainingContext context;
    context.current = state;
    context.selector = selector;
    context.tf = field;
    context.normal_selector_optimizer = &selector_optimizer;
    context.normal_tf_optimizer = &tf_optimizer;
    context.ratios = {0.5f};
    auto report = distiller.execute(event, capture, context);
    check(report.selector_steps == 1 && report.tf_steps == 1, "LR test requires one CD step");
    std::array<std::vector<torch::Tensor>, 3> groups = {
        selector->parameters(), field->mlpParameters(), field->gridParameters()};
    for (std::size_t g = 0; g < groups.size(); ++g) {
      bool changed = false;
      for (std::size_t i = 0; i < groups[g].size(); ++i) {
        if (scenario == 0) reference[g].push_back(groups[g][i].detach().clone());
        else changed |= !torch::equal(groups[g][i], reference[g][i]);
      }
      if (scenario > 0)
        check(changed == (scenario == static_cast<int>(g) + 2),
              "CD learning rate did not control exactly its own parameter group");
    }
    check(selector_optimizer.state().empty() && tf_optimizer.state().empty(),
          "CD used normal Adam state");
    check(static_cast<const torch::optim::AdamOptions&>(selector_optimizer.param_groups()[0].options()).lr() == normal_lr &&
          static_cast<const torch::optim::AdamOptions&>(tf_optimizer.param_groups()[0].options()).lr() == normal_lr &&
          static_cast<const torch::optim::AdamOptions&>(tf_optimizer.param_groups()[1].options()).lr() == normal_lr * 2,
          "CD changed normal learning rates");
  }
  for (auto member : {&ChangeDistillConfig::selector_learning_rate,
                      &ChangeDistillConfig::tf_mlp_learning_rate,
                      &ChangeDistillConfig::tf_grid_learning_rate}) {
    for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
      ChangeDistillConfig config;
      config.*member = invalid;
      bool rejected = false;
      try { ChangeDistillManager invalid_manager(config); }
      catch (const c10::Error&) { rejected = true; }
      check(rejected, "Invalid CD learning rate was accepted");
    }
  }
}

} // namespace

int main() {
  torch::manual_seed(19);
  eventNames();
  transport();
  anchorsAndSelector();
  aabbOnlyAndOutside();
  invalidCorrespondence();
  pressurePlanning();
  robustPressurePlanning();
  geometryScopes();
  splitEpochCompatibility();
  skippedDistillLogging();
  globalCorrectionPopulation();
  expansionWithoutDistill();
  independentLearningRates();
}
