#include "include/ultra_light_schedule.h"
#include "include/ultrafast_rotation.h"
#include "include/gradient_constraint_clock.h"
#include <torch/torch.h>
#include <stdexcept>
#include <vector>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void cadenceAndShutdown() {
  UltrafastRotation schedule;
  schedule.enabled = true;
  for (std::uint64_t mapping = 0; mapping < 12; ++mapping) {
    check(schedule.phase() == mapping % 5, "Ultra-light did not use ratio rotation");
    schedule.complete(1.0, 0.5, true, !schedule.full(), !schedule.full());
  }
  schedule.endOnline(12);
  check(schedule.tailSteps() == 0, "Online boundary already contains tail steps");
  schedule.completeRotatingTail(1.0, .5, true, true, true, true);
  check(schedule.tailSteps() == 1 && schedule.tailRatioIndex() == 3 && schedule.balanced() &&
        schedule.full_render_count == 2 && schedule.lod_render_count == 11 &&
        schedule.online_end_mapping_iteration == 12, "Tail failed to continue the next LoD slot");
}
void periodicMappingClocks() {
  DeferredMappingTask task;
  std::vector<std::uint64_t> executed;
  for (std::uint64_t mapping = 1; mapping <= 2107; ++mapping) {
    task.observe(mapping, 1000);
    if (task.take(true)) executed.push_back(mapping);
  }
  check(executed == std::vector<std::uint64_t>({1000, 2000}),
        "Refresh interval was rescaled to ratio cycles");
  task.observe(3000, 1000);
  check(!task.take(false) && task.pending, "A deferred task consumed pending work");
  check(task.take(true) && !task.take(true), "Deferred work did not execute exactly once");
  task.observe(0, 1);
  task.observe(10, 0);
  check(!task.pending, "Disabled/zero mapping scheduled a task");
}
void sharedAdamSkipsAuxiliaryState() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
  auto gaussian = torch::ones({2}, options);
  auto selector = torch::ones({2}, options);
  auto field = torch::ones({2}, options);
  torch::optim::Adam shared({gaussian}, torch::optim::AdamOptions(0.01));
  shared.add_param_group(torch::optim::OptimizerParamGroup({selector}));
  shared.add_param_group(torch::optim::OptimizerParamGroup({field}));
  torch::optim::Adam selector_optimizer({selector}, torch::optim::AdamOptions(0.01));
  torch::optim::Adam field_optimizer({field}, torch::optim::AdamOptions(0.01));
  (gaussian.square().sum() + selector.square().sum() + field.square().sum()).backward();
  shared.step();
  const auto selector_before = selector.detach().clone();
  const auto field_before = field.detach().clone();
  auto& selector_state = static_cast<torch::optim::AdamParamState&>(
      *shared.state().at(selector.unsafeGetTensorImpl()));
  auto& field_state = static_cast<torch::optim::AdamParamState&>(
      *shared.state().at(field.unsafeGetTensorImpl()));
  const auto selector_moment = selector_state.exp_avg().clone();
  const auto field_moment = field_state.exp_avg().clone();
  const auto selector_step = selector_state.step(), field_step = field_state.step();
  shared.zero_grad(true);
  // Simulate leftover auxiliary gradients from an event before a Full mapping.
  (selector.sum() + field.sum()).backward();
  selector_optimizer.zero_grad(true);
  field_optimizer.zero_grad(true);
  MappingUpdateClock clock;
  gaussian.square().sum().backward();
  shared.step();
  clock.record(true, false);
  shared.zero_grad(true);
  check(selector.equal(selector_before) && field.equal(field_before),
        "Full mapping updated auxiliary parameters through Adam momentum");
  check(selector_state.step() == selector_step && field_state.step() == field_step &&
        selector_state.exp_avg().equal(selector_moment) && field_state.exp_avg().equal(field_moment),
        "Full mapping advanced auxiliary Adam state");
  check(clock.tf_updates == 0 && clock.gaussianLRStep(1) == 1,
        "Full mapping advanced TF warmup or stalled the Gaussian schedule");
  (gaussian.square().sum() + selector.square().sum() + field.square().sum()).backward();
  shared.step();
  clock.record(true, true);
  check(clock.tf_updates == 1 && selector_state.step() == selector_step + 1,
        "The LOD mapping did not resume auxiliary updates");
}
}  // namespace
int main() {
  cadenceAndShutdown();
  periodicMappingClocks();
  sharedAdamSkipsAuxiliaryState();
}
