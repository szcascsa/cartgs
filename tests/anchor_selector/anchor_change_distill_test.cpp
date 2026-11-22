#include "include/change_management.h"
#include <iostream>

int main(int argc, char** argv) {
  torch::manual_seed(2026);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto xyz = torch::tensor({{.01f,.02f,.03f},{.08f,.04f,.07f},{.21f,.1f,.12f},{.42f,.2f,.1f}}, options);
  const bool adaptive = argc > 2 && std::string(argv[2]) == "adaptive";
  if (adaptive) xyz = torch::cat({xyz,xyz.slice(0,0,1).repeat({150,1})});
  const auto n = xyz.size(0);
  GaussianAttrState before{xyz, torch::zeros({n,3}), torch::tensor({{1.f,0.f,0.f,0.f}}).repeat({n,1})};
  const bool attributes = argc > 1;
  auto model = improvements::selector::GumbelNetwork(.1, attributes,argc>3 && std::string(argv[3])=="global_xyz");
  if (adaptive) model->anchors()->enableAdaptive();
  auto dc = torch::randn({n,1,3}, options), opacity = torch::full({n,1}, .6f, options);
  model->anchors()->reset(xyz);
  model->anchors()->refine(xyz);
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(.001));
  ChangeDistillConfig config; config.enabled = true; config.epochs = 2;
  ChangeDistillManager manager(config);
  ChangeCoordinator coordinator(nullptr, &manager);
  auto capture = coordinator.captureBeforeGeometryChange(ChangeFlag::LOCAL_GEOMETRY,
      torch::ones({n}, torch::kBool), before, {}, 0, 0, 0);
  capture.teacher_selector = improvements::selector::GumbelNetwork(model->cloneFrozen());
  if (attributes) {
    capture.old_selector_sh_dc = dc.clone();
    capture.old_selector_opacity = opacity.clone();
  }
  auto frozen = capture.teacher_selector;
  const auto teacher_probability = frozen->forwardProbability(xyz, {}, before.log_scaling.exp(), .1f, 1, {}, dc, opacity).detach().clone();
  GaussianTransformResult correction;
  correction.indices = torch::tensor({0}, torch::kLong);
  correction.population_count = n;
  correction.transform_4x4 = torch::eye(4);
  correction.transform_4x4.index_put_({3,0}, .15f);
  correction.rotation_wxyz = torch::tensor({1.f,0.f,0.f,0.f});
  correction.changed_attributes = toBits(GaussianAttrFlag::XYZ);
  coordinator.addCorrection(capture, correction);
  auto after = xyz.clone(); after.index_put_({0}, xyz[0] + torch::tensor({.15f,0.f,0.f}));
  model->anchors()->reassociate(after, {&optimizer});
  model->anchors()->refine(after, {&optimizer});
  ChangeTrainingContext context;
  context.current = {after, before.log_scaling, before.rotation};
  context.selector_sh_dc = dc;
  context.selector_opacity = opacity;
  context.selector = model;
  context.normal_selector_optimizer = &optimizer;
  context.ratios = {.01f,.05f,.1f,.15f};
  auto report = coordinator.finalizeGeometryChange(std::move(capture), context, 0, nullptr);
  TORCH_CHECK(report.selector_count == (adaptive ? 151 : 2) && report.selector_steps > 0,
              "Partial correction must distill both members of the old Anchor");
  TORCH_CHECK(torch::equal(teacher_probability, frozen->forwardProbability(xyz, {}, before.log_scaling.exp(), .1f, 1, {}, dc, opacity)), "Frozen teacher changed");
  TORCH_CHECK(torch::equal(after, context.current.xyz), "Selector CD changed explicit geometry");
  std::cout << "PASS: partial Anchor loop correction distills all co-members; frozen teacher and explicit geometry unchanged\n";
}
