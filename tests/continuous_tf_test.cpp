#include "include/transform_field.h"
#include "include/transform_field_cuda_optimizer.h"
#include "include/gaussian_protection.h"
#include "include/change_management.h"
#include <stdexcept>

void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main(int argc, char**) {
  const bool cuda = argc > 1;
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(cuda ? torch::kCUDA : torch::kCPU);
  torch::manual_seed(0);
  TransformFieldConfig config;
  config.backend = cuda ? "cuda" : "legacy";
  config.hexplane.spatial_resolution = 4; config.hexplane.ratio_resolution = 4;
  config.hexplane.feature_dim = 2; config.hexplane.multires = {1}; config.hidden_dim = 8;
  TransformField field(torch::full({3}, -2., options), torch::full({3}, 2., options), config);
  field->to(options.device());
  {
    torch::NoGradGuard guard;
    for (const auto& p : field->parameters()) p.zero_();
    const auto params = field->named_parameters();
    params["position_out.bias"].fill_(.3);
    params["scale_out.bias"].fill_(.2);
    params["rotation_out.bias"].select(0, 1).fill_(.6);
  }
  const auto xyz = torch::zeros({3,3}, options);
  const auto scale = torch::ones({3,3}, options);
  auto rotation = torch::zeros({3,4}, options); rotation.select(1,0).fill_(1);
  auto maturity = torch::tensor({0.f,.5f,1.f}, options).requires_grad_(true);
  auto result = field->applyResidual(xyz, scale, rotation, .1, maturity);
  const auto full = field->applyResidual(xyz, scale, rotation, .1);
  check(result.xyz[0].equal(xyz[0]) && result.scaling[0].equal(scale[0]) &&
      result.rotation[0].equal(rotation[0]), "m=0 must be strict identity");
  check(torch::allclose(result.xyz[1], .5 * full.xyz[1]) &&
      torch::allclose(result.scaling[1].log(), .5 * full.scaling[1].log()), "m=.5 displacement/log scale");
  check(result.rotation[1][1].item<float>() > 0 &&
      result.rotation[1][1].item<float>() < full.rotation[1][1].item<float>(), "m=.5 intermediate rotation");
  check(torch::allclose(result.xyz[2], full.xyz[2]) && torch::allclose(result.scaling[2], full.scaling[2]) &&
      torch::allclose(result.rotation[2], full.rotation[2]), "m=1 must match full correction");
  (result.xyz.sum() + result.scaling.sum() + result.rotation.sum()).backward();
  check(!maturity.grad().defined(), "maturity must be detached");
  field->zero_grad(true);
  auto zero = field->applyResidual(xyz, scale, rotation, .1, torch::zeros({3},options));
  (zero.xyz.sum()+zero.scaling.sum()+zero.rotation.sum()).backward();
  for (const auto& p : field->parameters())
    check(!p.grad().defined() || p.grad().abs().sum().item<float>() == 0, "m=0 TF gradient must vanish");
  const auto m = torch::tensor({0.f,.049f,.05f,1.f},options);
  const auto master_xyz = torch::zeros({4,3},options);
  check(field->candidateMask(master_xyz,m).all().item<bool>(), "epsilon=0 must admit all eligible points");
  config.maturity_compute_epsilon=.05;
  TransformField cutoff(torch::full({3},-2.,options),torch::full({3},2.,options),config);
  cutoff->to(options.device());
  const auto mask = cutoff->candidateMask(master_xyz,m);
  check(mask.equal(torch::tensor({false,false,true,true},options.dtype(torch::kBool))), "epsilon boundary");
  check(!cutoff->candidateMask(torch::full({4,3},3.,options),m).any().item<bool>(), "outside AABB must be identity");
  const auto master_scale=torch::ones({4,3},options);
  const auto master_rotation=torch::cat({rotation,rotation.slice(0,0,1)},0);
  const auto backup_xyz=master_xyz.clone(), backup_scale=master_scale.clone(), backup_rotation=master_rotation.clone();
  const auto indices=torch::nonzero(mask).reshape({-1});
  const auto output=cutoff->applyResidual(master_xyz.index_select(0,indices),master_scale.index_select(0,indices),
      master_rotation.index_select(0,indices),.1,m.index_select(0,indices));
  const auto rendered=master_xyz.index_copy(0,indices,output.xyz);
  check(rendered.slice(0,0,2).equal(master_xyz.slice(0,0,2)), "cutoff points must remain identity");
  check(master_xyz.equal(backup_xyz)&&master_scale.equal(backup_scale)&&master_rotation.equal(backup_rotation),
      "cutoff must not mutate master attributes");
  config.start_iter=3000;
  check(!config.ready(2999,10)&&config.ready(3000,-1), "fixed iteration semantics");
  config.start_mode="first_maturity_cycle";
  GaussianProtection protection; protection.reset(3,options);
  protection.mapping_iteration=5; protection.markTouched(torch::tensor({0},options.dtype(torch::kLong)));
  protection.observeFull(xyz,scale.log(),rotation);
  check(protection.first_valid_maturity_cycle_iteration<0 && !config.ready(5,-1), "first snapshot is not a cycle");
  protection.mapping_iteration=10;
  protection.observeFull(xyz,scale.log(),rotation);
  check(protection.first_valid_maturity_cycle_iteration<0, "untouched snapshots are not valid observations");
  protection.mapping_iteration=15; protection.markTouched(torch::tensor({0},options.dtype(torch::kLong)));
  protection.observeFull(xyz,scale.log(),rotation);
  check(protection.first_valid_maturity_cycle_iteration==15 && config.ready(15,15), "first genuine cycle starts TF");
  // TF domain expansion must work even when the separate CD population is empty.
  DynamicAABBConfig domain_config; domain_config.outside_count_threshold=1;
  domain_config.bound_quantile_low=0; domain_config.bound_quantile_high=1;
  DynamicAABBManager domain_manager(domain_config);
  ChangeCoordinator coordinator(&domain_manager,nullptr);
  const auto domain_xyz=torch::full({3,3},3.,options);
  GaussianAttrState domain_state{domain_xyz,scale.log(),rotation};
  AABB3D domain{field->aabbMin(),field->aabbMax()};
  const auto plan=domain_manager.makeExpansionPlan(domain_xyz,domain,domain_manager.evaluate(domain_xyz,domain));
  auto capture=coordinator.captureBeforeGeometryChange(ChangeFlag::FIELD_DOMAIN,
      torch::zeros({3},options.dtype(torch::kBool)),domain_state,domain,0,0,15);
  ChangeTrainingContext context; context.current=domain_state; context.tf=field;
  const auto expanded=coordinator.finalizeGeometryChange(std::move(capture),context,0,&plan);
  check(expanded.aabb_changed && field->insideMask(domain_xyz).all().item<bool>(), "CD compatibility gate blocked TF AABB");
  // Warm Adam momentum, then a skipped TF step must leave both parameters and state untouched.
  auto gaussian=torch::ones({1},options).requires_grad_(true);
  torch::optim::Adam optimizer({gaussian},torch::optim::AdamOptions(.01));
  optimizer.add_param_group(field->parameters());
  optimizer.zero_grad(true); (field->applyResidual(xyz,scale,rotation,.1,maturity.detach()).xyz.sum()+gaussian.sum()).backward();
  if(cuda) tf_cuda::tfCudaAdamMixedStep(optimizer,{1}); else optimizer.step();
  const auto first=field->parameters().front();
  const auto parameter_before=first.detach().clone();
  const auto step_before=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(first.unsafeGetTensorImpl())).step();
  optimizer.zero_grad(true); gaussian.sum().backward();
  if(cuda) tf_cuda::tfCudaAdamMixedStep(optimizer,{}); else optimizer.step();
  check(first.equal(parameter_before)&&static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(first.unsafeGetTensorImpl())).step()==step_before,
      "skipped TF must not advance Adam momentum or parameters");
  return 0;
}
