#include "include/change_management.h"
#include <iostream>

int main(int argc,char** argv) {
  torch::manual_seed(2026);
  auto device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  const auto width=argc>2&&std::string(argv[2])=="legacy"?10:14;
  const int64_t feature_dim=argc>2&&std::string(argv[2])=="feature32"?32:16;
  auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto model=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},feature_dim);model->to(device);
  TORCH_CHECK(model->anchors()->features().size(1)==feature_dim,"Feature dimension not applied");
  auto bank=model->anchors();bank->enableVariableMembers();
  auto observed=torch::tensor({{.01f,.02f,2.f},{.04f,.02f,2.f},{.31f,.02f,2.f}},opts);
  auto seeds=bank->observeBlocks(observed,torch::full_like(observed,.4));
  auto xyz=seeds.xyz,scale=seeds.sigma,dc=seeds.color.unsqueeze(1),alpha=torch::full({xyz.size(0),1},.2,opts);
  auto opacity_logits=torch::logit(alpha);
  auto rotation=torch::zeros({xyz.size(0),4},opts);rotation.select(1,0).fill_(1);
  GaussianAttrState state{xyz,scale.log(),rotation};
  std::vector<torch::Tensor> old;
  for(float r:{.01f,.05f,.1f,.15f})old.push_back(model->forwardAnchorProbability(xyz,r,1,{},scale,dc,alpha).detach().clone());
  // Head initialization must not change the subsequent cached-uniform RNG stream.
  torch::manual_seed(99);auto expected=torch::rand({10});torch::manual_seed(99);
  model->enableCompensation(width);TORCH_CHECK(torch::equal(expected,torch::rand({10})),"Head changed RNG stream");
  int j=0;
  for(float r:{.01f,.05f,.1f,.15f}) {
    auto encoded=model->forwardSelectionFeatures(xyz,r,1,{},scale,dc,alpha);
    TORCH_CHECK(torch::equal(old[j++],encoded.probability),"Shared encoding changed selection probability");
    auto residual=model->compensationResidual(encoded.shared);
    TORCH_CHECK(residual.size(1)==width,"Wrong compensation output width");
    TORCH_CHECK(residual.count_nonzero().item<int64_t>()==0,"Head initial output not exact zero");
    auto after=model->applyLogCompensation(state,r,{},dc,alpha);
    TORCH_CHECK(torch::equal(after.xyz,xyz)&&torch::equal(after.log_scaling,state.log_scaling)&&
                torch::equal(after.rotation,rotation),"Initial compensation not identity");
    auto appearance=model->applyAppearanceResidual(dc,opacity_logits,residual);
    TORCH_CHECK(torch::equal(appearance.first,dc)&&torch::equal(appearance.second,opacity_logits)&&
                torch::equal(appearance.second.sigmoid(),opacity_logits.sigmoid()),"Initial appearance not identity");
    auto saturated=torch::full_like(opacity_logits,100);saturated[0].fill_(-100);
    auto saturated_after=model->applyAppearanceResidual(dc,saturated,residual);
    TORCH_CHECK(torch::equal(saturated_after.second,saturated)&&torch::isfinite(saturated_after.second).all().item<bool>(),
                "Saturated finite logits were reconstructed through logit(sigmoid)");
  }
  torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(.001));
  auto ids=torch::tensor({0,2},opts.dtype(torch::kLong));
  for(int step=0;step<3;++step) {
    optimizer.zero_grad(true);
    auto encoded=model->forwardSelectionFeatures(xyz,.05,1,{},scale,dc,alpha);
    auto residual=model->compensationResidual(encoded.shared.index_select(0,ids));
    TORCH_CHECK(residual.size(0)==ids.numel(),"Head decoded unselected rows");
    auto appearance=model->applyAppearanceResidual(dc.index_select(0,ids),opacity_logits.index_select(0,ids),residual);
    auto loss=(residual.slice(1,0,10)-.03).square().mean();
    if(width==14)loss=loss+(appearance.second.sigmoid()-.3).square().mean()+(appearance.first-.5).square().mean();
    loss.backward();
    TORCH_CHECK(model->named_parameters()["compensation.2.weight"].grad().abs().sum().item<float>()>0,"Head gradient missing");
    if(width==14)for(int k=10;k<14;++k)
      TORCH_CHECK(model->named_parameters()["compensation.2.weight"].grad()[k].abs().sum().item<float>()>0,
                  "Opacity/color channel gradient missing");
    if(step>0)for(auto name:{"local_emd.0.weight","anchors.features","compensation.0.weight"}) {
      auto grad=model->named_parameters()[name].grad();
      TORCH_CHECK(grad.defined()&&torch::isfinite(grad).all().item<bool>()&&grad.abs().sum().item<float>()>0,
                  "Compensation upstream gradient missing: ",name);
    }
    optimizer.step();
  }
  optimizer.zero_grad(true);
  auto frozen=improvements::selector::GumbelNetwork(model->cloneFrozen());
  auto before=frozen->applyLogCompensation(state,.05,{},dc,alpha);
  auto teacher_residual=frozen->forwardLogCompensationResidual(state,.05,{},dc,alpha).detach().clone();
  auto dc_before=dc.clone(),logits_before=opacity_logits.clone();
  ChangeDistillConfig config;config.enabled=true;config.epochs=3;
  ChangeDistillManager manager(config);ChangeCoordinator coordinator(nullptr,&manager);
  AABB3D aabb; aabb.min_xyz=std::get<0>(xyz.min(0)); aabb.max_xyz=std::get<0>(xyz.max(0));
  auto capture=coordinator.captureBeforeGeometryChange(ChangeFlag::LOCAL_GEOMETRY,
      torch::ones({xyz.size(0)},opts.dtype(torch::kBool)),state,aabb,0,0,0);
  capture.teacher_selector=frozen;capture.old_selector_sh_dc=dc.clone();capture.old_selector_opacity=alpha.clone();
  capture.old_selector_opacity_logits=opacity_logits.clone();
  GaussianTransformResult correction;correction.indices=ids;correction.population_count=xyz.size(0);
  correction.transform_4x4=torch::eye(4,opts);correction.transform_4x4[3][0]=.1;
  correction.rotation_wxyz=torch::tensor({1.f,0.f,0.f,0.f},opts);correction.changed_attributes=toBits(GaussianAttrFlag::XYZ);
  coordinator.addCorrection(capture,correction);
  auto current=xyz.clone();current.index_put_({ids},current.index_select(0,ids)+torch::tensor({.1f,0.f,0.f},opts));
  bank->beginBlockGeometry();bank->transportBlocks(correction);bank->finishBlockGeometry();
  auto normal_moment=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg().clone();
  ChangeTrainingContext context;context.current={current,state.log_scaling,rotation};
  context.selector=model;context.selector_sh_dc=dc;context.selector_opacity=alpha;
  context.selector_opacity_logits=opacity_logits;
  context.normal_selector_optimizer=&optimizer;context.selector_compensation_active=true;context.ratios={.01f,.05f,.1f,.15f};
  auto report=coordinator.finalizeGeometryChange(std::move(capture),context,0,nullptr);
  TORCH_CHECK(report.compensation_steps>0&&report.tf_steps==0,"Geometry compensation CD did not run without TF");
  TORCH_CHECK(torch::equal(context.current.xyz,current)&&torch::equal(state.xyz,xyz),"CD changed Master geometry");
  TORCH_CHECK(torch::equal(before.xyz,frozen->applyLogCompensation(state,.05,{},dc,alpha).xyz),"CD changed frozen teacher");
  TORCH_CHECK(torch::equal(teacher_residual,frozen->forwardLogCompensationResidual(state,.05,{},dc,alpha)),
              "CD changed teacher geometry/appearance output");
  TORCH_CHECK(torch::equal(dc,dc_before)&&torch::equal(opacity_logits,logits_before),"CD changed Master appearance");
  TORCH_CHECK(torch::equal(normal_moment,static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg()),"CD changed normal Adam");
  torch::serialize::OutputArchive output;model->save(output);output.save_to("compensation_selector.pt");
  auto loaded=improvements::selector::GumbelNetwork(.1,true,false,true);loaded->to(device);
  torch::serialize::InputArchive input;input.load_from("compensation_selector.pt",device);loaded->load(input);
  auto result=loaded->applyLogCompensation(context.current,.05,{},dc,alpha);
  TORCH_CHECK(loaded->compensationOutputWidth()==width,"Checkpoint changed output schema");
  TORCH_CHECK(torch::equal(result.xyz,model->applyLogCompensation(context.current,.05,{},dc,alpha).xyz),"Checkpoint mismatch");
  auto selection=loaded->forwardSelectionFeatures(current,.05,1,{},scale,dc,alpha);
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",current);fixture.write("scale",scale);fixture.write("dc",dc);fixture.write("alpha",alpha);
  fixture.write("rotation",rotation);fixture.write("probability",selection.probability);fixture.write("shared",selection.shared);
  fixture.write("result_xyz",result.xyz);fixture.write("result_scale",result.log_scaling.exp());fixture.write("result_rotation",result.rotation);
  auto residual=loaded->compensationResidual(selection.shared);
  auto appearance=loaded->applyAppearanceResidual(dc,opacity_logits,residual);
  fixture.write("opacity_logits",opacity_logits);fixture.write("result_dc",appearance.first);
  fixture.write("result_opacity_logits",appearance.second);
  fixture.save_to("compensation_fixture.pt");
  std::cout<<"PASS: exact selection/zero init/RNG, selected-only head, finite head/trunk/Feature gradients, frozen Master/teacher/normal Adam in geometry CD, checkpoint round trip\n";
}
