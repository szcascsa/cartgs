#include "include/gaussian_mapper.h"
#include "include/local_importance.h"
#include "include/view_budget_training.h"
#include <iostream>

using namespace improvements::selector;

void verifyUnit(torch::Device device,const std::filesystem::path& output,int64_t schema=1) {
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(device);
  auto cpu_rng=at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state().clone();
  net->enableSelectorViewResidual(schema);
  TORCH_CHECK(cpu_rng.equal(at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state()),"New branch consumed global RNG");
  int64_t count=0;for(auto p:net->selectorViewResidual()->parameters())count+=p.numel();
  TORCH_CHECK(count==(schema==2?321:657),"Unexpected residual parameter count");
  auto shared=torch::randn({12,32},o).requires_grad_(true),logits=torch::randn({12,2},o).requires_grad_(true);
  auto code=torch::randn({12,7},o).requires_grad_(true);
  auto physical=torch::randn({12,11},o).requires_grad_(true);
  GumbelNetworkImpl::SelectionFeatures global{{},shared,logits,physical};
  auto initial=net->applyViewResidual(global,code,false);
  TORCH_CHECK(initial.logits.equal(logits),"Zero initialization changed logits");
  auto eligible=torch::ones({12},o.dtype(torch::kBool)),noise=torch::ones_like(logits);
  TORCH_CHECK(viewBudgetSelection(initial.logits,eligible,noise,1.).mask.equal(
      viewBudgetSelection(logits,eligible,noise,1.).mask),"Zero branch changed selected identities");
  auto residual=net->selectorViewResidual();
  {torch::NoGradGuard guard;residual->output->weight.fill_(.02);}
  auto corrected=net->applyViewResidual(global,code,false);
  auto sampled=viewBudgetSelection(corrected.logits,eligible,noise,1.);
  auto labels=torch::zeros_like(sampled.mask);labels.slice(0,0,3).fill_(1);
  auto loss=.1*localImportanceLoss(sampled.mask,labels);
  auto parameters=residual->parameters();
  auto expected=torch::autograd::grad({loss},parameters,{},true);
  {torch::NoGradGuard guard;
   for(auto p:parameters)p.mutable_grad()=torch::full_like(p,.125);
   for(auto p:{shared,logits,code,physical})p.mutable_grad()=torch::full_like(p,.125);}
  accumulateLocalImportanceGradients(loss,parameters);
  for(size_t i=0;i<parameters.size();++i)
    TORCH_CHECK(parameters[i].grad().equal(expected[i]+.125),"LI did not add only residual gradients");
  for(auto p:{shared,logits,code,physical})TORCH_CHECK(p.grad().equal(torch::full_like(p,.125)),"LI leaked into base/geometry");
  // Photo/Budget uses the same live base logits; only residual inputs detach.
  auto budget=selectorGumbelRatioLoss(sampled.mask,.1,true);
  (sampled.mask.square().mean()+.01*budget).backward();
  TORCH_CHECK(!logits.grad().equal(torch::full_like(logits,.125)),"Local Photo/Budget detached base logits");
  TORCH_CHECK(shared.grad().equal(torch::full_like(shared,.125))&&code.grad().equal(torch::full_like(code,.125))&&
      physical.grad().equal(torch::full_like(physical,.125)),
      "Residual input gradient escaped detach");
  TORCH_CHECK(residual->hidden->weight.grad().abs().sum().item<float>()>0,"Residual hidden layer cannot train");
  std::filesystem::create_directories(output);
  torch::save(net,(output/"selector.pt").string());
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(device);
  torch::serialize::InputArchive archive;archive.load_from((output/"selector.pt").string(),device);restored->load(archive);
  TORCH_CHECK(restored->applyViewResidual(global,code,false).logits.equal(corrected.logits),"Checkpoint roundtrip changed outputs");
  TORCH_CHECK(restored->selectorViewResidual()->schema()==schema&&
      restored->cloneFrozen()->applyViewResidual(global,code,false).logits.equal(corrected.logits),"Frozen schema/output changed");
  bool rejected=false;try {restored->enableSelectorViewResidual(schema==2?1:2);}catch(const c10::Error&) {rejected=true;}
  TORCH_CHECK(rejected,"Mismatched schema silently accepted");
  torch::serialize::OutputArchive fixture;
  fixture.write("physical",physical.detach(),true);fixture.write("shared",shared.detach(),true);
  fixture.write("code",code.detach(),true);fixture.write("base",logits.detach(),true);
  fixture.write("corrected",corrected.logits.detach(),true);fixture.save_to((output/"fixture.pt").string());
  std::cout<<"PASS unit "<<device<<" schema="<<schema<<": "<<count<<" params, zero identity/selection, RNG, residual-only additive LI VJP, live base Photo/Budget, checkpoint/frozen\n";
}

class ViewBudgetResidualMapperTest:public GaussianMapper {
 public:
  ViewBudgetResidualMapperTest(const std::filesystem::path& out,bool raw=false):GaussianMapper(nullptr,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",out,7,torch::kCUDA,true,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/(raw?"cfg/selector/spatial_offset3_residual48_global_view_raw18_li.yaml":
        "cfg/selector/spatial_offset3_residual48_global_view_residual_li.yaml")) {}
  void verify(const std::string& mode) {
    if(mode=="no_li")view_budget_li_weight_=0;
    const bool empty=mode=="empty";
    auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<80;++i) {
      Point3D p;p.xyz_=Eigen::Vector3d((i%10-5)*.025+.013+(empty||i>=40?5.:0.),((i/10)%8-4)*.025+.017,2.02+(i%3)*.005);
      p.color_=Eigen::Vector3f(.4,.3,.2);points[i]=p;
    }
    gaussians_->createFromPcd(points,1);gaussians_->trainingSetup(opt_params_);
    gaussians_->active_sh_degree_=default_sh_=1;scene_->cameras_extent_=1;
    opt_params_.densify_from_iter_=1000000;opt_params_.opacity_reset_interval_=0;
    do_gaus_pyramid_training_=false;initial_mapped_=true;SLAM_ended_=true;
    keyframe_record_interval_=all_keyframes_record_interval_=training_report_interval_=0;
    auto camera=std::make_shared<GaussianKeyframe>(0);camera->camera_id_=0;
    camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    camera->image_height_=32;camera->image_width_=40;camera->FoVx_=camera->FoVy_=1;
    camera->znear_=.01f;camera->world_view_transform_=torch::eye(4,o);
    camera->full_proj_transform_=camera->getProjectionMatrix(.01f,100.f,1.f,1.f,torch::kCUDA).transpose(0,1).contiguous();
    camera->camera_center_=torch::zeros({3},o);camera->original_image_=torch::full({3,32,40},.1,o);
    undistort_mask_[0]=torch::ones({32,40},o);scene_->addKeyframe(camera,&kfid_shuffled_);
    real_lod_.online_steps=12;real_lod_.phase=RealLodSchedule::Phase::TailFull;real_lod_.beginLod();
    auto& branch=selector_network_->selectorViewResidual();branch->initialize(gaussians_->getXYZ());
    if(mode=="isolation") {
      auto base=selector_network_->forwardFlexgsCUDA(gaussians_->getXYZ(),.1f,gaussians_->getScalingActivation(),
          gaussians_->features_dc_,gaussians_->getOpacityActivation());
      if(branch->schema()==2) {
        auto rows=torch::arange(gaussians_->xyz_.size(0)-1,-1,-3,o.dtype(torch::kLong));
        for(const auto& ids:{torch::arange(gaussians_->xyz_.size(0),o.dtype(torch::kLong)),rows,rows.slice(0,0,0)}) {
          auto x=gaussians_->getXYZ().index_select(0,ids),s=gaussians_->getScalingActivation().index_select(0,ids);
          auto dc=gaussians_->features_dc_.index_select(0,ids),op=gaussians_->getOpacityActivation().index_select(0,ids);
          auto owner=selector_network_->anchors()->rows().index_select(0,ids);
          auto packed=mapping_cuda::flexgsInput11(x,s,dc,op,owner,selector_network_->anchors()->centers(),
              selector_network_->anchors()->offsetScales(),.1f);
          auto actual=selector_network_->forwardFlexgsCUDA(x,.1f,s,dc,op,ids);
          TORCH_CHECK(actual.physical.equal(packed)&&!actual.physical.requires_grad(),"Raw18 did not reuse exact compact Selector input");
          auto aten=selector_network_->forwardSelectionFeatures(x,.1f,1.,ids,s,dc,op);
          auto expected=torch::cat({(x-selector_network_->anchors()->centers().index_select(0,owner))/
              selector_network_->anchors()->offsetScales().index_select(0,owner),s,dc.squeeze(1),op,
              torch::full({x.size(0),1},.1f,o).log()},1);
          TORCH_CHECK(aten.physical.equal(expected)&&!aten.physical.requires_grad(),"Raw18 ATen input packing changed");
        }
      }
      auto code=anchorViewCode(selector_network_->anchors()->centers(),selector_network_->anchors()->offsetScales(),
          camera->camera_center_,camera->world_view_transform_,camera->full_proj_transform_,32,40,.01f,branch->scene_scale);
      code=code.index_select(0,selector_network_->anchors()->rows());
      {torch::NoGradGuard guard;branch->output->weight.fill_(.02);}
      auto local=selector_network_->applyViewResidual(base,code,false);
      auto eligible=torch::ones({gaussians_->xyz_.size(0)},o.dtype(torch::kBool));
      auto sampled=viewBudgetSelection(local.logits,eligible,torch::ones_like(local.logits),1.);
      auto labels=torch::zeros_like(sampled.mask);labels.slice(0,0,12).fill_(1);
      auto loss=.1*localImportanceLoss(sampled.mask,labels);
      auto private_parameters=branch->parameters();auto expected=torch::autograd::grad({loss},private_parameters,{},true);
      auto all=selector_network_->parameters();
      for(auto p:{gaussians_->xyz_,gaussians_->scaling_,gaussians_->rotation_,gaussians_->opacity_,gaussians_->features_dc_,gaussians_->features_rest_})all.push_back(p);
      {torch::NoGradGuard guard;for(auto p:all)p.mutable_grad()=torch::full_like(p,.125);}
      accumulateLocalImportanceGradients(loss,private_parameters);
      for(auto p:all) {
        size_t i=0;while(i<private_parameters.size()&&p.unsafeGetTensorImpl()!=private_parameters[i].unsafeGetTensorImpl())++i;
        TORCH_CHECK(p.grad().equal(i<private_parameters.size()?expected[i]+.125:torch::full_like(p,.125)),
            "LI changed existing base/Master/Compensation/Feature gradients");
      }
      gaussians_->optimizer_->zero_grad(true);selector_optimizer_->zero_grad(true);
      (sampled.mask.square().mean()+.01*selectorGumbelRatioLoss(sampled.mask,.1,true)).backward();
      bool base_gradient=false;
      for(auto p:selector_network_->named_parameters())if(p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0)
        base_gradient|=p.value().grad().defined()&&p.value().grad().abs().sum().item<float>()>0;
      TORCH_CHECK(base_gradient&&gaussians_->xyz_.grad().defined(),"Fused local Photo/Budget lost original input/MLP path");
      std::cout<<"PASS fused isolation: bitwise unchanged original gradients, four residual tensors only; local base/Master VJP live\n";return;
    }
    auto owners=selector_network_->anchors()->rows().clone();auto topology=topology_version_;
    auto original=branch->output->weight.detach().clone();
    while(real_lod_.optimizingLod())trainRealLodIteration(camera);
    TORCH_CHECK(real_lod_full_renders_==12&&view_budget_renders_==(empty?0:12)&&flexgs_lod_master_updates_==12&&
        flexgs_lod_selector_updates_==12&&topology==topology_version_&&owners.equal(selector_network_->anchors()->rows()),"Exposure/Adam/topology changed");
    TORCH_CHECK(view_budget_li_backwards_==((mode=="no_li"||empty)?0:12),"Unexpected LI VJP count");
    TORCH_CHECK(empty||!original.equal(branch->output->weight),"View branch did not learn");
    if(empty)TORCH_CHECK(original.equal(branch->output->weight),"Empty branch updated residual weights");
    for(auto p:branch->parameters()) {
      auto& state=gaussians_->optimizer_->state();auto it=state.find(p.unsafeGetTensorImpl());
      TORCH_CHECK(empty?it==state.end():(it!=state.end()&&static_cast<const torch::optim::AdamParamState&>(*it->second).step()==12),
          "View branch advanced Adam more than once per Mapping");
    }
    savePly(result_dir_/"checkpoint");saveRealLodState(result_dir_);
    std::cout<<"PASS Mapping "<<mode<<": Full/global/local, one Adam, learned residual, LI scope, empty candidates, fixed topology/export\n";
  }
};

int main(int argc,char** argv) {
  torch::set_num_threads(2);torch::manual_seed(7);at::globalContext().setAllowTF32CuBLAS(false);
  std::string mode=argc>1?argv[1]:"unit";
  const bool raw=mode.rfind("raw_",0)==0;if(raw)mode=mode.substr(4);
  const std::filesystem::path out=argc>2?argv[2]:"/tmp/cartgs_view_budget_residual_"+mode;
  if(mode=="unit") {
    verifyUnit(torch::kCPU,out/"cpu",raw?2:1);
    if(torch::cuda::is_available())verifyUnit(torch::kCUDA,out/"cuda",raw?2:1);
    return 0;
  }
  if(!torch::cuda::is_available())return 77;
  ViewBudgetResidualMapperTest mapper(out,raw);mapper.verify(mode);
}
