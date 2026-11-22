#include "include/gaussian_mapper.h"
#include "include/gaussian_renderer.h"
#include "include/local_importance.h"
#include "include/view_budget_training.h"
#include "cuda_rasterizer/anchor_si_capture.h"
#include <iostream>

void verifyTeacherAndVjp(torch::Device device) {
  auto o=torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto contribution=torch::tensor({0.f,3.f,2.f,0.f,4.f,9.f},o);
  auto global=torch::tensor({8.f,1.f,1.f,6.f,2.f,0.f},o);
  auto ids=torch::arange(6,o.dtype(torch::kLong));
  auto eligible=torch::tensor({true,true,true,true,true,false},o.dtype(torch::kBool));
  auto labels=localImportanceTarget(contribution,global,ids,eligible,.5);
  TORCH_CHECK(labels.equal(torch::tensor({0.f,1.f,1.f,0.f,1.f,0.f},o))&&!labels.requires_grad(),"LI budget/eligibility");
  auto tied=localImportanceTarget(torch::ones_like(contribution),global,ids,eligible,.5);
  TORCH_CHECK(tied.equal(torch::tensor({1.f,0.f,0.f,1.f,1.f,0.f},o)),"LI global tie break");
  TORCH_CHECK(localImportanceTarget(contribution,global,ids,eligible,1.).sum().item<int>()==5,"LI infeasible K");
  TORCH_CHECK(!localImportanceTarget(torch::zeros_like(contribution),global,ids,eligible,.5).defined(),"Zero LI must skip");
  TORCH_CHECK(!localImportanceTarget(contribution.slice(0,0,0),global.slice(0,0,0),ids.slice(0,0,0),eligible.slice(0,0,0),.1).defined(),"Empty LI");
  auto input=torch::randn({6,3},o).requires_grad_(true),weight=torch::randn({3,2},o).requires_grad_(true);
  auto feature=torch::randn({3,2},o).requires_grad_(true);
  auto sample=viewBudgetSelection(input.matmul(weight),eligible,torch::ones({6,2},o),1.);
  auto loss=.1*localImportanceLoss(sample.mask,labels);
  auto expected=torch::autograd::grad({loss},{weight},{},true)[0];
  {torch::NoGradGuard guard;input.mutable_grad()=torch::full_like(input,.125);
   weight.mutable_grad()=torch::full_like(weight,.125);feature.mutable_grad()=torch::full_like(feature,.125);}
  accumulateLocalImportanceGradients(loss,{weight});
  TORCH_CHECK(input.grad().equal(torch::full_like(input,.125))&&feature.grad().equal(torch::full_like(feature,.125))&&
      weight.grad().equal(expected+.125),"LI leaked into Master/Feature or cleared old gradients");
  (input.matmul(feature).square().mean()).backward();
  TORCH_CHECK(!input.grad().equal(torch::full_like(input,.125))&&!feature.grad().equal(torch::full_like(feature,.125)),"Photo gradient path detached");
  std::cout<<"PASS teacher/VJP "<<device<<": exact teacher K, ties, inactive/empty/zero, Selector-only additive gradients\n";
}

class LocalImportanceMapperTest:public GaussianMapper {
 public:
  LocalImportanceMapperTest(const std::filesystem::path& out):GaussianMapper(nullptr,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",out,7,torch::kCUDA,true,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/selector/spatial_offset3_residual48_global_view_joint.yaml") {}
  void verify(const std::string& mode) {
    view_budget_li_weight_=mode=="off"?0:.1f;
    auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<80;++i) {
      Point3D p;p.xyz_=Eigen::Vector3d((i%10-5)*.025+.013+(i>=40?5.:0.),((i/10)%8-4)*.025+.017,2.02+(i%3)*.005);
      p.color_=Eigen::Vector3f(.4,.3,.2);points[i]=p;
    }
    gaussians_->createFromPcd(points,1);gaussians_->trainingSetup(opt_params_);
    gaussians_->active_sh_degree_=default_sh_=1;scene_->cameras_extent_=1;
    opt_params_.densify_from_iter_=1000000;opt_params_.opacity_reset_interval_=0;opt_params_.iterations_=1;
    do_gaus_pyramid_training_=false;initial_mapped_=true;SLAM_ended_=true;
    keyframe_record_interval_=all_keyframes_record_interval_=training_report_interval_=0;
    auto camera=std::make_shared<GaussianKeyframe>(0);camera->camera_id_=0;
    camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    camera->image_height_=32;camera->image_width_=40;camera->FoVx_=camera->FoVy_=1;
    camera->world_view_transform_=torch::eye(4,o);
    camera->full_proj_transform_=camera->getProjectionMatrix(.01f,100.f,1.f,1.f,torch::kCUDA).transpose(0,1).contiguous();
    camera->camera_center_=torch::zeros({3},o);camera->original_image_=torch::full({3,32,40},.1,o);
    undistort_mask_[0]=torch::ones({32,40},o);scene_->addKeyframe(camera,&kfid_shuffled_);
    real_lod_.online_steps=12;real_lod_.phase=RealLodSchedule::Phase::TailFull;real_lod_.beginLod();
    if(mode=="isolation") {
      auto input=selector_network_->forwardFlexgsCUDA(gaussians_->getXYZ(),.1f,gaussians_->getScalingActivation(),
          gaussians_->features_dc_,gaussians_->getOpacityActivation());
      auto eligible=torch::ones({gaussians_->xyz_.size(0)},o.dtype(torch::kBool));
      auto sampled=viewBudgetSelection(input.logits,eligible,torch::ones_like(input.logits),1.);
      auto labels=torch::zeros_like(sampled.mask);labels.slice(0,0,12).fill_(1);
      auto loss=.1*localImportanceLoss(sampled.mask,labels);
      std::vector<torch::Tensor> mlp;
      for(const auto& p:selector_network_->named_parameters())
        if(p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0)mlp.push_back(p.value());
      auto expected=torch::autograd::grad({loss},mlp,{},true);
      auto all=selector_network_->parameters();
      for(auto p:{gaussians_->xyz_,gaussians_->scaling_,gaussians_->rotation_,gaussians_->opacity_,gaussians_->features_dc_,gaussians_->features_rest_})all.push_back(p);
      {torch::NoGradGuard guard;for(auto p:all)p.mutable_grad()=torch::full_like(p,.125);}
      accumulateLocalImportanceGradients(loss,mlp);
      for(auto p:all) {
        size_t i=0;while(i<mlp.size()&&p.unsafeGetTensorImpl()!=mlp[i].unsafeGetTensorImpl())++i;
        TORCH_CHECK(p.grad().equal(i<mlp.size()?expected[i]+.125:torch::full_like(p,.125)),"Fused LI gradient isolation/accumulation failed");
      }
      std::cout<<"PASS actual fused Selector VJP: Master, Compensation and all Feature gradients bitwise intact; six MLP gradients added\n";return;
    }
    if(mode=="collection") {
      RenderPackage plain,captured,raw;
      {anchor_si::CaptureScope scope(false);plain=GaussianRenderer::render(camera,32,40,gaussians_,pipe_params_,background_,override_color_,1.f,false,{},false,false);}
      {anchor_si::CaptureScope scope(true);captured=GaussianRenderer::render(camera,32,40,gaussians_,pipe_params_,background_,override_color_,1.f,false,{},false,true);}
      {anchor_si::CaptureScope scope(false);raw=GaussianRenderer::render(camera,32,40,gaussians_,pipe_params_,background_,override_color_,1.f,false,{},false,true);}
      TORCH_CHECK(plain.image.equal(captured.image)&&plain.radii.equal(captured.radii),"LI collection changed RGB/radii");
      TORCH_CHECK(captured.frame_importance.sum().item<double>()>0&&raw.contribution_count.numel()==gaussians_->xyz_.size(0)&&
          !captured.contribution_count.numel()&&!anchor_si::captureEnabled(),"Wrong LI/global statistic or leaked capture scope");
      auto truth=camera->original_image_;accumulateRealLodFullGradients(camera,truth);
      TORCH_CHECK(!view_budget_li_.requires_grad()&&view_budget_li_.numel()==gaussians_->xyz_.size(0)&&
          torch::allclose(view_budget_li_,captured.frame_importance/(32.*40.),1e-5,1e-8)&&
          view_budget_li_step_==1&&view_budget_li_full_renders_==1,"Full LI normalization/identity");
      std::cout<<"PASS Full capture: unchanged RGB/radii, alpha*T/HW, detached, scoped statistic, no extra Full\n";return;
    }
    auto owners=selector_network_->anchors()->rows().clone();auto topology=topology_version_;
    while(real_lod_.optimizingLod())trainRealLodIteration(camera);
    TORCH_CHECK(real_lod_full_renders_==12&&view_budget_renders_==12&&flexgs_lod_master_updates_==12&&
        flexgs_lod_selector_updates_==12&&topology==topology_version_&&owners.equal(selector_network_->anchors()->rows()),"LI changed exposure/Adam/topology");
    TORCH_CHECK(view_budget_li_full_renders_==(mode=="off"?0:12)&&view_budget_li_backwards_==(mode=="off"?0:12),"Missing LI update/disabled path");
    for(const auto& p:selector_network_->named_parameters())if(p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0) {
      auto& state=gaussians_->optimizer_->state();auto it=state.find(p.value().unsafeGetTensorImpl());
      TORCH_CHECK(it!=state.end()&&static_cast<const torch::optim::AdamParamState&>(*it->second).step()==12,"LI advanced Adam separately");
    }
    savePly(result_dir_/"checkpoint");saveRealLodState(result_dir_);
    std::cout<<"PASS Mapping "<<mode<<": Full/global/local12, Adam12, LI only six Selector tensors, fixed topology/checkpoint\n";
  }
};

int main(int argc,char** argv) {
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);
  const std::string mode=argc>1?argv[1]:"unit";
  if(mode=="unit") {verifyTeacherAndVjp(torch::kCPU);if(torch::cuda::is_available())verifyTeacherAndVjp(torch::kCUDA);return 0;}
  if(!torch::cuda::is_available())return 77;
  LocalImportanceMapperTest mapper(argc>2?argv[2]:"/tmp/cartgs_li_"+mode);mapper.verify(mode);
}
