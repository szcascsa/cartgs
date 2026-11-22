#include "include/gaussian_mapper.h"
#include "include/anchor_frustum.h"
#include "include/view_budget_training.h"
#include "cuda_rasterizer/render_workspace.h"
#include "cuda_rasterizer/ssim_backend.h"
#include <iostream>

void verifySampling(torch::Device device) {
  auto o=torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto logits=torch::tensor({{0.f,2.f},{0.f,-2.f},{0.f,1.f},{0.f,3.f}},o).requires_grad_(true);
  auto eligible=torch::tensor({true,true,true,false},o.dtype(torch::kBool));
  auto out=viewBudgetSelection(logits,eligible,torch::ones_like(logits),1.);
  TORCH_CHECK(out.mask.equal(torch::tensor({1.f,0.f,1.f,0.f},o))&&
      out.chosen.equal(torch::tensor({0,2},o.dtype(torch::kLong))),"Gumbel-ST/Slot forward");
  // 0.1*4 is deliberately fractional; the budget is neither floor(K) nor Top-K.
  auto loss=selectorGumbelRatioLoss(out.mask,.1f,true);
  TORCH_CHECK(std::abs(loss.item<double>()-1.6)<1e-6,"Budget population or rounding changed");
  loss.backward();
  auto p=logits.detach().softmax(-1).select(1,1);
  auto expected=p*(1-p)*eligible;
  TORCH_CHECK(torch::allclose(logits.grad().select(1,1),expected,1e-5,1e-7)&&
      torch::allclose(logits.grad().select(1,0),-expected,1e-5,1e-7),"Budget VJP/Slot isolation");
  for(double r:{.01,.05,.1,.2,.3,.4,.5,.7}) {
    auto value=selectorGumbelRatioLoss(out.mask.detach(),r,true).item<double>();
    const double target=double(float(r))*4;
    TORCH_CHECK(std::abs(value-std::abs(2.-target)/std::max(1.,target))<1e-6,"Ratio budget mismatch");
  }
  auto empty=viewBudgetSelection(torch::empty({0,2},o),torch::empty({0},o.dtype(torch::kBool)),torch::empty({0,2},o),1.);
  TORCH_CHECK(!empty.chosen.numel()&&selectorGumbelRatioLoss(empty.mask,.1,true).item<float>()==0,"Empty sampler");
  auto generator=at::globalContext().defaultGenerator(device).clone();generator.set_current_seed(91);
  auto before=at::globalContext().defaultGenerator(device).get_state().clone();
  auto private_before=generator.get_state().clone();
  auto draw=torch::empty({100,2},o).exponential_(1.,generator);
  auto saved=generator.get_state().clone();auto second=torch::empty_like(draw).exponential_(1.,generator);
  generator.set_state(saved);auto restored=torch::empty_like(draw).exponential_(1.,generator);
  TORCH_CHECK(!draw.equal(second)&&second.equal(restored)&&!private_before.equal(saved)&&
      before.equal(at::globalContext().defaultGenerator(device).get_state()),"Private RNG/restart changed global sampling");
  std::cout<<"PASS sampling "<<device<<": ST forward/VJP, fractional budget, inactive/empty, private advancing RNG\n";
}

class ViewBudgetMapperTest:public GaussianMapper {
 public:
  ViewBudgetMapperTest(const std::filesystem::path& out):GaussianMapper(nullptr,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",out,7,torch::kCUDA,true,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/selector/selector_config.yaml") {}

  void verify(const std::string& mode) {
    TORCH_CHECK(view_budget_enabled_&&view_budget_loss_weight_==1.f&&selector_temperature_==1.f&&
        !selector_view_input_enabled_&&!selector_view_residual_enabled_&&!selector_anchor_frustum_culling_,"Joint configuration");
    TORCH_CHECK(selector_target_ratios_==std::vector<float>({.01f,.05f,.1f,.3f,.5f,.7f}),"Default six-ratio configuration");
    const auto ratio_count=selector_target_ratios_.size(),steps=2*ratio_count;
    if(mode=="off"||mode=="off_repeat")view_budget_enabled_=false;
    if(mode=="zero")view_budget_loss_weight_=0;
    view_budget_global_control_=mode=="control";
    auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<80;++i) {
      Point3D p;p.xyz_=Eigen::Vector3d((i%10-5)*.025+.013+(i>=40||mode=="empty"?5.:0.),
          ((i/10)%8-4)*.025+.017,2.02+(i%3)*.005);
      p.color_=Eigen::Vector3f(.4,.3,.2);points[i]=p;
    }
    gaussians_->createFromPcd(points,1);gaussians_->trainingSetup(opt_params_);
    gaussians_->active_sh_degree_=default_sh_=1;scene_->cameras_extent_=1;
    opt_params_.densify_from_iter_=1000000;opt_params_.opacity_reset_interval_=0;opt_params_.iterations_=1;
    do_gaus_pyramid_training_=false;initial_mapped_=true;SLAM_ended_=true;
    keyframe_record_interval_=all_keyframes_record_interval_=training_report_interval_=0;
    auto camera=std::make_shared<GaussianKeyframe>(0);
    camera->camera_id_=0;camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    camera->image_height_=32;camera->image_width_=40;camera->FoVx_=camera->FoVy_=1;
    camera->world_view_transform_=torch::eye(4,o);
    camera->full_proj_transform_=camera->getProjectionMatrix(.01f,100.f,1.f,1.f,torch::kCUDA).transpose(0,1).contiguous();
    camera->camera_center_=torch::zeros({3},o);camera->original_image_=torch::full({3,32,40},.1,o);
    undistort_mask_[0]=torch::ones({32,40},o);scene_->addKeyframe(camera,&kfid_shuffled_);
    real_lod_.online_steps=steps;real_lod_.phase=RealLodSchedule::Phase::TailFull;real_lod_.beginLod();
    const auto owners=selector_network_->anchors()->rows().clone();const auto topology=topology_version_;
    const auto parameters=selector_network_->parameters();int64_t parameter_count=0;
    for(auto p:parameters)parameter_count+=p.numel();
    const auto visible_n=anchorNominalFrustumMask(selector_network_->anchors()->centers(),
        selector_network_->anchors()->offsetScales(),camera->full_proj_transform_).index_select(0,owners).sum().item<int64_t>();
    TORCH_CHECK(mode=="empty"?visible_n==0:visible_n>0&&visible_n<owners.numel(),"Reduced/empty view fixture");
    if(mode=="health") {verifyHealth(camera);return;}
    if(mode=="gradient") {
      trainRealLodIteration(camera);verifyAccumulation(camera);return;
    }
    if(mode=="empty") {
      view_budget_health_.global_zero_streak=3;
      for(size_t i=0;i<selector_target_ratios_.size();++i)
        view_budget_health_.zero_streak[i]=view_budget_health_.saturation_streak[i]=3;
    }
    while(real_lod_.optimizingLod())trainRealLodIteration(camera);
    const bool auxiliary=view_budget_enabled_&&view_budget_loss_weight_>0;
    const bool rendered=auxiliary&&mode!="empty";
    TORCH_CHECK(real_lod_full_renders_==steps&&real_lod_full_backwards_==steps&&real_lod_.lod_steps==steps&&
        flexgs_lod_master_updates_==steps&&flexgs_lod_selector_updates_==steps&&flexgs_lod_nonfinite_==0,"Full/global/Adam exposure");
    TORCH_CHECK(view_budget_renders_==(rendered?steps:0)&&view_budget_backwards_==view_budget_renders_,"Auxiliary exposure");
    TORCH_CHECK(view_budget_generator_.defined()==auxiliary,"Disabled auxiliary consumed RNG");
    if(mode=="empty") {
      TORCH_CHECK(view_budget_health_.global_zero_streak==0,"Empty view did not reset health history");
      for(size_t i=0;i<selector_target_ratios_.size();++i)
        TORCH_CHECK(view_budget_health_.zero_streak[i]==0&&view_budget_health_.saturation_streak[i]==0,
            "Empty view counted as consecutive zero selection");
    }
    TORCH_CHECK(owners.equal(selector_network_->anchors()->rows())&&topology_version_==topology,"Topology changed");
    TORCH_CHECK(real_lod_ratio_counts_==std::vector<uint64_t>(ratio_count,2),"Global ratio cursor changed");
    TORCH_CHECK(view_budget_ratio_counts_==std::vector<uint64_t>(ratio_count,auxiliary?2:0),"Auxiliary ratio cursor");
    for(const auto& p:selector_network_->named_parameters()) {
      if(p.key().rfind("local_emd.",0)&&p.key().rfind("soft_net.",0)&&p.key().rfind("spatial_compensation.",0))continue;
      auto it=gaussians_->optimizer_->state().find(p.value().unsafeGetTensorImpl());
      TORCH_CHECK(it!=gaussians_->optimizer_->state().end()&&
          static_cast<const torch::optim::AdamParamState&>(*it->second).step()==steps,"Network Adam advanced twice");
    }
    const auto tables=selector_network_->anchors()->compensationFeatureTables();
    TORCH_CHECK(tables.size()==ratio_count,"One Base48 plus five residual tables");
    for(size_t j=0;j<tables.size();++j) {
      auto it=gaussians_->optimizer_->state().find(tables[j].unsafeGetTensorImpl());
      TORCH_CHECK(it!=gaussians_->optimizer_->state().end()&&
          static_cast<const torch::optim::AdamParamState&>(*it->second).step()==2*(ratio_count-j),"Residual Feature Adam advanced per branch");
    }
    std::ifstream input(result_dir_/"view_budget_steps.jsonl");std::string line;int count=0;bool fluctuated=false;
    while(std::getline(input,line)) {
      Json::Value v;Json::Reader reader;TORCH_CHECK(reader.parse(line,v),"Invalid auxiliary trace");
      auto expected_n=view_budget_global_control_?owners.numel():visible_n;
      TORCH_CHECK(v["candidate_N"].asInt64()==expected_n&&v["K"].asInt64()<=v["eligible_N"].asInt64()&&
          !v["strict_topk"].asBool()&&v["gi_weight"].asInt()==0&&v["adam_commits_this_step"].asInt()==1,"Wrong auxiliary selection semantics");
      TORCH_CHECK(std::abs(v["target_K"].asDouble()-v["ratio"].asDouble()*expected_n)<1e-8,"Budget was rounded or uses full-map N");
      const double target=v["target_K"].asDouble();
      TORCH_CHECK(std::abs(v["budget_loss"].asDouble()-std::abs(v["K"].asDouble()-target)/std::max(1.,target))<1e-5,"Actual sampled mask must drive budget");
      fluctuated|=v["K"].asInt64()!=int64_t(std::floor(target));++count;
    }
    TORCH_CHECK(count==(auxiliary?steps:0)&&(!rendered||fluctuated),"Missing trace or an unexpected exact-K projection");
    savePly(result_dir_/"checkpoint");saveRealLodState(result_dir_);
    torch::save(at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state(),
        (result_dir_/"global_cpu_rng.pt").string());
    torch::save(at::globalContext().defaultGenerator(torch::Device(torch::kCUDA)).get_state(),
        (result_dir_/"global_cuda_rng.pt").string());
    auto restored=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(torch::kCUDA);
    torch::serialize::InputArchive archive;archive.load_from((result_dir_/"checkpoint/selector.pt").string(),torch::Device(torch::kCUDA));restored->load(archive);
    auto a=selector_network_->named_parameters(),b=restored->named_parameters();int64_t restored_count=0;
    for(auto p:b) {restored_count+=p.value().numel();TORCH_CHECK(p.value().equal(a[p.key()]),"Checkpoint parameter changed");}
    TORCH_CHECK(restored_count==parameter_count,"Auxiliary introduced new trainable parameters");
    std::cout<<"PASS Mapping "<<mode<<": Full"<<steps<<"/global"<<steps<<"/aux"<<view_budget_renders_
        <<", Adam"<<steps<<", six-ratio Feature clocks12/10/8/6/4/2, unchanged topology/checkpoint\n";
  }
 private:
  std::vector<torch::Tensor> allParameters() {
    std::vector<torch::Tensor> result;
    for(const auto& group:gaussians_->optimizer_->param_groups())for(auto p:group.params())result.push_back(p);
    return result;
  }
  void verifyAccumulation(const std::shared_ptr<GaussianKeyframe>& camera) {
    renderer_workspace::Scope workspace(reuse_renderer_workspace_);ssim_backend::Scope ssim(optimized_ssim_backward_);
    ssim_backend::CacheScope cache(cached_ssim_statistics_);mapping_cuda::L1FusionScope l1(fused_l1_loss_);
    auto parameters=allParameters();auto truth=camera->original_image_;auto state=view_budget_generator_.get_state().clone();
    auto global=at::globalContext().defaultGenerator(torch::Device(torch::kCUDA)).get_state().clone();
    gaussians_->optimizer_->zero_grad(true);selector_optimizer_->zero_grad(true);
    auto isolated=accumulateViewBudgetGradients(camera,truth,1);
    TORCH_CHECK(isolated.finite_loss&&isolated.rendered,"Isolated auxiliary failed");
    std::vector<torch::Tensor> gradients;
    for(auto p:parameters)gradients.push_back(p.grad().defined()?p.grad().detach().clone():torch::zeros_like(p));
    {torch::NoGradGuard guard;for(auto p:parameters)p.mutable_grad()=torch::full_like(p,.125);}
    view_budget_generator_.set_state(state);
    auto accumulated=accumulateViewBudgetGradients(camera,truth,1);
    TORCH_CHECK(accumulated.finite_loss,"Accumulated auxiliary failed");double max_error=0;
    for(size_t i=0;i<parameters.size();++i)if(parameters[i].numel()) {
      auto expected=gradients[i]+.125;
      max_error=std::max(max_error,(parameters[i].grad()-expected).abs().max().item<double>());
      TORCH_CHECK(torch::allclose(parameters[i].grad(),expected,1e-4,1e-5),"Auxiliary cleared or corrupted pre-existing gradients");
    }
    TORCH_CHECK(global.equal(at::globalContext().defaultGenerator(torch::Device(torch::kCUDA)).get_state()),"Auxiliary consumed global CUDA RNG");
    for(auto p:parameters) {
      auto it=gaussians_->optimizer_->state().find(p.unsafeGetTensorImpl());
      if(it!=gaussians_->optimizer_->state().end())TORCH_CHECK(static_cast<const torch::optim::AdamParamState&>(*it->second).step()==1,"Auxiliary committed Adam");
    }
    std::cout<<"PASS accumulation: retained Master/Selector/Compensator/Feature gradients, private RNG; max_abs_error="<<max_error
        <<" (repeated rasterizer VJP atomics; not a bitwise equivalence claim)\n";
  }
  void verifyHealth(const std::shared_ptr<GaussianKeyframe>& camera) {
    lod_health_guard_.enabled=false;view_budget_health_.enabled=true;view_budget_health_.zero_patience=1;
    {torch::NoGradGuard guard;for(auto p:selector_network_->named_parameters())
      if(p.key()=="soft_net.bias")p.value()[1].fill_(-100);}
    auto parameters=allParameters();std::vector<torch::Tensor> before;
    for(auto p:parameters)before.push_back(p.detach().clone());bool rejected=false;
    try {trainRealLodIteration(camera);} catch(const c10::Error& e) {
      rejected=std::string(e.what()).find("view_budget_consecutive_zero")!=std::string::npos;
    }
    TORCH_CHECK(rejected&&real_lod_.lod_steps==0&&gaussians_->optimizer_->state().empty(),"Local failure advanced Adam");
    for(size_t i=0;i<parameters.size();++i)TORCH_CHECK(parameters[i].equal(before[i]),"Failure changed parameters");
    std::ifstream in(result_dir_/"lod_health_failure.json");Json::Value v;in>>v;
    TORCH_CHECK(v["branch"].asString()=="view_budget","Local failure diagnostics overwritten by global branch");
    std::cout<<"PASS local zero-selection abort: correct diagnostic, no Adam/parameter update\n";
  }
};

int main(int argc,char** argv) {
  const std::string mode=argc>1?argv[1]:"sampling";
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);
  if(mode=="sampling") {verifySampling(torch::kCPU);if(torch::cuda::is_available())verifySampling(torch::kCUDA);return 0;}
  if(!torch::cuda::is_available())return 77;
  ViewBudgetMapperTest mapper(argc>2?argv[2]:"/tmp/cartgs_view_budget_"+mode);mapper.verify(mode);
}
