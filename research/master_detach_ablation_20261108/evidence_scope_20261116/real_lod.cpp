#include "include/gaussian_mapper.h"
#include "include/gaussian_renderer.h"
#include "include/loss_utils.h"
#include "include/mapping_cuda_optimizer.h"
#include "cuda_rasterizer/anchor_si_capture.h"
#include "cuda_rasterizer/render_workspace.h"
#include "cuda_rasterizer/ssim_backend.h"
#include "include/kernel_fusion.h"
#include <cmath>

void GaussianMapper::failLodHealth(const std::string& reason) {
  lod_health_guard_.record(result_dir_,reason);
  torch::NoGradGuard guard;
  torch::save(selector_network_,(result_dir_/"lod_health_failed_selector.pt").string());
  gaussians_->savePly(result_dir_/"lod_health_failed_master.ply");
  TORCH_CHECK(false,"LoD health failure before Adam: ",reason,"; see ",(result_dir_/"lod_health_failure.json").string());
}

// Called with mutex_render_ held. Official prune_list sums every training view
// on the current, uncompensated Master. It does not average, EMA or normalize.
void GaussianMapper::refreshFlexgsImportance() {
  torch::NoGradGuard no_grad;
  const auto start=std::chrono::steady_clock::now();
  auto scores=torch::zeros({gaussians_->getXYZ().size(0)},gaussians_->getXYZ().options());
  auto views=scene_->getAllKeyframes();
  TORCH_CHECK(!views.empty(),"FlexGS importance requires training keyframes");
  anchor_si::CaptureScope raw_opacity_transmittance(false);
  std::uint64_t count=0;
  for(auto it=views.rbegin();it!=views.rend();++it) {
    auto camera=it->second;
    TORCH_CHECK(camera&&camera->original_image_.defined(),"Invalid FlexGS training keyframe");
    auto rendered=GaussianRenderer::render(camera,camera->image_height_,camera->image_width_,
        gaussians_,pipe_params_,background_,override_color_,1.f,false,{},true,true,RasterizeMode::Inference);
    TORCH_CHECK(rendered.frame_importance.sizes()==scores.sizes(),"FlexGS importance row mismatch");
    scores.add_(rendered.frame_importance);
    ++count;
  }
  TORCH_CHECK(torch::isfinite(scores).all().item<bool>(),"Nonfinite FlexGS importance");
  flexgs_importance_=scores.detach();
  flexgs_importance_topology_=topology_version_;
  flexgs_importance_views_+=count;
  ++flexgs_importance_version_;
  const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  flexgs_importance_ms_+=ms;
  std::cout<<"[FlexGSImportance] lod_step="<<real_lod_.lod_steps+1<<" version="<<flexgs_importance_version_
           <<" views="<<count<<" gaussians="<<scores.numel()<<" ms="<<ms
           <<" statistic=sum_views_sum_valid_pixels_opacity_times_T\n";
}

double GaussianMapper::accumulateRealLodFullGradients(
    const std::shared_ptr<GaussianKeyframe>& camera, torch::Tensor& truth) {
  TORCH_CHECK(real_lod_.optimizingLod() && camera && real_lod_full_loss_weight_ > 0.f, "Full Fine supervision requires an active Fine phase and positive weight");
  RenderPackage rendered;
  const bool collect_li=view_budget_enabled_&&view_budget_loss_weight_>0&&view_budget_li_weight_>0;
  const bool collect_view=(selector_view_residual_enabled_||selector_view_input_enabled_)&&
      real_lod_.viewResidualRound(selector_target_ratios_.size());
  {
    auto timing=mapping_profiler_.scope(MappingProfiler::FullRender);
    std::unique_ptr<anchor_si::CaptureScope> raw;
    if(collect_view||collect_li)raw=std::make_unique<anchor_si::CaptureScope>(collect_li);
    rendered=GaussianRenderer::render(camera,camera->image_height_,camera->image_width_,
        gaussians_,pipe_params_,background_,override_color_,1.f,false,{},selector_detach_gaussian_parameters_,collect_view||collect_li);
    ++real_lod_full_renders_;
  }
  if(collect_view) {
    TORCH_CHECK(rendered.frame_importance.numel()==gaussians_->getXYZ().size(0),"Current Full view GI row mismatch");
    view_importance_=rendered.frame_importance.detach();
  }
  if(collect_li) {
    TORCH_CHECK(rendered.frame_importance.numel()==gaussians_->getXYZ().size(0),"Full LI row mismatch");
    view_budget_li_=rendered.frame_importance.detach()/double(camera->image_height_*camera->image_width_);
    view_budget_li_step_=real_lod_.lod_steps+1;view_budget_li_camera_=camera->fid_;
    view_budget_li_topology_=topology_version_;++view_budget_li_full_renders_;
  }
  torch::Tensor full_loss;
  {
    auto timing=mapping_profiler_.scope(MappingProfiler::FullLoss);
    auto l1_loss=opt_params_.smooth_l1_?loss_utils::smooth_l1_loss:loss_utils::l1_loss;
    const float lambda=lambdaDssim();
    full_loss=real_lod_full_loss_weight_*((1.f-lambda)*l1_loss(rendered.image,truth,1.f)+
        lambda*(1.f-loss_utils::fast_ssim(rendered.image,truth)));
    // This is the existing Full regularizer, once per Fine step, not once per branch.
    if(opt_params_.opacity_reg_)
      full_loss+=opt_params_.opacity_reg_*(selector_detach_gaussian_parameters_?
          gaussians_->getOpacityActivation().detach():gaussians_->getOpacityActivation()).abs().mean();
  }
  const double value=full_loss.detach().item<double>();
  if(!std::isfinite(value))++flexgs_lod_nonfinite_;
  TORCH_CHECK(std::isfinite(value),"RealLoD nonfinite Full loss; Adam not advanced");
  {
    auto timing=mapping_profiler_.scope(MappingProfiler::FullBackward);
    full_loss.backward();
    ++real_lod_full_backwards_;
  }
  return value;
}

void GaussianMapper::trainRealLodIteration(const std::shared_ptr<GaussianKeyframe>& camera) {
  TORCH_CHECK(real_lod_.optimizingLod()&&camera,"RealLoD step outside its phase");
  const auto start=std::chrono::steady_clock::now();
  renderer_workspace::Scope workspace(reuse_renderer_workspace_);
  ssim_backend::Scope ssim(optimized_ssim_backward_);
  ssim_backend::CacheScope ssim_cache(cached_ssim_statistics_);
  mapping_cuda::L1FusionScope l1(fused_l1_loss_);
  std::unique_lock<std::mutex> lock(mutex_render_);
  // Match official first fine iteration, then iterations divisible by interval.
  if(flexgs_importance_enabled_ && (!flexgs_importance_.defined() ||
      (real_lod_.lod_steps+1)%flexgs_importance_interval_==0 || flexgs_importance_topology_!=topology_version_))
    refreshFlexgsImportance();
  ensureSharedMappingOptimizer();
  view_importance_=torch::Tensor{};
  view_budget_li_=torch::Tensor{};
  if(selector_view_residual_enabled_||view_budget_view_residual_)
    selector_network_->selectorViewResidual()->initialize(gaussians_->getXYZ());
  if(selector_view_input_enabled_||view_budget_conditioning_)selector_network_->initializeSelectorViewInput(gaussians_->getXYZ());
  gaussians_->optimizer_->zero_grad(true);
  selector_optimizer_->zero_grad(true);
  increaseIteration(1);
  if(!selector_detach_gaussian_parameters_) {
    gaussians_->updateLearningRate(getIteration());
    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());
  }
  mapping_profiler_.begin(device_type_==torch::kCUDA);
  mapping_profiler_.setPhase(true);
  auto index=real_lod_.ratioIndex(selector_target_ratios_.size());
  if(selector_view_input_enabled_&&real_lod_.viewResidualRound(selector_target_ratios_.size())&&
      !selector_view_ratio_indices_.empty())
    index=selector_view_ratio_indices_[index%selector_view_ratio_indices_.size()];
  const auto topology=topology_version_;
  const auto population=gaussians_->getXYZ().size(0);
  auto truth=camera->original_image_.to(device_type_);
  // Backward frees Full's render/SSIM caches before LoD reuses the workspace.
  // No zero_grad, parameter update or topology mutation may occur between branches.
  const double full_loss=real_lod_full_loss_weight_>0.f?accumulateRealLodFullGradients(camera,truth):0.;
  auto lod=(mixed_ratio_config_.enabled||selector_view_residual_enabled_||selector_view_input_enabled_)?accumulateSpatialRatioGradients(camera,truth,index):
      accumulateLodGradients(camera,camera->image_height_,camera->image_width_,truth,
          index,selector_detach_gaussian_parameters_,true,nullptr,false,true);
  if(!lod.finite_loss)++flexgs_lod_nonfinite_;
  TORCH_CHECK(lod.rendered&&lod.trained&&lod.finite_loss,"RealLoD forward/loss failed; no update counted");
  LodGradientResult auxiliary;
  if(view_budget_enabled_&&view_budget_loss_weight_>0) {
    auxiliary=accumulateViewBudgetGradients(camera,truth,index);
    if(!auxiliary.finite_loss)++flexgs_lod_nonfinite_;
    TORCH_CHECK(auxiliary.trained&&auxiliary.finite_loss,"ViewBudget failed; Adam not advanced");
  }
  std::vector<torch::Tensor> checks;
  bool master_gradient=false,selector_gradient=false;
  for(size_t g=0;g<gaussians_->optimizer_->param_groups().size();++g) {
    const auto& group=gaussians_->optimizer_->param_groups()[g];
    for(const auto& parameter:group.params())if(parameter.grad().defined()) {
      TORCH_CHECK(!selector_detach_gaussian_parameters_||g>=6,
                  "Frozen RealLoD received a Master gradient in group ",g);
      checks.push_back(torch::isfinite(parameter.grad()).all());
      if(static_cast<const torch::optim::AdamOptions&>(group.options()).lr()!=0) {
        master_gradient|=g<6;selector_gradient|=g==shared_selector_group_index_;
      }
    }
  }
  const bool finite=checks.empty()||torch::stack(checks).all().item<bool>();
  if(!finite)++flexgs_lod_nonfinite_;
  TORCH_CHECK(finite,"RealLoD nonfinite gradient; Adam not advanced");
  bool committed=false;
  stepWithHistoricalGuard(camera,true,&committed);
  TORCH_CHECK(committed,"RealLoD Adam did not commit");
  flexgs_lod_master_updates_+=master_gradient;
  flexgs_lod_selector_updates_+=selector_gradient;
  selector_step_+=selector_gradient;
  gaussians_->optimizer_->zero_grad(true);
  TORCH_CHECK(topology==topology_version_&&population==gaussians_->getXYZ().size(0),
      "RealLoD must not mutate Gaussian/Anchor topology");
  ++real_lod_ratio_counts_[index];
  if(view_budget_enabled_&&view_budget_loss_weight_>0) {
    ++view_budget_ratio_counts_[index];
    auto& plan=auxiliary.spatial_plan;
    plan["global_K"]=Json::Int64(lod.active_gaussians);
    plan["global_loss"]=lod.loss;plan["full_loss"]=full_loss;
    plan["joint_loss"]=full_loss+lod.loss+auxiliary.loss;
    plan["optimizer_committed"]=true;plan["adam_commits_this_step"]=1;
    Json::StreamWriterBuilder writer;writer["indentation"]="";
    std::ofstream trace(result_dir_/"view_budget_steps.jsonl",std::ios::app);
    trace<<Json::writeString(writer,plan)<<'\n';
  }
  if(selector_view_residual_enabled_||selector_view_input_enabled_) {
    ++(real_lod_.viewResidualRound(selector_target_ratios_.size())?residual_view_ratio_counts_:global_view_ratio_counts_)[index];
    lod.spatial_plan["full_loss"]=full_loss;lod.spatial_plan["lod_loss"]=lod.loss;
    lod.spatial_plan["master_update"]=master_gradient;lod.spatial_plan["selector_update"]=selector_gradient;
    lod.spatial_plan["optimizer_committed"]=true;
    Json::StreamWriterBuilder writer;writer["indentation"]="";
    std::ofstream trace(result_dir_/(selector_view_input_enabled_?"view_input_steps.jsonl":"view_residual_steps.jsonl"),std::ios::app);
    trace<<Json::writeString(writer,lod.spatial_plan)<<'\n';
  }
  if(mixed_ratio_config_.enabled) {
    ++(real_lod_.mixedRound(selector_target_ratios_.size())?mixed_ratio_counts_:uniform_ratio_counts_)[index];
    lod.spatial_plan["master_update"]=master_gradient;
    lod.spatial_plan["selector_update"]=selector_gradient;
    lod.spatial_plan["full_loss"]=full_loss;
    lod.spatial_plan["lod_loss"]=lod.loss;
    lod.spatial_plan["ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    lod.spatial_plan["optimizer_committed"]=true;
    Json::StreamWriterBuilder writer;writer["indentation"]="";
    std::ofstream mixed_trace(result_dir_/"mixed_ratio_steps.jsonl",std::ios::app);
    mixed_trace<<Json::writeString(writer,lod.spatial_plan)<<'\n';
  }
  real_lod_.completeLod();
  ++mapping_iter_;
  const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  real_lod_training_ms_+=ms;
  mapping_profiler_.end(mapping_iter_);
  std::ofstream trace(result_dir_/"real_lod_steps.csv",std::ios::app);
  if(trace.tellp()==0)trace<<"step,ratio,N,K,photo_and_regularization,master_update,selector_update,xyz_lr,ms,full_loss,joint_loss\n";
  trace<<real_lod_.lod_steps<<','<<selector_target_ratios_[index]<<','<<population<<','<<lod.active_gaussians
       <<','<<lod.loss<<','<<master_gradient<<','<<selector_gradient<<','
       <<static_cast<const torch::optim::AdamOptions&>(gaussians_->optimizer_->param_groups()[0].options()).lr()<<','<<ms
       <<','<<full_loss<<','<<full_loss+lod.loss+auxiliary.loss<<'\n';
  if(real_lod_.lod_steps<=4||real_lod_.lod_steps%1000==0)
    std::cout<<"[RealLoD] step="<<real_lod_.lod_steps<<'/'<<real_lod_.target_steps
             <<" ratio="<<selector_target_ratios_[index]<<" N="<<population<<" K="<<lod.active_gaussians
             <<" master="<<master_gradient<<" selector="<<selector_gradient
             <<" full_loss="<<full_loss<<" lod_loss="<<lod.loss
             <<" view_loss="<<auxiliary.loss<<" loss="<<full_loss+lod.loss+auxiliary.loss<<'\n';
}

void GaussianMapper::runRealLodOptimization() {
  if(!real_lod_.enabled||isStopped())return;
  TORCH_CHECK(real_lod_.phase==RealLodSchedule::Phase::TailFull,"RealLoD phase transition repeated");
  if(!real_lod_.online_steps) {
    real_lod_.beginLod();saveRealLodState(result_dir_);return;
  }
  renderAndRecordAllKeyframes("_full_end");
  {
    std::unique_lock<std::mutex> lock(mutex_render_);
    savePly(result_dir_/(std::to_string(getIteration())+"_full_end")/"ply");
    real_lod_.beginLod();
    gaussians_->optimizer_->zero_grad(true);
    selector_optimizer_->zero_grad(true);
  }
  std::cout<<"[RealLoD BEGIN] online="<<real_lod_.online_steps<<" full_tail="<<real_lod_.tail_steps
           <<" lod_target="<<real_lod_.target_steps<<" topology=fixed master="
           <<(selector_detach_gaussian_parameters_?"frozen":"joint")
           <<" full_loss_weight="<<real_lod_full_loss_weight_<<" adam_commits_per_step=1\n";
  while(real_lod_.optimizingLod()&&!isStopped()) {
    auto camera=useOneRandomKeyframe(); // All saved training keyframes, no exhausted-use filtering.
    TORCH_CHECK(camera,"RealLoD lost all training keyframes");
    trainRealLodIteration(camera);
    if(real_lod_.lod_steps%1000==0)saveRealLodState(result_dir_);
  }
  saveRealLodState(result_dir_);
}

void GaussianMapper::saveRealLodState(const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);
  Json::Value state;
  state["phase"]=real_lod_.name();
  state["online_full_steps"]=Json::UInt64(real_lod_.online_steps);
  state["tail_full_steps"]=Json::UInt64(real_lod_.tail_steps);
  state["lod_target_steps"]=Json::UInt64(real_lod_.target_steps);
  state["lod_steps"]=Json::UInt64(real_lod_.lod_steps);
  state["master_updates"]=Json::UInt64(flexgs_lod_master_updates_);
  state["selector_updates"]=Json::UInt64(flexgs_lod_selector_updates_);
  state["lod_nonfinite"]=Json::UInt64(flexgs_lod_nonfinite_);
  state["importance_refreshes"]=Json::UInt64(flexgs_importance_version_);
  state["importance_full_renders"]=Json::UInt64(flexgs_importance_views_);
  state["importance_ms"]=flexgs_importance_ms_;
  state["lod_training_ms_including_importance"]=real_lod_training_ms_;
  state["importance_statistic"]="sum_views_sum_valid_pixels_opacity_times_T";
  state["importance_detached"]=true;
  state["topology_frozen_in_lod"]=true;
  state["master_frozen_in_lod"]=selector_detach_gaussian_parameters_;
  state["fine_objective"]=real_lod_full_loss_weight_>0.f?"full_plus_lod":"lod_only";
  if(view_budget_enabled_&&view_budget_loss_weight_>0)
    state["fine_objective"]=view_budget_global_control_?"full_plus_global_lod_plus_extra_global_lod":
        "full_plus_global_lod_plus_view_lod";
  state["full_photo_weight"]=real_lod_full_loss_weight_;
  state["photo_dssim_weight"]=lambdaDssim();
  state["full_opacity_regularization"]=real_lod_full_loss_weight_>0.f?opt_params_.opacity_reg_:0.f;
  state["full_training_renders"]=Json::UInt64(real_lod_full_renders_);
  state["full_training_backwards"]=Json::UInt64(real_lod_full_backwards_);
  state["lod_training_renders"]=Json::UInt64(real_lod_.lod_steps);
  state["lod_training_backwards"]=Json::UInt64(real_lod_.lod_steps);
  state["view_budget_enabled"]=view_budget_enabled_;
  state["view_budget_view_residual"]=view_budget_view_residual_;
  state["view_budget_conditioning"]=view_budget_conditioning_;
  if(view_budget_view_residual_) {
    state["view_budget_residual_input"]=view_budget_residual_input_;
    state["view_budget_residual_parameters"]=view_budget_residual_input_=="physical11"?321:657;
    state["view_budget_view_encoding"]=view_budget_residual_input_=="physical11"?
        "detached Anchor view7; detached Selector physical11; 18->16->1 keep-logit residual":
        "detached Anchor view7; detached Selector hidden32; 39->16->1 keep-logit residual";
    state["view_budget_base_photo_budget_gradients"]=true;
  }
  state["view_budget_loss_weight"]=view_budget_loss_weight_;
  state["view_budget_sampler"]="gumbel_st_soft_budget_no_topk";
  state["view_budget_gi_weight"]=0;
  state["view_budget_li_weight"]=view_budget_li_weight_;
  state["view_budget_li_statistic"]="current_uncompensated_full_sum_valid_pixels_alpha_times_T_div_HW";
  state["view_budget_li_gradient_scope"]=view_budget_view_residual_?
      "selector_view_residual_only_targeted_vjp":"local_emd_and_soft_net_only_targeted_vjp";
  state["view_budget_li_full_renders"]=Json::UInt64(view_budget_li_full_renders_);
  state["view_budget_li_backwards"]=Json::UInt64(view_budget_li_backwards_);
  state["view_budget_seed"]=view_budget_seed_;
  state["view_budget_global_control"]=view_budget_global_control_;
  state["view_budget_training_renders"]=Json::UInt64(view_budget_renders_);
  state["view_budget_training_backwards"]=Json::UInt64(view_budget_backwards_);
  state["total_training_renders"]=Json::UInt64(real_lod_full_renders_+real_lod_.lod_steps+view_budget_renders_);
  for(auto count:view_budget_ratio_counts_)state["view_budget_ratio_steps"].append(Json::UInt64(count));
  if(view_budget_generator_.defined()) {
    torch::save(view_budget_generator_.get_state(),(directory/"view_budget_rng.pt").string());
    state["view_budget_rng_state"]="view_budget_rng.pt";
  }
  state["optimizer_commits"]=Json::UInt64(real_lod_.lod_steps);
  state["maturity_loss"]=false;
  state["maturity_gate"]=false;
  for(auto r:selector_target_ratios_)state["ratios"].append(r);
  for(auto n:real_lod_ratio_counts_)state["ratio_steps"].append(Json::UInt64(n));
  state["mixed_ratio_enabled"]=mixed_ratio_config_.enabled;
  state["view_residual_enabled"]=selector_view_residual_enabled_;
  state["view_input_enabled"]=selector_view_input_enabled_;
  if(selector_view_residual_enabled_||selector_view_input_enabled_) {
    state["view_residual_round"]="global_full_rotation_then_view_residual_full_rotation";
    state["view_residual_scope"]="full_map_both_rounds";
    state["view_residual_current_gi"]="current_uncompensated_full_opacity_times_T";
    state["view_residual_global_selector_detached_in_view"]=!selector_view_input_enabled_;
    if(selector_view_input_enabled_) {
      state["view_input_encoding"]="physical11 + (global: zeros7,mode0; current: view7,mode1); shared19->32->32->2";
      state["view_input_gi"]="exact_K_eligible; primaryGI/globalGI/fixed_topology_row_UID; deterministic_logit_BCE";
      state["view_input_global_gi_weight"]=selector_gi_loss_weight_;
      state["view_input_single_view_gi_weight"]=selector_view_gi_weight_;
      state["view_input_single_view_gi_lowest_two_only"]=selector_view_gi_lowest_two_only_;
      state["view_input_visible_point_budget"]=selector_view_visible_budget_;
      state["view_input_budget_population"]=selector_view_visible_budget_?"nominal_visible_anchor_members; full_map_candidates":"full_map";
      if(!selector_view_ratio_indices_.empty())
        for(auto index:selector_view_ratio_indices_)state["view_input_target_ratios"].append(selector_target_ratios_[index]);
      state["view_input_round_steps"]=Json::UInt64(selector_target_ratios_.size());
    }
    for(auto n:global_view_ratio_counts_)state["global_view_ratio_steps"].append(Json::UInt64(n));
    for(auto n:residual_view_ratio_counts_)state["residual_view_ratio_steps"].append(Json::UInt64(n));
  }
  if(mixed_ratio_config_.enabled) {
    state["mixed_ratio_seed"]=mixed_ratio_config_.seed;
    state["mixed_ratio_scope"]="full_map_both_rounds";
    state["mixed_ratio_anchor_frustum_culling"]=false;
    state["mixed_ratio_projection"]="longest_pixel_side_divided_by_height";
    state["mixed_ratio_round"]="uniform_full_rotation_then_mixed_full_rotation";
    for(auto n:uniform_ratio_counts_)state["uniform_ratio_steps"].append(Json::UInt64(n));
    for(auto n:mixed_ratio_counts_)state["mixed_ratio_steps"].append(Json::UInt64(n));
  }
  state["optimizer_resume_complete"]=false; // Existing model exports are not complete SLAM/Adam/RNG resumes.
  std::ofstream stream(directory/"real_lod_state.json");stream<<state;
  if(flexgs_importance_.defined())torch::save(flexgs_importance_,(directory/"flexgs_importance.pt").string());
}
