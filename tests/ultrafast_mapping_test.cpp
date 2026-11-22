#include "include/gaussian_mapper.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

std::vector<torch::Tensor> snapshot(const std::vector<torch::Tensor>& parameters) {
  std::vector<torch::Tensor> result;
  for (const auto& p : parameters) result.push_back(p.detach().clone());
  return result;
}
bool changed(const std::vector<torch::Tensor>& parameters,
             const std::vector<torch::Tensor>& before) {
  for (std::size_t i = 0; i < parameters.size(); ++i)
    if (!parameters[i].equal(before[i])) return true;
  return false;
}

// Exercise the production KF sampler and Mapping body with real shared Adam.
class RotationMapper : public GaussianMapper {
  static std::filesystem::path selectorConfig(const std::filesystem::path& output, bool verbose) {
    std::filesystem::create_directories(output);
    std::ifstream input(std::filesystem::path(CARTGS_TEST_SOURCE_DIR) /
#ifdef CARTGS_TEST_SPATIAL_COMPENSATION
        "cfg/selector/selector_config.yaml");
#else
        "cfg/selector/fast.yaml");
#endif
    check(input.is_open(), "Cannot read existing fast config");
    const auto path = output / "selector_config.yaml";
    std::ofstream fixture(path);
    check(fixture.is_open(), "Cannot write test config");
    const std::map<std::string, std::string> overrides = {
        {"Mapping.iteration_logging:", verbose ? "1" : "0"},
        {"Training.ultra_light_mode:", "1"},
#ifdef CARTGS_TEST_TAIL_R3
        {"ChangeDistill.enable:", "0"},
#endif
#ifdef CARTGS_TEST_SELECTOR_COMPENSATION
        {"TransformField.enable:", "0"},
#endif
#ifdef CARTGS_TEST_ANCHOR_SELECTOR
        {"GlobalGI.mode:", "disabled"},
        {"Selector.architecture:", "\"anchor\""},
#ifdef CARTGS_TEST_ANCHOR_ATTRIBUTES
        {"Selector.anchor_gaussian_attributes:", "1"},
#endif
#ifdef CARTGS_TEST_ANCHOR_ADAPTIVE
        {"Selector.anchor_adaptive:", "1"},
#endif
#ifdef CARTGS_TEST_ANCHOR_GLOBAL_XYZ
        {"Selector.anchor_global_xyz:", "1"},
#endif
#ifdef CARTGS_TEST_EXPLICIT_ANCHORS
        {"Selector.explicit_anchor_blocks:", "1"},
#endif
#else
        {"GlobalGI.mode:", "recent"},
        {"Selector.architecture:", "\"legacy\""},
        {"Selector.protection_enabled:", "1"},
#endif
        {"Protection.drift_temperature:", "0.1"}};
    for (std::string line; std::getline(input, line);) {
      for (const auto& replacement : overrides)
        if (line.rfind(replacement.first, 0) == 0)
          line = replacement.first + " " + replacement.second;
      fixture << line << '\n';
    }
    fixture << "RecentWindow.enable: 1\n";
#ifdef CARTGS_TEST_ANCHOR_TOPOLOGY_POLICY
    fixture << "Selector.anchor_only_pruning: 1\nSelector.anchor_refine_requires_full: 1\n";
#endif
#ifdef CARTGS_TEST_SELECTOR_COMPENSATION
    fixture << "Selector.compensation: 1\nSelector.compensation_start_iter: 3000\n";
#endif
#ifdef CARTGS_TEST_TAIL_R3
    fixture << "Selector.compensation_method: \"r3_independent14\"\n"
            << "CUDA.fused_dense_adam: 1\nCUDA.fused_compensation_decode: 1\n"
            << "CUDA.fused_anchor_input_pack: 1\nCUDA.fused_anchor_full_stats: 1\n"
            << "CUDA.fused_selector_compaction: 1\nCUDA.reuse_renderer_workspace: 1\n"
            << "CUDA.optimized_ssim_backward: 1\nSelector.anchor_only_pruning: 0\n";
#endif
#ifdef CARTGS_TEST_VARIABLE_ANCHORS
    fixture << "Selector.variable_anchor_members: 1\n";
#endif
#ifdef CARTGS_TEST_RATIO_ANCHOR_FEATURES
    fixture << "Selector.anchor_ratio_features: 1\n";
#endif
#ifdef CARTGS_TEST_ANCHOR_FEATURE32
    fixture << "Selector.anchor_feature_dim: 32\nSelector.anchor_ratio_features: 0\n";
#endif
#ifdef CARTGS_TEST_SLOT_DEACTIVATION
    fixture << "Selector.slot_deactivation: 1\n";
#endif
#ifdef CARTGS_TEST_FULL_RECOVERY
    fixture << "Training.post_topology_full_steps: 100\n";
#endif
#ifdef CARTGS_TEST_SERIAL_FULL_LOD
    fixture << "Training.serial_full_lod: 1\nTraining.post_topology_full_steps: 0\n";
#endif
#ifdef CARTGS_TEST_SERIAL_JOINT
    fixture << "Training.serial_lod_detach_master: 0\n";
#endif
    return path;
  }
 public:
  std::vector<torch::Tensor> first_cycle_parameters;
  std::vector<torch::Tensor> serial_master_history;
  std::vector<int64_t> serial_adam_steps;
  explicit RotationMapper(const std::filesystem::path& output, bool verbose)
      : GaussianMapper(nullptr,
          std::filesystem::path(CARTGS_TEST_SOURCE_DIR) / "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
          output, 0, torch::kCUDA, true,
          selectorConfig(output, verbose)) {}

  void verify(bool full_only_reference = false) {
#ifdef CARTGS_TEST_FLEXGS_SELECTION
    check(selector_selection_mode_ == "flexgs" && selector_ratio_loss_weight_ == .01f,
          "FlexGS sampling mode/budget weight missing from config");
    check(selector_gumbel_budget_normalize_, "Normalized Gumbel budget missing from config");
    saveSelectorConfig(result_dir_ / "flexgs_roundtrip");
    cv::FileStorage saved((result_dir_ / "flexgs_roundtrip/selector_config.yaml").string(), cv::FileStorage::READ);
    std::string saved_mode; saved["Selector.selection_mode"] >> saved_mode;
    check(saved_mode == "flexgs" && std::abs(double(saved["Selector.ratio_loss_weight"]) - .01) < 1e-7,
          "Saved Selector config lost FlexGS sampling/budget settings");
    check(int(saved["Selector.gumbel_budget_normalize"]) == 1,
          "Saved Selector config lost normalized Gumbel budget");
#endif
#ifdef CARTGS_TEST_ANCHOR_TOPOLOGY_POLICY
    check(selector_anchor_only_pruning_ && selector_anchor_refine_requires_full_,"Mapper lost Anchor topology policy flags");
#endif
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto& model = *gaussians_;
    model.xyz_ = torch::tensor({{-0.1f,0.0f,2.0f},{0.1f,0.0f,2.0f},{0.0f,0.1f,2.1f}}, opts).set_requires_grad(true);
    model.features_dc_ = torch::full({3,1,3}, 0.2f, opts).set_requires_grad(true);
    model.features_rest_ = torch::full({3,15,3}, 0.01f, opts).set_requires_grad(true);
    model.opacity_ = torch::zeros({3,1}, opts).set_requires_grad(true);
    model.scaling_ = torch::full({3,3}, -2.7f, opts).set_requires_grad(true);
    model.rotation_ = torch::tensor({{1.0f,0.0f,0.0f,0.0f}}, opts).repeat({3,1}).set_requires_grad(true);
    model.active_sh_degree_ = default_sh_ = 1;
    model.spatial_lr_scale_ = 1.0f;
    model.max_radii2D_ = torch::zeros({3}, opts);
    model.exist_since_iter_ = torch::zeros({3}, opts.dtype(torch::kInt32));
    model.protection_.reset(3, opts);
    model.protection_.maturity.fill_(1);
    model.protection_.mapping_iteration = 6000;
    model.last_elastic_supervised_iter_ = torch::zeros({3}, opts.dtype(torch::kLong));
    model.Tensor_vec_xyz_ = {model.xyz_}; model.Tensor_vec_feature_dc_ = {model.features_dc_};
    model.Tensor_vec_feature_rest_ = {model.features_rest_}; model.Tensor_vec_opacity_ = {model.opacity_};
    model.Tensor_vec_scaling_ = {model.scaling_}; model.Tensor_vec_rotation_ = {model.rotation_};
#ifdef CARTGS_TEST_EXPLICIT_ANCHORS
    std::map<point3D_id_t,Point3D> observed;
    for(int i=0;i<3;++i) {
      Point3D p;p.xyz_=Eigen::Vector3d(i*.08-.08,.02,2.02);p.color_=Eigen::Vector3f(.4,.4,.4);
      observed[i]=p;
    }
    model.createFromPcd(observed,1);
#ifdef CARTGS_TEST_SLOT_DEACTIVATION
    {
      torch::NoGradGuard guard;
      model.opacity_[0].fill_(-12);
      auto bank=selector_network_->anchors();
      auto enough=torch::ones({bank->size()},opts.dtype(torch::kBool));
      auto visits=torch::full({bank->size()},5,opts.dtype(torch::kLong));
      auto below=torch::zeros({bank->size(),10},visits.options());below[0][0].fill_(5);
      bank->closeSlotWindow(enough,visits,below,model.getOpacityActivation());
      check(bank->inactiveSlots().sum().item<int64_t>()==1,"Mapping fixture failed to deactivate one low slot");
    }
#endif
#endif
    model.trainingSetup(opt_params_);
    if (selector_network_->anchorConditioned()) {
      if(!selector_explicit_anchor_blocks_) selector_network_->anchors()->reset(model.xyz_, {selector_optimizer_.get()});
      check(!global_gi_ && !geometry_marginal_ && !recent_window_config_.collect_importance &&
            selector_gi_loss_weight_ == 0, "Anchor path retained explicit SI/GI");
    }
    recent_window_importance_->resetForGaussianCount(model.xyz_.size(0));
    iteration_ = 6000; mapping_iter_ = 6000;
    scene_->cameras_extent_ = 1.0f;
    do_gaus_pyramid_training_ = false;
    selector_detach_gaussian_parameters_ = true; // Rotation must always override this.
    selector_gi_loss_weight_ = 0.0f;
    // Fixture only: isolate the real opacity-ST photo gradient to the Anchor
    // network/features instead of allowing the budget term to satisfy this test.
    if (selector_network_->anchorConditioned()) selector_ratio_loss_weight_ = 0.0f;
    selector_replay_enabled_ = false;
    gradient_constraint_config_.enabled = false;
    opt_params_.iterations_ = 10000;
    opt_params_.densify_until_iter_ = -1;
    opt_params_.densify_from_iter_ = 100000;
    opt_params_.opacity_reset_interval_ = 0;
    transform_field_config_.hexplane.spatial_resolution = 4;
    transform_field_config_.hexplane.ratio_resolution = 4;
    transform_field_config_.hexplane.feature_dim = 2;
    transform_field_config_.hexplane.multires = {1};
    transform_field_config_.hidden_dim = 4;
    keyframe_record_interval_ = all_keyframes_record_interval_ = training_report_interval_ = 0;
    {
      torch::NoGradGuard no_grad;
      selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.0f,4.0f}, opts));
    }
    initializeTransformFieldIfNeeded();
    ensureSharedMappingOptimizer();

    const std::array<std::size_t, 5> ids = {1,8,3,11,2};
    for (auto id : ids) {
      auto camera = std::make_shared<GaussianKeyframe>(id);
      camera->camera_id_ = 0;
      camera->setPose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
      camera->image_height_ = 32; camera->image_width_ = 40;
      camera->FoVx_ = camera->FoVy_ = 1.0f;
      camera->world_view_transform_ = torch::eye(4, opts);
      camera->full_proj_transform_ = camera->getProjectionMatrix(
          0.01f,100.0f,1.0f,1.0f,torch::kCUDA).transpose(0,1).contiguous();
      camera->camera_center_ = torch::zeros({3}, opts);
      camera->original_image_ = torch::full({3,32,40}, 0.1f, opts);
      camera->remaining_times_of_use_ = 1;
      scene_->addKeyframe(camera, &kfid_shuffled_);
    }
    // Map order is 1,2,3,8,11. Each iteration obtains a new sampler result.
    kfid_shuffle_ = {0,3,2,4,1};
    kfid_shuffle_idx_ = 4; kfid_shuffled_ = true;
    for (auto& entry : scene_->keyframes()) entry.second->remaining_times_of_use_ = 0;
    check(useOneRandomSlidingWindowKeyframe() != nullptr && kfs_loss_.empty(),
          "Adaptive sampler failed before the first Full quality estimate");
    for (auto& entry : scene_->keyframes()) entry.second->remaining_times_of_use_ = 1;
    kfid_shuffle_idx_ = 4;
    kfs_used_times_.clear();
#ifdef CARTGS_TEST_SERIAL_FULL_LOD
    check(ultrafast_rotation_.serial_full_lod && !ultrafast_rotation_.post_topology_full_steps,
          "Serial config did not disable Full100");
    const bool joint_lod = !ultrafast_rotation_.serial_lod_detach_master;
    // The reference uses the exact same production Full stage, without LoD.
    // Comparing parameters AND both Adam moments catches detached-input or
    // stale-momentum leaks, even when the combined loss looks reasonable.
    selector_enabled_ = !full_only_reference;
    const auto auxiliary_before = snapshot(selector_network_->parameters());
    for (int i=0;i<13;++i) {
      if (i==8) markUltrafastOnlineEnd();
      if (i==12) {
        opt_params_.densify_from_iter_=0;opt_params_.densification_interval_=1;
        opt_params_.densify_grad_threshold_=1e6f;densify_min_opacity_=0;
      }
      auto camera=useOneRandomSlidingWindowKeyframe();
      const auto g_before=snapshot(model.optimizer_->param_groups()[1].params());
      const auto s_before=snapshot(selector_network_->parameters());
      const auto anchor_before=selector_network_->anchors()->features().detach().clone();
      const auto tf_before=snapshot(transform_field_->parameters());
      const auto full_before=ultrafast_rotation_.full_render_count;
      const auto lod_before=ultrafast_rotation_.lod_render_count;
      ++iteration_;trainMappingIteration(camera,true,i>=8);++mapping_iter_;
      check(ultrafast_rotation_.full_render_count==full_before+1,"Serial mapping omitted Full");
      check(ultrafast_rotation_.lod_render_count==lod_before+(!full_only_reference),"Serial mapping LoD count");
      check(changed(model.optimizer_->param_groups()[1].params(),g_before),"Full did not update Master");
      if(!full_only_reference && i<12) {
        check(changed(selector_network_->parameters(),s_before),"Serial LoD did not update Selector");
        check(!anchor_before.equal(selector_network_->anchors()->features()),"Serial LoD did not update Anchor feature");
        check(changed(transform_field_->parameters(),tf_before),"Serial LoD did not update TF");
      }
      for(std::size_t g=0;g<6;++g) for(const auto& p:model.optimizer_->param_groups()[g].params()) {
        check(!p.grad().defined(),"Serial left Master gradients after LoD");
        serial_master_history.push_back(p.detach().cpu().clone());
        const auto& state=static_cast<const torch::optim::AdamParamState&>(*model.optimizer_->state().at(p.unsafeGetTensorImpl()));
        serial_adam_steps.push_back(state.step());
        serial_master_history.push_back(state.exp_avg().cpu().clone());
        serial_master_history.push_back(state.exp_avg_sq().cpu().clone());
      }
      check(ultrafast_rotation_.balanced(),"Serial ratio balance or Mapping clock incorrect");
    }
    const auto commits = joint_lod && !full_only_reference ? 2 : 1;
    check(mode2_online_.full_renders==8 && mode2_tail_.full_renders==5 &&
          mode2_online_.gaussian_updates==8*commits && mode2_tail_.gaussian_updates==5*commits,
          "Serial Master stage update counts incorrect");
    if(!full_only_reference) {
      check(ultrafast_rotation_.counts[0]==2 && ultrafast_rotation_.counts[1]==2 &&
            ultrafast_rotation_.counts[2]==2 && ultrafast_rotation_.counts[3]==2 &&
            ultrafast_rotation_.tail_ratio_counts==std::array<std::uint64_t,4>{2,1,1,1},"Serial ratio order online/tail");
      check(mode2_tail_.topology==1 && ultrafast_rotation_.recovery_windows==0,"Topology or disabled Full100 incorrect");
    } else check(!changed(selector_network_->parameters(),auxiliary_before),"Full-only reference changed auxiliary parameters");
    for(const auto& group:model.optimizer_->param_groups()) for(const auto& p:group.params())
      first_cycle_parameters.push_back(p.detach().cpu().clone());
    if(!full_only_reference && !joint_lod) {
      // Check isolation within one execution: unlike two independent CUDA
      // Full renders, this comparison does not depend on atomic reduction order.
      std::vector<torch::Tensor> frozen;
      std::vector<int64_t> steps;
      auto capture=[&](bool verify) {
        std::size_t j=0,k=0;
        for(std::size_t g=0;g<6;++g) for(const auto& p:model.optimizer_->param_groups()[g].params()) {
          const auto& state=static_cast<const torch::optim::AdamParamState&>(*model.optimizer_->state().at(p.unsafeGetTensorImpl()));
          for(const auto& v:{p,state.exp_avg(),state.exp_avg_sq()}) {
            if(verify) check(torch::equal(v,frozen[j++]),"Detached LoD modified Master parameter/Adam moment");
            else frozen.push_back(v.detach().clone());
          }
          if(verify) check(state.step()==steps[k++],"Detached LoD advanced Master Adam step");
          else steps.push_back(state.step());
        }
      };
      capture(false);
      auto camera=scene_->keyframes().begin()->second;
      auto gt=camera->original_image_;
      const auto auxiliary=snapshot(selector_network_->parameters());
      auto result=accumulateLodGradients(camera,32,40,gt,1,true,true,nullptr,false,true);
      check(result.rendered && result.finite_loss,"Isolated LoD render/backward failed");
      for(std::size_t g=0;g<6;++g) for(const auto& p:model.optimizer_->param_groups()[g].params())
        check(!p.grad().defined(),"Isolated LoD created Master gradient");
      stepWithHistoricalGuard(camera,false);capture(true);
      check(changed(selector_network_->parameters(),auxiliary),"Isolated LoD did not update auxiliary parameters");
      model.optimizer_->zero_grad(true);
      std::cout<<"PASS: isolated LoD Master parameters, Adam steps and both moments EXACT unchanged\n";
    }
    if(!full_only_reference && joint_lod) {
      auto camera=scene_->keyframes().begin()->second;
      auto gt=camera->original_image_;
      const auto color=model.features_dc_.detach().clone();
      const auto selector=snapshot(selector_network_->parameters());
      const auto anchor=selector_network_->anchors()->features().detach().clone();
      const auto tf=snapshot(transform_field_->parameters());
      const auto key=model.features_dc_.unsafeGetTensorImpl();
      const auto& state=static_cast<const torch::optim::AdamParamState&>(*model.optimizer_->state().at(key));
      const auto step=state.step();const auto first=state.exp_avg().clone(),second=state.exp_avg_sq().clone();
      const auto statistics=model.denom_.clone(),visits=model.anchorVisits().clone();
      auto result=accumulateLodGradients(camera,32,40,gt,1,false,true,nullptr,false,true);
      check(result.rendered && result.finite_loss && model.features_dc_.grad().defined() &&
            torch::isfinite(model.features_dc_.grad()).all().item<bool>() &&
            model.features_dc_.grad().abs().sum().item<float>()>0,"Joint LoD did not create finite Master gradient");
      stepWithHistoricalGuard(camera,false);
      const auto& updated=static_cast<const torch::optim::AdamParamState&>(*model.optimizer_->state().at(key));
      check(!torch::equal(color,model.features_dc_) && updated.step()==step+1 &&
            !torch::equal(first,updated.exp_avg()) && !torch::equal(second,updated.exp_avg_sq()),
            "Joint LoD did not update Master parameters/Adam states");
      check(changed(selector_network_->parameters(),selector) &&
            !torch::equal(anchor,selector_network_->anchors()->features()) && changed(transform_field_->parameters(),tf),
            "Joint LoD did not update Selector/Anchor/TF");
      check(torch::equal(statistics,model.denom_) && torch::equal(visits,model.anchorVisits()),
            "Joint LoD contaminated Full-only topology evidence");
      model.optimizer_->zero_grad(true);
      std::cout<<"PASS: joint LoD finite Master gradient, parameter and Adam moments update; Selector/Anchor/TF update; Full-only statistics preserved\n";
    }
    std::cout<<"PASS: serial Full then "<<(joint_lod?"joint":"detached")<<" LoD, online/tail ratios, Master and auxiliary updates, topology, no Full100\n";
    return;
#endif
    for (std::size_t i = 0; i < ids.size(); ++i) {
      auto camera = useOneRandomSlidingWindowKeyframe();
      check(camera->fid_ == ids[i], "Ratio rotation changed KF sampler results");
      const auto selector_before = snapshot(selector_network_->parameters());
#ifdef CARTGS_TEST_RATIO_ANCHOR_FEATURES
      const auto feature_before = snapshot(selector_network_->anchors()->featureTables());
#endif
      const auto anchor_before = selector_network_->anchorConditioned()
          ? selector_network_->anchors()->features().detach().clone() : torch::Tensor{};
      const auto tf_before = transform_field_ ? snapshot(transform_field_->parameters()) : std::vector<torch::Tensor>{};
      const auto gaussian_before = snapshot(model.optimizer_->param_groups()[1].params());
      const auto losses = kfs_loss_;
      const auto ema = ema_loss_for_log_;
      const auto accum = model.xyz_gradient_accum_.clone(), denom = model.denom_.clone();
      const auto radii = model.max_radii2D_.clone();
      const auto anchor_visits = selector_explicit_anchor_blocks_ ? model.anchorVisits().clone() : torch::Tensor{};
      const auto anchor_opacity = selector_explicit_anchor_blocks_ ? model.anchorOpacityAccum().clone() : torch::Tensor{};
      const auto topology = topology_version_, tf_steps = normal_update_clock_.tf_updates;
      std::vector<int64_t> adam_steps;
      std::vector<torch::Tensor> moments;
      if (i == 4) {
        for (std::size_t group = 6; group < model.optimizer_->param_groups().size(); ++group)
          for (const auto& p : model.optimizer_->param_groups()[group].params()) {
            const auto found = model.optimizer_->state().find(p.unsafeGetTensorImpl());
            adam_steps.push_back(found == model.optimizer_->state().end() ? 0 :
                static_cast<const torch::optim::AdamParamState&>(*found->second).step());
            moments.push_back(found == model.optimizer_->state().end() ? torch::Tensor() :
                static_cast<const torch::optim::AdamParamState&>(*found->second).exp_avg().clone());
            // A geometry event may leave stale gradients before Full.
            p.mutable_grad() = torch::ones_like(p);
          }
      }
      ++iteration_;
      trainMappingIteration(camera, true, false);
      ++mapping_iter_;
      check(changed(model.optimizer_->param_groups()[1].params(), gaussian_before),
            "Normal render did not update Master Gaussian color");
      check(topology_version_ == topology, "Unexpected topology decision");
      if (i < 4) {
#ifdef CARTGS_TEST_RATIO_ANCHOR_FEATURES
        const auto tables=selector_network_->anchors()->featureTables();
        check(tables.size()==5 && tables[0].size(1)==8,"Mapping did not use ratio Anchor features");
        for(size_t feature=1;feature<tables.size();++feature)
          if(feature!=i+1)check(torch::equal(tables[feature],feature_before[feature]),"Production LoD changed an inactive ratio Feature");
#endif
        if (selector_explicit_anchor_blocks_)
          check(torch::equal(anchor_visits,model.anchorVisits())&&torch::equal(anchor_opacity,model.anchorOpacityAccum()),
                "LoD contaminated Full-only Anchor pruning evidence");
        check(kfs_loss_ == losses && ema_loss_for_log_ == ema, "LOD contaminated KF loss/EMA");
        check(model.xyz_gradient_accum_.equal(accum) && model.denom_.equal(denom) &&
              model.max_radii2D_.equal(radii), "LOD contaminated densification statistics");
        const auto ratio = selector_target_ratios_.at(i);
        const bool tf_executed = continuous_tf_statistics_.ratios.at(ratio).executed > 0;
        check(changed(selector_network_->parameters(), selector_before), "LOD failed to update Selector");
        if (anchor_before.defined() && anchor_before.numel())
          check(!anchor_before.equal(selector_network_->anchors()->features()), "LoD failed to update Anchor features");
        check((transform_field_ && changed(transform_field_->parameters(), tf_before)) == tf_executed,
              "TF changed on a skipped branch or failed to update an executed branch");
        check(normal_update_clock_.tf_updates == tf_steps + (tf_executed ? 1 : 0), "LOD TF count wrong");
      } else {
        if (selector_explicit_anchor_blocks_)
          check((model.anchorVisits()>anchor_visits).any().item<bool>()&&
                (model.anchorOpacityAccum()>anchor_opacity).any().item<bool>(),"Full did not accumulate Anchor opacity/visits");
        check((selector_network_->anchorConditioned() ? (model.protection_.maturity == 1).all().item<bool>() :
              model.protection_.snapshot_valid.all().item<bool>()) && !model.protection_.touched_in_cycle.any().item<bool>(),
              "Committed Full failed to create the cycle snapshot/reset touched");
        check(kfs_loss_.size() == 1 && kfs_loss_.count(camera->fid_) == 1,
              "Full did not update KF loss exactly once");
        check((model.denom_ > denom).any().item<bool>(), "Full did not accumulate visibility statistics");
        check(!changed(selector_network_->parameters(), selector_before) &&
              !(transform_field_ && changed(transform_field_->parameters(), tf_before)) &&
              normal_update_clock_.tf_updates == tf_steps, "Full updated Selector/TF");
        std::size_t j = 0;
        for (std::size_t group = 6; group < model.optimizer_->param_groups().size(); ++group)
          for (const auto& p : model.optimizer_->param_groups()[group].params()) {
            const auto found = model.optimizer_->state().find(p.unsafeGetTensorImpl());
            if (adam_steps[j] == 0) check(found == model.optimizer_->state().end(), "Full created auxiliary Adam state");
            else {
              const auto& state = static_cast<const torch::optim::AdamParamState&>(*found->second);
              check(state.step() == adam_steps[j] && state.exp_avg().equal(moments[j]),
                    "Full advanced auxiliary Adam momentum");
            }
            ++j;
          }
      }
    }
    check(ultrafast_rotation_.full_render_count == 1 && ultrafast_rotation_.lod_render_count == 4 &&
          ultrafast_rotation_.selector_updates == 4 &&
          ultrafast_rotation_.tf_updates == continuous_tf_statistics_.ratios.at(.01f).executed +
              continuous_tf_statistics_.ratios.at(.05f).executed + continuous_tf_statistics_.ratios.at(.10f).executed +
              continuous_tf_statistics_.ratios.at(.15f).executed,
          "Five normal mappings did not produce exactly 4 LOD + 1 Full");

    for (const auto& group : model.optimizer_->param_groups())
      for (const auto& parameter : group.params())
        first_cycle_parameters.push_back(parameter.detach().cpu().clone());

#ifdef CARTGS_TEST_SELECTOR_COMPENSATION
    check(!transform_field_ && !transform_optimizer_ && compensation_forwards_==4,
          "Compensation must replace TF and run only on LoD");
#if !defined(CARTGS_TEST_TAIL_R3) && !defined(CARTGS_TEST_SPATIAL_COMPENSATION)
    check(selector_network_->named_parameters()["compensation.2.weight"].abs().sum().item<float>()>0,
          "Rendering loss did not train compensation head");
    if(selector_network_->compensationOutputWidth()==14)
      for(int channel=10;channel<14;++channel)
        check(selector_network_->named_parameters()["compensation.2.weight"][channel].abs().sum().item<float>()>0,
              "Rendering loss did not train each opacity/color output");
#endif
#ifdef CARTGS_TEST_SPATIAL_COMPENSATION
    check(selector_network_->spatialCompensationEnabled()&&selector_network_->spatialCompensation()->ready(),"Spatial path/normalization missing");
    check(selector_network_->named_parameters()["spatial_compensation.heads.other.2.weight"].abs().sum().item<float>()>0,"Photo loss failed to train Spatial appearance head");
#endif
    markUltrafastOnlineEnd();
    const auto frozen_online = ultrafast_rotation_.counts;
    for(int i=0;i<10;++i) {
      auto camera=useOneRandomSlidingWindowKeyframe();
#ifdef CARTGS_TEST_TAIL_R3
      const auto phase=ultrafast_rotation_.phase();
      auto tables=selector_network_->privateCompensation()->tables();
      std::vector<int64_t> before_steps;
      for(const auto& table:tables) {
        auto it=model.optimizer_->state().find(table.unsafeGetTensorImpl());
        before_steps.push_back(it==model.optimizer_->state().end()?0:
            static_cast<const torch::optim::AdamParamState&>(*it->second).step());
      }
#endif
      ++iteration_;trainMappingIteration(camera,true,true);++mapping_iter_;
#ifdef CARTGS_TEST_TAIL_R3
      for(std::size_t k=0;k<tables.size();++k) {
        const auto& state=static_cast<const torch::optim::AdamParamState&>(
            *model.optimizer_->state().at(tables[k].unsafeGetTensorImpl()));
        check(state.step()==before_steps[k]+(phase==k),"Tail updated the wrong R3 ratio Adam clock");
        check(!tables[k].grad().defined(),"Tail failed to clear R3 gradients to None");
      }
#endif
    }
    check(compensation_forwards_==12 && ultrafast_rotation_.tail_lod_attempts==8 &&
          mode2_tail_.full_renders==2 && mode2_tail_.lod_renders==8 &&
          mode2_tail_.gaussian_updates==10 && ultrafast_rotation_.counts==frozen_online &&
          ultrafast_rotation_.tail_phase_counts==std::array<std::uint64_t,5>{2,2,2,2,2},
          "Compensation tail must execute exactly one branch per five-slot rotation");
    reportUltrafastRotation("tail_compensation_test");
    std::cout<<"PASS: compensation photo gradient, joint LoD Master/Selector/Feature updates, Full parameter and Adam isolation, original online/tail ratios, no TF\n";
    return;
#endif

#ifdef CARTGS_TEST_FULL_RECOVERY
    // A real topology pass arms the NEXT 100 Mapping calls, including at tail.
    opt_params_.densify_from_iter_=0;opt_params_.densification_interval_=5;
    opt_params_.densify_grad_threshold_=1e6f;densify_min_opacity_=0;
    for(int i=0;i<5;++i) {
      auto camera=useOneRandomSlidingWindowKeyframe();
      ++iteration_;trainMappingIteration(camera,true,false);++mapping_iter_;
    }
    check(ultrafast_rotation_.recovery_remaining==100,"Topology pass did not arm 100 next mappings");
    opt_params_.densify_from_iter_=100000;
    const auto next_phase=ultrafast_rotation_.phase();
    const auto previous_full=ultrafast_rotation_.full_render_count;
    const auto previous_lod=ultrafast_rotation_.lod_render_count;
    const auto previous_selector=selector_step_;
    const auto previous_tf=normal_update_clock_.tf_updates;
    const auto selector_frozen=snapshot(selector_network_->parameters());
    const auto tf_frozen=snapshot(transform_field_->parameters());
    const auto geometry=model.xyz_.detach().clone();
    for(int i=0;i<100;++i) {
      auto camera=useOneRandomSlidingWindowKeyframe();
      ++iteration_;trainMappingIteration(camera,true,false);++mapping_iter_;
      check(ultrafast_rotation_.recovery_remaining==99-i,"Full recovery has an off-by-one error");
      check(ultrafast_rotation_.phase()==next_phase,"Recovery consumed a normal ratio slot");
    }
    check(ultrafast_rotation_.full_render_count==previous_full+100 && ultrafast_rotation_.lod_render_count==previous_lod &&
          selector_step_==previous_selector && normal_update_clock_.tf_updates==previous_tf &&
          !changed(selector_network_->parameters(),selector_frozen) && !changed(transform_field_->parameters(),tf_frozen),
          "Full recovery ran LoD or modified Selector/Anchor/TF");
    check(!torch::equal(geometry,model.xyz_.detach()),"Full recovery did not optimize Master geometry");
    {auto camera=useOneRandomSlidingWindowKeyframe();++iteration_;trainMappingIteration(camera,true,false);++mapping_iter_;}
    check(ultrafast_rotation_.lod_render_count==previous_lod+1,"LoD did not resume on mapping 101");
    // Tail Full+LoD must also pause LoD while keeping the same ratio ready.
    markUltrafastOnlineEnd();ultrafast_rotation_.beginRecovery();
    const auto tail_phase=ultrafast_rotation_.tailRatioIndex();
    const auto tail_lod=ultrafast_rotation_.tail_lod_attempts;
    for(int i=0;i<100;++i) {
      auto camera=useOneRandomSlidingWindowKeyframe();++iteration_;trainMappingIteration(camera,true,true);++mapping_iter_;
      check(ultrafast_rotation_.tail_lod_attempts==tail_lod && ultrafast_rotation_.tailRatioIndex()==tail_phase,
            "Tail recovery advanced LoD");
    }
    {auto camera=useOneRandomSlidingWindowKeyframe();++iteration_;trainMappingIteration(camera,true,true);++mapping_iter_;}
    check(ultrafast_rotation_.tail_lod_attempts==tail_lod+1 && ultrafast_rotation_.balanced(),"Tail LoD did not resume");
    reportUltrafastRotation("recovery_test");
    std::cout<<"PASS: real topology trigger, exact 100 Full, paused ratio, no Selector/Anchor/TF update, Master update, online/tail resume\n";
    return;
#endif

    // Ordinary deadlines land on LOD; clone/split/prune wait for Full.
    opt_params_.densify_from_iter_ = 0;
    opt_params_.densification_interval_ = 3;
    opt_params_.densify_grad_threshold_ = 1e6f;
    densify_min_opacity_ = 0.0f;
    const auto topology_before = topology_version_;
    for (int i = 0; i < 5; ++i) {
      auto camera = useOneRandomSlidingWindowKeyframe();
      const auto color_before = model.features_dc_.detach().clone();
      ++iteration_; trainMappingIteration(camera, true, false); ++mapping_iter_;
      if (i < 4) check(topology_version_ == topology_before, "LOD triggered clone/split/prune");
      else check(!model.features_dc_.equal(color_before), "Full topology discarded Gaussian calibration gradients");
    }
    check(topology_version_ > topology_before && !ultrafast_densify_task_.pending,
          "Full missed deferred topology deadline");
    check(selector_network_->anchorConditioned() ?
          ((model.protection_.maturity == 1).all().item<bool>() && !model.protection_.drift_initialized.any().item<bool>()) :
          model.protection_.drift_initialized.any().item<bool>(),
          "Second committed Full did not observe a touched Full-to-Full drift");

    // No mature points: preserve existing readiness, still render only LOD.
    transform_field_ = nullptr; transform_optimizer_.reset();
    shared_transform_optimizer_configured_ = false;
    transform_field_config_.enabled = false;
    {
      torch::NoGradGuard no_grad;
      if (!selector_network_->anchorConditioned()) model.protection_.maturity.zero_();
      model.protection_.cached_uniform.fill_(1.0f); // Empty admitted compact population.
    }
    const auto renders_before = ultrafast_rotation_.lod_render_count;
    auto camera = useOneRandomSlidingWindowKeyframe();
    ++iteration_; trainMappingIteration(camera, true, false); ++mapping_iter_;
    check(!transform_field_ && ultrafast_rotation_.lod_render_count == renders_before + 1,
          "Unready TF or empty LOD caused a Full fallback");

    // Online end freezes statistics, but the production entry keeps training.
    transform_field_config_.enabled = true;
    model.protection_.maturity.fill_(1);
    model.protection_.cached_uniform.zero_();
    initializeTransformFieldIfNeeded();
    ensureSharedMappingOptimizer();
    markUltrafastOnlineEnd();
    const auto mappings = mapping_iter_, slots = ultrafast_rotation_.step;
    const auto online_counts = ultrafast_rotation_.counts;
    const auto full_renders = ultrafast_rotation_.full_render_count;
    const auto lod_renders = ultrafast_rotation_.lod_render_count;
    opt_params_.densify_from_iter_ = 100000;
    do_gaus_pyramid_training_ = true;
    num_gaus_pyramid_sub_levels_ = 1;
    scene_->cameras_[0].gaus_pyramid_undistort_mask_ = {torch::ones({16,20},opts)};
    for (auto& entry : scene_->keyframes()) {
      auto& c = entry.second;
      c->num_gaus_pyramid_sub_levels_ = 1;
      c->gaus_pyramid_times_of_use_ = {100};
      c->gaus_pyramid_height_ = {16}; c->gaus_pyramid_width_ = {20};
      c->gaus_pyramid_original_image_ = {torch::full({3,16,20},0.3f,opts)};
    }
    // This fixture tests tail optimizer commits, not geometry leaving the TF domain.
    transform_field_->setAABB({torch::tensor({-1.0f,-1.0f,1.0f},opts),
                               torch::tensor({1.0f,1.0f,3.0f},opts)});
    check(transform_field_->insideMask(model.xyz_).all().item<bool>(), "Tail fixture outside TF domain");
    const auto color_before_tail = model.features_dc_.detach().clone();
    const auto tf_before_tail = snapshot(transform_field_->parameters());
    tail_optimization_schedule_.enabled = true;
    tail_optimization_schedule_.steps = 7;
    opt_params_.iterations_ = 1; // Forced tail must bypass the optimizer cap.
    beginForcedTail(getIteration());
    check(hasNormalOptimizationBudget(), "Mode1 forced tail retained the optimizer cap");
    int expected_full = 0;
    for (int i = 0; i < 9; ++i) {
      auto& color = model.features_dc_;
      const auto before = color.detach().clone();
      const auto& old_state = static_cast<const torch::optim::AdamParamState&>(*model.optimizer_->state().at(color.unsafeGetTensorImpl()));
      const auto adam_step = old_state.step();
      const bool full = ultrafast_rotation_.full();
      if (i < 7) expected_full += full;
      const auto maturity = model.protection_.maturity.clone();
      const auto full_snapshot = model.protection_.prev_full_xyz.clone();
      int backwards = 0;
      const auto hook = color.register_hook([&](torch::Tensor gradient) {
        ++backwards;
        return gradient;
      });
      trainForOneIteration();
      color.remove_hook(hook);
      const auto& state = static_cast<const torch::optim::AdamParamState&>(*model.optimizer_->state().at(color.unsafeGetTensorImpl()));
      if (i < 7) {
        check(backwards == 1 && state.step() == adam_step+1,"Tail failed to execute one Gaussian backward/Adam commit");
        check(!color.equal(before),"Tail failed to update Master");
        if (!full) check(model.protection_.maturity.equal(maturity) &&
            model.protection_.prev_full_xyz.equal(full_snapshot),"Tail LoD recomputed Full maturity/snapshots");
      } else check(backwards == 0 && state.step() == adam_step,"Completed tail still trained");
    }
    check(ultrafast_rotation_.full_render_count == full_renders+expected_full &&
          ultrafast_rotation_.lod_render_count == lod_renders+7-expected_full && ultrafast_rotation_.counts == online_counts &&
          mode2_tail_.full_renders == expected_full && mode2_tail_.lod_renders == 7-expected_full && mode2_tail_.gaussian_updates == 7,
          "Tail did not execute exactly one rotating branch per Mapping or changed online counts");
    int remaining_pyramid_uses = 0;
    for (const auto& entry : scene_->keyframes()) remaining_pyramid_uses += entry.second->gaus_pyramid_times_of_use_[0];
    check(remaining_pyramid_uses == 500-expected_full,"Tail Full skipped GausPyramid or LoD consumed a pyramid level");
    // Native LoD uses original GT even when the Full input is a pyramid image.
    auto pyramid_gt = camera->gaus_pyramid_original_image_[0];
    auto native_lod = accumulateLodGradients(camera,16,20,pyramid_gt,0,false,true,nullptr,false,true);
    check(native_lod.render_height == 32 && native_lod.render_width == 40,"Tail LoD did not render native dimensions");
    model.optimizer_->zero_grad(true);
    check(mapping_iter_ == mappings + 7 && ultrafast_rotation_.step == slots + 7 &&
          ultrafast_rotation_.tailSteps() == 7 && tail_optimization_schedule_.completed(mapping_iter_) == 7,
          "Mode1 fixed tail did not execute exactly seven real mappings");
    check(!model.features_dc_.equal(color_before_tail) && changed(transform_field_->parameters(), tf_before_tail),
          "Mode1 tail advanced clocks without Gaussian/TF optimizer updates");
    check(ultrafast_rotation_.online_end_mapping_iteration == mappings &&
          ultrafast_rotation_.online_end_rotation_step == slots, "Tail moved the online snapshot");

    tail_optimization_schedule_ = TailOptimizationSchedule{};
    tail_optimization_schedule_.enabled = true;
    tail_optimization_schedule_.steps = 0;
    beginForcedTail(getIteration());
    const auto zero_mapping = mapping_iter_;
    const auto zero_iteration = getIteration();
    const auto zero_color = model.features_dc_.detach().clone();
    trainForOneIteration();
    check(mapping_iter_ == zero_mapping && getIteration() == zero_iteration && model.features_dc_.equal(zero_color),
          "Mode1 zero-step tail performed training");

    tail_optimization_schedule_ = TailOptimizationSchedule{};
    selector_max_mapping_iterations_ = 0;
    opt_params_.iterations_ = -1;
    for (int i = 0; i < 3; ++i) trainForOneIteration();
    check(mapping_iter_ == zero_mapping + 3 && ultrafast_rotation_.tailSteps() == 10 &&
          !tail_optimization_schedule_.active, "Mode1 legacy tail was blocked by online end");
    // A disabled Selector on a scheduled Full must preserve Full and skip auxiliaries.
    while (!ultrafast_rotation_.full()) trainForOneIteration();
    const auto attempts = ultrafast_rotation_.tail_lod_attempts;
    selector_enabled_ = false;
    const auto before_disabled = static_cast<const torch::optim::AdamParamState&>(
        *model.optimizer_->state().at(model.features_dc_.unsafeGetTensorImpl())).step();
    trainForOneIteration();
    const auto after_disabled = static_cast<const torch::optim::AdamParamState&>(
        *model.optimizer_->state().at(model.features_dc_.unsafeGetTensorImpl())).step();
    check(after_disabled == before_disabled+1 && ultrafast_rotation_.tail_lod_attempts == attempts,
          "Scheduled tail Full updated auxiliaries or failed to update Master");
    selector_enabled_ = true;
    if (selector_network_->anchorConditioned()) {
      model.protection_.maturity.fill_(1);
      const auto old_xyz = model.xyz_.detach().clone();
      auto capture = captureGeometryChange(ChangeFlag::GLOBAL_GEOMETRY);
      check(capture.teacher_selector && capture.teacher_selector->anchorConditioned(), "Anchor CD snapshot missing");
      auto frozen_teacher = capture.teacher_selector;
      const auto teacher_scale = capture.old_state.log_scaling.exp();
      const auto teacher_dc = capture.old_selector_sh_dc;
      const auto teacher_opacity = capture.old_selector_opacity;
      const auto teacher_probability = capture.teacher_selector->forwardProbability(old_xyz, {}, teacher_scale,
          .1f, 1, {}, teacher_dc, teacher_opacity).detach().clone();
      auto correction = model.applyScaledTransformation(1.02f, Sophus::SE3f());
      capture.has_geometry_mutation = true;
      change_coordinator_->addCorrection(capture, correction);
      const auto selector_cd_before = cd_selector_steps_, tf_cd_before = cd_tf_steps_;
      finishGeometryChange(std::move(capture));
      check(teacher_probability.equal(frozen_teacher->forwardProbability(old_xyz, {}, teacher_scale,
          .1f, 1, {}, teacher_dc, teacher_opacity)), "CD mutated frozen Anchor teacher");
      check(cd_selector_steps_ > selector_cd_before && cd_tf_steps_ > tf_cd_before,
            "Geometry event did not distill Anchor Selector and TF");
      // Append/prune use the actual Gaussian callbacks and both optimizer owners.
      auto extra_xyz = model.xyz_.slice(0,0,1).detach().clone() + .12f;
      auto dc = model.features_dc_.slice(0,0,1).detach().clone();
      auto rest = model.features_rest_.slice(0,0,1).detach().clone();
      auto opacity = model.opacity_.slice(0,0,1).detach().clone();
      auto scaling = model.scaling_.slice(0,0,1).detach().clone();
      auto rotation = model.rotation_.slice(0,0,1).detach().clone();
      auto since = model.exist_since_iter_.slice(0,0,1).clone();
      if(selector_explicit_anchor_blocks_) {
        auto color=(dc.squeeze(1)*0.28209479177387814+.5).clamp(0,1);
        model.increasePcd(extra_xyz,color,getIteration());
      } else model.densificationPostfix(extra_xyz, dc, rest, opacity, scaling, rotation, since);
      check(selector_network_->anchors()->rows().numel() == model.xyz_.size(0), "Append callback lost Anchor mapping");
      check((model.protection_.maturity == 1).all().item<bool>(), "Append restored maturity dynamics");
      auto remove = torch::zeros({model.xyz_.size(0)}, opts.dtype(torch::kBool));
      remove.index_put_({0}, true);
      model.prunePoints(remove);
      if(selector_explicit_anchor_blocks_ && !model.variableAnchorMembers()) {
        check(!remove.any().item<bool>(),"Partial block unexpectedly pruned");
        remove.slice(0,0,10).fill_(true);model.prunePoints(remove);
        selector_network_->anchors()->assertBlocks(model.xyz_.size(0));
        bool rejected=false;
        auto grad=torch::ones({model.xyz_.size(0),1},opts);
        try { model.densifyAndClone(grad,.1,1); } catch(const c10::Error&) {rejected=true;}
        check(rejected,"Explicit map allowed Gaussian clone");
      } else if(model.variableAnchorMembers()) {
        selector_network_->anchors()->assertBlocks(model.xyz_.size(0));
        check(remove[0].item<bool>(),"Variable Anchor rejected pointwise prune");
      }
      check(selector_network_->anchors()->rows().numel() == model.xyz_.size(0), "Prune callback lost Anchor mapping");
      check((model.protection_.maturity == 1).all().item<bool>(), "Prune changed fixed maturity");
      // Force a shared-Anchor partial correction: student association may split,
      // teacher keeps the complete old feature/row map, and CD trains current leaves.
      model.protection_.maturity.fill_(1);
      auto local_capture = captureGeometryChange(ChangeFlag::LOCAL_GEOMETRY);
      GaussianTransformResult local;
      local.indices = torch::tensor({0}, opts.dtype(torch::kLong));
      local.population_count = model.xyz_.size(0);
      local.transform_4x4 = torch::eye(4, opts);
      local.transform_4x4.index_put_({3,0}, .15f);
      local.rotation_wxyz = torch::tensor({1.f,0.f,0.f,0.f}, opts);
      local.changed_attributes = toBits(GaussianAttrFlag::XYZ);
      {
        torch::NoGradGuard no_grad;
        model.xyz_.index_put_({0}, model.xyz_[0] + torch::tensor({.15f,0.f,0.f}, opts));
      }
      local_capture.has_geometry_mutation = true;
      if(selector_explicit_anchor_blocks_) selector_network_->anchors()->transportBlocks(local);
      change_coordinator_->addCorrection(local_capture, local);
      const auto local_steps = cd_selector_steps_;
      finishGeometryChange(std::move(local_capture));
      check(cd_selector_steps_ > local_steps, "Local loop correction failed Anchor Selector CD");
      check(selector_network_->anchors()->inputs(model.xyz_).first.size(0) == model.xyz_.size(0), "Post-CD Anchor association");
      std::cout << "PASS: Anchor global/local geometry transport, Selector+TF CD, append/prune optimizer migration\n";
    }
    reportUltrafastRotation("tail_test");
    check(mapping_profiler_.enabled == mapping_iteration_logging_ &&
          ultrafast_rotation_.elapsed.empty() == !mapping_iteration_logging_,
          "Iteration logging switch did not disable profiler/timing collection");
    std::cout << "PASS: production KF independence, one branch, KF loss/EMA, topology statistics/decisions, "
                 "G/S/TF updates, Full Adam freeze, pyramid Full and native joint LoD in five-slot rotation, one Gaussian commit, "
                 "Full-only maturity, fixed/zero/legacy mode1 tail, frozen online boundary\n";
  }
};
} // namespace

int main(int argc, char** argv) {
  if (!torch::cuda::is_available()) { std::cout << "SKIP: CUDA unavailable\n"; return 77; }
  const auto output = std::filesystem::temp_directory_path() /
      ("cartgs-ultrafast-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  const bool verbose = argc > 1 && std::string(argv[1]) == "--verbose";
  const bool parity = argc > 1 && std::string(argv[1]) == "--parity";
  std::ostringstream output_log;
  auto* previous = std::cout.rdbuf(output_log.rdbuf());
  try {
    torch::manual_seed(20261016);
    RotationMapper mapper(output, verbose);
    mapper.verify();
#if defined(CARTGS_TEST_SERIAL_FULL_LOD) && !defined(CARTGS_TEST_SERIAL_JOINT)
    torch::manual_seed(20261016);
    RotationMapper reference(output / "full_reference",verbose);
    reference.verify(true);
    check(mapper.serial_adam_steps==reference.serial_adam_steps,"LoD advanced Master Adam steps");
    check(mapper.serial_master_history.size()==reference.serial_master_history.size(),"Serial Master history size");
    for(std::size_t i=0;i<mapper.serial_master_history.size();++i)
      check(torch::allclose(mapper.serial_master_history[i],reference.serial_master_history[i],1e-5,1e-6),
            "Serial Master parameters/moments differ from Full-only oracle");
    std::cout<<"PASS: serial Master parameters/moments match independent Full-only reference at every Mapping (rtol=1e-5, atol=1e-6), Adam steps exact\n";
#endif
    if (parity) {
      torch::manual_seed(20261016);
      RotationMapper measured(output / "verbose", true);
      measured.verify();
      check(mapper.first_cycle_parameters.size() == measured.first_cycle_parameters.size(),
            "Disabling logging changed optimizer parameter groups");
      for (std::size_t i = 0; i < mapper.first_cycle_parameters.size(); ++i)
        check(torch::allclose(mapper.first_cycle_parameters[i], measured.first_cycle_parameters[i], 1e-5, 1e-6),
              "Disabling iteration logging/timing changed Gaussian/Selector/TF updates");
    }
  } catch (...) {
    std::cout.rdbuf(previous);
    std::cerr << output_log.str();
    throw;
  }
  std::cout.rdbuf(previous);
  check((output_log.str().find("[UltrafastRotation] normal_training_step=") != std::string::npos) == (verbose || parity),
        "Production iteration log output does not match the configured switch");
  if (verbose || parity)
#if defined(CARTGS_TEST_SELECTOR_COMPENSATION)
    check(output_log.str().find("[Selector Compensation]")!=std::string::npos,"Compensation logging missing");
#elif !defined(CARTGS_TEST_FULL_RECOVERY) && !defined(CARTGS_TEST_SERIAL_FULL_LOD)
    check(output_log.str().find("full_height=16 full_width=20") != std::string::npos &&
          output_log.str().find("lod_height=32 lod_width=40") != std::string::npos,
          "Production tail failed to rotate pyramid Full and native LoD");
#elif defined(CARTGS_TEST_FULL_RECOVERY)
    check(output_log.str().find("FULL_RECOVERY")!=std::string::npos,"Recovery logging missing");
#else
    check(output_log.str().find("phase=FULL_LOD")!=std::string::npos,"Serial logging missing");
#endif
  std::cout << output_log.str();
}
