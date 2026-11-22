#include "include/gaussian_mapper.h"
#include "include/gaussian_renderer.h"
#include "include/loss_utils.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
using Overrides = std::map<std::string, std::string>;

std::filesystem::path config(const std::filesystem::path& output, const Overrides& overrides) {
  std::filesystem::create_directories(output);
  std::ifstream input(std::filesystem::path(CARTGS_TEST_SOURCE_DIR) / "cfg/selector/selector_config.yaml");
  check(input.is_open(), "Cannot read canonical configuration");
  auto pending = overrides;
  const auto path = output / "selector_config.yaml";
  std::ofstream fixture(path);
  for (std::string line; std::getline(input, line);) {
    const auto colon = line.find(':');
    if (colon != std::string::npos) {
      const auto found = pending.find(line.substr(0, colon));
      if (found != pending.end()) {
        if (!found->second.empty()) fixture << found->first << ": " << found->second << '\n';
        pending.erase(found);
        continue;
      }
    }
    fixture << line << '\n';
  }
  for (const auto& kv : pending) if (!kv.second.empty()) fixture << kv.first << ": " << kv.second << '\n';
  return path;
}

struct ParameterState {
  torch::Tensor value, mean, variance;
  int64_t step = -1;
};
using Snapshot = std::vector<ParameterState>;
bool bitwise(const torch::Tensor& a, const torch::Tensor& b) {
  if (!a.defined() || !b.defined()) return a.defined() == b.defined();
  if (a.sizes() != b.sizes() || a.scalar_type() != b.scalar_type()) return false;
  const auto ac = a.detach().cpu().contiguous(), bc = b.detach().cpu().contiguous();
  return std::memcmp(ac.const_data_ptr(), bc.const_data_ptr(), ac.numel() * ac.element_size()) == 0;
}
Snapshot snapshot(torch::optim::Adam& optimizer, std::size_t first, std::size_t end) {
  Snapshot result;
  for (auto g = first; g < end; ++g) for (const auto& p : optimizer.param_groups()[g].params()) {
    ParameterState item;
    item.value = p.detach().clone();
    const auto state = optimizer.state().find(p.unsafeGetTensorImpl());
    if (state != optimizer.state().end()) {
      const auto& s = static_cast<const torch::optim::AdamParamState&>(*state->second);
      item.step = s.step(); item.mean = s.exp_avg().clone(); item.variance = s.exp_avg_sq().clone();
    }
    result.push_back(std::move(item));
  }
  return result;
}
void same(const Snapshot& a, const Snapshot& b, bool exact = true) {
  check(a.size() == b.size(), "Parameter population changed");
  for (std::size_t i = 0; i < a.size(); ++i) {
    check(a[i].step == b[i].step, "Adam step/state presence differs");
    for (const auto pair : {std::make_pair(a[i].value,b[i].value),
                           std::make_pair(a[i].mean,b[i].mean),
                           std::make_pair(a[i].variance,b[i].variance)}) {
      if (exact) check(bitwise(pair.first,pair.second), "Parameter or Adam moment changed bitwise");
      else if (pair.first.defined()) check(torch::allclose(pair.first,pair.second,1e-5,1e-7), "Full-only numerical parity failed");
      else check(!pair.second.defined(), "Unexpected Adam moment");
    }
  }
}
double gradNorm(const std::vector<torch::Tensor>& parameters) {
  double result = 0;
  for (const auto& p : parameters) if (p.grad().defined()) result += p.grad().abs().sum().item<double>();
  return result;
}

class Mode2Mapper : public GaussianMapper {
 public:
  Mode2Mapper(const std::filesystem::path& output, Overrides overrides, torch::DeviceType device = torch::kCUDA)
      : GaussianMapper(nullptr,
          std::filesystem::path(CARTGS_TEST_SOURCE_DIR) / "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
          output, 17, device, std::nullopt, config(output, overrides)) {}

  void configIs(int mode, float scale, int start) {
    check(ultra_light_mode_ == mode && ultrafast_rotation_.enabled == (mode == 1), "Mode/default selection wrong");
    check(selector_lod_render_scale_ == scale && transform_field_config_.start_iter == start, "Mode2 defaults wrong");
    saveSelectorConfig(result_dir_);
    cv::FileStorage saved((result_dir_/"selector_config.yaml").string(),cv::FileStorage::READ);
    check(saved["Training.ultra_light_mode"].operator int() == mode &&
          saved["Training.ultra_light_enabled"].empty() && saved["Training.ultra_light_interval"].empty(),
          "Export retained obsolete ultra-light compatibility keys");
  }
  void setup(int n = 32) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto& m = *gaussians_;
    m.xyz_ = torch::stack({torch::linspace(-0.35,0.35,n,opts),
                         torch::linspace(-0.2,0.2,n,opts).sin(),
                         torch::linspace(1.8,2.3,n,opts)},1).set_requires_grad(true);
    m.features_dc_ = torch::full({n,1,3},0.21f,opts).set_requires_grad(true);
    m.features_rest_ = torch::full({n,15,3},0.01f,opts).set_requires_grad(true);
    m.scaling_ = torch::tensor({{-3.0f,-2.6f,-3.4f}},opts).repeat({n,1}).set_requires_grad(true);
    m.rotation_ = torch::tensor({{0.96f,0.1f,0.2f,0.15f}},opts).repeat({n,1}).set_requires_grad(true);
    m.opacity_ = torch::zeros({n,1},opts).set_requires_grad(true);
    m.max_radii2D_ = torch::zeros({n},opts);
    m.exist_since_iter_ = torch::zeros({n},opts.dtype(torch::kInt32));
    m.last_elastic_supervised_iter_ = torch::zeros({n},opts.dtype(torch::kLong));
    m.protection_.reset(n,opts);
    m.protection_.maturity.fill_(1);
    m.protection_.mapping_iteration = 999;
    m.Tensor_vec_xyz_ = {m.xyz_}; m.Tensor_vec_feature_dc_ = {m.features_dc_};
    m.Tensor_vec_feature_rest_ = {m.features_rest_}; m.Tensor_vec_scaling_ = {m.scaling_};
    m.Tensor_vec_rotation_ = {m.rotation_}; m.Tensor_vec_opacity_ = {m.opacity_};
    m.active_sh_degree_ = default_sh_ = 0; m.spatial_lr_scale_ = 1.0f;
    m.trainingSetup(opt_params_);
    scene_->cameras_extent_ = 1.0f;
    recent_window_importance_->resetForGaussianCount(n);
    iteration_ = 999; mapping_iter_ = 999;
    opt_params_.iterations_ = -1; opt_params_.densify_until_iter_ = -1;
    opt_params_.densify_from_iter_ = 0; opt_params_.densification_interval_ = 3;
    opt_params_.densify_grad_threshold_ = 1e6f; opt_params_.opacity_reset_interval_ = 7;
    densify_min_opacity_ = 0.0f; prune_big_point_after_iter_ = -1;
    selector_replay_enabled_ = false; selector_gi_loss_weight_ = 0;
    keyframe_record_interval_ = all_keyframes_record_interval_ = training_report_interval_ = 0;
    record_keyframe_used_times_ = false; do_gaus_pyramid_training_ = false;
    // A one-KF fixture would hit the original adaptive sampler's empty-loss
    // boundary on its very first wrap. Use the same fixed sampler in both arms.
    opt_params_.auto_distribute_ = 0;
    transform_field_config_.hexplane.spatial_resolution = 4;
    transform_field_config_.hexplane.ratio_resolution = 4;
    transform_field_config_.hexplane.feature_dim = 2;
    transform_field_config_.hexplane.multires = {1}; transform_field_config_.hidden_dim = 8;
    camera_ = camera(64,80);
    camera_->remaining_times_of_use_ = 100;
    scene_->addKeyframe(camera_, &kfid_shuffled_);
    kfid_shuffle_ = {0}; kfid_shuffle_idx_ = 0; kfid_shuffled_ = true;
  }
  std::shared_ptr<GaussianKeyframe> camera(int h, int w) {
    auto opts = gaussians_->xyz_.options();
    auto c = std::make_shared<GaussianKeyframe>(1);
    c->camera_id_ = 0; c->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    c->image_height_ = h; c->image_width_ = w; c->FoVx_ = c->FoVy_ = 1;
    c->world_view_transform_ = torch::eye(4,opts);
    c->full_proj_transform_ = c->getProjectionMatrix(0.01f,100,1,1,torch::kCUDA).transpose(0,1).contiguous();
    c->camera_center_ = torch::zeros({3},opts);
    const auto x = torch::linspace(0.03,0.19,w,opts).reshape({1,1,w});
    c->original_image_ = x.expand({3,h,w}).contiguous();
    return c;
  }
  Snapshot master() { return snapshot(*gaussians_->optimizer_,0,6); }
  Snapshot auxiliary() { return snapshot(*gaussians_->optimizer_,6,gaussians_->optimizer_->param_groups().size()); }
  void tick() { trainForOneIteration(); }
  void enablePyramid() {
    do_gaus_pyramid_training_ = true; num_gaus_pyramid_sub_levels_ = 1;
    camera_->num_gaus_pyramid_sub_levels_ = 1;
    camera_->gaus_pyramid_times_of_use_ = {2};
    camera_->gaus_pyramid_height_ = {32}; camera_->gaus_pyramid_width_ = {40};
    camera_->gaus_pyramid_original_image_ = {torch::full({3,32,40},0.07f,gaussians_->xyz_.options())};
    scene_->cameras_[0].gaus_pyramid_undistort_mask_ = {torch::ones({32,40},gaussians_->xyz_.options())};
  }

  // Test-only oracle transcribed from baseline 547c9055, mapper lines 621-764.
  // Uses the unchanged production Gaussian/topology/rasterizer math; no mode/Guard/LoD logic.
  void originalFull() {
    ++iteration_;
    auto c = useOneRandomSlidingWindowKeyframe();
    if (!c) { --iteration_; return; }
    auto& m = *gaussians_;
    if (iteration_ % 1000 == 0 && default_sh_ < model_params_.sh_degree_) ++default_sh_;
    m.setShDegree(default_sh_);
    m.updateLearningRate(iteration_); m.setFeatureLearningRate(featureLearningRate());
    m.setOpacityLearningRate(opacityLearningRate()); m.setScalingLearningRate(scalingLearningRate());
    m.setRotationLearningRate(rotationLearningRate());
    const int level = isdoingGausPyramidTraining() ? c->getCurrentGausPyramidLevel() : num_gaus_pyramid_sub_levels_;
    const bool native = level == num_gaus_pyramid_sub_levels_;
    const auto height = native ? c->image_height_ : c->gaus_pyramid_height_[level];
    const auto width = native ? c->image_width_ : c->gaus_pyramid_width_[level];
    auto gt = native ? c->original_image_ : c->gaus_pyramid_original_image_[level];
    auto pkg = GaussianRenderer::render(c,height,width,gaussians_,pipe_params_,background_,override_color_);
    auto l1 = opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
    auto loss = (1-lambdaDssim())*l1(pkg.image,gt,1.0f) + lambdaDssim()*(1-loss_utils::fast_ssim(pkg.image,gt));
    if (opt_params_.opacity_reg_) loss += opt_params_.opacity_reg_*m.getOpacityActivation().abs().mean();
    loss.backward();
    torch::NoGradGuard no_grad;
    kfs_loss_[c->fid_] = loss.item<float>(); ema_loss_for_log_ = 0.4f*loss.item<float>() + 0.6f*ema_loss_for_log_;
    if (iteration_ < opt_params_.densify_until_iter_ || opt_params_.densify_until_iter_ == -1) {
      m.max_radii2D_.index_put_({pkg.visibility_indices},torch::max(m.max_radii2D_.index({pkg.visibility_indices}),pkg.radii.index({pkg.visibility_indices})));
      m.addDensificationStats(pkg.viewspace_points,pkg.visibility_indices);
      if (iteration_ > opt_params_.densify_from_iter_ && iteration_ % densifyInterval() == 0)
        m.densifyAndPrune(densifyGradThreshold(),densify_min_opacity_,scene_->cameras_extent_,
                         (iteration_ < prune_big_point_after_iter_ || prune_big_point_after_iter_ == -1) ? 0 : 20);
      if (opacityResetInterval() && (iteration_ % opacityResetInterval() == 0 ||
          (model_params_.white_background_ && iteration_ == opt_params_.densify_from_iter_))) m.resetOpacity();
    }
    if (loop_closure_iteration_) loop_closure_iteration_ = false;
    if (iteration_ < opt_params_.iterations_ || opt_params_.iterations_ == -1) {
      m.optimizer_->step(); m.optimizer_->zero_grad(true);
    }
    ++mapping_iter_;
  }
  void topologyCase(int which) {
    opt_params_.densify_grad_threshold_ = which == 1 ? 0.0f : 1e6f;
    if (which == 2) {
      torch::NoGradGuard no_grad;
      gaussians_->opacity_.select(0,0).fill_(-30);
      densify_min_opacity_ = 0.005f;
    }
  }
  void parity(Mode2Mapper& reference) {
    for (int i = 0; i < 10; ++i) {
      if (i == 5) { topologyCase(1); reference.topologyCase(1); }
      if (i == 8) { topologyCase(2); reference.topologyCase(2); }
      torch::manual_seed(500+i); reference.originalFull();
      const auto old_identity = gaussians_->xyz_.unsafeGetTensorImpl();
      const auto optimized = gaussians_->protection_.maturity.clone();
      const auto updates = mode2_online_.gaussian_updates;
      torch::manual_seed(500+i); tick();
      same(master(),reference.master(),false);
      check(kfs_used_times_ == reference.kfs_used_times_ && iteration_ == reference.iteration_ &&
            mapping_iter_ == reference.mapping_iter_, "KF/iteration schedule mismatch");
      check(std::abs(kfs_loss_.at(1)-reference.kfs_loss_.at(1)) < 1e-6 &&
            std::abs(ema_loss_for_log_-reference.ema_loss_for_log_) < 1e-6, "Full loss/EMA parity failed");
      for (const auto pair : {std::make_pair(gaussians_->xyz_gradient_accum_,reference.gaussians_->xyz_gradient_accum_),
                             std::make_pair(gaussians_->denom_,reference.gaussians_->denom_),
                             std::make_pair(gaussians_->max_radii2D_,reference.gaussians_->max_radii2D_)})
        check(torch::allclose(pair.first,pair.second,1e-5,1e-7), "Full statistics parity failed");
      if (i == 2) {
        check(gaussians_->xyz_.unsafeGetTensorImpl() != old_identity, "Empty topology failed to replace parameter");
        check(mode2_online_.gaussian_updates == updates && bitwise(optimized,gaussians_->protection_.maturity),
              "Empty topology step created false mode2 maturity changes");
      }
    }
    check(normal_lod_branch_count_ == 0 && selector_step_ == 0 && normal_update_clock_.tf_updates == 0, "Full-only advanced auxiliary clocks");
    check(mode2_online_.full_renders == 10 && mode2_online_.full_backwards == 10, "Full skipped an iteration");
    std::cout << "PASS Full-only: fixed schedule, SH, loss/EMA, KF, all six params/m/v/step, topology-zero/split/prune/reset/stats\n";
  }
  void gradients() {
    gaussians_->setShDegree(1); default_sh_ = 1;
    opt_params_.densify_from_iter_ = 100000; opt_params_.opacity_reset_interval_ = 0;
    initializeTransformFieldIfNeeded(); ensureSharedMappingOptimizer();
    {
      torch::NoGradGuard no_grad;
      selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.0f,4.0f},gaussians_->xyz_.options()));
    }
    selector_ratio_loss_weight_ = selector_gi_loss_weight_ = 0;
    // Test photo-only, not a norm supplied by plane regularization.
    auto cfg = transform_field_config_;
    cfg.spatial_smooth_weight = cfg.ratio_smooth_weight = cfg.ratio_l1_weight = 0;
    transform_field_config_ = cfg;
    transform_field_ = nullptr; transform_optimizer_.reset();
    // Existing groups point at the old module; recreate setup before attach.
    gaussians_->trainingSetup(opt_params_);
    shared_selector_optimizer_configured_ = shared_transform_optimizer_configured_ = false;
    initializeTransformFieldIfNeeded(); ensureSharedMappingOptimizer();
    auto& optimizer = *gaussians_->optimizer_;
    for (int repeat = 0; repeat < 2; ++repeat) {
      optimizer.zero_grad(true);
      auto pkg = GaussianRenderer::render(camera_,64,80,gaussians_,pipe_params_,background_,override_color_);
      loss_utils::l1_loss(pkg.image,camera_->original_image_,1.0f).backward();
      for (int g = 0; g < 6; ++g) check(gradNorm(optimizer.param_groups()[g].params()) > 0, "Full Gaussian grad is zero");
      const auto aux_before = auxiliary(); const auto master_before = master();
      check(commitMode2Stage(true).gaussian, "Full commit not recognized");
      check(!bitwise(master_before[1].value,master()[1].value), "Full did not change Gaussian color");
      same(aux_before,auxiliary()); optimizer.zero_grad(true);
      const auto master_full = master();
      const auto stats = gaussians_->xyz_gradient_accum_.clone(), radii = gaussians_->max_radii2D_.clone();
      const auto denom = gaussians_->denom_.clone(), evidence = gaussians_->protection_.maturity.clone();
      auto gt = camera_->original_image_;
      const auto lod = accumulateLodGradients(camera_,64,80,gt,repeat,true,true,nullptr,true);
      check(lod.rendered && lod.render_height == 32 && lod.render_width == 40 && lod.transformed_gaussians > 0, "LoD/TF missing or wrong dimensions");
      for (int g = 0; g < 6; ++g) for (const auto& p : optimizer.param_groups()[g].params())
        check(!p.grad().defined(), "LoD wrote Gaussian grad before clear");
      check(gradNorm(selector_network_->parameters()) > 0, "Selector photo ST gradient missing");
      check(gradNorm(transform_field_->gridParameters()) > 0 && gradNorm(transform_field_->mlpParameters()) > 0, "TF plane/MLP photo gradient missing");
      for (const auto name : {"position_out.weight","scale_out.weight","rotation_out.weight"})
        check(gradNorm({transform_field_->named_parameters()[name]}) > 0, "TF photo head gradient missing");
      const auto aux_grad_before = auxiliary();
      auto committed = commitMode2Stage(false);
      check(committed.selector && committed.tf && committed.finite, "Auxiliary commit missing");
      same(master_full,master());
      const auto aux_after = auxiliary();
      std::size_t aux_index = 0;
      bool selector_changed = false, planes_changed = false, mlp_changed = false;
      for (std::size_t g = 6; g < optimizer.param_groups().size(); ++g)
        for (const auto& p : optimizer.param_groups()[g].params()) {
          const auto& old = aux_grad_before.at(aux_index);
          const auto& current = aux_after.at(aux_index++);
          if (p.grad().defined())
            check(current.step == (old.step < 0 ? 1 : old.step+1), "Aux Adam skipped/doubled defined-gradient state step");
          else check(current.step == old.step, "Undefined auxiliary gradient advanced Adam state");
          const bool changed = !bitwise(old.value,current.value);
          if (g == shared_selector_group_index_) selector_changed |= changed;
          if (g == shared_transform_grid_group_index_) planes_changed |= changed;
          if (g == shared_transform_mlp_group_index_) mlp_changed |= changed;
        }
      check(selector_changed && planes_changed && mlp_changed, "Selector/TF plane/MLP parameters did not actually update");
      check(bitwise(stats,gaussians_->xyz_gradient_accum_) && bitwise(radii,gaussians_->max_radii2D_) &&
            bitwise(denom,gaussians_->denom_) && bitwise(evidence,gaussians_->protection_.maturity), "LoD contaminated Full statistics/maturity");
      optimizer.zero_grad(true);
    }
    // A stale Gaussian grad is rejected, not silently applied a second time.
    gaussians_->xyz_.mutable_grad() = torch::ones_like(gaussians_->xyz_);
    bool rejected = false;
    try { commitMode2Stage(false); } catch (const c10::Error&) { rejected = true; }
    check(rejected,"Auxiliary stage accepted a stale Gaussian gradient"); optimizer.zero_grad(true);
    // Existing auxiliary moments and stale gradients must survive Full unchanged.
    const auto aux_old = auxiliary();
    for (std::size_t g = 6; g < optimizer.param_groups().size(); ++g)
      for (const auto& p : optimizer.param_groups()[g].params()) p.mutable_grad() = torch::ones_like(p);
    selector_enabled_ = false; tick(); same(aux_old,auxiliary()); selector_enabled_ = true;
    for (const auto dims : {std::make_pair(480,640),std::make_pair(479,641),std::make_pair(37,53)}) {
      auto c = camera(dims.first,dims.second);
      auto gt = c->original_image_;
      const auto lod = accumulateLodGradients(c,dims.first,dims.second,gt,0,true,true,nullptr,true);
      check(lod.render_height == dims.first/2 && lod.render_width == dims.second/2,"Half-resolution floor dimensions wrong");
      optimizer.zero_grad(true);
    }
    std::cout << "PASS two stages: actual renderer + photo-only Selector/TF heads, undefined Master grads, bitwise Master/Adam freeze, stale-grad protection, sizes\n";
  }
  void clocks() {
    opt_params_.densify_from_iter_ = 100000; opt_params_.opacity_reset_interval_ = 0;
    transform_field_config_.start_iter = mapping_iter_+2;
    const auto count = normal_lod_branch_count_;
    check(!shared_selector_optimizer_configured_,"Fixture unexpectedly attached Selector before first LoD");
    selector_network_->parameters().front().mutable_grad() = torch::ones_like(selector_network_->parameters().front());
    tick();
    check(!selector_network_->parameters().front().grad().defined(),"Stale unattached Selector gradient survived Mapping");
    check(normal_lod_branch_count_ == count+1 && selector_step_ == 1 && !transform_field_ &&
          normal_update_clock_.tf_updates == 0,"TF start blocked early Selector-only LoD");
    tick(); check(normal_lod_branch_count_ == count+2 && !transform_field_,"TF initialized before its own start");
    for (int i = 0; i < 6; ++i) tick();
    check(normal_lod_branch_count_ == count+8,"Ratio attempt count wrong");
    for (auto n : mode2_online_.ratios) check(n == 2,"Ratio cycle is not two complete rounds");
    check(transform_field_ && normal_update_clock_.tf_updates > 0,"TF failed to join LoD after its own start");
    transform_field_config_.enabled = false;
    transform_field_config_.start_iter = 100000; tick();
    check(normal_lod_branch_count_ == count+9,"Disabled TF blocked Selector-only branch");
    selector_enabled_ = false; tick(); check(normal_lod_branch_count_ == count+9,"Disabled Selector consumed ratio");
    selector_enabled_ = true; opt_params_.iterations_ = iteration_+1; tick();
    check(normal_lod_branch_count_ == count+9,"Optimization cap consumed ratio");
    opt_params_.iterations_ = -1;
    const auto full = mode2_tail_.full_renders;
    ++iteration_; trainMappingIteration(camera_,true,true); ++mapping_iter_;
    check(mode2_tail_.full_renders == full+1 && mode2_tail_.lod_attempts == 1,"Tail partition lost normal Full/LoD");
    std::cout << "PASS clocks: immediate Selector LoD, independent TF start, ratios/disabled/cap/tail\n";
  }
  void maturity() {
    opt_params_.densify_from_iter_ = 100000; opt_params_.opacity_reset_interval_ = 0;
    transform_field_config_.start_iter = 100000;
    gaussians_->protection_.reset(gaussians_->xyz_.size(0),gaussians_->xyz_.options());
    tick();
    check(gaussians_->protection_.snapshot_valid.any().item<bool>() &&
          gaussians_->protection_.maturity.eq(0).all().item<bool>(),"First Full did not seed maturity snapshots");
    tick();
    check((gaussians_->protection_.maturity > 0).any().item<bool>() &&
          gaussians_->protection_.first_valid_maturity_cycle_iteration == static_cast<int64_t>(mapping_iter_),
          "Mode2 Full commits failed to grow fresh maturity");
    check(!gaussians_->protection_.touched_in_cycle.any().item<bool>(),"Full left pending maturity evidence");
    check(normal_lod_branch_count_ == 2 && !transform_field_,"Fresh maturity or TF start blocked Selector LoD");
    std::cout << "PASS maturity: fresh mode2 Full snapshots and committed Full-to-Full drift\n";
  }
  void specialAndNonfinite() {
    opt_params_.densify_from_iter_ = 100000; opt_params_.opacity_reset_interval_ = 0;
    gaussians_->setShDegree(1); default_sh_ = 1;
    initializeTransformFieldIfNeeded(); ensureSharedMappingOptimizer();
    const auto normal = normal_lod_branch_count_, s = selector_step_, t = normal_update_clock_.tf_updates;
    const auto before = master();
    check(trainJointReplay({1,1},camera_),"Mode2 broke joint Replay commit");
    check(!bitwise(before[1].value,master()[1].value),"Mode2 detached Replay Master update");
    check(normal_lod_branch_count_ == normal && selector_step_ == s && normal_update_clock_.tf_updates == t,"Replay consumed normal auxiliary clocks");
    auto capture = captureGeometryChange(ChangeFlag::GLOBAL_GEOMETRY);
    check(capture.snapshot_count == gaussians_->xyz_.size(0),"CD failed to capture Mode2 Master");
    {
      torch::NoGradGuard no_grad;
      auto correction = gaussians_->applyScaledTransformation(1.01f,Sophus::SE3f());
      capture.has_geometry_mutation = true;
      change_coordinator_->addCorrection(capture,correction);
    }
    const auto corrected = master();
    const auto cd_steps = cd_tf_steps_;
    finishGeometryChange(std::move(capture));
    check(cd_tf_steps_ > cd_steps,"Geometry CD did not train TF");
    evaluateDriftStability(true);
    same(corrected,master());
    check(normal_lod_branch_count_ == normal && selector_step_ == s && normal_update_clock_.tf_updates == t,
          "CD/DSD consumed normal auxiliary clocks");
    auto& optimizer = *gaussians_->optimizer_;
    const auto all = snapshot(optimizer,0,optimizer.param_groups().size());
    selector_network_->parameters().front().mutable_grad() = torch::full_like(selector_network_->parameters().front(),std::numeric_limits<float>::quiet_NaN());
    const auto commit = commitMode2Stage(false);
    check(!commit.finite && !commit.selector && !commit.tf,"Nonfinite auxiliary gradient was committed");
    same(all,snapshot(optimizer,0,optimizer.param_groups().size())); optimizer.zero_grad(true);
    // A zero photo loss cannot be manufactured by clearing final grads: inspect
    // the branch failure before the actual two-stage wrapper clears them.
    selector_render_loss_weight_ = std::numeric_limits<float>::quiet_NaN();
    const auto branches = normal_lod_branch_count_, nonfinite = mode2_online_.lod_nonfinite;
    tick();
    check(normal_lod_branch_count_ == branches+1 && mode2_online_.lod_nonfinite == nonfinite+1,
          "Nonfinite attempted branch did not consume exactly one ratio");
    check(selector_step_ == s && normal_update_clock_.tf_updates == t,"Nonfinite branch advanced auxiliary clocks");
    for (const auto& group : optimizer.param_groups()) for (const auto& p : group.params())
      check(!p.grad().defined(),"Failed branch leaked gradients");
    std::cout << "PASS special/nonfinite: Replay stays joint, actual geometry CD and DSD probe preserve Master/clocks, nonfinite attempt-only ratio\n";
  }
  void topologyLod() {
    opt_params_.opacity_reset_interval_ = 0; opt_params_.densify_grad_threshold_ = 0;
    tick(); tick();
    const auto n = gaussians_->xyz_.size(0);
    const auto cursor = normal_lod_branch_count_;
    const auto updates = mode2_online_.gaussian_updates;
    const auto old = gaussians_->xyz_.unsafeGetTensorImpl();
    tick(); // 1002: split all parents; step sees replacement tensors without grad.
    check(gaussians_->xyz_.size(0) == 2*n && gaussians_->xyz_.unsafeGetTensorImpl() != old,"Split topology not observed");
    check(normal_lod_branch_count_ == cursor+1 && mode2_online_.gaussian_updates == updates,"Topology changed ratio/Full update evidence incorrectly");
    check(!gaussians_->protection_.snapshot_valid.any().item<bool>() && !gaussians_->protection_.touched_in_cycle.any().item<bool>(), "New split children inherited runtime snapshots/touched");
    check(gaussians_->xyz_gradient_accum_.sum().item<float>() == 0 && gaussians_->denom_.sum().item<float>() == 0 &&
          gaussians_->max_radii2D_.sum().item<float>() == 0,"Post-topology LoD polluted Full statistics");
    check(gaussians_->protection_.maturity.size(0) == 2*n && gaussians_->last_elastic_supervised_iter_.size(0) == 2*n,
          "Topology metadata not aligned before LoD");
    opt_params_.densify_grad_threshold_ = 1e6f; densify_min_opacity_ = 0.005f;
    { torch::NoGradGuard no_grad; gaussians_->opacity_.select(0,0).fill_(-30); }
    tick(); tick(); tick(); // 1005: prune, then LoD reads the surviving population.
    check(gaussians_->xyz_.size(0) == 2*n-1 && gaussians_->protection_.maturity.size(0) == 2*n-1,
          "Prune survivor indices/metadata were stale at LoD");
    std::cout << "PASS topology LoD: latest split/prune population, new children evidence, post-topology stats isolation\n";
  }
  void benchmark(const std::string& label) {
    opt_params_.densify_from_iter_ = 100000; opt_params_.opacity_reset_interval_ = 0;
    default_sh_ = 1; gaussians_->setShDegree(1);
    camera_->image_height_ = 480; camera_->image_width_ = 640;
    camera_->original_image_ = camera(480,640)->original_image_;
    if (selector_enabled_) initializeTransformFieldIfNeeded();
    for (int i = 0; i < 15; ++i) {
      if (label == "original_full_only") originalFull();
      else { ++iteration_; trainMappingIteration(camera_,true,false); ++mapping_iter_; }
    }
    torch::cuda::synchronize(); // Benchmark batch boundary, not production/per iteration.
    std::vector<double> times;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; ++i) {
      const auto start = std::chrono::steady_clock::now();
      mapping_profiler_.begin(true); mapping_profiler_.setPhase(false);
      if (label == "original_full_only") originalFull();
      else { ++iteration_; trainMappingIteration(camera_,true,false); ++mapping_iter_; }
      mapping_profiler_.end(mapping_iter_);
      times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    torch::cuda::synchronize();
    const auto batch_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    std::sort(times.begin(),times.end());
    std::cout << "[Mode2 MICRO] label=" << label << " gaussians=" << gaussians_->xyz_.size(0)
              << " measured_cycles=50 warmup_cycles=15 complete_mean_ms=" << batch_ms/50
              << " host_p95_ms=" << times[47] << '\n';
    reportMode2Mapping("micro_warmup_included"); reportUltrafastRotation("micro_warmup_included");
    mapping_profiler_.report(mapping_iter_);
  }
 protected:
  std::shared_ptr<GaussianKeyframe> camera_;
};

Overrides settings(bool selector, const std::string& backend = "cuda") {
  return {{"Training.ultra_light_mode","2"},
          {"Selector.enabled",selector?"1":"0"},{"GlobalGI.mode","recent"},
          {"RecentWindow.enable","1"},
          {"Selector.lod_render_scale","0.5"},
          {"TransformField.backend",backend},{"TransformField.start_iter","0"},
          {"GradientConstraint.enable","0"},{"ChangeDistill.enable","0"},{"DriftStability.enable","0"}};
}
void configTests(const std::filesystem::path& root) {
  auto defaults = settings(false); defaults["Selector.lod_render_scale"] = ""; defaults["TransformField.start_iter"] = "";
  Mode2Mapper(root/"defaults",defaults,torch::kCPU).configIs(2,0.5,3000);
  auto modes = settings(true); modes["Training.ultra_light_mode"] = "";
  modes["Selector.lod_render_scale"] = "1";
  Mode2Mapper(root/"default1",modes,torch::kCPU).configIs(1,1,0);
  modes["Training.ultra_light_mode"] = "1";
  Mode2Mapper(root/"explicit1",modes,torch::kCPU).configIs(1,1,0);
  modes["Training.ultra_light_mode"] = "0";
  Mode2Mapper(root/"baseline0",modes,torch::kCPU).configIs(0,1,0);
  for (const auto kv : {std::make_pair("Training.ultra_light_mode","3"),std::make_pair("Training.ultra_light_mode","1.5"),
                       std::make_pair("Selector.target_ratios","[]"),std::make_pair("Selector.target_ratios","[0.01,0]"),
                       std::make_pair("GradientConstraint.enable","1")}) {
    auto bad = settings(false); bad[kv.first] = kv.second;
    bool rejected = false;
    try { Mode2Mapper m(root/"invalid",bad,torch::kCPU); } catch (const std::exception&) { rejected = true; }
    check(rejected,"Invalid mode/ratios/Guard combination silently accepted");
  }
  std::cout << "PASS config: default mode1, explicit modes, mode2 defaults, invalid values/Guard\n";
}
} // namespace

int main(int argc, char** argv) {
  const auto output = std::filesystem::temp_directory_path() /
      ("cartgs-mode2-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  const std::string section = argc > 1 ? argv[1] : "all";
  if (section == "config" || section == "all") configTests(output);
  if (section == "config") return 0;
  if (!torch::cuda::is_available()) { std::cout << "SKIP: CUDA unavailable\n"; return 77; }
  if (section == "parity" || section == "all") {
    Mode2Mapper candidate(output/"parity",settings(false)), reference(output/"reference",settings(false));
    candidate.setup(); reference.setup(); candidate.parity(reference);
    Mode2Mapper pyramid(output/"pyramid",settings(false)), pyramid_reference(output/"pyramid_reference",settings(false));
    pyramid.setup(); pyramid_reference.setup(); pyramid.enablePyramid(); pyramid_reference.enablePyramid();
    pyramid.parity(pyramid_reference);
  }
  if (section == "stages" || section == "all") for (const auto backend : {"legacy","cuda"}) {
    Mode2Mapper m(output/backend,settings(true,backend)); m.setup(); m.gradients();
  }
  if (section == "clocks" || section == "all") {
    Mode2Mapper m(output/"clocks",settings(true)); m.setup(); m.clocks();
    Mode2Mapper topology(output/"topology",settings(true)); topology.setup(); topology.topologyLod();
    Mode2Mapper fresh(output/"maturity",settings(true)); fresh.setup(); fresh.maturity();
  }
  if (section == "special" || section == "all") {
    auto opts = settings(true); opts["ChangeDistill.enable"] = "1";
    opts["ChangeDistill.epochs"] = "1"; opts["DriftStability.enable"] = "1";
    Mode2Mapper m(output/"special",opts); m.setup(); m.specialAndNonfinite();
  }
  if (section == "benchmark") for (const std::string label :
      {"original_full_only","mode1","mode2_full_only","mode2_half_lod","mode2_native_lod"}) {
    auto opts = settings(label != "original_full_only" && label != "mode2_full_only");
    opts["Training.ultra_light_mode"] = label == "original_full_only" ? "0" : label == "mode1" ? "1" : "2";
    opts["Selector.lod_render_scale"] = label == "mode2_native_lod" ? "1" : "0.5";
    Mode2Mapper m(output/label,opts); m.setup(2048); m.benchmark(label);
  }
}
