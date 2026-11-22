#include "include/gaussian_mapper.h"

#include <chrono>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

// Exercises the real Mapper Replay entry point, not a second test-only optimizer.
class ReplayMapper : public GaussianMapper {
 public:
  explicit ReplayMapper(const std::filesystem::path& output, const std::string& backend = "legacy")
      : GaussianMapper(nullptr,
          std::filesystem::path(CARTGS_TEST_SOURCE_DIR) / "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
          output, 0, torch::kCUDA, true,
          std::filesystem::path(CARTGS_TEST_SOURCE_DIR) / "cfg/selector/fast.yaml") {
    transform_field_config_.backend = backend;
#ifdef CARTGS_TEST_SELECTOR_CUDA
    selector_backend_ = "cuda";
    selector_network_->setBackend("cuda");
#endif
    // This fixture's normal-Mapping assertions exercise joint mode0. Do not
    // inherit an experimenter's mode1 switch from the mutable fast config.
    ultra_light_mode_ = 0;
    ultrafast_rotation_.enabled = false;
  }

  void verify() {
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto& model = *gaussians_;
    model.xyz_ = torch::tensor({{-0.1f, 0.0f, 2.0f}, {0.1f, 0.0f, 2.0f}, {0.0f, 0.1f, 2.1f}}, opts).set_requires_grad(true);
    model.features_dc_ = torch::full({3, 1, 3}, 0.2f, opts).set_requires_grad(true);
    model.features_rest_ = torch::full({3, 15, 3}, 0.01f, opts).set_requires_grad(true);
    model.opacity_ = torch::zeros({3, 1}, opts).set_requires_grad(true);
    model.scaling_ = torch::full({3, 3}, -2.7f, opts).set_requires_grad(true);
    model.rotation_ = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts).repeat({3, 1}).set_requires_grad(true);
    model.active_sh_degree_ = 1;
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
    model.trainingSetup(opt_params_);
    recent_window_importance_->resetForGaussianCount(3);
    iteration_ = 6000; mapping_iter_ = 6000; selector_step_ = 12;
    selector_detach_gaussian_parameters_ = true; // Replay must override this normal-LOD option.
    selector_gi_loss_weight_ = 0.0f;
    transform_field_config_.hexplane.spatial_resolution = 4;
    transform_field_config_.hexplane.ratio_resolution = 4;
    transform_field_config_.hexplane.feature_dim = 2;
    transform_field_config_.hexplane.multires = {1};
    transform_field_config_.hidden_dim = 4;
    {
      torch::NoGradGuard no_grad;
      selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.0f, 4.0f}, opts));
    }
    initializeTransformFieldIfNeeded();
    ensureSharedMappingOptimizer();
    normal_update_clock_.tf_updates = 1000;

    auto camera = std::make_shared<GaussianKeyframe>(7);
    camera->setPose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    camera->image_height_ = 32; camera->image_width_ = 40;
    camera->FoVx_ = camera->FoVy_ = 1.0f;
    camera->world_view_transform_ = torch::eye(4, opts);
    camera->full_proj_transform_ = camera->getProjectionMatrix(0.01f, 100.0f, 1.0f, 1.0f, torch::kCUDA).transpose(0, 1).contiguous();
    camera->camera_center_ = torch::zeros({3}, opts);
    camera->original_image_ = torch::full({3, 32, 40}, 0.1f, opts);
    camera->gaus_pyramid_times_of_use_ = {8, 8};
    camera->remaining_times_of_use_ = 3;
    scene_->addKeyframe(camera, &kfid_shuffled_);
    kfs_used_times_[7] = 4; kfs_loss_[7] = 0.5f;

    // LOD alone must send gradients into master GS even when the normal option detaches.
    auto gt = camera->original_image_;
    model.optimizer_->zero_grad(true);
    const auto lod = accumulateLodGradients(camera, 32, 40, gt, 0, false, false);
    check(lod.trained && lod.finite_loss && model.xyz_.grad().defined() &&
              model.xyz_.grad().abs().sum().item<float>() > 0.0f,
          "Joint Replay LOD detached master Gaussian gradients");
    model.optimizer_->zero_grad(true);
    const auto maturity = model.protection_.maturity.clone(), radii = model.max_radii2D_.clone();
    const auto drift = model.protection_.drift_ema.clone();
    const auto accum = model.xyz_gradient_accum_.clone(), denom = model.denom_.clone();
    const auto gi_version = recent_window_importance_->version();
    const auto gi_topology = recent_window_importance_->topologyVersion();
    const auto tf_updates = normal_update_clock_.tf_updates;
    const auto rejected = normal_update_clock_.rejected_updates;
    const auto topology = topology_version_, correspondence = correspondence_version_;
    std::vector<torch::Tensor> before;
    for (const auto& group : model.optimizer_->param_groups())
      for (const auto& parameter : group.params()) before.push_back(parameter.detach().clone());
    ultra_light_mode_ = 1;
    check(trainJointReplay({7, 1}, camera), "Real Replay did not commit");
    check(model.protection_.touched_in_cycle.all().item<bool>(), "Committed mode1 Replay did not mark touched rows");
    model.protection_.touched_in_cycle.zero_();
    std::size_t index = 0;
    std::vector<bool> changed(model.optimizer_->param_groups().size(), false);
    for (std::size_t g = 0; g < model.optimizer_->param_groups().size(); ++g)
      for (const auto& parameter : model.optimizer_->param_groups()[g].params()) {
        changed[g] = changed[g] || !parameter.equal(before[index++]);
        check(!parameter.grad().defined(), "Replay left gradients for the next normal iteration");
        const auto state = model.optimizer_->state().find(parameter.unsafeGetTensorImpl());
        if (state != model.optimizer_->state().end())
          check(static_cast<const torch::optim::AdamParamState&>(*state->second).step() == 1,
                "Replay advanced shared Adam state more than once");
      }
    check((changed[0] || changed[1] || changed[3] || changed[4]) &&
              changed[shared_selector_group_index_] && changed[shared_transform_mlp_group_index_],
          "Replay failed to jointly update GS, Selector and TF");
    check(iteration_ == 6000 && mapping_iter_ == 6000 && selector_step_ == 12 &&
              normal_update_clock_.tf_updates == tf_updates && normal_update_clock_.rejected_updates == rejected &&
              topology_version_ == topology && correspondence_version_ == correspondence && gradient_guard_attempt_ == 0 &&
              gradient_would_cancel_count_ == 0 && replay_commit_count_ == 1 && replay_ratio_cursor_ == 1,
          "Replay polluted normal clocks, topology, or recursively invoked Guard");
    check(model.protection_.maturity.equal(maturity) &&
              model.protection_.drift_ema.equal(drift) && model.max_radii2D_.equal(radii) &&
              model.xyz_gradient_accum_.equal(accum) && model.denom_.equal(denom) &&
              recent_window_importance_->version() == gi_version &&
              recent_window_importance_->topologyVersion() == gi_topology &&
              camera->gaus_pyramid_times_of_use_ == std::vector<int>({8, 8}) &&
              camera->remaining_times_of_use_ == 3 && kfs_used_times_[7] == 4 && kfs_loss_[7] == 0.5f,
          "Replay executed mapping lifecycle, GI, optimization evidence or keyframe bookkeeping");

    before.clear();
    std::vector<torch::Tensor> moments;
    std::vector<int64_t> steps;
    for (const auto& group : model.optimizer_->param_groups()) for (const auto& parameter : group.params()) {
      before.push_back(parameter.detach().clone());
      const auto found = model.optimizer_->state().find(parameter.unsafeGetTensorImpl());
      if (found == model.optimizer_->state().end()) { moments.push_back(torch::Tensor()); steps.push_back(0); }
      else {
        const auto& state = static_cast<const torch::optim::AdamParamState&>(*found->second);
        moments.push_back(state.exp_avg().clone()); steps.push_back(state.step());
      }
    }
    camera->original_image_ = torch::full_like(camera->original_image_, std::numeric_limits<float>::quiet_NaN());
    check(!trainJointReplay({7, 2}, camera), "Nonfinite Replay loss was committed");
    check(!model.protection_.touched_in_cycle.any().item<bool>(), "Rejected Replay marked candidate rows touched");
    index = 0;
    for (const auto& group : model.optimizer_->param_groups()) for (const auto& parameter : group.params()) {
      check(parameter.equal(before[index]) && !parameter.grad().defined(), "Failed Replay changed parameters or leaked gradients");
      const auto found = model.optimizer_->state().find(parameter.unsafeGetTensorImpl());
      if (steps[index] == 0) check(found == model.optimizer_->state().end(), "Failed Replay created Adam state");
      else {
        const auto& state = static_cast<const torch::optim::AdamParamState&>(*found->second);
        check(state.step() == steps[index] && state.exp_avg().equal(moments[index]), "Failed Replay changed Adam state");
      }
      ++index;
    }
    // Mode0 keeps convergence state frozen, including exhausted-budget Full.
    ultra_light_mode_ = 0;
    camera->original_image_ = torch::full_like(camera->original_image_, .1f);
    do_gaus_pyramid_training_ = false;
    opt_params_.densify_until_iter_ = 0;
    opt_params_.iterations_ = 10000;
    const auto normal_before = model.protection_.maturity.clone();
    ++iteration_;
    trainMappingIteration(camera, false, false);
    ++mapping_iter_;
    check(model.protection_.maturity.equal(normal_before), "Mode0 must keep geometric maturity frozen");
    const auto no_budget_before = model.protection_.maturity.clone();
    opt_params_.iterations_ = 0;
    ++iteration_;
    trainMappingIteration(camera, false, false);
    ++mapping_iter_;
    check(model.protection_.maturity.equal(no_budget_before),
          "Full Render without an optimizer commit accumulated evidence");
    model.optimizer_->zero_grad(true);
    selector_network_ = improvements::selector::GumbelNetwork(nullptr);
    selector_enabled_ = false;
    opt_params_.iterations_ = 10000;
    ++iteration_;
    trainMappingIteration(camera, false, false);
    ++mapping_iter_;
    check(model.protection_.maturity.equal(no_budget_before),
          "Unsupported mode mutated continuous maturity");
  }
};
}

int main(int argc, char** argv) {
  if (!torch::cuda::is_available()) return 77;
  const auto output = std::filesystem::temp_directory_path() /
      ("cartgs-joint-replay-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  ReplayMapper(output, argc == 2 ? argv[1] : "legacy").verify();
}
