#include "include/gaussian_model.h"
#include "include/online_gi.h"
#include "include/global_gi.h"

#include <torch/torch.h>

#include <cmath>
#include <memory>
#include <chrono>
#include <stdexcept>

namespace {

const auto kFloatOptions =
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
const auto kIntOptions =
    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
const auto kLongOptions =
    torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

void expectEqual(const torch::Tensor& actual,
                 const torch::Tensor& expected,
                 const char* message) {
  if (!actual.equal(expected)) throw std::runtime_error(message);
}

void expectClose(const torch::Tensor& actual,
                 const torch::Tensor& expected,
                 const char* message) {
  if (!torch::allclose(actual, expected)) throw std::runtime_error(message);
}

void initializeSingleGaussian(GaussianModel& model, float activated_scale) {
  model.xyz_ = torch::zeros({1, 3}, kFloatOptions).requires_grad_();
  model.features_dc_ = torch::zeros({1, 1, 3}, kFloatOptions).requires_grad_();
  model.features_rest_ =
      torch::zeros({1, 3, 3}, kFloatOptions).requires_grad_();
  model.opacity_ = torch::zeros({1, 1}, kFloatOptions).requires_grad_();
  model.scaling_ =
      torch::full({1, 3}, static_cast<float>(std::log(activated_scale)),
                  kFloatOptions)
          .requires_grad_();
  model.rotation_ =
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, kFloatOptions)
          .requires_grad_();
  model.max_radii2D_ = torch::zeros({1}, kFloatOptions);
  model.xyz_gradient_accum_ = torch::zeros({1, 1}, kFloatOptions);
  model.denom_ = torch::ones({1, 1}, kFloatOptions);
  model.exist_since_iter_ = torch::tensor({5}, kIntOptions);
  model.protection_.reset(1, kFloatOptions);
  model.protection_.maturity.fill_(.9f);
  model.protection_.drift_ema.fill_(.01f);
  model.protection_.drift_initialized.fill_(true);

  model.Tensor_vec_xyz_ = {model.xyz_};
  model.Tensor_vec_feature_dc_ = {model.features_dc_};
  model.Tensor_vec_feature_rest_ = {model.features_rest_};
  model.Tensor_vec_opacity_ = {model.opacity_};
  model.Tensor_vec_scaling_ = {model.scaling_};
  model.Tensor_vec_rotation_ = {model.rotation_};

  GaussianOptimizationParams optimization;
  optimization.percent_dense_ = 0.01f;
  model.trainingSetup(optimization);
}

void initializeGI(OnlineGIManager& manager, GaussianModel& model) {
  manager.resetForGaussianCount(1);
  FrameGIScore frame;
  frame.score = torch::tensor({99.0f}, kFloatOptions);
  frame.observed_mask =
      torch::ones({1}, torch::TensorOptions().dtype(torch::kBool));
  frame.observed_indices = torch::zeros({1}, kLongOptions);
  frame.num_observed = 1;
  manager.updateFast(frame, 10);
  manager.updateSlow(frame);

  model.setOnlineGIStateCallbacks(
      [&manager, &model](std::int64_t count) {
        manager.appendGaussians(count);
        manager.assertAligned(model.getXYZ());
      },
      [&manager, &model](const torch::Tensor& survivor_mask) {
        manager.pruneGaussians(survivor_mask);
        manager.assertAligned(model.getXYZ());
      },
      [&manager, &model](std::int64_t count) {
        manager.resetForGaussianCount(count);
        manager.assertAligned(model.getXYZ());
      });
}

std::unique_ptr<GaussianModel> makeCUDAModel() {
  GaussianModelParams parameters({}, {}, {}, 1, "images", -1.0f, false,
                                 "cuda", false);
  return std::make_unique<GaussianModel>(parameters);
}

void testCloneChildIsFresh() {
  auto model = makeCUDAModel();
  initializeSingleGaussian(*model, 0.001f);
  OnlineGIManager manager(torch::kCUDA, OnlineGIConfig{});
  initializeGI(manager, *model);

  auto grads = torch::ones({1, 1}, kFloatOptions);
  {
    torch::NoGradGuard no_grad;
    model->protection_.mapping_iteration = 42;
    model->densifyAndClone(grads, 0.5f, 1.0f);
  }

  expectClose(model->protection_.maturity, torch::full_like(model->protection_.maturity, .9f),
              "Clone/split children must inherit parent maturity");
  expectClose(model->protection_.drift_ema, torch::full_like(model->protection_.drift_ema, .01f),
              "Clone/split children must inherit parent EMA");
  if (model->protection_.snapshot_valid.any().item<bool>() || model->protection_.touched_in_cycle.any().item<bool>())
    throw std::runtime_error("Children inherited Full snapshots/touched state");
  expectEqual(model->exist_since_iter_, torch::tensor({5, 5}, kIntOptions),
              "Clone child must preserve CaRtGS existence metadata");
  expectClose(manager.getCombinedGI(),
              torch::tensor({99.0f, 0.0f}, kFloatOptions),
              "Clone child must start with fresh GI state");
}

void testSplitChildrenAreFreshAndParentIsRemoved() {
  auto model = makeCUDAModel();
  initializeSingleGaussian(*model, 1.0f);
  OnlineGIManager manager(torch::kCUDA, OnlineGIConfig{});
  initializeGI(manager, *model);

  auto grads = torch::ones({1, 1}, kFloatOptions);
  torch::manual_seed(0);
  {
    torch::NoGradGuard no_grad;
    model->protection_.mapping_iteration = 43;
    model->densifyAndSplit(grads, 0.5f, 1.0f, 2);
  }

  expectClose(model->protection_.maturity, torch::full_like(model->protection_.maturity, .9f),
              "Clone/split children must inherit parent maturity");
  expectClose(model->protection_.drift_ema, torch::full_like(model->protection_.drift_ema, .01f),
              "Clone/split children must inherit parent EMA");
  if (model->protection_.snapshot_valid.any().item<bool>() || model->protection_.touched_in_cycle.any().item<bool>())
    throw std::runtime_error("Children inherited Full snapshots/touched state");
  expectEqual(model->exist_since_iter_, torch::tensor({5, 5}, kIntOptions),
              "Split children must preserve CaRtGS existence metadata");
  expectClose(manager.getCombinedGI(), torch::zeros({2}, kFloatOptions),
              "Split must delete parent GI and keep fresh child GI state");
  if (manager.topologyVersion() != 2)
    throw std::runtime_error("Split must execute one GI append and one prune");
}

void testGlobalGITopologyCallbacks() {
  for (bool split : {false, true}) {
    auto model = makeCUDAModel();
    initializeSingleGaussian(*model, split ? 1.0f : 0.001f);
    GlobalGIConfig config;
    config.mode = "streaming"; config.view_ratio = 1; config.performance_logging = false;
    GlobalGIManager gi(torch::kCUDA, config);
    gi.resetForGaussianCount(1);
    gi.updateBank({{1, {1}}}, [](uint64_t) { return torch::tensor({99.f}, kFloatOptions); });
    int append_events = 0, prune_events = 0;
    model->setOnlineGIStateCallbacks({},
        [&](const torch::Tensor& keep) { gi.pruneGaussians(keep); ++prune_events; },
        [&](int64_t count) { gi.resetForGaussianCount(count); },
        [&](int64_t count, GaussianModel::AppendKind kind) {
          if (kind != (split ? GaussianModel::AppendKind::Split : GaussianModel::AppendKind::Clone))
            throw std::runtime_error("GaussianModel reported wrong GI topology origin");
          gi.appendGaussians(count, split ? GITopologyKind::Split : GITopologyKind::Clone);
          ++append_events;
        });
    auto grads = torch::ones({1, 1}, kFloatOptions);
    torch::NoGradGuard guard;
    if (split) model->densifyAndSplit(grads, 0.5f, 1.0f, 2);
    else model->densifyAndClone(grads, 0.5f, 1.0f);
    gi.assertAligned(model->getXYZ());
    if (append_events != 1 || prune_events != int(split))
      throw std::runtime_error("GaussianModel did not eagerly notify every GI topology operation");
    expectEqual(gi.validMask(), split ? torch::zeros({2}, kFloatOptions.dtype(torch::kBool))
        : torch::tensor({true, false}, kFloatOptions.dtype(torch::kBool)),
        "Clone/split children inherited parent GI validity");
    expectClose(gi.importance(), split ? torch::zeros({2}, kFloatOptions)
        : torch::tensor({99.f, 0.f}, kFloatOptions), "Clone/split copied parent GI contribution");
  }
}

void testTransformResultMatchesMutation() {
  torch::NoGradGuard no_grad;
  // Translation, scale, rotation, and a translation below the attribute epsilon.
  for (int scenario = 0; scenario < 4; ++scenario) {
    auto model = makeCUDAModel();
    initializeSingleGaussian(*model, 0.1f);
    model->xyz_.copy_(torch::tensor({{1.0f, 0.0f, 0.0f}}, kFloatOptions));
    model->rotation_.mul_(2.0f); // Raw rotations must not be rewritten by translation.
    for (const auto& group : model->optimizer_->param_groups()) {
      const auto& p = group.params().front();
      auto state = std::make_unique<torch::optim::AdamParamState>();
      state->step(3);
      state->exp_avg(torch::ones_like(p));
      state->exp_avg_sq(torch::ones_like(p));
      model->optimizer_->state()[p.unsafeGetTensorImpl()] = std::move(state);
    }
    auto old_scaling = model->scaling_.clone();
    auto old_rotation_raw = model->rotation_.clone();
    auto scale_key = model->scaling_.unsafeGetTensorImpl();
    auto rotation_key = model->rotation_.unsafeGetTensorImpl();
    GaussianAttrState before{model->getXYZ().clone(), model->scaling_.clone(),
                             model->getRotationActivation().clone()};
    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    if (scenario == 0) translation.x() = 0.5f;
    if (scenario == 3) translation.x() = 1e-7f;
    if (scenario == 2) rotation << 0.0f, -1.0f, 0.0f,
                                  1.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f;
    const auto result = model->applyScaledTransformation(
        scenario == 1 ? 2.0f : 1.0f, Sophus::SE3f(rotation, translation));
    const auto expected_flags = toBits(GaussianAttrFlag::XYZ) |
        (scenario == 1 ? toBits(GaussianAttrFlag::SCALE) : 0u) |
        (scenario == 2 ? toBits(GaussianAttrFlag::ROTATION) : 0u);
    if (!result.affects_all || result.population_count != 1 ||
        result.indices.defined() || result.changed_attributes != expected_flags)
      throw std::runtime_error("Global transform result does not describe the actual mutation");
    const auto target = applyGaussianTransform(before, result);
    expectClose(model->getXYZ(), target.xyz, "XYZ differs from teacher transport");
    expectClose(model->scaling_, target.log_scaling, "Scale differs from teacher transport");
    expectClose(model->getRotationActivation(), target.rotation, "Rotation differs from teacher transport");
    if (scenario == 0 || scenario == 3)
      expectEqual(model->getXYZ(), before.xyz +
                  torch::tensor({{translation.x(), 0.0f, 0.0f}}, kFloatOptions),
                  "Exact translation was dropped");
    if (scenario == 1)
      expectClose(model->getXYZ(), torch::tensor({{2.0f, 0.0f, 0.0f}}, kFloatOptions),
                  "Global scale changed XYZ incorrectly");
    if (scenario == 2 && !torch::allclose(model->getXYZ(),
        torch::tensor({{0.0f, 1.0f, 0.0f}}, kFloatOptions), 1e-5, 1e-6))
      throw std::runtime_error("Global rotation uses the wrong matrix convention");
    auto unchanged = [&](const torch::Tensor& parameter, void* old_key,
                         const torch::Tensor& old_value) {
      if (parameter.unsafeGetTensorImpl() != old_key)
        throw std::runtime_error("Unchanged attribute replaced its Parameter");
      expectEqual(parameter, old_value, "Unchanged attribute was modified");
      const auto& state = static_cast<const torch::optim::AdamParamState&>(
          *model->optimizer_->state().at(old_key));
      expectEqual(state.exp_avg(), torch::ones_like(parameter),
                  "Unchanged attribute lost its optimizer momentum");
    };
    if (scenario != 1) unchanged(model->scaling_, scale_key, old_scaling);
    if (scenario != 2) unchanged(model->rotation_, rotation_key, old_rotation_raw);
  }
}

void testProtectionCheckpointRoundTrip() {
  auto model = makeCUDAModel();
  initializeSingleGaussian(*model, .1f);
  model->protection_.mapping_iteration = (int64_t(1) << 35) + 500;
  const auto path = std::filesystem::temp_directory_path() /
      ("cartgs-protection-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ply");
  model->saveSelectorMetadataPly(path);
  auto restored = makeCUDAModel();
  initializeSingleGaussian(*restored, .1f);
  restored->loadSelectorMetadataPly(path);
  expectEqual(restored->protection_.maturity, model->protection_.maturity, "Checkpoint lost maturity");
  expectEqual(restored->protection_.drift_ema, model->protection_.drift_ema, "Checkpoint lost EMA");
  expectEqual(restored->protection_.cached_uniform, model->protection_.cached_uniform, "Checkpoint lost sampling randomness");
  if (restored->protection_.snapshot_valid.any().item<bool>() || restored->protection_.touched_in_cycle.any().item<bool>())
    throw std::runtime_error("Checkpoint restored runtime snapshots/touched");
  if (restored->protection_.mapping_iteration != model->protection_.mapping_iteration)
    throw std::runtime_error("Checkpoint restored the wrong Mapping clock");
  std::filesystem::remove(path);
}

}  // namespace

int main() {
  if (!torch::cuda::is_available())
    throw std::runtime_error(
        "Gaussian lifecycle test requires a CUDA device");
  testCloneChildIsFresh();
  testSplitChildrenAreFreshAndParentIsRemoved();
  testGlobalGITopologyCallbacks();
  testTransformResultMatchesMutation();
  testProtectionCheckpointRoundTrip();
  return 0;
}
