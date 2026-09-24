#include "include/gaussian_model.h"
#include "include/online_gi.h"

#include <torch/torch.h>

#include <cmath>
#include <memory>
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
  model.selector_birth_iter_ = torch::tensor({1}, kIntOptions);
  model.selector_seen_count_ = torch::tensor({100}, kIntOptions);

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
  frame.observed_indices = torch::tensor({0LL}, kLongOptions);
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
    model->densifyAndClone(grads, 0.5f, 1.0f, 42);
  }

  expectEqual(model->selector_birth_iter_, torch::tensor({1, 42}, kIntOptions),
              "Clone child must use its current Selector birth iteration");
  expectEqual(model->selector_seen_count_, torch::tensor({100, 0}, kIntOptions),
              "Clone child must restart its Selector seen count");
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
    model->densifyAndSplit(grads, 0.5f, 1.0f, 2, 43);
  }

  expectEqual(model->selector_birth_iter_, torch::tensor({43, 43}, kIntOptions),
              "Split children must use their current Selector birth iteration");
  expectEqual(model->selector_seen_count_, torch::tensor({0, 0}, kIntOptions),
              "Split children must restart their Selector seen count");
  expectEqual(model->exist_since_iter_, torch::tensor({5, 5}, kIntOptions),
              "Split children must preserve CaRtGS existence metadata");
  expectClose(manager.getCombinedGI(), torch::zeros({2}, kFloatOptions),
              "Split must delete parent GI and keep fresh child GI state");
  if (manager.topologyVersion() != 2)
    throw std::runtime_error("Split must execute one GI append and one prune");
}

}  // namespace

int main() {
  if (!torch::cuda::is_available())
    throw std::runtime_error(
        "Gaussian lifecycle test requires a CUDA device");
  testCloneChildIsFresh();
  testSplitChildrenAreFreshAndParentIsRemoved();
  return 0;
}
