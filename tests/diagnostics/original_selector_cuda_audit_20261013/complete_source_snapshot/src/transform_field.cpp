#include "include/transform_field.h"

#include <torch/nn/init.h>
#include "include/quaternion_utils.h"
#include "include/model_snapshot.h"
#include "include/transform_field_cuda.h"

TransformFieldImpl::TransformFieldImpl(
    const torch::Tensor& aabb_min,
    const torch::Tensor& aabb_max,
    const TransformFieldConfig& config)
    : config_(config),
      grid_(register_module(
          "grid", HexPlaneField(aabb_min, aabb_max, config.hexplane))),
      feature_out_(register_module(
          "feature_out",
          torch::nn::Linear(grid_->outputDim(), config.hidden_dim))),
      position_hidden_(register_module(
          "position_hidden",
          torch::nn::Linear(config.hidden_dim, config.hidden_dim))),
      position_out_(register_module(
          "position_out", torch::nn::Linear(config.hidden_dim, 3))),
      scale_hidden_(register_module(
          "scale_hidden",
          torch::nn::Linear(config.hidden_dim, config.hidden_dim))),
      scale_out_(register_module(
          "scale_out", torch::nn::Linear(config.hidden_dim, 3))),
      rotation_hidden_(register_module(
          "rotation_hidden",
          torch::nn::Linear(config.hidden_dim, config.hidden_dim))),
      rotation_out_(register_module(
          "rotation_out", torch::nn::Linear(config.hidden_dim, 4))) {
  TORCH_CHECK(config_.backend == "legacy" || config_.backend == "cuda",
              "TransformField.backend must be legacy or cuda");
  initializeLinear(feature_out_);
  initializeLinear(position_hidden_);
  initializeLinear(scale_hidden_);
  initializeLinear(rotation_hidden_);
  // Starting close to identity prevents a discontinuity when the online field
  // is enabled while keeping a non-zero path into the HexPlane parameters.
  initializeLinear(position_out_, 1e-4f);
  initializeLinear(scale_out_, 1e-4f);
  initializeLinear(rotation_out_, 1e-4f);
}

void TransformFieldImpl::initializeLinear(const torch::nn::Linear& layer,
                                          float weight_scale) {
  torch::NoGradGuard no_grad;
  torch::nn::init::xavier_uniform_(layer->weight);
  layer->weight.mul_(weight_scale);
  if (layer->bias.defined()) layer->bias.zero_();
}

TransformResidual TransformFieldImpl::forward(
    const torch::Tensor& xyz,
    float ratio) {
  auto ratio_tensor = torch::full(
      {xyz.size(0), 1}, ratio, xyz.options());
  if (config_.backend == "cuda") {
    auto feature = tf_cuda::query(xyz, ratio_tensor, aabbMin(), aabbMax(), gridParameters());
    // Same-machine complete training measurements favor Warp32 through 10k;
    // cuBLAS wins at 20k/50k and serves all other architectures.
    const auto mlp_backend = feature.size(1) == 16 && config_.hidden_dim == 32 && xyz.size(0) <= 10000
        ? tf_cuda::MlpBackend::Warp32 : tf_cuda::MlpBackend::Cublas;
    auto residual = tf_cuda::neural(feature, mlpParameters(), mlp_backend);
    return {residual[0], residual[1], residual[2]};
  }
  auto feature = grid_->forward(xyz, ratio_tensor);
  auto hidden = feature_out_->forward(feature);
  auto delta_xyz = position_out_->forward(
      torch::relu(position_hidden_->forward(torch::relu(hidden))));
  auto delta_log_scale = scale_out_->forward(
      torch::relu(scale_hidden_->forward(torch::relu(hidden))));
  auto delta_rotation_raw = rotation_out_->forward(
      torch::relu(rotation_hidden_->forward(torch::relu(hidden))));
  return {delta_xyz, delta_log_scale, delta_rotation_raw};
}

ElasticGaussianAttributes TransformFieldImpl::applyResidual(
    const torch::Tensor& xyz,
    const torch::Tensor& world_scaling,
    const torch::Tensor& world_rotation,
    float ratio) {
  auto residual = forward(xyz, ratio);
  if (config_.backend == "cuda") {
    auto output = tf_cuda::output(xyz, world_scaling, world_rotation,
        {residual.delta_xyz, residual.delta_log_scale, residual.delta_rotation_raw});
    return {output[0], output[1], output[2]};
  }
  auto transformed_xyz = xyz + residual.delta_xyz;
  auto transformed_scaling =
      world_scaling * torch::exp(residual.delta_log_scale);
  auto identity = torch::zeros_like(residual.delta_rotation_raw);
  identity.index_put_({torch::indexing::Slice(), 0}, 1.0f);
  auto transformed_rotation = quaternion_utils::normalize(
      quaternion_utils::multiply(
          quaternion_utils::normalize(identity + residual.delta_rotation_raw),
          world_rotation));
  return {transformed_xyz, transformed_scaling, transformed_rotation};
}

torch::Tensor TransformFieldImpl::regularizationLoss() const {
  if (config_.backend == "cuda")
    return tf_cuda::regularization(gridParameters(), config_.spatial_smooth_weight,
        config_.ratio_smooth_weight, config_.ratio_l1_weight);
  return config_.spatial_smooth_weight * grid_->spatialSmoothness() +
         config_.ratio_smooth_weight * grid_->ratioSmoothness() +
         config_.ratio_l1_weight * grid_->ratioL1();
}

std::shared_ptr<TransformFieldImpl> TransformFieldImpl::cloneFrozen() const {
  ScopedTorchRNG rng(torch::Device(torch::kCPU));
  auto teacher = std::make_shared<TransformFieldImpl>(
      aabbMin().cpu(), aabbMax().cpu(), config_);
  teacher->to(aabbMin().device());
  copyFrozenModule(*this, *teacher);
  return teacher;
}

GaussianAttrState TransformFieldImpl::applyLogResidual(
    const GaussianAttrState& state, float ratio) {
  auto indices = config_.outside_identity
                     ? torch::nonzero(insideMask(state.xyz)).reshape({-1})
                     : torch::arange(state.xyz.size(0), state.xyz.options().dtype(torch::kLong));
  if (indices.numel() == 0) return state;
  auto selected = state.select(indices);
  auto residual = forward(selected.xyz, ratio);
  if (config_.backend == "cuda") {
    auto output = tf_cuda::output(selected.xyz, selected.log_scaling, selected.rotation,
        {residual.delta_xyz, residual.delta_log_scale, residual.delta_rotation_raw}, true);
    return {state.xyz.index_copy(0, indices, output[0]),
            state.log_scaling.index_copy(0, indices, output[1]),
            state.rotation.index_copy(0, indices, output[2])};
  }
  auto identity = torch::zeros_like(residual.delta_rotation_raw);
  identity.select(1, 0).fill_(1.0);
  auto rotation = quaternion_utils::normalize(quaternion_utils::multiply(
      quaternion_utils::normalize(identity + residual.delta_rotation_raw),
      selected.rotation));
  return {state.xyz.index_copy(0, indices, selected.xyz + residual.delta_xyz),
          state.log_scaling.index_copy(0, indices, selected.log_scaling + residual.delta_log_scale),
          state.rotation.index_copy(0, indices, rotation)};
}

std::vector<torch::Tensor> TransformFieldImpl::mlpParameters() const {
  std::vector<torch::Tensor> result;
  const std::array<torch::nn::Linear, 7> layers = {
      feature_out_,      position_hidden_, position_out_, scale_hidden_,
      scale_out_,        rotation_hidden_, rotation_out_};
  for (const auto& layer : layers) {
    const auto parameters = layer->parameters();
    result.insert(result.end(), parameters.begin(), parameters.end());
  }
  return result;
}
