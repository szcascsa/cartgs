#include "include/transform_field.h"

#include <torch/nn/init.h>
#include "include/quaternion_utils.h"

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
  return config_.spatial_smooth_weight * grid_->spatialSmoothness() +
         config_.ratio_smooth_weight * grid_->ratioSmoothness() +
         config_.ratio_l1_weight * grid_->ratioL1();
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
