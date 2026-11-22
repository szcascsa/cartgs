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
  TORCH_CHECK(config_.start_mode == "fixed_iteration" || config_.start_mode == "first_maturity_cycle" ||
      config_.start_mode == "maturity_population",
              "Unknown TransformField.start_mode");
  TORCH_CHECK(config_.elastic_entry_maturity >= 0 && config_.elastic_entry_maturity < 1 &&
      std::isfinite(config_.elastic_entry_maturity) && std::isfinite(config_.ready_maturity) &&
      config_.ready_maturity >= 0 && config_.ready_maturity <= 1 && config_.min_ready_gaussians > 0,
      "Invalid Elastic entry or TF readiness configuration");
  TORCH_CHECK(std::isfinite(config_.maturity_compute_epsilon) &&
      config_.maturity_compute_epsilon >= 0 && config_.maturity_compute_epsilon <= 1,
      "TransformField.maturity_compute_epsilon must be in [0,1]");
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
    float ratio, const torch::Tensor& maturity) {
  auto residual = forward(xyz, ratio);
  if (maturity.defined()) {
    TORCH_CHECK(maturity.dim() == 1 && maturity.size(0) == xyz.size(0), "TF maturity size mismatch");
    const auto weight = maturity.detach().unsqueeze(1);
    residual.delta_xyz = weight * residual.delta_xyz;
    residual.delta_log_scale = weight * residual.delta_log_scale;
    auto identity = torch::zeros_like(residual.delta_rotation_raw);
    identity.select(1, 0).fill_(1);
    auto correction = quaternion_utils::normalize(identity + residual.delta_rotation_raw);
    // q and -q describe the same rotation. Use the identity hemisphere for nlerp.
    correction = torch::where(correction.slice(1, 0, 1) < 0, -correction, correction);
    const auto interpolated = quaternion_utils::normalize((1 - weight) * identity + weight * correction);
    residual.delta_rotation_raw = torch::where(weight == 1, residual.delta_rotation_raw, interpolated - identity);
  }
  ElasticGaussianAttributes transformed;
  if (config_.backend == "cuda") {
    auto output = tf_cuda::output(xyz, world_scaling, world_rotation,
        {residual.delta_xyz, residual.delta_log_scale, residual.delta_rotation_raw});
    transformed = {output[0], output[1], output[2]};
  } else {
    auto transformed_xyz = xyz + residual.delta_xyz;
    auto transformed_scaling = world_scaling * torch::exp(residual.delta_log_scale);
    auto identity = torch::zeros_like(residual.delta_rotation_raw);
    identity.index_put_({torch::indexing::Slice(), 0}, 1.0f);
    auto transformed_rotation = quaternion_utils::normalize(
        quaternion_utils::multiply(
            quaternion_utils::normalize(identity + residual.delta_rotation_raw), world_rotation));
    transformed = {transformed_xyz, transformed_scaling, transformed_rotation};
  }
  if (maturity.defined()) {
    const auto zero = maturity.detach().unsqueeze(1) == 0;
    transformed.xyz = torch::where(zero, xyz, transformed.xyz);
    transformed.scaling = torch::where(zero, world_scaling, transformed.scaling);
    transformed.rotation = torch::where(zero, world_rotation, transformed.rotation);
  }
  return transformed;
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

std::vector<GaussianAttrState> TransformFieldImpl::applyLogResidualMany(
    const GaussianAttrState& state, const std::vector<float>& ratios) {
  std::vector<GaussianAttrState> result;
  if (ratios.empty()) return result;
  if (config_.backend != "cuda") {
    for (auto ratio : ratios) result.push_back(applyLogResidual(state, ratio));
    return result;
  }
  auto indices = config_.outside_identity
      ? torch::nonzero(insideMask(state.xyz)).reshape({-1})
      : torch::arange(state.xyz.size(0), state.xyz.options().dtype(torch::kLong));
  if (indices.numel() == 0) return std::vector<GaussianAttrState>(ratios.size(), state);
  auto selected = state.select(indices);
  auto residuals = forwardMany(selected.xyz, ratios);
  for (std::size_t r = 0; r < ratios.size(); ++r) {
    const auto& residual = residuals[r];
    auto output = tf_cuda::output(selected.xyz, selected.log_scaling, selected.rotation,
        {residual.delta_xyz, residual.delta_log_scale, residual.delta_rotation_raw}, true);
    result.push_back({state.xyz.index_copy(0, indices, output[0]),
                     state.log_scaling.index_copy(0, indices, output[1]),
                     state.rotation.index_copy(0, indices, output[2])});
  }
  return result;
}

std::vector<TransformResidual> TransformFieldImpl::forwardMany(
    const torch::Tensor& xyz, const std::vector<float>& ratios) {
  std::vector<TransformResidual> result;
  if (config_.backend != "cuda") {
    for (auto ratio : ratios) result.push_back(forward(xyz, ratio));
    return result;
  }
  std::vector<torch::Tensor> features;
  for (auto ratio : ratios) {
    auto ratio_tensor = torch::full({xyz.size(0),1}, ratio, xyz.options());
    features.push_back(tf_cuda::query(xyz, ratio_tensor, aabbMin(), aabbMax(), gridParameters()));
  }
  if (ratios.empty()) return result;
  for (std::size_t r = 0; r < ratios.size(); ++r) {
      const auto& feature = features[r];
      (void)feature;
  }
  const auto n = xyz.size(0);
  const auto mlp_backend = features[0].size(1) == 16 && config_.hidden_dim == 32 && n * ratios.size() <= 10000
      ? tf_cuda::MlpBackend::Warp32 : tf_cuda::MlpBackend::Cublas;
  auto residual = tf_cuda::neural(torch::cat(features, 0), mlpParameters(), mlp_backend);
  for (std::size_t r = 0; r < ratios.size(); ++r)
    result.push_back({residual[0].narrow(0,r*n,n),residual[1].narrow(0,r*n,n),residual[2].narrow(0,r*n,n)});
  return result;
}
