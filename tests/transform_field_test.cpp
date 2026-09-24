#include "include/transform_field.h"

#include <torch/torch.h>

#include <stdexcept>

int main() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  TransformFieldConfig config;
  config.hexplane.spatial_resolution = 4;
  config.hexplane.ratio_resolution = 4;
  config.hexplane.feature_dim = 2;
  config.hexplane.multires = {1};
  config.hidden_dim = 8;
  TransformField field(torch::full({3}, -2.0f, options),
                       torch::full({3}, 2.0f, options), config);
  auto xyz = torch::tensor({{0.0f, 0.0f, 0.0f}}, options);
  auto world_scale = torch::ones({1, 3}, options);
  auto identity_rotation = torch::tensor(
      {{1.0f, 0.0f, 0.0f, 0.0f}}, options);
  if (!field->insideMask(xyz).item<bool>() ||
      field->insideMask(torch::full({1, 3}, 3.0f, options)).item<bool>())
    throw std::runtime_error("Transform Field AABB mask is incorrect");

  auto residual = field->forward(xyz, 0.1f);
  auto transformed = field->applyResidual(
      xyz, world_scale, identity_rotation, 0.1f);
  if (transformed.xyz.sizes() != xyz.sizes() ||
      transformed.scaling.sizes() != world_scale.sizes() ||
      transformed.rotation.sizes() != identity_rotation.sizes())
    throw std::runtime_error("Transform Field output shape is incorrect");
  if (!torch::allclose(transformed.xyz, xyz + residual.delta_xyz))
    throw std::runtime_error(
        "Transform Field position residual must use current coordinates");
  auto loss = transformed.xyz.sum() + transformed.scaling.sum() +
              transformed.rotation.sum() + field->regularizationLoss();
  loss.backward();
  if (!field->gridParameters().front().grad().defined() ||
      !field->mlpParameters().front().grad().defined())
    throw std::runtime_error("Transform Field gradients are disconnected");
  return 0;
}
