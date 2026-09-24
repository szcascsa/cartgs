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
  auto canonical_xyz = torch::tensor({{0.0f, 0.0f, 0.0f}}, options);
  auto world_scale = torch::ones({1, 3}, options);
  auto identity_rotation = torch::tensor(
      {{1.0f, 0.0f, 0.0f, 0.0f}}, options);
  auto frame_scale = torch::ones({1, 1}, options);
  auto frame_translation = torch::zeros({1, 3}, options);
  if (!field->insideMask(canonical_xyz).item<bool>() ||
      field->insideMask(torch::full({1, 3}, 3.0f, options)).item<bool>())
    throw std::runtime_error("Transform Field AABB mask is incorrect");

  auto transformed = field->applyResidual(
      canonical_xyz, world_scale, identity_rotation, frame_scale,
      identity_rotation, frame_translation, 0.1f);
  if (transformed.xyz.sizes() != canonical_xyz.sizes() ||
      transformed.scaling.sizes() != world_scale.sizes() ||
      transformed.rotation.sizes() != identity_rotation.sizes())
    throw std::runtime_error("Transform Field output shape is incorrect");
  auto loss = transformed.xyz.sum() + transformed.scaling.sum() +
              transformed.rotation.sum() + field->regularizationLoss();
  loss.backward();
  if (!field->gridParameters().front().grad().defined() ||
      !field->mlpParameters().front().grad().defined())
    throw std::runtime_error("Transform Field gradients are disconnected");
  return 0;
}
