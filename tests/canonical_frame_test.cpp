#include "include/canonical_frame.h"

#include <torch/torch.h>

#include <cmath>
#include <stdexcept>

int main() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto scale = torch::ones({1, 1}, options);
  auto rotation = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, options);
  auto translation = torch::zeros({1, 3}, options);
  auto point = torch::tensor({{1.0f, 2.0f, 3.0f}}, options);

  auto world = canonical_frame::canonicalToWorld(
      point, scale, rotation, translation);
  auto canonical = canonical_frame::worldToCanonical(
      world, scale, rotation, translation);
  if (!torch::allclose(point, canonical))
    throw std::runtime_error("identity canonical frame round-trip failed");

  auto delta_scale = torch::full({1, 1}, 2.0f, options);
  auto delta_rotation = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, options);
  auto delta_translation = torch::tensor({{3.0f, 4.0f, 5.0f}}, options);
  auto [composed_scale, composed_rotation, composed_translation] =
      canonical_frame::compose(scale, rotation, translation, delta_scale,
                               delta_rotation, delta_translation);
  auto composed_world = canonical_frame::canonicalToWorld(
      point, composed_scale, composed_rotation, composed_translation);
  auto expected = 2.0f * point + delta_translation;
  if (!torch::allclose(composed_world, expected))
    throw std::runtime_error("Sim(3) composition failed");

  const float half_angle = static_cast<float>(std::acos(-1.0) / 4.0);
  auto quarter_turn = torch::tensor(
      {{std::cos(half_angle), 0.0f, 0.0f, std::sin(half_angle)}}, options);
  auto shifted = torch::tensor({{2.0f, -1.0f, 3.0f}}, options);
  auto [rotated_scale, rotated_rotation, rotated_translation] =
      canonical_frame::compose(composed_scale, composed_rotation,
                               composed_translation, delta_scale,
                               quarter_turn, shifted);
  auto rotated_world = canonical_frame::canonicalToWorld(
      point, rotated_scale, rotated_rotation, rotated_translation);
  auto expected_rotated_world = 2.0f * canonical_frame::rotate(
      quarter_turn, composed_world) + shifted;
  if (!torch::allclose(rotated_world, expected_rotated_world, 1e-5, 1e-5))
    throw std::runtime_error("rotating Sim(3) composition failed");
  auto recovered = canonical_frame::worldToCanonical(
      rotated_world, rotated_scale, rotated_rotation, rotated_translation);
  if (!torch::allclose(recovered, point, 1e-5, 1e-5))
    throw std::runtime_error("canonical position changed after Sim(3) update");
  return 0;
}
