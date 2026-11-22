#include "include/gaussian_renderer.h"

#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void rendererModes(int count, int height, int width, int mode) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto xyz = torch::zeros({count, 3}, opts);
  if (count > 0) {
    xyz.select(1, 2).fill_(2.0f);
    if (count > 2) {
      xyz.index_put_({1, 0}, 50.0f); // off screen
      xyz.index_put_({2, 2}, -2.0f); // behind camera
    }
    if (count > 3) xyz.index_put_({3, 0}, std::numeric_limits<float>::quiet_NaN());
  }
  GaussianRenderInput input{xyz, torch::full({count, 1}, 0.6f, opts),
      torch::full({count, 3}, 0.07f, opts),
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts).repeat({count, 1}),
      torch::full({count, 1, 3}, 0.2f, opts), torch::full({count, 3, 3}, 0.03f, opts), 1, 1};
  auto camera = std::make_shared<GaussianKeyframe>(0);
  camera->FoVx_ = camera->FoVy_ = 1.0f;
  camera->world_view_transform_ = torch::eye(4, opts);
  camera->full_proj_transform_ = camera->getProjectionMatrix(
      0.01f, 100.0f, 1.0f, 1.0f, torch::kCUDA).transpose(0, 1).contiguous();
  camera->camera_center_ = torch::zeros({3}, opts);
  GaussianPipelineParams pipe;
  pipe.compute_cov3D_ = mode == 1 || mode == 2;
  pipe.convert_SHs_ = mode == 2;
  pipe.separate_sh_ = mode == 3;
  auto background = torch::tensor({0.1f, 0.2f, 0.3f}, opts);
  torch::Tensor color;
  torch::NoGradGuard no_grad;
  auto legacy = GaussianRenderer::render(camera, height, width, input, pipe, background, color);
  auto inference = GaussianRenderer::render(camera, height, width, input, pipe, background,
      color, 1.0f, false, false, RasterizeMode::Inference);
  auto visibility = GaussianRenderer::render(camera, height, width, input, pipe, background,
      color, 1.0f, false, false, RasterizeMode::Visibility);
  check(torch::allclose(legacy.image, inference.image, 1e-5, 1e-6),
        "Inference changed original pixel accumulation");
  check(legacy.radii.equal(inference.radii) && legacy.radii.equal(visibility.radii) &&
        legacy.visibility_indices.equal(visibility.visibility_indices),
        "Read-only mode changed original visibility/radii");
  check(visibility.image.numel() == 0 && !inference.viewspace_points.defined() &&
        !inference.visibility_indices.defined() && !inference.image.requires_grad(),
        "Read-only renderer allocated unused outputs or retained an autograd graph");
  auto training_gi = GaussianRenderer::render(camera, height, width, input, pipe,
      background, color, 1.0f, false, true, RasterizeMode::Training);
  auto inference_gi = GaussianRenderer::render(camera, height, width, input, pipe,
      background, color, 1.0f, false, true, RasterizeMode::Inference);
  check(torch::allclose(training_gi.frame_importance, inference_gi.frame_importance, 1e-4, 1e-5) &&
        training_gi.contribution_count.equal(inference_gi.contribution_count),
        "Read-only GI changed the existing raw contribution kernel");
  check(!inference_gi.viewspace_points.defined() && !inference_gi.image.requires_grad(),
        "GI inference retained training state");
}
void auxiliaryBuffers() {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  const auto empty = torch::empty({0}, opts);
  const auto xyz = torch::tensor({{0.0f, 0.0f, 2.0f}}, opts);
  const auto identity = torch::eye(4, opts);
  const auto color = torch::ones({1, 3}, opts);
  const auto opacity = torch::full({1, 1}, 0.6f, opts);
  const auto scale = torch::full({1, 3}, 0.07f, opts);
  const auto rotation = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts);
  const auto center = torch::zeros({3}, opts);
  for (const auto mode : {RasterizeMode::Inference, RasterizeMode::Visibility}) {
    const auto result = RasterizeGaussiansCUDAWithImportance(center, xyz, color,
        opacity, scale, rotation, 1.0f, empty, identity, identity, 0.5f, 0.5f,
        32, 48, empty, empty, 0, center, false, false, false, mode);
    check(std::get<1>(result) == 0 && std::get<9>(result).numel() == 0,
          "Read-only rasterization created backward bucket/sample state");
    if (mode == RasterizeMode::Visibility)
      check(std::get<7>(result).numel() == 0 && std::get<8>(result).numel() == 0,
            "Visibility-only rasterization allocated binning/image state");
    else
      check(std::get<8>(result).numel() <= 6 * 8,
            "Inference image state must contain tile ranges only");
  }
}
}
int main() {
  if (!torch::cuda::is_available()) return 77;
  for (const int count : {0, 1, 5})
    for (const int mode : {0, 1, 2, 3})
      for (const int resolution : {32, 64}) rendererModes(count, resolution, resolution + 7, mode);
  auxiliaryBuffers();
}
