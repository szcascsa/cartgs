/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM and modified by Dapeng Feng in 2024, as part of CaRtGS.
 */

#include "include/gaussian_renderer.h"

#include "include/general_utils.h"

/**
 * @brief
 *
 * @return RenderPackage with the render result, CaRtGS visibility data, and
 * optional non-differentiable raw importance statistics.
 */
RenderPackage GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height, int image_width, std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe, torch::Tensor& bg_color,
    torch::Tensor& override_color,
    float scaling_modifier,
    bool use_override_color,
    torch::Tensor selector_mask,
    bool detach_gaussian_parameters,
    bool collect_importance) {
  GaussianRenderInput input{
      pc->getXYZ(),
      pc->getOpacityActivation(),
      pc->getScalingActivation(),
      pc->getRotationActivation(),
      pc->features_dc_,
      pc->features_rest_,
      pc->active_sh_degree_,
      pc->max_sh_degree_};
  if (detach_gaussian_parameters) {
    input.xyz = input.xyz.detach();
    input.opacity = input.opacity.detach();
    input.scaling = input.scaling.detach();
    input.rotation = input.rotation.detach();
    input.features_dc = input.features_dc.detach();
    input.features_rest = input.features_rest.detach();
  }
  if (selector_mask.defined()) {
    if (selector_mask.dim() != 1 ||
        selector_mask.size(0) != input.opacity.size(0)) {
      throw std::runtime_error(
          "Selector mask must have one value per Gaussian.");
    }
    input.opacity =
        (input.opacity * selector_mask.unsqueeze(1)).contiguous();
  }
  return render(viewpoint_camera, image_height, image_width, input, pipe,
                bg_color, override_color, scaling_modifier,
                use_override_color, collect_importance);
}

RenderPackage GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    const GaussianRenderInput& input,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    float scaling_modifier,
    bool use_override_color,
    bool collect_importance) {
  if (!input.xyz.defined() || input.xyz.dim() != 2 || input.xyz.size(1) != 3)
    throw std::runtime_error("Render xyz must have shape [N,3]");
  const auto count = input.xyz.size(0);
  if (!input.opacity.defined() || input.opacity.dim() != 2 ||
      input.opacity.size(0) != count || input.opacity.size(1) != 1 ||
      !input.scaling.defined() || input.scaling.dim() != 2 ||
      input.scaling.size(0) != count || input.scaling.size(1) != 3 ||
      !input.rotation.defined() || input.rotation.dim() != 2 ||
      input.rotation.size(0) != count || input.rotation.size(1) != 4 ||
      !input.features_dc.defined() || input.features_dc.size(0) != count ||
      !input.features_rest.defined() || input.features_rest.size(0) != count)
    throw std::runtime_error("Render attributes must align with xyz");

  auto screenspace_points = torch::zeros_like(
      input.xyz, input.xyz.options().requires_grad(true));
  try {
    screenspace_points.retain_grad();
  } catch (const std::exception&) {
  }

  const float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
  const float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);
  GaussianRasterizationSettings raster_settings(
      image_height, image_width, tanfovx, tanfovy, bg_color, scaling_modifier,
      viewpoint_camera->world_view_transform_,
      viewpoint_camera->full_proj_transform_, input.active_sh_degree,
      viewpoint_camera->camera_center_, false, false, collect_importance);
  GaussianRasterizer rasterizer(raster_settings);

  auto means3D = input.xyz.contiguous();
  auto means2D = screenspace_points;
  auto opacity = input.opacity.contiguous();

  /* If precomputed 3d covariance is provided, use it. If not, then it will be
     computed from scaling / rotation by the rasterizer.
   */
  torch::Tensor scales, rotations, cov3D_precomp;
  if (pipe.compute_cov3D_) {
    auto rotation_copy = input.rotation;
    auto rotation_matrix = general_utils::build_rotation(rotation_copy);
    auto scaled = scaling_modifier * input.scaling;
    auto scaling_matrix = torch::diag_embed(scaled);
    auto basis = rotation_matrix.matmul(scaling_matrix);
    cov3D_precomp = basis.matmul(basis.transpose(1, 2)).contiguous();
  } else {
    scales = input.scaling.contiguous();
    rotations = input.rotation.contiguous();
  }

  /* If precomputed colors are provided, use them. Otherwise, if it is desired
     to precompute colors from SHs in Python, do it. If not, then SH -> RGB
     conversion will be done by rasterizer.
   */
  torch::Tensor dc, shs, colors_precomp;
  if (use_override_color) {
    colors_precomp = override_color;
  } else {
    if (pipe.convert_SHs_) {
      const int max_sh_degree = input.max_sh_degree + 1;
      auto features =
          torch::cat({input.features_dc, input.features_rest}, /*dim=*/1);
      torch::Tensor shs_view = features.transpose(1, 2).view(
          {-1, 3, max_sh_degree * max_sh_degree});
      torch::Tensor dir_pp =
          (means3D - viewpoint_camera->camera_center_.repeat(
                         {features.size(0), 1}));
      auto dir_pp_normalized =
          dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
      auto sh2rgb =
          sh_utils::eval_sh(input.active_sh_degree, shs_view,
                            dir_pp_normalized);
      colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
    } else {
      if (pipe.separate_sh_) {
        dc = input.features_dc.contiguous();
        shs = input.features_rest.contiguous();
      } else {
        shs = torch::cat({input.features_dc, input.features_rest}, /*dim=*/1)
                  .contiguous();
      }
    }
  }

  // Rasterize visible Gaussians to image, obtain their radii (on screen).
  auto rasterizer_result =
      rasterizer.forward(means3D, means2D, opacity, dc, shs, colors_precomp,
                         scales, rotations, cov3D_precomp);
  auto rendered_image = rasterizer_result.image;
  auto radii = rasterizer_result.radii;

  /* Those Gaussians that were frustum culled or had a radius of 0 were not
     visible. They will be excluded from value updates used in the splitting
     criteria.
   */
  return {rendered_image,
          screenspace_points,
          (radii > 0).nonzero().reshape({-1}),
          radii,
          rasterizer_result.frame_importance,
          rasterizer_result.contribution_count};
}
