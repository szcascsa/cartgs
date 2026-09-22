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
 * as part of Photo-SLAM.
 */

#pragma once

#include <torch/torch.h>

#include <memory>
#include <tuple>

#include "gaussian_keyframe.h"
#include "gaussian_model.h"
#include "gaussian_parameters.h"
#include "gaussian_rasterizer.h"
#include "sh_utils.h"

struct RenderPackage {
  torch::Tensor image;
  torch::Tensor viewspace_points;
  torch::Tensor visibility_indices;
  torch::Tensor radii;

  // Raw non-differentiable per-Gaussian statistics. They are empty unless the
  // caller enables collection.
  torch::Tensor frame_importance;
  torch::Tensor contribution_count;
};

class GaussianRenderer {
 public:
  static RenderPackage render(
      std::shared_ptr<GaussianKeyframe> viewpoint_camera,
      int image_height,
      int image_width,
      std::shared_ptr<GaussianModel> gaussians,
      GaussianPipelineParams& pipe,
      torch::Tensor& bg_color,
      torch::Tensor& override_color,
      float scaling_modifier = 1.0f,
      bool has_override_color = false,
      torch::Tensor selector_mask = torch::Tensor(),
      bool detach_gaussian_parameters = false,
      bool collect_importance = false);
};
