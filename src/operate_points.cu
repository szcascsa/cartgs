/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University
 * of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Photo-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iostream>
#include <numeric>

#include "include/operate_points.h"
namespace cg = cooperative_groups;

#include "cuda_rasterizer/operate_points.h"

__global__ void transform_points(const int P,
                                 const float* __restrict__ orig_points,
                                 const float* __restrict__ transformmatrix,
                                 float* __restrict__ trans_points) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  float3 p_trans = transform_point(idx, orig_points, transformmatrix);
  insert_point_to_pcd(idx, p_trans, trans_points);
}

void transformPoints(torch::Tensor& points, torch::Tensor& transformmatrix) {
  if (points.ndimension() != 2 || points.size(1) != 3) {
    AT_ERROR("points must have dimensions (num_points, 3)");
  }

  const int P = points.size(0);
  torch::Tensor transformed_points = torch::zeros_like(points);

  if (P != 0) {
    transform_points<<<(P + 255) / 256, 256>>>(
        P, points.contiguous().data_ptr<float>(),
        transformmatrix.contiguous().data_ptr<float>(),
        transformed_points.contiguous().data_ptr<float>());

    points = transformed_points;
  }
}

torch::Tensor markVisibleUntransformedPoints(
    const torch::Tensor& points, const torch::Tensor& remaining,
    const torch::Tensor& unstable, const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix) {
  TORCH_CHECK(remaining.size(0) == points.size(0) &&
                  unstable.size(0) == points.size(0), "Correction mask size mismatch");
  auto xyz = points, view = viewmatrix, projection = projmatrix;
  return remaining & unstable & markVisible(xyz, view, projection);
}
