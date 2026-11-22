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
 * All the performance optimizations are released under the MIT License with
 * {Mallick and Goel} and Kerbl, Bernhard and Vicente Carrasco, Francisco and
 * Steinberger, Markus and De La Torre, Fernando
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>

#include <algorithm>
#include <cub/cub.cuh>

#include "auxiliary.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "forward.h"
#include <c10/cuda/CUDAStream.h>
namespace cg = cooperative_groups;


namespace preprocess_sort_study {
// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx,
                                        int deg,
                                        int max_coeffs,
                                        const glm::vec3* means,
                                        glm::vec3 campos,
                                        const float* dc,
                                        const float* shs,
                                        bool* clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir = dir / glm::length(dir);

  glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
  glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
  glm::vec3 result = SH_C0 * direct_color[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[3] + SH_C2[1] * yz * sh[4] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
               SH_C2[3] * xz * sh[6] + SH_C2[4] * (xx - yy) * sh[7];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
                 SH_C3[1] * xy * z * sh[9] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
                 SH_C3[5] * z * (xx - yy) * sh[13] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
      }
    }
  }
  result += 0.5f;

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean,
                               float focal_x,
                               float focal_y,
                               float tan_fovx,
                               float tan_fovy,
                               const float* cov3D,
                               const float* viewmatrix) {
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  float3 t = transformPoint4x3(mean, viewmatrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J =
      glm::mat3(focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f,
                focal_y / t.z, -(focal_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3(viewmatrix[0], viewmatrix[4], viewmatrix[8],
                          viewmatrix[1], viewmatrix[5], viewmatrix[9],
                          viewmatrix[2], viewmatrix[6], viewmatrix[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3],
                            cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale,
                             float mod,
                             const glm::vec4 rot,
                             float* cov3D) {
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot;  // / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),
                          2.f * (x * z + r * y), 2.f * (x * y + r * z),
                          1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                          2.f * (x * z - r * y), 2.f * (y * z + r * x),
                          1.f - 2.f * (x * x + y * y));

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C, bool Keys = false, bool Strided = false, bool Metadata = false>
__device__ __forceinline__ void preprocessOne(int idx,int P,
                               int D,
                               int M,
                               const float* orig_points,
                               const glm::vec3* scales,
                               const float scale_modifier,
                               const glm::vec4* rotations,
                               const float* opacities,
                               const float* dc,
                               const float* shs,
                               bool* clamped,
                               const float* cov3D_precomp,
                               const float* colors_precomp,
                               const float* viewmatrix,
                               const float* projmatrix,
                               const glm::vec3* cam_pos,
                               const int W,
                               int H,
                               const float tan_fovx,
                               float tan_fovy,
                               const float focal_x,
                               float focal_y,
                               int* radii,
                               float2* points_xy_image,
                               float* depths,
                               float* cov3Ds,
                               float* rgb,
                               float4* conic_opacity,
                               const dim3 grid,
                               uint32_t* tiles_touched,
                               bool prefiltered,uint32_t* depth_keys=nullptr,uint32_t* depth_ids=nullptr,
  int v0=0,int v1=0,int p0=0,int p1=0,int c0=0,uint4* footprint_rects=nullptr,float* footprint_limits=nullptr) {
  if (idx >= P) return;
  if(Keys){depth_keys[idx]=0xffffffff;depth_ids[idx]=idx;}

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx] = 0;
  tiles_touched[idx] = 0;

  // Perform near culling, quit if outside.
  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered,
                  p_view))
    return;
  if (!isfinite(p_view.x) || !isfinite(p_view.y) || !isfinite(p_view.z) ||
      !isfinite(opacities[idx]))
    return;

  // Transform point by projecting
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1],
                   orig_points[3 * idx + 2]};
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  if (!isfinite(p_hom.x) || !isfinite(p_hom.y) || !isfinite(p_hom.z) ||
      !isfinite(p_hom.w))
    return;
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  if (!isfinite(p_proj.x) || !isfinite(p_proj.y) || !isfinite(p_proj.z))
    return;

  // If 3D covariance matrix is precomputed, use it, otherwise compute
  // from scaling and rotation parameters.
  const float* cov3D;
  if (cov3D_precomp != nullptr) {
    cov3D = cov3D_precomp + idx * 6;
  } else {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D,
                            viewmatrix);
  if (!isfinite(cov.x) || !isfinite(cov.y) || !isfinite(cov.z)) return;

  // Invert covariance (EWA algorithm)
  float det = (cov.x * cov.z - cov.y * cov.y);
  if (!isfinite(det) || det <= 0.0f) return;
  float det_inv = 1.f / det;
  float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};
  if (!isfinite(conic.x) || !isfinite(conic.y) || !isfinite(conic.z)) return;

  // Compute extent in screen space (by finding eigenvalues of
  // 2D covariance matrix). Use extent to compute a bounding rectangle
  // of screen-space tiles that this Gaussian overlaps with. Quit if
  // rectangle covers 0 tiles.
  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
  if (!isfinite(my_radius) || my_radius <= 0.0f)
    return;
  // A finite radius larger than the image extent already covers every tile;
  // clamp it before the float-to-int rectangle conversion.
  my_radius = min(my_radius, 2.0f * max(W, H));
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
  if (!isfinite(point_image.x) || !isfinite(point_image.y)) return;
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

  // If colors have been precomputed, use them, otherwise convert
  // spherical harmonics coefficients to RGB color.
  if (colors_precomp == nullptr) {
    glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points,
                                          *cam_pos, dc, shs, clamped);
    if (!isfinite(result.x) || !isfinite(result.y) || !isfinite(result.z))
      return;
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }

  // Store some useful helper data for the next steps.
  depths[idx] = p_view.z;
  if(Keys)depth_keys[idx]=__float_as_uint(p_view.z);
  radii[idx] = my_radius;
  points_xy_image[idx] = point_image;

  // Inverse 2D covariance and opacity neatly pack into one float4
  conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
  if(Metadata) {
    footprint_rects[idx]=make_uint4(rect_min.x,rect_min.y,rect_max.x,rect_max.y);
    footprint_limits[idx]=logf(opacities[idx]/(1.0f/255.0f));
  }
}


__global__ void preprocessSortCUDA(int P,
                               int D,
                               int M,
                               const float* orig_points,
                               const glm::vec3* scales,
                               const float scale_modifier,
                               const glm::vec4* rotations,
                               const float* opacities,
                               const float* dc,
                               const float* shs,
                               bool* clamped,
                               const float* cov3D_precomp,
                               const float* colors_precomp,
                               const float* viewmatrix,
                               const float* projmatrix,
                               const glm::vec3* cam_pos,
                               const int W,
                               int H,
                               const float tan_fovx,
                               float tan_fovy,
                               const float focal_x,
                               float focal_y,
                               int* radii,
                               float2* points_xy_image,
                               float* depths,
                               float* cov3Ds,
                               float* rgb,
                               float4* conic_opacity,
                               const dim3 grid,
                               uint32_t* tiles_touched,
                               bool prefiltered,uint32_t* depth_keys=nullptr,uint32_t* depth_ids=nullptr,
  int v0=0,int v1=0,int p0=0,int p1=0,int c0=0,uint4* footprint_rects=nullptr,float* footprint_limits=nullptr,uint64_t* temporary=nullptr,uint32_t* sorted_keys=nullptr,uint32_t* sorted_ids=nullptr,bool final=false) {
  {
    __shared__ float camera_values[35];
    int t=threadIdx.x;
    if(t<16)camera_values[t]=viewmatrix[(t/4)*v0+(t%4)*v1];
    else if(t<32){int j=t-16;camera_values[t]=projmatrix[(j/4)*p0+(j%4)*p1];}
    else if(t<35)camera_values[t]=reinterpret_cast<const float*>(cam_pos)[(t-32)*c0];
    __syncthreads();
    viewmatrix=camera_values;projmatrix=camera_values+16;
    cam_pos=reinterpret_cast<const glm::vec3*>(camera_values+32);
  }

  #pragma unroll 1
  for(int item=0;item<4;++item) {
    int idx=blockIdx.x*512+threadIdx.x+item*128;
    preprocessOne<3,true,false,false>(idx,P,D,M,orig_points,scales,scale_modifier,rotations,opacities,dc,shs,
        clamped,cov3D_precomp,colors_precomp,viewmatrix,projmatrix,cam_pos,W,H,tan_fovx,tan_fovy,focal_x,focal_y,
        radii,points_xy_image,depths,cov3Ds,rgb,conic_opacity,grid,tiles_touched,prefiltered,depth_keys,depth_ids);
  }
  __syncthreads();
  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[4],values[4];
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    keys[j]=i<unsigned(P)?depth_keys[i]:0xffffffff;values[j]=i;
  }
  Sort(storage).Sort(keys,values);
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    if(i<unsigned(P)) {
      if(final){sorted_keys[i]=keys[j];sorted_ids[i]=values[j];}
      else temporary[i]=(uint64_t(keys[j])<<32)|values[j];
    }
  }
}

}
void sixhour_preprocess_sort(int P,
                         int D,
                         int M,
                         const float* means3D,
                         const glm::vec3* scales,
                         const float scale_modifier,
                         const glm::vec4* rotations,
                         const float* opacities,
                         const float* dc,
                         const float* shs,
                         bool* clamped,
                         const float* cov3D_precomp,
                         const float* colors_precomp,
                         const float* viewmatrix,
                         const float* projmatrix,
                         const glm::vec3* cam_pos,
                         const int W,
                         int H,
                         const float focal_x,
                         float focal_y,
                         const float tan_fovx,
                         float tan_fovy,
                         int* radii,
                         float2* means2D,
                         float* depths,
                         float* cov3Ds,
                         float* rgb,
                         float4* conic_opacity,
                         const dim3 grid,
                         uint32_t* tiles_touched,
                         bool prefiltered,uint32_t* depth_keys,uint32_t* depth_ids,int v0,int v1,int p0,int p1,int c0,uint64_t* temporary,uint32_t* sorted_keys,uint32_t* sorted_ids,bool final) {
  preprocess_sort_study::preprocessSortCUDA<<<(P+511)/512,128,0,c10::cuda::getCurrentCUDAStream()>>>(P, D, M, means3D, scales, scale_modifier, rotations, opacities, dc, shs,
      clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos,
      W, H, tan_fovx, tan_fovy, focal_x, focal_y, radii, means2D, depths,
      cov3Ds, rgb, conic_opacity, grid, tiles_touched, prefiltered,depth_keys,depth_ids,v0,v1,p0,p1,c0,nullptr,nullptr,temporary,sorted_keys,sorted_ids,final);
}
