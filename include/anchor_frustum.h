#pragma once
#include <torch/torch.h>

// Same row-vector, z in [0,w] clip convention as python/anchor_frustum.py.
// The nominal cell is center +/- offsetScale/2; no Gaussian or compensation padding.
inline torch::Tensor anchorNominalFrustumMask(const torch::Tensor& centers,
    const torch::Tensor& offset_scales, const torch::Tensor& full_projection) {
  torch::NoGradGuard guard;
  TORCH_CHECK(centers.dim()==2 && centers.size(1)==3 &&
      offset_scales.sizes()==torch::IntArrayRef({centers.size(0),1}) &&
      full_projection.sizes()==torch::IntArrayRef({4,4}), "Invalid nominal Anchor bounds/projection");
  TORCH_CHECK(centers.device()==offset_scales.device() && centers.device()==full_projection.device() &&
      centers.scalar_type()==offset_scales.scalar_type() && centers.scalar_type()==full_projection.scalar_type(),
      "Anchor bounds and camera projection must share device/dtype");
  const auto w=full_projection.select(1,3), x=full_projection.select(1,0);
  const auto y=full_projection.select(1,1), z=full_projection.select(1,2);
  const auto planes=torch::stack({w+x,w-x,w+y,w-y,z,w-z});
  const auto normals=planes.slice(1,0,3);
  const auto distance=centers.matmul(normals.transpose(0,1))+planes.select(1,3);
  const auto radius=(offset_scales*.5)*normals.abs().sum(1);
  return (distance+radius>=0).all(1);
}
