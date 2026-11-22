#pragma once

#include <torch/torch.h>

struct AABB3D {
  torch::Tensor min_xyz;
  torch::Tensor max_xyz;
  bool defined() const {
    return min_xyz.defined() && max_xyz.defined() &&
           min_xyz.numel() == 3 && max_xyz.numel() == 3;
  }
  torch::Tensor extent() const { return max_xyz - min_xyz; }
  torch::Tensor center() const { return (min_xyz + max_xyz) * 0.5; }
  torch::Tensor diagonal() const { return extent().norm(); }
  torch::Tensor normalize(const torch::Tensor& xyz) const {
    return 2.0 * (xyz - min_xyz) / extent() - 1.0;
  }
  torch::Tensor containsMask(const torch::Tensor& xyz) const {
    return ((xyz >= min_xyz) & (xyz <= max_xyz)).all(-1);
  }
  AABB3D clone() const {
    return defined() ? AABB3D{min_xyz.detach().clone(), max_xyz.detach().clone()}
                     : AABB3D{};
  }
};

struct PlaneChangeMask {
  bool xy = false;
  bool xz = false;
  bool yz = false;
  bool xe = false;
  bool ye = false;
  bool ze = false;
  // Actual storage order: xy, xz, xe, yz, ye, ze.
  bool at(std::size_t index) const {
    const bool flags[] = {xy, xz, xe, yz, ye, ze};
    return flags[index];
  }
};
