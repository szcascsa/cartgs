#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct HexPlaneConfig {
  int spatial_resolution = 64;
  int ratio_resolution = 150;
  int feature_dim = 16;
  std::vector<int> multires = {1, 2, 4};
};

class HexPlaneFieldImpl : public torch::nn::Module {
 public:
  HexPlaneFieldImpl(const torch::Tensor& aabb_min,
                    const torch::Tensor& aabb_max,
                    const HexPlaneConfig& config);

  torch::Tensor forward(const torch::Tensor& canonical_xyz,
                        const torch::Tensor& ratio);
  torch::Tensor insideMask(const torch::Tensor& canonical_xyz) const;

  torch::Tensor spatialSmoothness() const;
  torch::Tensor ratioSmoothness() const;
  torch::Tensor ratioL1() const;

  const torch::Tensor& aabbMin() const { return aabb_min_; }
  const torch::Tensor& aabbMax() const { return aabb_max_; }
  int outputDim() const {
    return config_.feature_dim * static_cast<int>(config_.multires.size());
  }
  std::vector<torch::Tensor> gridParameters() const { return planes_; }

 private:
  static constexpr std::array<std::array<int, 2>, 6> kCoordinatePairs = {{
      {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}},
  }};
  static constexpr std::array<const char*, 6> kPlaneNames = {
      "xy", "xz", "xe", "yz", "ye", "ze"};

  torch::Tensor samplePlane(const torch::Tensor& plane,
                            const torch::Tensor& coordinates,
                            int first_coordinate,
                            int second_coordinate) const;
  torch::Tensor planeSmoothness(const torch::Tensor& plane) const;
  std::size_t planeOffset(std::size_t level, std::size_t plane) const {
    return level * kCoordinatePairs.size() + plane;
  }

  HexPlaneConfig config_;
  torch::Tensor aabb_min_;
  torch::Tensor aabb_max_;
  std::vector<torch::Tensor> planes_;
};

TORCH_MODULE(HexPlaneField);
