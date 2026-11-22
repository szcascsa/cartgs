#include "include/hexplane_field.h"

#include <torch/nn/functional/vision.h>

#include <stdexcept>

using torch::indexing::Slice;

HexPlaneFieldImpl::HexPlaneFieldImpl(const torch::Tensor& aabb_min,
                                     const torch::Tensor& aabb_max,
                                     const HexPlaneConfig& config)
    : config_(config) {
  if (!aabb_min.defined() || !aabb_max.defined() || aabb_min.numel() != 3 ||
      aabb_max.numel() != 3)
    throw std::invalid_argument("HexPlane AABB must contain three values");
  if (config_.spatial_resolution < 2 || config_.ratio_resolution < 2 ||
      config_.feature_dim <= 0 || config_.multires.empty())
    throw std::invalid_argument("Invalid HexPlane configuration");

  aabb_min_ = register_buffer("aabb_min", aabb_min.reshape({3}).detach().clone());
  aabb_max_ = register_buffer("aabb_max", aabb_max.reshape({3}).detach().clone());
  if (!(aabb_max_ > aabb_min_).all().item<bool>())
    throw std::invalid_argument("HexPlane AABB must have positive extent");

  torch::NoGradGuard no_grad;
  for (std::size_t level = 0; level < config_.multires.size(); ++level) {
    const int multiplier = config_.multires[level];
    if (multiplier <= 0)
      throw std::invalid_argument("HexPlane multires values must be positive");
    const std::array<int, 4> resolution = {
        config_.spatial_resolution * multiplier,
        config_.spatial_resolution * multiplier,
        config_.spatial_resolution * multiplier,
        config_.ratio_resolution};
    for (std::size_t plane_index = 0;
         plane_index < kCoordinatePairs.size(); ++plane_index) {
      const auto pair = kCoordinatePairs[plane_index];
      auto plane = torch::empty(
          {1, config_.feature_dim, resolution[pair[1]], resolution[pair[0]]},
          aabb_min_.options().requires_grad(true));
      initializePlane(plane, pair[0] == 3 || pair[1] == 3);
      planes_.push_back(register_parameter(
          "L" + std::to_string(level) + "_" + kPlaneNames[plane_index],
          plane, true));
    }
  }
}

torch::Tensor HexPlaneFieldImpl::insideMask(
    const torch::Tensor& xyz) const {
  if (xyz.dim() != 2 || xyz.size(1) != 3)
    throw std::invalid_argument("xyz must have shape [N,3]");
  return AABB3D{aabb_min_, aabb_max_}.containsMask(xyz);
}

void HexPlaneFieldImpl::initializePlane(torch::Tensor plane, bool ratio_plane) {
  torch::NoGradGuard no_grad;
  if (ratio_plane) plane.fill_(1.0f);
  else plane.uniform_(0.1f, 0.5f);
}

std::vector<torch::Tensor> HexPlaneFieldImpl::affectedParameters(
    const PlaneChangeMask& affected) const {
  std::vector<torch::Tensor> result;
  for (std::size_t level = 0; level < config_.multires.size(); ++level)
    for (std::size_t p = 0; p < kCoordinatePairs.size(); ++p)
      if (affected.at(p)) result.push_back(planes_[planeOffset(level, p)]);
  return result;
}

void HexPlaneFieldImpl::setAABB(const AABB3D& aabb) {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(aabb.defined() && torch::isfinite(aabb.min_xyz).all().item<bool>() &&
                  torch::isfinite(aabb.max_xyz).all().item<bool>() &&
                  (aabb.extent() > 0).all().item<bool>(), "Invalid TF AABB");
  aabb_min_.copy_(aabb.min_xyz);
  aabb_max_.copy_(aabb.max_xyz);
}

void HexPlaneFieldImpl::resampleDomain(
    const AABB3D& old_aabb, const AABB3D& new_aabb,
    const PlaneChangeMask& affected) {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(old_aabb.defined() && new_aabb.defined(), "Undefined resampling domain");
  TORCH_CHECK(torch::isfinite(new_aabb.min_xyz).all().item<bool>() &&
                  torch::isfinite(new_aabb.max_xyz).all().item<bool>() &&
                  (new_aabb.extent() > 0).all().item<bool>(), "Invalid new domain");
  TORCH_CHECK(torch::equal(aabb_min_, old_aabb.min_xyz) &&
                  torch::equal(aabb_max_, old_aabb.max_xyz), "Stale resampling plan");
  TORCH_CHECK((new_aabb.min_xyz <= old_aabb.min_xyz).all().item<bool>() &&
                  (new_aabb.max_xyz >= old_aabb.max_xyz).all().item<bool>(),
              "TF domain must expand monotonically");
  std::array<bool, 3> changed_axes;
  for (int axis = 0; axis < 3; ++axis)
    changed_axes[axis] = ((old_aabb.min_xyz[axis] != new_aabb.min_xyz[axis]) |
                          (old_aabb.max_xyz[axis] != new_aabb.max_xyz[axis])).item<bool>();
  for (std::size_t p = 0; p < kCoordinatePairs.size(); ++p) {
    const auto pair = kCoordinatePairs[p];
    TORCH_CHECK(affected.at(p) == (changed_axes[pair[0]] ||
                    (pair[1] < 3 && changed_axes[pair[1]])),
                "Affected-plane mask does not match AABB change");
  }
  for (std::size_t level = 0; level < config_.multires.size(); ++level) {
    for (std::size_t p = 0; p < kCoordinatePairs.size(); ++p) {
      const auto pair = kCoordinatePairs[p];
      if (!affected.at(p)) continue;
      auto plane = planes_[planeOffset(level, p)];
      const auto height = plane.size(2), width = plane.size(3);
      auto remap = [&](int axis, int64_t size) {
        auto u = torch::linspace(-1.0, 1.0, size, plane.options().requires_grad(false));
        if (axis == 3) return u;  // Ratio coordinates do not change.
        auto world = new_aabb.min_xyz[axis] + (u + 1.0) *
                     (0.5 * new_aabb.extent()[axis]);
        return 2.0 * (world - old_aabb.min_xyz[axis]) / old_aabb.extent()[axis] - 1.0;
      };
      auto u = remap(pair[0], width).view({1, width}).expand({height, width});
      auto v = remap(pair[1], height).view({height, 1}).expand({height, width});
      auto grid = torch::stack({u, v}, -1).unsqueeze(0);
      auto sampled = torch::nn::functional::grid_sample(
          plane, grid, torch::nn::functional::GridSampleFuncOptions()
                           .mode(torch::kBilinear).padding_mode(torch::kBorder)
                           .align_corners(true));
      auto inside = ((u >= -1.0) & (u <= 1.0) & (v >= -1.0) & (v <= 1.0))
                        .unsqueeze(0).unsqueeze(0);
      auto initialized = torch::empty_like(plane);
      initializePlane(initialized, pair[0] == 3 || pair[1] == 3);
      // Preserve the Parameter/TensorImpl referenced by the live optimizer.
      plane.copy_(torch::where(inside, sampled, initialized));
    }
  }
}

torch::Tensor HexPlaneFieldImpl::samplePlane(
    const torch::Tensor& plane,
    const torch::Tensor& coordinates,
    int first_coordinate,
    int second_coordinate) const {
  auto grid = torch::stack(
                  {coordinates.select(1, first_coordinate),
                   coordinates.select(1, second_coordinate)},
                  /*dim=*/1)
                  .view({1, 1, coordinates.size(0), 2});
  auto sampled = torch::nn::functional::grid_sample(
      plane, grid,
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .padding_mode(torch::kBorder)
          .align_corners(true));
  return sampled.view({config_.feature_dim, coordinates.size(0)})
      .transpose(0, 1);
}

torch::Tensor HexPlaneFieldImpl::forward(
    const torch::Tensor& xyz,
    const torch::Tensor& ratio) {
  if (xyz.dim() != 2 || xyz.size(1) != 3)
    throw std::invalid_argument("xyz must have shape [N,3]");
  if (xyz.size(0) == 0)
    return torch::empty({0, outputDim()}, xyz.options());

  torch::Tensor ratio_column = ratio;
  if (ratio_column.dim() == 0)
    ratio_column = ratio_column.expand({xyz.size(0), 1});
  else if (ratio_column.dim() == 1)
    ratio_column = ratio_column.unsqueeze(1);
  if (ratio_column.dim() != 2 || ratio_column.size(1) != 1 ||
      ratio_column.size(0) != xyz.size(0))
    throw std::invalid_argument("ratio must have shape [N,1]");

  auto normalized_xyz = AABB3D{aabb_min_, aabb_max_}.normalize(xyz);
  auto coordinates = torch::cat({normalized_xyz, ratio_column}, /*dim=*/1);
  std::vector<torch::Tensor> level_features;
  level_features.reserve(config_.multires.size());
  for (std::size_t level = 0; level < config_.multires.size(); ++level) {
    torch::Tensor level_feature;
    for (std::size_t plane_index = 0;
         plane_index < kCoordinatePairs.size(); ++plane_index) {
      const auto pair = kCoordinatePairs[plane_index];
      auto feature = samplePlane(planes_[planeOffset(level, plane_index)],
                                 coordinates, pair[0], pair[1]);
      level_feature = level_feature.defined() ? level_feature * feature : feature;
    }
    level_features.push_back(level_feature);
  }
  return torch::cat(level_features, /*dim=*/1);
}

torch::Tensor HexPlaneFieldImpl::planeSmoothness(
    const torch::Tensor& plane) const {
  if (plane.size(2) < 3) return plane.sum() * 0.0;
  auto first_difference =
      plane.index({Slice(), Slice(), Slice(1, plane.size(2)), Slice()}) -
      plane.index({Slice(), Slice(), Slice(0, plane.size(2) - 1), Slice()});
  auto second_difference =
      first_difference.index(
          {Slice(), Slice(), Slice(1, first_difference.size(2)), Slice()}) -
      first_difference.index(
          {Slice(), Slice(), Slice(0, first_difference.size(2) - 1), Slice()});
  return second_difference.square().mean();
}

torch::Tensor HexPlaneFieldImpl::spatialSmoothness() const {
  torch::Tensor loss;
  constexpr std::array<std::size_t, 3> spatial_planes = {0, 1, 3};
  for (std::size_t level = 0; level < config_.multires.size(); ++level) {
    for (const auto plane : spatial_planes) {
      auto value = planeSmoothness(planes_[planeOffset(level, plane)]);
      loss = loss.defined() ? loss + value : value;
    }
  }
  return loss;
}

torch::Tensor HexPlaneFieldImpl::ratioSmoothness() const {
  torch::Tensor loss;
  constexpr std::array<std::size_t, 3> ratio_planes = {2, 4, 5};
  for (std::size_t level = 0; level < config_.multires.size(); ++level) {
    for (const auto plane : ratio_planes) {
      auto value = planeSmoothness(planes_[planeOffset(level, plane)]);
      loss = loss.defined() ? loss + value : value;
    }
  }
  return loss;
}

torch::Tensor HexPlaneFieldImpl::ratioL1() const {
  torch::Tensor loss;
  constexpr std::array<std::size_t, 3> ratio_planes = {2, 4, 5};
  for (std::size_t level = 0; level < config_.multires.size(); ++level) {
    for (const auto plane : ratio_planes) {
      auto value = (1.0 - planes_[planeOffset(level, plane)]).abs().mean();
      loss = loss.defined() ? loss + value : value;
    }
  }
  return loss;
}
