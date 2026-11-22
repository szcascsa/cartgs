#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "hexplane_field.h"
#include "gaussian_transform_ops.h"
#include "elastic_entry.h"

struct TransformFieldConfig {
  std::string backend = "legacy";
  bool enabled = true;
  int start_iter = 5000;
  std::string start_mode = "fixed_iteration";
  float maturity_compute_epsilon = 0.0f;
  float elastic_entry_maturity = 0.0f;
  float ready_maturity = 0.30f;
  int min_ready_gaussians = 500;
  bool ready(std::uint64_t iteration, std::int64_t first_valid_cycle,
             int64_t ready_count = -1) const {
    if (start_mode == "maturity_population")
      return first_valid_cycle >= 0 && ready_count >= min_ready_gaussians;
    return start_mode == "first_maturity_cycle" ? first_valid_cycle >= 0
        : iteration >= static_cast<std::uint64_t>(std::max(0, start_iter));
  }
  float aabb_scale = 1.3f;
  HexPlaneConfig hexplane;
  int hidden_dim = 64;
  float grid_lr_init = 0.0016f;
  float grid_lr_final = 0.00016f;
  float mlp_lr_init = 0.00016f;
  float mlp_lr_final = 0.000016f;
  float lr_delay_mult = 0.01f;
  int lr_max_steps = 20000;
  float spatial_smooth_weight = 0.0002f;
  float ratio_smooth_weight = 0.001f;
  float ratio_l1_weight = 0.0001f;
  bool outside_identity = true;
};

struct TransformResidual {
  torch::Tensor delta_xyz;
  torch::Tensor delta_log_scale;
  torch::Tensor delta_rotation_raw;
};

struct ElasticGaussianAttributes {
  torch::Tensor xyz;
  torch::Tensor scaling;
  torch::Tensor rotation;
};

class TransformFieldImpl : public torch::nn::Module {
 public:
  TransformFieldImpl(const torch::Tensor& aabb_min,
                     const torch::Tensor& aabb_max,
                     const TransformFieldConfig& config);

  TransformResidual forward(const torch::Tensor& xyz,
                            float ratio);
  ElasticGaussianAttributes applyResidual(
      const torch::Tensor& xyz,
      const torch::Tensor& world_scaling,
      const torch::Tensor& world_rotation,
      float ratio, const torch::Tensor& maturity = {});
  GaussianAttrState applyLogResidual(const GaussianAttrState& state, float ratio);
  std::vector<GaussianAttrState> applyLogResidualMany(
      const GaussianAttrState& state, const std::vector<float>& ratios);
  std::shared_ptr<TransformFieldImpl> cloneFrozen() const;
  void resampleDomain(const AABB3D& old_aabb, const AABB3D& new_aabb,
                      const PlaneChangeMask& affected) {
    grid_->resampleDomain(old_aabb, new_aabb, affected);
  }
  void setAABB(const AABB3D& aabb) { grid_->setAABB(aabb); }
  std::vector<torch::Tensor> affectedParameters(const PlaneChangeMask& mask) const {
    return grid_->affectedParameters(mask);
  }

  torch::Tensor insideMask(const torch::Tensor& xyz) const {
    return grid_->insideMask(xyz);
  }
  torch::Tensor candidateMask(const torch::Tensor& xyz, const torch::Tensor& maturity) const {
    auto candidates = insideMask(xyz) & (maturity.detach() >= config_.maturity_compute_epsilon);
    return config_.elastic_entry_maturity > 0
        ? candidates & (maturity.detach() >= config_.elastic_entry_maturity) : candidates;
  }
  torch::Tensor regularizationLoss() const;

  std::vector<torch::Tensor> gridParameters() const {
    return grid_->gridParameters();
  }
  std::vector<torch::Tensor> mlpParameters() const;

  const torch::Tensor& aabbMin() const { return grid_->aabbMin(); }
  const torch::Tensor& aabbMax() const { return grid_->aabbMax(); }
  const TransformFieldConfig& config() const { return config_; }

 private:
  void initializeLinear(const torch::nn::Linear& layer,
                        float weight_scale = 1.0f);

  TransformFieldConfig config_;
  HexPlaneField grid_ = nullptr;
  torch::nn::Linear feature_out_ = nullptr;
  torch::nn::Linear position_hidden_ = nullptr;
  torch::nn::Linear position_out_ = nullptr;
  torch::nn::Linear scale_hidden_ = nullptr;
  torch::nn::Linear scale_out_ = nullptr;
  torch::nn::Linear rotation_hidden_ = nullptr;
  torch::nn::Linear rotation_out_ = nullptr;
};

TORCH_MODULE(TransformField);
