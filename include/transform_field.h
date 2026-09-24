#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "hexplane_field.h"

struct TransformFieldConfig {
  bool enabled = true;
  int start_iter = 5000;
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
  bool mature_only = true;
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
      float ratio);

  torch::Tensor insideMask(const torch::Tensor& xyz) const {
    return grid_->insideMask(xyz);
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
