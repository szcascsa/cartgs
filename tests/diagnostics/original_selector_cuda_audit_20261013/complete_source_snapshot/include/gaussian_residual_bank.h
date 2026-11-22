#pragma once

#include "gaussian_renderer.h"
#include "transform_field.h"

// Attribute-block TF residuals. The caller supplies only selected, mature, in-domain
// rows and their master indices; this module never changes Gaussian topology.
class GaussianResidualBankImpl : public torch::nn::Module {
 public:
  GaussianResidualBankImpl(int64_t gaussian_count, int64_t hidden_dim,
                          int64_t sh_rest_columns,
                          const torch::TensorOptions& options,
                          bool attribute_blocks = true);

  torch::Tensor residual(const torch::Tensor& hidden,
                         const torch::Tensor& global_indices);
  GaussianRenderInput apply(const GaussianRenderInput& base,
                            const torch::Tensor& opacity_logits,
                            const torch::Tensor& global_indices,
                            TransformField& field, float ratio);
  int64_t gaussianCount() const { return embedding_.size(0); }
  std::vector<torch::Tensor> headParameters() const { return head_->parameters(); }
  torch::Tensor embedding() const { return embedding_; }
  void append(int64_t count, torch::optim::Adam& optimizer, std::size_t group);
  void prune(const torch::Tensor& survivors, torch::optim::Adam& optimizer,
             std::size_t group);
  std::shared_ptr<GaussianResidualBankImpl> cloneFrozen() const;

 private:
  torch::Tensor embedding_, projection_mask_, attribute_units_;
  torch::nn::Linear head_ = nullptr;
  void replaceEmbedding(const torch::Tensor& value, const torch::Tensor& survivors,
                        int64_t appended, torch::optim::Adam& optimizer,
                        std::size_t group);
};

TORCH_MODULE(GaussianResidualBank);
