#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace improvements::selector {

class GumbelNetworkImpl : public torch::nn::Module {
 public:
  static constexpr int64_t kEmbeddingWidth = 32;

  GumbelNetworkImpl()
      : pos_emd_(register_module("pos_emd", makeEmbedding(3))),
        rotation_emd_(register_module("rotation_emd", makeEmbedding(4))),
        scale_emd_(register_module("scale_emd", makeEmbedding(3))),
        time_emd_(register_module("time_emd", makeEmbedding(1))),
        soft_net_(register_module(
            "soft_net", torch::nn::Linear(kEmbeddingWidth * 4, 2))) {}

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& position,
      const torch::Tensor& rotation,
      const torch::Tensor& scale,
      const torch::Tensor& target_ratio,
      const double tau = 1.0) {
    auto gumbel_input = torch::cat(
        {pos_emd_->forward(position), rotation_emd_->forward(rotation),
         scale_emd_->forward(scale), time_emd_->forward(target_ratio)},
        1);
    auto logits = soft_net_->forward(gumbel_input);
    auto hard_output = gumbelSoftmax(logits, tau, true);
    auto soft_output = gumbelSoftmax(logits, tau, false);

    return {logits.select(1, 1), hard_output.select(1, 1),
            soft_output.select(1, 1)};
  }

 private:
  static torch::nn::Sequential makeEmbedding(const int64_t input_width) {
    return torch::nn::Sequential(
        torch::nn::Linear(input_width, kEmbeddingWidth), torch::nn::ReLU(),
        torch::nn::Linear(kEmbeddingWidth, kEmbeddingWidth),
        torch::nn::ReLU());
  }

  static torch::Tensor gumbelSoftmax(const torch::Tensor& logits,
                                     const double tau,
                                     const bool hard) {
    auto gumbel_noise = -torch::empty_like(logits).exponential_().log();
    auto soft = ((logits + gumbel_noise) / tau).softmax(-1);
    if (!hard) {
      return soft;
    }

    auto indices = std::get<1>(soft.max(-1, true));
    auto one_hot = torch::zeros_like(logits).scatter_(-1, indices, 1.0);
    return one_hot - soft.detach() + soft;
  }

  torch::nn::Sequential pos_emd_;
  torch::nn::Sequential rotation_emd_;
  torch::nn::Sequential scale_emd_;
  torch::nn::Sequential time_emd_;
  torch::nn::Linear soft_net_;
};

TORCH_MODULE(GumbelNetwork);

}  // namespace improvements::selector
