#pragma once

#include <torch/torch.h>
#include "include/model_snapshot.h"
#ifdef CARTGS_EXPERIMENTAL_SELECTOR_CUDA
#include "include/selector_cuda.h"
#endif

#include <cstdint>
#include <tuple>

namespace improvements::selector {

class GumbelNetworkImpl : public torch::nn::Module {
 public:
  static constexpr int64_t kEmbeddingWidth = 32;

  // The mapper config selects CUDA by default; bare CPU/oracle modules stay legacy.
  void setBackend(const std::string& selector_backend) {
    TORCH_CHECK(selector_backend == "legacy" || selector_backend == "cuda",
                "Selector.backend must be legacy or cuda");
#ifndef CARTGS_EXPERIMENTAL_SELECTOR_CUDA
    TORCH_CHECK(selector_backend == "legacy",
                "Selector CUDA backend was not compiled; rebuild with "
                "-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON");
#endif
    selector_backend_ = selector_backend;
  }
  const std::string& backend() const { return selector_backend_; }

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
    auto logits = logitsFor(position, rotation, scale, target_ratio);
    auto hard_output = gumbelSoftmax(logits, tau, true);
    auto soft_output = gumbelSoftmax(logits, tau, false);

    return {logits.select(1, 1), hard_output.select(1, 1),
            soft_output.select(1, 1)};
  }

  // LOD only needs the hard straight-through sample, not a second soft sample.
  torch::Tensor forwardHard(const torch::Tensor& position,
                            const torch::Tensor& rotation,
                            const torch::Tensor& scale, float ratio,
                            double tau = 1.0,
                            torch::Tensor* keep_probability = nullptr) {
    auto logits = logitsFor(position, rotation, scale,
        torch::full({1, 1}, ratio, position.options()));
    // Gumbel argmax has softmax(logits) probabilities, independent of tau.
    // Reuse the logits for deterministic normal-training budget supervision.
    if (keep_probability) *keep_probability = logits.softmax(-1).select(1, 1);
    return gumbelSoftmax(logits, tau, true).select(1, 1);
  }

  // Deterministic keep probability used by Change Distill. It deliberately
  // avoids Gumbel noise so the teacher/student comparison is stable.
  torch::Tensor forwardKeepLogit(const torch::Tensor& position,
                                const torch::Tensor& rotation,
                                const torch::Tensor& scale, float ratio,
                                double tau = 1.0) {
    const auto logits = logitsFor(position, rotation, scale,
        torch::full({1, 1}, ratio, position.options()));
    return (logits.select(1, 1) - logits.select(1, 0)) / tau;
  }

  // Historical guard only: share one uniform draw across baseline and all
  // candidates. The forward has the SAME Bernoulli law as Gumbel argmax;
  // the backward is a straight-through relaxation. Normal training is unchanged.
  torch::Tensor forwardHardWithUniform(const torch::Tensor& position,
                                      const torch::Tensor& rotation,
                                      const torch::Tensor& scale, float ratio,
                                      const torch::Tensor& uniform,
                                      double tau = 1.0) {
    TORCH_CHECK(tau > 0.0 && uniform.numel() == position.size(0),
                "Fixed-uniform Selector input mismatch");
    auto logit = forwardKeepLogit(position, rotation, scale, ratio, 1.0);
    auto u = uniform.reshape({-1}).clamp(1e-7, 1.0 - 1e-7);
    auto threshold = u.log() - (1.0 - u).log();
    auto soft = ((logit - threshold) / tau).sigmoid();
    auto hard = (uniform.reshape({-1}) < logit.sigmoid()).to(soft.scalar_type());
    return hard - soft.detach() + soft;
  }

  torch::Tensor forwardProbability(const torch::Tensor& position,
                                   const torch::Tensor& rotation,
                                   const torch::Tensor& scale,
                                   const torch::Tensor& target_ratio,
                                   const double tau = 1.0) {
    const auto logits = logitsFor(position, rotation, scale, target_ratio);
    return (logits / tau).softmax(-1).select(1, 1);
  }

  torch::Tensor forwardProbability(const torch::Tensor& position,
                                   const torch::Tensor& rotation,
                                   const torch::Tensor& scale, float ratio,
                                   double tau = 1.0) {
    return forwardProbability(position, rotation, scale,
        torch::full({1, 1}, ratio, position.options()), tau);
  }

  std::shared_ptr<GumbelNetworkImpl> cloneFrozen() const {
    ScopedTorchRNG rng(torch::Device(torch::kCPU));
    auto teacher = std::make_shared<GumbelNetworkImpl>();
    teacher->to(parameters().front().device());
    copyFrozenModule(*this, *teacher);
    teacher->setBackend(selector_backend_);
    return teacher;
  }

 private:
  torch::Tensor logitsFor(const torch::Tensor& position,
                          const torch::Tensor& rotation,
                          const torch::Tensor& scale,
                          const torch::Tensor& target_ratio) {
#ifdef CARTGS_EXPERIMENTAL_SELECTOR_CUDA
    if (selector_backend_ == "cuda")
      return selector_cuda::network({position, rotation, scale, target_ratio},
          parameters(), selector_cuda::Candidate::CublasStrict);
#endif
    auto ratio_embedding = time_emd_->forward(target_ratio);
    if (ratio_embedding.size(0) == 1 && position.size(0) != 1)
      ratio_embedding = ratio_embedding.expand({position.size(0), kEmbeddingWidth});
    auto gumbel_input = torch::cat(
        {pos_emd_->forward(position), rotation_emd_->forward(rotation),
         scale_emd_->forward(scale), ratio_embedding},
        1);
    return soft_net_->forward(gumbel_input);
  }

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
  std::string selector_backend_ = "legacy";
};

TORCH_MODULE(GumbelNetwork);

}  // namespace improvements::selector
