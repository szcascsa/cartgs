#pragma once
#include <torch/torch.h>
namespace mapping_cuda {
// Probability (possibly stride=2), cached uniform, activated opacity and optional
// inactive bool. Returns effective probability, ST mask, stable selected IDs.
// Budget deliberately retains the original ATen sum/abs/div implementation.
std::vector<torch::Tensor> selectorSampleCompact(const torch::Tensor& probability,
    const torch::Tensor& uniform,const torch::Tensor& opacity={},const torch::Tensor& inactive={});
std::vector<torch::Tensor> selectorSamplingForward(const torch::Tensor& probability,
    const torch::Tensor& uniform,const torch::Tensor& opacity,const torch::Tensor& inactive);
torch::Tensor selectorSamplingBackward(const torch::Tensor& eligible,const torch::Tensor& mask_gradient,
    const torch::Tensor& probability_gradient);
}
