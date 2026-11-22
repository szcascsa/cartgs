#pragma once

#include <torch/torch.h>
#include <vector>

namespace tf_cuda {
// TensorList is flattened by the C++ autograd bridge, preserving parameter edges.
torch::Tensor query(const torch::Tensor& xyz, const torch::Tensor& ratio,
                    const torch::Tensor& lo, const torch::Tensor& hi,
                    const std::vector<torch::Tensor>& planes);
torch::Tensor queryForward(const std::vector<torch::Tensor>& inputs);
std::vector<torch::Tensor> queryBackward(
    const std::vector<torch::Tensor>& inputs, const torch::Tensor& gradient);

enum class MlpBackend { Cublas, Warp32 };
// Minimal training context: feature, shared ReLU, three head ReLUs, three heads.
// No per-plane samples or product intermediates are retained.
std::vector<torch::Tensor> mlpForward(
    const torch::Tensor& feature, const std::vector<torch::Tensor>& parameters,
    MlpBackend backend = MlpBackend::Cublas);
std::vector<torch::Tensor> outputForward(
    const torch::Tensor& xyz, const torch::Tensor& scale,
    const torch::Tensor& quaternion, const std::vector<torch::Tensor>& residual,
    bool log_scale = false);
// mlpBackward returns dFeature followed by the fourteen W/b gradients.
std::vector<torch::Tensor> mlpBackward(
    const std::vector<torch::Tensor>& context,
    const std::vector<torch::Tensor>& parameters,
    const std::vector<torch::Tensor>& gradients);
// outputBackward returns dXYZ/dScale/dBaseQ/dDX/dDS/dDQ.
std::vector<torch::Tensor> outputBackward(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& gradients, bool log_scale = false);
std::vector<torch::Tensor> neural(const torch::Tensor& feature,
    const std::vector<torch::Tensor>& parameters,
    MlpBackend backend = MlpBackend::Cublas);
std::vector<torch::Tensor> output(const torch::Tensor& xyz,
    const torch::Tensor& scale, const torch::Tensor& quaternion,
    const std::vector<torch::Tensor>& residual, bool log_scale = false);
torch::Tensor regularization(const std::vector<torch::Tensor>& planes,
    float spatial_weight,float ratio_weight,float l1_weight);
torch::Tensor regularizationForward(const std::vector<torch::Tensor>& planes,
    float spatial_weight,float ratio_weight,float l1_weight);
std::vector<torch::Tensor> regularizationBackward(
    const std::vector<torch::Tensor>& planes,const torch::Tensor& gradient,
    float spatial_weight,float ratio_weight,float l1_weight);
}  // namespace tf_cuda
