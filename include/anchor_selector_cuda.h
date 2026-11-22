#pragma once
#include <torch/torch.h>

namespace mapping_cuda {
// Base rotation is already normalized by GaussianModel's original activation.
// Outputs: XYZ, activated scale, normalized quaternion, gated opacity, SH DC.
std::vector<torch::Tensor> decode14(const torch::Tensor& xyz,
    const torch::Tensor& scale,const torch::Tensor& rotation,
    const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask);
// Legacy Log-Scale path: add the residual to the Master log-scale before exp.
std::vector<torch::Tensor> decode14LogScale(const torch::Tensor& xyz,
    const torch::Tensor& log_scale,const torch::Tensor& rotation,
    const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask);
std::vector<torch::Tensor> decode14Forward(const std::vector<torch::Tensor>& inputs);
std::vector<torch::Tensor> decode14Backward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& gradients);
std::vector<torch::Tensor> decode14LogScaleForward(const std::vector<torch::Tensor>& inputs);
std::vector<torch::Tensor> decode14LogScaleBackward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& gradients);
}

namespace mapping_cuda {
// Input is Master raw log-scale. Physical residual is added after exp.
std::vector<torch::Tensor> decode14AdditiveScale(const torch::Tensor& xyz,
    const torch::Tensor& log_scale,const torch::Tensor& rotation,
    const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask);
std::vector<torch::Tensor> decode14AdditiveScaleForward(const std::vector<torch::Tensor>& inputs);
std::vector<torch::Tensor> decode14AdditiveScaleBackward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& gradients);
// Relative Softplus: exp(master_log_scale) * softplus(log(expm1(1)) + residual).
// Floor only at the smallest normal FP32 value to prevent numerical zero.
std::vector<torch::Tensor> decode14SoftplusScale(const torch::Tensor& xyz,
    const torch::Tensor& log_scale,const torch::Tensor& rotation,
    const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask);
std::vector<torch::Tensor> decode14SoftplusScaleForward(const std::vector<torch::Tensor>& inputs);
std::vector<torch::Tensor> decode14SoftplusScaleBackward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& gradients);
}
