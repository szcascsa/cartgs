#pragma once
#include <torch/torch.h>
namespace diagnostic_cuda {
torch::Tensor residual48Pack(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const std::vector<torch::Tensor>&,const std::vector<double>&);
std::vector<torch::Tensor> residual48PackForward(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const std::vector<torch::Tensor>&,const std::vector<double>&);
torch::Tensor residual48XYZBackward(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
}
