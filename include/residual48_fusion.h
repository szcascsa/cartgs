#pragma once
#include <torch/torch.h>

namespace mapping_cuda {
torch::Tensor residual48TrainingPack(const torch::Tensor& xyz,const torch::Tensor& center,
    const torch::Tensor& reference,const torch::Tensor& mean,const torch::Tensor& stddev,
    const std::vector<torch::Tensor>& features,const std::vector<double>& weights);
std::vector<torch::Tensor> residual48TrainingPackForward(const torch::Tensor& xyz,const torch::Tensor& center,
    const torch::Tensor& reference,const torch::Tensor& mean,const torch::Tensor& stddev,
    const std::vector<torch::Tensor>& features,const std::vector<double>& weights);
torch::Tensor residual48TrainingXYZBackward(const torch::Tensor& reference,const torch::Tensor& stddev,
    const torch::Tensor& saved,const torch::Tensor& gradient);
std::vector<torch::Tensor> residual48HeadsInference(const torch::Tensor& hidden,torch::nn::ModuleDict& heads);
torch::Tensor residual48AddReluInference(const torch::Tensor& encoded,const torch::Tensor& ratio);
torch::Tensor residual48OutputInference(const std::vector<torch::Tensor>& input,const torch::Tensor& reference);
torch::Tensor residual48InferencePack(const torch::Tensor& xyz,const torch::Tensor& center,
    const torch::Tensor& reference,const torch::Tensor& mean,const torch::Tensor& stddev,
    const torch::Tensor& owners,const std::vector<torch::Tensor>& tables,const std::vector<double>& weights);
}
