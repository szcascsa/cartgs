#pragma once
#include <torch/torch.h>
namespace mapping_cuda {
torch::Tensor spatialInput55(const torch::Tensor& raw,const torch::Tensor& coords,
    const torch::Tensor& mean,const torch::Tensor& stddev,const torch::Tensor& freq);
torch::Tensor spatialFilm(const torch::Tensor& h,const torch::Tensor& gain,const torch::Tensor& bias);
torch::Tensor spatialLiveInput55(const std::vector<torch::Tensor>& values);
torch::Tensor spatialLiveLogScaleInput55(const std::vector<torch::Tensor>& values);
std::vector<torch::Tensor> spatialLiveForward(const std::vector<torch::Tensor>& values);
std::vector<torch::Tensor> spatialLiveBackward(const std::vector<torch::Tensor>& values,const torch::Tensor& saved,const torch::Tensor& gradient);
std::vector<torch::Tensor> spatialLiveLogScaleForward(const std::vector<torch::Tensor>& values);
std::vector<torch::Tensor> spatialLiveLogScaleBackward(const std::vector<torch::Tensor>& values,const torch::Tensor& saved,const torch::Tensor& gradient);
torch::Tensor spatialInputForward(const std::vector<torch::Tensor>& values);
std::vector<torch::Tensor> spatialInputBackward(const std::vector<torch::Tensor>& values,const torch::Tensor& gradient);
torch::Tensor spatialFilmForward(const std::vector<torch::Tensor>& values);
std::vector<torch::Tensor> spatialFilmBackward(const std::vector<torch::Tensor>& values,const torch::Tensor& output,const torch::Tensor& gradient);
}
