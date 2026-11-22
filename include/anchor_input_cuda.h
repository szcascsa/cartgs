#pragma once
#include <torch/torch.h>

namespace mapping_cuda {
// Layout: Feature16, (XYZ-center)/reference_scale, activated Scale3, DC3,
// activated Opacity1, log(ratio). Geometry/feature values are read each call.
torch::Tensor anchorInput27(const torch::Tensor& xyz,const torch::Tensor& feature,
    const torch::Tensor& owner,const torch::Tensor& center,const torch::Tensor& reference_scale,
    const torch::Tensor& scale,const torch::Tensor& dc,const torch::Tensor& opacity,double ratio);
torch::Tensor anchorInput27Forward(const std::vector<torch::Tensor>& inputs,double ratio);
std::vector<torch::Tensor> anchorInput27Backward(const std::vector<torch::Tensor>& inputs,
    const torch::Tensor& gradient,const std::vector<bool>& needs);
}
