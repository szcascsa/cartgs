#pragma once
#include <torch/torch.h>
namespace mapping_cuda {
// Validates every original Variable Anchor condition, returns sorted [A,10]
// member rows. Invalid owner/capacity never cause an out-of-range access.
torch::Tensor validateAnchors(const torch::Tensor& owner,const torch::Tensor& counts,
    const torch::Tensor& levels,const torch::Tensor& centers,const torch::Tensor& scales);
void anchorFullStats(const torch::Tensor& owner,const torch::Tensor& members,
    const torch::Tensor& counts,const torch::Tensor& visible,const torch::Tensor& opacity,
    const std::vector<torch::Tensor>& statistics,const torch::Tensor& inactive={},
    const torch::Tensor& recovered={});
}
