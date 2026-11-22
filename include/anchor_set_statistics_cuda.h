#pragma once
#include <torch/torch.h>

namespace mapping_cuda {
// Input order: XYZ, activated Scale, DC, activated Opacity, owners, centers,
// offsetScales, detached slot eligibility, one non-trainable ratio.
torch::Tensor anchorSetStatistics21(const std::vector<torch::Tensor>& inputs);
std::vector<torch::Tensor> anchorSetStatisticsForward(const std::vector<torch::Tensor>& inputs);
std::vector<torch::Tensor> anchorSetStatisticsBackward(const std::vector<torch::Tensor>& saved,
    const torch::Tensor& gradient);
}
