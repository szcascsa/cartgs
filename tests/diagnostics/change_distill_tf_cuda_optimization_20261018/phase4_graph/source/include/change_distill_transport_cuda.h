#pragma once
#include <torch/torch.h>
#include "include/gaussian_transform_ops.h"

namespace cd_transport_cuda {
torch::Tensor gatherScaledXYZ(const torch::Tensor& xyz,
                             const torch::Tensor& positions, float scale);
void scatter(const GaussianAttrState& input, const GaussianAttrState& output,
             const torch::Tensor& positions, const torch::Tensor& product,
             const GaussianTransformResult& correction);
}

namespace cd_transport_cuda {
torch::Tensor compactXYZ(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void scatterCompact(const GaussianAttrState&, const GaussianAttrState&,
                    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
}
