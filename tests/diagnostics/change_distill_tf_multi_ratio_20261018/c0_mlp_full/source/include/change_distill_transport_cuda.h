#pragma once
#include "include/gaussian_transform_ops.h"

namespace cd_transport_cuda {
// positions and groups are aligned contiguous int64 vectors in stable group order.
torch::Tensor compactXYZ(const torch::Tensor& xyz, const torch::Tensor& positions,
                         const torch::Tensor& groups, const torch::Tensor& metadata);
void scatterCompact(const GaussianAttrState& input, const GaussianAttrState& output,
                    const torch::Tensor& positions, const torch::Tensor& groups,
                    const torch::Tensor& metadata, const torch::Tensor& product);
}
