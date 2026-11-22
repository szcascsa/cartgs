#pragma once
#include <torch/torch.h>
#include "protection_config.h"

// One row kernel: support, exact distinct-view bitset, drift, maturity and snapshot.
void reliabilityMaturityUpdate(const std::vector<torch::Tensor>& state,
    const torch::Tensor& xyz, const torch::Tensor& scale, const torch::Tensor& rotation,
    const torch::Tensor& radii, const std::vector<torch::Tensor>& gradients,
    int64_t view_ordinal, const ProtectionConfig& config,
    const torch::Tensor& contribution={},int64_t sample_pixels=0);
