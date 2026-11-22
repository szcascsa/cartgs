#pragma once
#include <torch/torch.h>

void advanceProtectionClockCUDA(const torch::Tensor& birth, const torch::Tensor& optimized,
    const torch::Tensor& unique, const torch::Tensor& transitions, int64_t before, int64_t after,
    int age_ready, int optimization_ready, int keyframe_ready, int stale_age);
void commitProtectionEvidenceCUDA(const torch::Tensor& visible, const torch::Tensor& birth,
    const torch::Tensor& optimized, const torch::Tensor& unique, const torch::Tensor& ids,
    const torch::Tensor& transitions, int64_t iteration, int64_t keyframe,
    int age_ready, int optimization_ready, int stale_age);
