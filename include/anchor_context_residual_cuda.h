#pragma once
#include <torch/torch.h>

namespace mapping_cuda {
// x11,w0,b0,w1,b1,w2,b2,projection32x17,global owners,slot eligibility.
std::vector<torch::Tensor> anchorContextMLP(const std::vector<torch::Tensor>& a,int64_t anchors);
// 1: cache ordered members/count/max ties; bit 2 fuses final VJP and Anchor sum;
// bit 4 packs integer cache metadata into otherwise unused high bits of members.
// The original entry point remains the numerical reference and rollback path.
std::vector<torch::Tensor> anchorContextMLPOptimized(const std::vector<torch::Tensor>& a,int64_t anchors,int64_t options=7);
std::vector<torch::Tensor> anchorContextMLPForward(const std::vector<torch::Tensor>& a,int64_t anchors);
std::vector<torch::Tensor> anchorContextMLPForwardOptimized(const std::vector<torch::Tensor>& a,int64_t anchors,int64_t options);
std::vector<torch::Tensor> anchorContextMLPBackward(const std::vector<torch::Tensor>& a,
    const torch::Tensor& logits_gradient,const torch::Tensor& shared_gradient);
std::vector<torch::Tensor> anchorContextMLPBackwardOptimized(const std::vector<torch::Tensor>& a,
    const torch::Tensor& logits_gradient,const torch::Tensor& shared_gradient,int64_t options);
}
