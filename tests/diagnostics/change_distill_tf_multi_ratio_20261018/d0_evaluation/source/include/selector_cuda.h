#pragma once

#include <torch/torch.h>
#include <vector>

namespace selector_cuda {

// Candidates remain diagnostic until the complete acceptance gate passes.
enum class Candidate : int64_t { Cublas = 0, Warp = 1, CublasStrict = 2 };
enum class BackwardStrategy : int64_t {
  Current = 0, FusedActivation = 1, BatchedWeight = 2, FusedBias = 3, GroupedBranch = 4,
  DirectFinal = 5, DirectFinalRelu = 6
};

// Input order: XYZ, activated quaternion, activated scale, ratio ([1,1] or [M,1]).
// Parameter order is the existing module registration order, never packed copies.
// Forward returns the eight ReLU activations followed by two-class logits.
std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& parameters, Candidate candidate,
    const std::vector<torch::Tensor>& spatial = {});
std::vector<torch::Tensor> spatialForward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& parameters);
std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& parameters,
    const std::vector<torch::Tensor>& activations, const torch::Tensor& upstream,
    BackwardStrategy strategy = BackwardStrategy::Current);
torch::Tensor network(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& parameters, Candidate candidate,
    BackwardStrategy strategy = BackwardStrategy::DirectFinalRelu);

// Reuse only this call's spatial forward activations. Each ratio retains the
// existing strict final GEMM and backward, including parameter-gradient sums.
std::vector<torch::Tensor> networkMany(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& ratios, const std::vector<torch::Tensor>& parameters);

}  // namespace selector_cuda
