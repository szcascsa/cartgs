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
    const std::vector<torch::Tensor>& parameters, Candidate candidate);
std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& parameters,
    const std::vector<torch::Tensor>& activations, const torch::Tensor& upstream,
    BackwardStrategy strategy = BackwardStrategy::Current);
torch::Tensor network(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& parameters, Candidate candidate,
    BackwardStrategy strategy = BackwardStrategy::DirectFinalRelu);

}  // namespace selector_cuda
