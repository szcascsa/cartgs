#pragma once
#include <torch/torch.h>
#include <vector>
#include "protection_profiler.h"

namespace tf_cuda {
// Registered parameters and ordinary AdamParamState tensors remain authoritative.
// The same descriptor will support read-only preparation with separate outputs.
struct AdamUpdate {
  torch::Tensor parameter,gradient,old_mean,old_variance,mean,variance,proposed,delta;
  float beta1,beta2,alpha1,alpha2,inverse_bias2,step_size,eps,weight_decay;
};
void adamUpdate(const std::vector<AdamUpdate>& updates);
void tfCudaAdamDirect(torch::optim::Adam& optimizer,
                      const std::vector<std::size_t>& groups);
void tfCudaAdamMixedStep(torch::optim::Adam& optimizer,
                         const std::vector<std::size_t>& groups,
                         ProtectionProfiler* profiler = nullptr);
struct AdamPreparedTensor {
  torch::Tensor parameter, proposed, delta;
  std::unique_ptr<torch::optim::AdamParamState> state;
  std::size_t group;
  uint32_t parameter_version;
  const torch::optim::OptimizerParamState* previous_state;
};
using AdamPreparation = std::vector<AdamPreparedTensor>;
AdamPreparation tfCudaAdamPrepare(torch::optim::Adam& optimizer,
                                  const std::vector<std::size_t>& groups);
void tfCudaAdamValidate(torch::optim::Adam& optimizer,const AdamPreparation& prepared,
                        const std::vector<torch::Tensor>& candidate);
void tfCudaAdamCommit(torch::optim::Adam& optimizer,AdamPreparation& prepared,
                      const std::vector<torch::Tensor>& candidate);
void tfCudaAdamDiscard(AdamPreparation& prepared);
std::vector<torch::Tensor> adamCandidate(const std::vector<torch::Tensor>& parameters,
                                       const std::vector<torch::Tensor>& direction,double alpha);
void adamCopy(const std::vector<torch::Tensor>& parameters,
              const std::vector<torch::Tensor>& candidate);
}  // namespace tf_cuda
