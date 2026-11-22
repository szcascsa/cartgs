#include "include/selector_cuda.h"
#include <torch/csrc/autograd/custom_function.h>

namespace selector_cuda {
namespace {
class NetworkFunction : public torch::autograd::Function<NetworkFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               at::TensorList tensors, int64_t candidate, int64_t strategy) {
    auto in=tensors.vec();
    TORCH_CHECK(in.size()==22,"Selector expects four inputs and eighteen parameters");
    auto stages=selector_cuda::forward({in.begin(),in.begin()+4},
        {in.begin()+4,in.end()},static_cast<Candidate>(candidate));
    ctx->saved_data["strategy"]=strategy;
    for(int i=0;i<8;++i) {
      // Second-layer activations are already present in the packed features.
      // Retain views for the new candidates instead of duplicate Mx32 storage.
      if(strategy && stages.size()>9 && i%2 && stages[i].size(0)==stages[9].size(0))
        in.push_back(stages[9].slice(1,(i/2)*32,(i/2+1)*32));
      else in.push_back(stages[i]);
    }
    if(stages.size()>9)in.push_back(stages[9]); // Strict final GEMM's packed input.
    ctx->save_for_backward(in);
    return stages[8];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list gradients) {
    auto saved=ctx->get_saved_variables();
    auto result=selector_cuda::backward({saved.begin(),saved.begin()+4},
        {saved.begin()+4,saved.begin()+22},{saved.begin()+22,saved.end()},gradients[0].contiguous(),
        static_cast<BackwardStrategy>(ctx->saved_data["strategy"].toInt()));
    result.push_back(torch::Tensor()); // Candidate enum has no gradient.
    result.push_back(torch::Tensor()); // Backward strategy has no gradient.
    return result;
  }
};
}  // namespace

torch::Tensor network(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& parameters, Candidate candidate, BackwardStrategy strategy) {
  std::vector<torch::Tensor> tensors=inputs;
  tensors.insert(tensors.end(),parameters.begin(),parameters.end());
  return NetworkFunction::apply(at::TensorList(tensors),static_cast<int64_t>(candidate),static_cast<int64_t>(strategy));
}
}  // namespace selector_cuda
