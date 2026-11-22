#include "include/selector_sampling_cuda.h"
namespace mapping_cuda {
namespace {
class SampleCompact : public torch::autograd::Function<SampleCompact> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& probability,const torch::Tensor& uniform,const torch::Tensor& opacity,const torch::Tensor& inactive){
    auto out=selectorSamplingForward(probability,uniform,opacity,inactive);ctx->save_for_backward({out[3]});
    ctx->mark_non_differentiable({out[2]});ctx->set_materialize_grads(false);out.resize(3);return out;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g){
    return {selectorSamplingBackward(ctx->get_saved_variables()[0],g[1].defined()?g[1].contiguous():g[1],g[0].defined()?g[0].contiguous():g[0]),{},{},{}};
  }
};
}
std::vector<torch::Tensor> selectorSampleCompact(const torch::Tensor& p,const torch::Tensor& u,const torch::Tensor& opacity,const torch::Tensor& inactive){
  return SampleCompact::apply(p,u,opacity.defined()?opacity:torch::empty({0},p.options()),
      inactive.defined()?inactive:torch::empty({0},p.options().dtype(torch::kBool)));
}
}
