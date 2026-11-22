#include "include/anchor_selector_cuda.h"

namespace mapping_cuda {
namespace {
template<int ScaleMode>
class Decode14 : public torch::autograd::Function<Decode14<ScaleMode>> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& opacity_logit,const torch::Tensor& dc,const torch::Tensor& delta,
      const torch::Tensor& mask) {
    std::vector<torch::Tensor> inputs{xyz.contiguous(),scale.contiguous(),rotation.contiguous(),
        opacity_logit.contiguous(),dc.contiguous(),delta.contiguous(),mask.contiguous()};
    ctx->save_for_backward(inputs);
    ctx->set_materialize_grads(false);
    if(ScaleMode==3)return decode14SoftplusScaleForward(inputs);
    return ScaleMode==2?decode14AdditiveScaleForward(inputs):(ScaleMode==1?decode14LogScaleForward(inputs):decode14Forward(inputs));
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list gradients) {
    for(auto& g:gradients)if(g.defined())g=g.contiguous();
    if(ScaleMode==3)return decode14SoftplusScaleBackward(ctx->get_saved_variables(),gradients);
    return ScaleMode==2?decode14AdditiveScaleBackward(ctx->get_saved_variables(),gradients):(ScaleMode==1?decode14LogScaleBackward(ctx->get_saved_variables(),gradients):decode14Backward(ctx->get_saved_variables(),gradients));
  }
};
}
std::vector<torch::Tensor> decode14(const torch::Tensor& xyz,const torch::Tensor& scale,
    const torch::Tensor& rotation,const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask) {
  return Decode14<0>::apply(xyz,scale,rotation,opacity_logit,dc,delta,mask);
}
std::vector<torch::Tensor> decode14AdditiveScale(const torch::Tensor& xyz,const torch::Tensor& log_scale,
    const torch::Tensor& rotation,const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask) {
  return Decode14<2>::apply(xyz,log_scale,rotation,opacity_logit,dc,delta,mask);
}
std::vector<torch::Tensor> decode14LogScale(const torch::Tensor& xyz,const torch::Tensor& log_scale,
    const torch::Tensor& rotation,const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask) {
  return Decode14<1>::apply(xyz,log_scale,rotation,opacity_logit,dc,delta,mask);
}
std::vector<torch::Tensor> decode14SoftplusScale(const torch::Tensor& xyz,const torch::Tensor& log_scale,
    const torch::Tensor& rotation,const torch::Tensor& opacity_logit,const torch::Tensor& dc,
    const torch::Tensor& delta,const torch::Tensor& mask) {
  return Decode14<3>::apply(xyz,log_scale,rotation,opacity_logit,dc,delta,mask);
}
}
