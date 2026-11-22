#include "include/anchor_context_residual_cuda.h"

namespace mapping_cuda {
namespace {
class ContextMLP : public torch::autograd::Function<ContextMLP> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,const torch::Tensor& w0,const torch::Tensor& b0,const torch::Tensor& w1,
      const torch::Tensor& b1,const torch::Tensor& w2,const torch::Tensor& b2,const torch::Tensor& p,
      const torch::Tensor& owners,const torch::Tensor& eligible,int64_t anchors) {
    std::vector<torch::Tensor> a{x.contiguous(),w0.contiguous(),b0.contiguous(),w1.contiguous(),b1.contiguous(),
        w2.contiguous(),b2.contiguous(),p.contiguous(),owners.contiguous(),eligible.contiguous()};
    auto out=anchorContextMLPForward(a,anchors);
    a.insert(a.end(),out.begin()+1,out.end());ctx->save_for_backward(a);ctx->set_materialize_grads(false);
    return {out[0],out[1]};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    for(auto& t:g)if(t.defined())t=t.contiguous();
    auto out=anchorContextMLPBackward(ctx->get_saved_variables(),g[0],g[1]);out.resize(11);return out;
  }
};
class CachedContextMLP : public torch::autograd::Function<CachedContextMLP> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,const torch::Tensor& w0,const torch::Tensor& b0,const torch::Tensor& w1,
      const torch::Tensor& b1,const torch::Tensor& w2,const torch::Tensor& b2,const torch::Tensor& p,
      const torch::Tensor& owners,const torch::Tensor& eligible,int64_t anchors,int64_t options) {
    std::vector<torch::Tensor> a{x.contiguous(),w0.contiguous(),b0.contiguous(),w1.contiguous(),b1.contiguous(),
        w2.contiguous(),b2.contiguous(),p.contiguous(),owners.contiguous(),eligible.contiguous()};
    auto out=anchorContextMLPForwardOptimized(a,anchors,options);
    a.insert(a.end(),out.begin()+1,out.end());ctx->save_for_backward(a);ctx->set_materialize_grads(false);
    ctx->saved_data["options"]=options;return {out[0],out[1]};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    for(auto& t:g)if(t.defined())t=t.contiguous();
    auto out=anchorContextMLPBackwardOptimized(ctx->get_saved_variables(),g[0],g[1],ctx->saved_data["options"].toInt());
    out.resize(12);return out;
  }
};
}
std::vector<torch::Tensor> anchorContextMLP(const std::vector<torch::Tensor>& a,int64_t anchors) {
  TORCH_CHECK(a.size()==10,"ContextMLP input count");
  return ContextMLP::apply(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],anchors);
}
std::vector<torch::Tensor> anchorContextMLPOptimized(const std::vector<torch::Tensor>& a,int64_t anchors,int64_t options) {
  TORCH_CHECK(a.size()==10 && (options==1 || options==3 || options==5 || options==7),"ContextMLP optimization options must be 1,3,5,7");
  if(!torch::GradMode::is_enabled())return anchorContextMLP(a,anchors);
  return CachedContextMLP::apply(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],anchors,options);
}
}
