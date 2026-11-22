#include "include/flexgs_selector_cuda.h"

namespace mapping_cuda {
namespace {
class Input11 : public torch::autograd::Function<Input11> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& x,
      const torch::Tensor& s,const torch::Tensor& dc,const torch::Tensor& op,const torch::Tensor& owner,
      const torch::Tensor& center,const torch::Tensor& ref,double ratio) {
    std::vector<torch::Tensor> a{x.contiguous(),s.contiguous(),dc.contiguous(),op.contiguous(),
        owner.contiguous(),center.contiguous(),ref.contiguous()};
    ctx->save_for_backward(a);return flexgsInputForward(a,ratio)[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    auto out=flexgsInputBackward(ctx->get_saved_variables(),g[0].contiguous());out.resize(8);return out;
  }
};
class MLP : public torch::autograd::Function<MLP> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,const torch::Tensor& w0,const torch::Tensor& b0,const torch::Tensor& w1,
      const torch::Tensor& b1,const torch::Tensor& w2,const torch::Tensor& b2) {
    std::vector<torch::Tensor> a{x.contiguous(),w0,b0,w1,b1,w2,b2};
    auto out=flexgsMLPForward(a);a.push_back(out[1]);a.push_back(out[2]);ctx->save_for_backward(a);
    ctx->set_materialize_grads(false);return {out[0],out[1]};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    for(auto& t:g)if(t.defined())t=t.contiguous();
    return flexgsMLPBackward(ctx->get_saved_variables(),g[0],g[1]);
  }
};
class Sample : public torch::autograd::Function<Sample> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& logits,
      const torch::Tensor& exponentials,const torch::Tensor& eligible,double tau) {
    auto out=flexgsSampleForward(logits,exponentials,eligible,tau);
    ctx->save_for_backward({out[3],out[4],eligible});ctx->saved_data["tau"]=tau;
    ctx->mark_non_differentiable({out[2]});ctx->set_materialize_grads(false);out.resize(3);return out;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    for(int i=0;i<2;++i)if(g[i].defined())g[i]=g[i].contiguous();
    return {flexgsSampleBackward(ctx->get_saved_variables(),g[0],g[1],ctx->saved_data["tau"].toDouble()),{},{},{}};
  }
};
class SampleWeights : public torch::autograd::Function<SampleWeights> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& logits,
      const torch::Tensor& exponentials,const torch::Tensor& eligible,double tau) {
    auto out=flexgsSampleForward(logits,exponentials,eligible,tau);
    ctx->save_for_backward({out[3],out[4],eligible});ctx->saved_data["tau"]=tau;
    ctx->mark_non_differentiable({out[2]});ctx->set_materialize_grads(false);
    return {out[0],out[1],out[2],out[3]};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    auto saved=ctx->get_saved_variables();const double tau=ctx->saved_data["tau"].toDouble();torch::Tensor result;
    if(g[0].defined()||g[1].defined()) {
      for(int i=0;i<2;++i)if(g[i].defined())g[i]=g[i].contiguous();
      result=flexgsSampleBackward(saved,g[0],g[1],tau);
    }
    if(g[3].defined()) {
      auto vjp=at::_softmax_backward_data(g[3],saved[0],-1,g[3].scalar_type())/tau;
      result=result.defined()?result+vjp:vjp;
    }
    return {result,{},{},{}};
  }
};
}
torch::Tensor flexgsInput11(const torch::Tensor& x,const torch::Tensor& s,const torch::Tensor& dc,
    const torch::Tensor& op,const torch::Tensor& owner,const torch::Tensor& c,const torch::Tensor& ref,double ratio) {
  return Input11::apply(x,s,dc,op,owner,c,ref,ratio);
}
std::vector<torch::Tensor> flexgsMLP(const torch::Tensor& x,const torch::Tensor& w0,const torch::Tensor& b0,
    const torch::Tensor& w1,const torch::Tensor& b1,const torch::Tensor& w2,const torch::Tensor& b2) {
  return MLP::apply(x,w0,b0,w1,b1,w2,b2);
}
std::vector<torch::Tensor> flexgsSampleCompact(const torch::Tensor& logits,const torch::Tensor& exp,
    const torch::Tensor& eligible,double tau) {return Sample::apply(logits,exp,eligible,tau);}
std::vector<torch::Tensor> flexgsSampleCompactWithWeights(const torch::Tensor& logits,const torch::Tensor& exp,
    const torch::Tensor& eligible,double tau) {return SampleWeights::apply(logits,exp,eligible,tau);}
}
