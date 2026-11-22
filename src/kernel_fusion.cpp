#include "include/kernel_fusion.h"
namespace mapping_cuda {
namespace {thread_local bool l1_fusion_enabled=false;}
bool l1FusionEnabled(){return l1_fusion_enabled;}
void setL1FusionEnabled(bool value){l1_fusion_enabled=value;}
namespace {
class AnchorPack : public torch::autograd::Function<AnchorPack> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& xyz,const torch::Tensor& center,const torch::Tensor& reference,
      const torch::Tensor& feature,const torch::Tensor& gate,const torch::Tensor& mean,const torch::Tensor& stddev) {
    std::vector<torch::Tensor> a{xyz.contiguous(),center.contiguous(),reference.contiguous(),
        feature.contiguous(),gate.contiguous(),mean.contiguous(),stddev.contiguous()};
    TORCH_CHECK(!center.requires_grad()&&!reference.requires_grad()&&!mean.requires_grad()&&!stddev.requires_grad(),
        "Anchor pack requires fixed Anchor geometry and normalization");
    auto out=anchor128PackForward(a);a.push_back(out[1]);ctx->save_for_backward(a);return out[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    auto a=ctx->get_saved_variables();auto out=anchor128PackBackward(a,a[7],g[0].contiguous());
    // Keep the exact ATen broadcast-reduction kernel and original tanh VJP outside.
    return {out[0],{},{},out[1],out[2].sum(0,true),{},{}};
  }
};
class MaturityLoss : public torch::autograd::Function<MaturityLoss> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& w,
      const torch::Tensor& m,const torch::Tensor& e,double delta) {
    auto md=m.detach().contiguous(),ed=e.detach().contiguous();
    auto v=maturityPrepare(w,md,ed);
    // Preserve the original contiguous ATen sum reductions and their row order.
    auto pool_num=v[2].sum(),pool_den=v[3].sum(),num=v[1].sum(),den=v[0].sum();
    auto out=maturityFinish({pool_num,pool_den,num,den},float(delta));
    ctx->save_for_backward({md,ed,out[1]});return out[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    auto v=ctx->get_saved_variables();return {maturityLossVjp(v[0],v[1],v[2],g[0].contiguous()),{},{},{}};
  }
};
class L1Loss : public torch::autograd::Function<L1Loss> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& x,const torch::Tensor& y) {
    auto v=l1Prepare(x.contiguous(),y.contiguous());ctx->save_for_backward({v[1]});
    ctx->saved_data["target_gradient"]=y.requires_grad();return v[0].mean();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    return l1Vjp(ctx->get_saved_variables()[0],g[0].contiguous(),ctx->saved_data["target_gradient"].toBool());
  }
};
}
torch::Tensor anchor128Pack(const torch::Tensor& x,const torch::Tensor& c,const torch::Tensor& r,
    const torch::Tensor& f,const torch::Tensor& g,const torch::Tensor& m,const torch::Tensor& s) {
  return AnchorPack::apply(x,c,r,f,g,m,s);
}
torch::Tensor maturityLossFused(const torch::Tensor& w,const torch::Tensor& m,const torch::Tensor& e,float delta) {
  return MaturityLoss::apply(w,m,e,double(delta));
}
torch::Tensor fusedL1Loss(const torch::Tensor& x,const torch::Tensor& y){return L1Loss::apply(x,y);}
}
