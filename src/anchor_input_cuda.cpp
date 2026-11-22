#include "include/anchor_input_cuda.h"

namespace mapping_cuda {
namespace {
class Input27 : public torch::autograd::Function<Input27> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& xyz,const torch::Tensor& feature,const torch::Tensor& owner,
      const torch::Tensor& center,const torch::Tensor& reference_scale,const torch::Tensor& scale,
      const torch::Tensor& dc,const torch::Tensor& opacity,double ratio) {
    std::vector<torch::Tensor> a{xyz.contiguous(),feature.contiguous(),owner.contiguous(),
        center.contiguous(),reference_scale.contiguous(),scale.contiguous(),dc.contiguous(),opacity.contiguous()};
    ctx->save_for_backward(a);
    ctx->saved_data["needs"]=std::vector<bool>{xyz.requires_grad(),feature.requires_grad(),false,
        false,false,scale.requires_grad(),dc.requires_grad(),opacity.requires_grad()};
    return anchorInput27Forward(a,ratio);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list gradients) {
    auto result=anchorInput27Backward(ctx->get_saved_variables(),gradients[0].contiguous(),
        ctx->saved_data["needs"].toBoolList().vec());
    result.push_back({}); // ratio is not a differentiable input.
    return result;
  }
};
}
torch::Tensor anchorInput27(const torch::Tensor& xyz,const torch::Tensor& feature,
    const torch::Tensor& owner,const torch::Tensor& center,const torch::Tensor& reference_scale,
    const torch::Tensor& scale,const torch::Tensor& dc,const torch::Tensor& opacity,double ratio) {
  return Input27::apply(xyz,feature,owner,center,reference_scale,scale,dc,opacity,ratio);
}
}
