#include "include/anchor_set_statistics_cuda.h"

namespace mapping_cuda {
namespace {
class Statistics21 : public torch::autograd::Function<Statistics21> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& x,
      const torch::Tensor& s,const torch::Tensor& dc,const torch::Tensor& op,const torch::Tensor& owner,
      const torch::Tensor& center,const torch::Tensor& ref,const torch::Tensor& eligible,const torch::Tensor& ratio) {
    std::vector<torch::Tensor> a{x.contiguous(),s.contiguous(),dc.contiguous(),op.contiguous(),
        owner.contiguous(),center.contiguous(),ref.contiguous(),eligible.contiguous(),ratio.contiguous()};
    auto out=anchorSetStatisticsForward(a);
    a.insert(a.end(),out.begin()+1,out.end());ctx->save_for_backward(a);
    return out[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list g) {
    return anchorSetStatisticsBackward(ctx->get_saved_variables(),g[0].contiguous());
  }
};
}
torch::Tensor anchorSetStatistics21(const std::vector<torch::Tensor>& a) {
  TORCH_CHECK(a.size()==9,"Statistics21 input count");
  return Statistics21::apply(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8]);
}
}
