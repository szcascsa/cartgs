#include "include/spatial_compensation_cuda.h"
namespace mapping_cuda {
namespace {
class Input : public torch::autograd::Function<Input> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& raw,
      const torch::Tensor& coords,const torch::Tensor& mean,const torch::Tensor& stddev,const torch::Tensor& freq) {
    std::vector<torch::Tensor> values{raw.contiguous(),coords.contiguous(),mean.contiguous(),stddev.contiguous(),freq.contiguous()};
    TORCH_CHECK(!mean.requires_grad()&&!stddev.requires_grad()&&!freq.requires_grad(),"Spatial normalization buffers must be fixed");
    ctx->save_for_backward(values);return spatialInputForward(values);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list gradients) {
    auto grads=spatialInputBackward(ctx->get_saved_variables(),gradients[0].contiguous());return {grads[0],grads[1],{},{},{}};
  }
};
class Film : public torch::autograd::Function<Film> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& h,const torch::Tensor& gain,const torch::Tensor& bias) {
    std::vector<torch::Tensor> values{h.contiguous(),gain.contiguous(),bias.contiguous()};
    auto result=spatialFilmForward(values);values.push_back(result);ctx->save_for_backward(values);return result;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list gradients) {
    auto values=ctx->get_saved_variables();auto result=spatialFilmBackward(values,values[3],gradients[0].contiguous());
    return {result[0],result[1].sum(0,true),result[2].sum(0,true)};
  }
};
template<bool LogScale>
class Live : public torch::autograd::Function<Live<LogScale>> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,const torch::Tensor& xyz,
      const torch::Tensor& scale,const torch::Tensor& q,const torch::Tensor& dc,const torch::Tensor& logits,
      const torch::Tensor& center,const torch::Tensor& reference,const torch::Tensor& wc,const torch::Tensor& ws,
      const torch::Tensor& mean,const torch::Tensor& stddev) {
    std::vector<torch::Tensor> a;for(const auto& t:{xyz,scale,q,dc,logits,center,reference,wc,ws,mean,stddev})a.push_back(t.contiguous());
    for(int j=7;j<11;++j)TORCH_CHECK(!a[j].requires_grad(),"Spatial population statistics must be fixed");
    auto result=LogScale?spatialLiveLogScaleForward(a):spatialLiveForward(a);a.push_back(result[1]);ctx->save_for_backward(a);return result[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list gradients) {
    auto a=ctx->get_saved_variables();auto g=LogScale?spatialLiveLogScaleBackward(a,a[11],gradients[0].contiguous()):spatialLiveBackward(a,a[11],gradients[0].contiguous());
    g.resize(11);return g;
  }
};
}
torch::Tensor spatialInput55(const torch::Tensor& raw,const torch::Tensor& coords,const torch::Tensor& mean,const torch::Tensor& stddev,const torch::Tensor& freq){return Input::apply(raw,coords,mean,stddev,freq);}
torch::Tensor spatialFilm(const torch::Tensor& h,const torch::Tensor& gain,const torch::Tensor& bias){return Film::apply(h,gain,bias);}
torch::Tensor spatialLiveInput55(const std::vector<torch::Tensor>& a){return Live<false>::apply(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10]);}
torch::Tensor spatialLiveLogScaleInput55(const std::vector<torch::Tensor>& a){return Live<true>::apply(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10]);}
}
