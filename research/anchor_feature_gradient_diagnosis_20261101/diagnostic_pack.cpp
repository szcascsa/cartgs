#include "diagnostic_pack.h"

namespace diagnostic_cuda {
namespace {
class ResidualPack : public torch::autograd::Function<ResidualPack> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
      const torch::Tensor& xyz,const torch::Tensor& center,const torch::Tensor& reference,
      const torch::Tensor& mean,const torch::Tensor& stddev,
      const torch::Tensor& f0,const torch::Tensor& f1,const torch::Tensor& f2,
      const torch::Tensor& f3,const torch::Tensor& f4,const torch::Tensor& f5,std::vector<double> weights) {
    std::vector<torch::Tensor> features{f0,f1,f2,f3,f4,f5};features.resize(weights.size());
    auto out=residual48PackForward(xyz,center,reference,mean,stddev,features,weights);
    ctx->save_for_backward({reference,stddev,out[1]});ctx->saved_data["weights"]=weights;
    return out[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list grad) {
    auto saved=ctx->get_saved_variables();auto weights=ctx->saved_data["weights"].toDoubleVector();
    auto xyz=residual48XYZBackward(saved[0],saved[1],saved[2],grad[0]);
    auto feature_gradient=grad[0].slice(1,3,51);
    // Preserve original cat view and scalar-mul VJP; original index_select nodes scatter to each table.
    torch::autograd::variable_list out{xyz,{},{},{},{},feature_gradient,{},{},{},{},{},{}};
    for(size_t j=1;j<weights.size();++j)out[5+j]=feature_gradient*weights[j];
    return out;
  }
};
}

torch::Tensor residual48Pack(const torch::Tensor& xyz,const torch::Tensor& center,
    const torch::Tensor& reference,const torch::Tensor& mean,const torch::Tensor& stddev,
    const std::vector<torch::Tensor>& features,const std::vector<double>& weights) {
  TORCH_CHECK(features.size()>=1&&features.size()<=6&&features.size()==weights.size(),"Residual48 prefix inputs");
  if(!xyz.is_cuda()||xyz.scalar_type()!=torch::kFloat32||center.requires_grad()||reference.requires_grad()||mean.requires_grad()||stddev.requires_grad()) {
    auto value=features[0];for(size_t j=1;j<features.size();++j)value=value+weights[j]*features[j];
    return torch::cat({((torch::asinh((xyz-center)/reference)-mean.slice(0,3,6))/stddev.slice(0,3,6)).clamp(-15,15),value},1);
  }
  auto f=features;for(auto& t:f)t=t.contiguous();
  while(f.size()<6)f.push_back(torch::empty({0},xyz.options()));
  return ResidualPack::apply(xyz.contiguous(),center.contiguous(),reference.contiguous(),mean.contiguous(),stddev.contiguous(),
                            f[0],f[1],f[2],f[3],f[4],f[5],weights);
}
}
