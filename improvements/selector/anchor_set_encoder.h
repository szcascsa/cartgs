#pragma once
#include <torch/torch.h>
#include <limits>
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/anchor_set_statistics_cuda.h"
#endif

namespace improvements::selector {

// A forward consumes complete, pre-selection member sets. Owners are global
// Anchor rows; Gaussian order/occupancy may change at every topology update.
class AnchorSetEncoderImpl : public torch::nn::Module {
 public:
  explicit AnchorSetEncoderImpl(int64_t schema=2) : schema_(schema) {
    TORCH_CHECK(schema>=1 && schema<=3,"Unknown Anchor Set Encoder schema");
    if(schema!=3)network=register_module("network",torch::nn::Sequential(
        torch::nn::Linear(10,16),torch::nn::ReLU(),
        torch::nn::Linear(16,16),torch::nn::ReLU()));
  }
  torch::Tensor forward(const torch::Tensor& xyz,const torch::Tensor& scale,
      const torch::Tensor& dc,const torch::Tensor& opacity,const torch::Tensor& owners,
      const torch::Tensor& offset_scales,const torch::Tensor& ratio,
      const torch::Tensor& centers,const torch::Tensor& eligible,bool fused=false) {
    const auto n=xyz.size(0),a=offset_scales.size(0);
    const auto color=dc.dim()==3?dc.squeeze(1):dc;
    TORCH_CHECK(xyz.dim()==2 && xyz.size(1)==3 && scale.sizes()==xyz.sizes() &&
        color.sizes()==xyz.sizes() && opacity.sizes()==torch::IntArrayRef({n,1}) &&
        owners.sizes()==torch::IntArrayRef({n}) && owners.scalar_type()==torch::kLong &&
        offset_scales.sizes()==torch::IntArrayRef({a,1}) && centers.sizes()==torch::IntArrayRef({a,3}) &&
        eligible.sizes()==torch::IntArrayRef({n}) && eligible.scalar_type()==torch::kBool,
        "Anchor set member attributes/owners/eligibility must align");
    if(schema_==3) {
#ifdef CARTGS_FUSED_MAPPING_CUDA
      if(fused && xyz.is_cuda() && xyz.scalar_type()==torch::kFloat32 && ratio.numel()==1)
        return mapping_cuda::anchorSetStatistics21({xyz,scale,dc,opacity,owners,centers,offset_scales,eligible,ratio});
#endif
      const auto valid=eligible.unsqueeze(1);
      const auto count=torch::zeros({a,1},xyz.options()).index_add(0,owners,valid.to(xyz.scalar_type()));
      const auto divisor=count.clamp_min(1);
      const auto local=((xyz-centers.index_select(0,owners))/offset_scales.index_select(0,owners)).masked_fill(~valid,0);
      const auto mean=torch::zeros({a,3},xyz.options()).index_add(0,owners,local)/divisor;
      const auto d=(local-mean.index_select(0,owners)).masked_fill(~valid,0);
      const auto x=d.select(1,0),y=d.select(1,1),z=d.select(1,2);
      const auto moments=torch::stack({x*x,x*y,x*z,y*y,y*z,z*z},1);
      const auto covariance=torch::zeros({a,6},xyz.options()).index_add(0,owners,moments)/divisor;
      const auto context=torch::cat({mean,covariance,count/10.},1).index_select(0,owners);
      return torch::cat({torch::cat({local,scale,color,opacity},1).masked_fill(~valid,0),
          ratio.log().expand({n,1}),context},1);
    }
    const auto valid=schema_==1?torch::ones_like(eligible):eligible;
    const auto count=torch::zeros({a,1},xyz.options()).index_add(0,owners,valid.to(xyz.scalar_type()).unsqueeze(1));
    const auto divisor=count.clamp_min(1);
    const auto origin=schema_==1?torch::zeros({a,3},xyz.options()).index_add(0,owners,xyz)/divisor:centers;
    const auto local=(xyz-origin.index_select(0,owners))/offset_scales.index_select(0,owners);
    auto input=torch::cat({local,scale,color,opacity},1);
    if(schema_==2)input=input.masked_fill(~valid.unsqueeze(1),0);
    auto hidden=network->forward(input);
    if(schema_==2)hidden=hidden.masked_fill(~valid.unsqueeze(1),0);
    const auto mean=torch::zeros({a,16},xyz.options()).index_add(0,owners,hidden)/divisor;
    const auto max_input=schema_==1?hidden:hidden.masked_fill(~valid.unsqueeze(1),-std::numeric_limits<float>::infinity());
    auto maximum=torch::zeros_like(mean).scatter_reduce(0,owners.unsqueeze(1).expand({n,16}),max_input,"amax",false);
    if(schema_==2)maximum=torch::where(count>0,maximum,torch::zeros_like(maximum));
    const auto context=torch::cat({mean,maximum,count/10.},1).index_select(0,owners);
    // One ratio for this LoD pass, with the original log-ratio conditioning.
    return torch::cat({hidden,context,ratio.log().expand({n,1})},1);
  }
  int64_t schema() const { return schema_; }
  torch::nn::Sequential network{nullptr};
 private:
  int64_t schema_;
};
TORCH_MODULE(AnchorSetEncoder);

}  // namespace improvements::selector
