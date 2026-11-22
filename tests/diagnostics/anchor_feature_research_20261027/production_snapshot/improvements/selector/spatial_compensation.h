#pragma once
#include <torch/torch.h>
#include <cmath>
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/spatial_compensation_cuda.h"
#endif

namespace improvements::selector {
// Shared SpatialLocalXYZ field. Only population-independent statistics are buffers.
class SpatialCompensationImpl : public torch::nn::Module {
 public:
  SpatialCompensationImpl() {
    encode=register_module("encode",torch::nn::Linear(55,64));
    body=register_module("body",torch::nn::Linear(64,64));
    ratio_net=register_module("ratio",torch::nn::Sequential(torch::nn::Linear(1,32),torch::nn::ReLU(),torch::nn::Linear(32,256)));
    heads=register_module("heads",torch::nn::ModuleDict());
    for(const auto& entry:std::vector<std::pair<std::string,int64_t>>{{"xyz",3},{"scale",3},{"rotation",4},{"other",4}}) {
      auto head=torch::nn::Sequential(torch::nn::Linear(64,64),torch::nn::ReLU(),torch::nn::Linear(64,entry.second));
      {torch::NoGradGuard guard;head->at<torch::nn::LinearImpl>(2).weight.zero_();head->at<torch::nn::LinearImpl>(2).bias.zero_();}
      heads->update(std::vector<std::pair<std::string,std::shared_ptr<torch::nn::Module>>>{{entry.first,head.ptr()}});
    }
    world_center=register_buffer("world_center",torch::zeros({3}));
    world_spread=register_buffer("world_spread",torch::ones({3}));
    mean=register_buffer("mean",torch::zeros({19}));stddev=register_buffer("stddev",torch::ones({19}));
    ready_buffer=register_buffer("normalization_ready",torch::tensor(0,torch::kLong));
  }
  void setFused(bool enabled){fused_=enabled;}
  void setLiveFused(bool enabled){live_fused_=enabled;}
  bool ready() const{return ready_;}
  void syncReady(){ready_=ready_buffer.item<int64_t>()!=0;}
  void load(torch::serialize::InputArchive& archive) override {
    torch::nn::Module::load(archive);ready_=ready_buffer.item<int64_t>()!=0;
  }
  static torch::Tensor physical(const torch::Tensor& xyz,const torch::Tensor& scale,
      const torch::Tensor& rotation,const torch::Tensor& dc,const torch::Tensor& logits,
      const torch::Tensor& anchor_center,const torch::Tensor& anchor_scale,
      const torch::Tensor& world_center,const torch::Tensor& world_spread) {
    auto q=rotation.unbind(1);auto w=q[0],x=q[1],y=q[2],z=q[3];
    auto R=torch::stack({1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),
        1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)},1).reshape({-1,3,3});
    auto cov=(R*scale.square().unsqueeze(1)).matmul(R.transpose(1,2));
    cov=cov/cov.diagonal(0,1,2).sum(1).clamp_min(1e-12).reshape({-1,1,1});
    auto shape=torch::stack({cov.select(1,0).select(1,0),cov.select(1,1).select(1,1),cov.select(1,2).select(1,2),
        cov.select(1,0).select(1,1),cov.select(1,0).select(1,2),cov.select(1,1).select(1,2)},1);
    return torch::cat({(xyz-world_center)/world_spread,torch::asinh((xyz-anchor_center)/anchor_scale),
        scale.log(),dc.reshape({xyz.size(0),3}),logits,shape},1);
  }
  void initialize(const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& dc,const torch::Tensor& logits,const torch::Tensor& anchor_center,const torch::Tensor& anchor_scale) {
    if(ready_)return;
    TORCH_CHECK(xyz.size(0)>0,"Spatial normalization requires a nonempty map");
    torch::NoGradGuard guard;
    world_center.copy_(std::get<0>(xyz.median(0)));
    world_spread.copy_((xyz-world_center).square().mean(0).sqrt().clamp_min(.1));
    auto raw=physical(xyz,scale,rotation,dc,logits,anchor_center,anchor_scale,world_center,world_spread);
    mean.copy_(raw.mean(0));
    // Same sample std as the frozen reference; define the single-point online boundary.
    stddev.copy_(xyz.size(0)>1?raw.std(std::vector<int64_t>{0},true,false).clamp_min(.1):torch::full_like(mean,.1));
    ready_buffer.fill_(1);ready_=true;
  }
  torch::Tensor input55(const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& dc,const torch::Tensor& logits,const torch::Tensor& center,const torch::Tensor& reference) {
    TORCH_CHECK(ready_,"Initialize Spatial normalization from the complete current map first");
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused_&&live_fused_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32)
      return mapping_cuda::spatialLiveInput55({xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread,mean,stddev});
#endif
    auto raw=physical(xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread);
    auto coords=(xyz-world_center)/world_spread;
    auto freq=torch::pow(2,torch::arange(6,xyz.options()))*M_PI;
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32)
      return mapping_cuda::spatialInput55(raw,coords,mean,stddev,freq);
#endif
    auto angle=coords.unsqueeze(2)*freq;
    return torch::cat({((raw-mean)/stddev).clamp(-15,15),angle.sin().flatten(1),angle.cos().flatten(1)},1);
  }
  torch::Tensor forward(const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& dc,const torch::Tensor& logits,const torch::Tensor& center,const torch::Tensor& reference,double ratio) {
    TORCH_CHECK(std::isfinite(ratio)&&ratio>0&&ratio<=1,"Invalid Spatial ratio");
    auto input=input55(xyz,scale,rotation,dc,logits,center,reference);
    auto condition=torch::full({1,1},2*(std::log(ratio)-std::log(.01))/(std::log(.15)-std::log(.01))-1,xyz.options());
    auto modulation=ratio_net->forward(condition).chunk(4,1);
    auto film=[&](const torch::Tensor& h,const torch::Tensor& g,const torch::Tensor& b){
#ifdef CARTGS_FUSED_MAPPING_CUDA
      if(fused_&&h.is_cuda()&&h.scalar_type()==torch::kFloat32)return mapping_cuda::spatialFilm(h,g,b);
#endif
      return torch::relu((1+g)*h+b);
    };
    auto hidden=film(encode->forward(input),modulation[0],modulation[1]);
    hidden=film(body->forward(hidden),modulation[2],modulation[3]);
    std::vector<torch::Tensor> result;
    for(const auto& name:{"xyz","scale","rotation","other"})result.push_back(heads->at<torch::nn::SequentialImpl>(name).forward(hidden));
    result[0]=result[0]*reference;
    return torch::cat(result,1);
  }
  torch::nn::Linear encode{nullptr},body{nullptr};
  torch::nn::Sequential ratio_net{nullptr};torch::nn::ModuleDict heads{nullptr};
  torch::Tensor world_center,world_spread,mean,stddev,ready_buffer;
 private: bool ready_=false,fused_=true,live_fused_=true;
};
TORCH_MODULE(SpatialCompensation);
}
