#pragma once
#include <torch/torch.h>
#include <cmath>
#include <deque>
#include "include/model_snapshot.h"
#include "include/kernel_fusion.h"
#include "include/residual48_fusion.h"
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/spatial_compensation_cuda.h"
#include <c10/cuda/CUDAStream.h>
#endif

namespace improvements::selector {
// Shared SpatialLocalXYZ field. Only population-independent statistics are buffers.
class SpatialCompensationImpl : public torch::nn::Module {
 public:
  explicit SpatialCompensationImpl(bool early16=false,bool direct_ratio=false,int64_t fourier_bands=6,int64_t ratio_feature_width=0,
      bool ratio_code=false,double ratio_min=.1,double ratio_max=.8,bool shared_anchor128=false,bool offset_only=false,bool bounded_xyz=false) {
    TORCH_CHECK(early16||!direct_ratio,"Direct ratio input requires Early16");
    direct_ratio_=direct_ratio;
    TORCH_CHECK(fourier_bands==0||fourier_bands==3||fourier_bands==6,"Fourier bands must be 0, 3 or 6");
    fourier_bands_=fourier_bands;
    TORCH_CHECK(ratio_feature_width==0 || ((ratio_feature_width==32 || ratio_feature_width==48 || (shared_anchor128&&ratio_feature_width==128))&&!early16&&!direct_ratio&&fourier_bands==0),"Unsupported compensation Feature layout");
    TORCH_CHECK(!shared_anchor128 || (ratio_feature_width==128&&ratio_code),"Shared Anchor128 requires ratio code and Feature128");
    TORCH_CHECK(ratio_feature_width!=48 || (offset_only&&ratio_code&&!shared_anchor128),"Ratio48 requires offset3 and continuous ratio code without a gate");
    TORCH_CHECK(!offset_only||shared_anchor128||ratio_feature_width==48,"Offset-only compensation requires Anchor128 or ratio48");
    TORCH_CHECK(!bounded_xyz||offset_only,"Bounded XYZ requires offset-only Anchor128");
    offset_only_=offset_only;
    bounded_xyz_=bounded_xyz;
    ratio_feature_width_=ratio_feature_width;
    encode=register_module("encode",torch::nn::Linear(ratio_feature_width?(offset_only?3:19)+ratio_feature_width:(direct_ratio?56:55),64));
    body=register_module("body",torch::nn::Linear(64,64));
    if(ratio_code) {
      TORCH_CHECK((ratio_feature_width==32||ratio_feature_width==48||shared_anchor128)&&ratio_min>0&&ratio_max>ratio_min&&ratio_max<=1,"Ratio code requires valid encoding bounds");
      ratio_min_=ratio_min;ratio_max_=ratio_max;
      ratio_bounds=register_buffer("ratio_encoding_bounds",torch::tensor({ratio_min,ratio_max},torch::kFloat64));
      ratio_projection=register_module("ratio_projection",torch::nn::Linear(torch::nn::LinearOptions(3,64).bias(false)));
      torch::NoGradGuard guard;ratio_projection->weight.zero_();
    }
    if(!direct_ratio&&!ratio_feature_width)ratio_net=register_module("ratio",torch::nn::Sequential(torch::nn::Linear(1,32),torch::nn::ReLU(),torch::nn::Linear(32,256)));
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
    if(fourier_bands!=6&&!ratio_feature_width) {
      auto mask=torch::zeros({55});mask.slice(0,0,19).fill_(1);
      for(int64_t axis=0;axis<3;++axis)for(int64_t k=0;k<fourier_bands;++k) {
        mask[19+axis*6+k]=1;mask[37+axis*6+k]=1;
      }
      fourier_mask=register_buffer("fourier_mask",mask);
    }
    if(early16) {
      early_feature=register_module("early_feature",torch::nn::Linear(torch::nn::LinearOptions(16,64).bias(false)));
      torch::NoGradGuard guard;early_feature->weight.normal_(0,.02);
    }
    if(shared_anchor128) {
      ratio_gate=register_module("ratio_gate",torch::nn::Sequential(torch::nn::Linear(3,32),torch::nn::ReLU(),torch::nn::Linear(32,128)));
      torch::NoGradGuard guard;ratio_gate->at<torch::nn::LinearImpl>(2).weight.zero_();ratio_gate->at<torch::nn::LinearImpl>(2).bias.zero_();
    }
  }
  bool physicalScaleAdditive() const{return named_buffers(false).contains("physical_scale_additive_schema");}
  bool viewInput() const{return static_cast<bool>(view_projection);}
  void enableViewInput() {
    TORCH_CHECK(offset_only_&&ratio_feature_width_==48&&ratio_projection,"View conditioning requires Residual48");
    if(viewInput())return;
    ScopedTorchRNG rng{torch::Device(torch::kCPU)};
    view_projection=register_module("view_projection",torch::nn::Linear(torch::nn::LinearOptions(8,64).bias(false)));
    view_projection->to(encode->weight.device(),encode->weight.scalar_type());
    {torch::NoGradGuard guard;view_projection->weight.zero_();}
    register_buffer("view_input_schema",torch::tensor(1,torch::TensorOptions().dtype(torch::kLong).device(encode->weight.device())));
  }
  bool softplusScale() const{return named_buffers(false).contains("softplus_scale_schema");}
  void enableSoftplusScale() {
    TORCH_CHECK(!physicalScaleAdditive(),"Softplus and physical additive Scale are mutually exclusive");
    if(!softplusScale())register_buffer("softplus_scale_schema",
        torch::tensor(1,torch::TensorOptions().dtype(torch::kInt64).device(encode->weight.device())));
  }
  void enablePhysicalScaleAdditive() {
    TORCH_CHECK(!softplusScale(),"Softplus and physical additive Scale are mutually exclusive");
    if(!physicalScaleAdditive())register_buffer("physical_scale_additive_schema",
        torch::tensor(1,torch::TensorOptions().dtype(torch::kInt64).device(encode->weight.device())));
  }
  torch::Tensor decodeScale(const torch::Tensor& master_log_scale,const torch::Tensor& residual) const {
    if(softplusScale())
      return (master_log_scale.exp()*torch::softplus(residual+0.5413248546129181)).clamp_min(1.1754943508222875e-38);
    if(physicalScaleAdditive())return master_log_scale.exp()+residual;
    return (master_log_scale+residual).exp();
  }
  int geometryHeadVariant() const{return geometry_variant_;}
  void configureGeometryHeads(int mode,int64_t seed=0) {
    TORCH_CHECK(mode>=0&&mode<=2,"Unknown geometry head variant");
    if(mode==geometry_variant_)return;
    TORCH_CHECK(geometry_variant_==0&&mode>0&&offset_only_&&
        (ratio_feature_width_==128||(mode==1&&ratio_feature_width_==48)),
        "Geometry ablation requires offset-only Anchor128; no-Other also supports Residual48");
    auto old=heads;unregister_module("heads");
    if(mode==1) {
      heads=register_module("heads",torch::nn::ModuleDict());
      for(const auto& name:{"xyz","scale","rotation"})
        heads->update(std::vector<std::pair<std::string,std::shared_ptr<torch::nn::Module>>>{{name,(*old)[name]}});
    } else {
      heads=nullptr;
      ScopedTorchRNG rng(torch::Device(torch::kCPU),uint64_t(seed));
      geometry_mlp=register_module("geometry_mlp",torch::nn::Sequential(
          torch::nn::Linear(64,176),torch::nn::ReLU(),torch::nn::Linear(176,10)));
      {torch::NoGradGuard guard;geometry_mlp->at<torch::nn::LinearImpl>(2).weight.zero_();geometry_mlp->at<torch::nn::LinearImpl>(2).bias.zero_();}
      geometry_mlp->to(encode->weight.device(),encode->weight.scalar_type());
    }
    geometry_variant_=mode;
    register_buffer("geometry_head_schema",torch::tensor(mode,torch::TensorOptions().dtype(torch::kInt64).device(encode->weight.device())));
  }
  void setFused(bool enabled){fused_=enabled;}
  void setLiveFused(bool enabled){live_fused_=enabled;}
  void setRatioCodeCache(bool enabled){cache_ratio_code_=enabled;if(!enabled)ratio_code_cache_.clear();}
  void setAnchorInputFusion(bool enabled){fuse_anchor_input_=enabled;}
  void setResidualHeadFusion(bool enabled){fuse_residual_heads_=enabled;}
  void setResidualHeadParallel(bool enabled){parallel_residual_heads_=enabled;}
  bool ready() const{return ready_;}
  bool directRatioInput() const{return direct_ratio_;}
  int64_t fourierBands() const{return fourier_bands_;}
  int64_t ratioFeatureWidth() const{return ratio_feature_width_;}
  bool ratioCodeInput() const{return static_cast<bool>(ratio_projection);}
  bool sharedAnchor128() const{return static_cast<bool>(ratio_gate);}
  bool offsetOnlyInput() const{return offset_only_;}
  bool boundedXYZ() const{return bounded_xyz_;}
  std::vector<double> ratioBounds() const{return {ratio_min_,ratio_max_};}
  torch::Tensor filterFourier(const torch::Tensor& input) const{return fourier_mask.defined()?input*fourier_mask:input;}
  void syncReady(){ready_=ready_buffer.item<int64_t>()!=0;if(ratio_bounds.defined()){auto cpu=ratio_bounds.to(torch::kCPU);ratio_min_=cpu[0].item<double>();ratio_max_=cpu[1].item<double>();}}
  void load(torch::serialize::InputArchive& archive) override {
    torch::Tensor view_schema;
    if(archive.try_read("view_input_schema",view_schema,true)) {
      TORCH_CHECK(view_schema.item<int64_t>()==1,"Unknown Compensation view schema");enableViewInput();
    } else TORCH_CHECK(!viewInput(),"Compensation view schema missing");
    torch::Tensor softplus_schema;
    if(archive.try_read("softplus_scale_schema",softplus_schema,true)) {
      TORCH_CHECK(softplus_schema.item<int64_t>()==1,"Unknown Softplus Scale schema");enableSoftplusScale();
    } else TORCH_CHECK(!softplusScale(),"Softplus Scale schema missing; cannot reinterpret an old checkpoint");
    torch::Tensor scale_schema;
    if(archive.try_read("physical_scale_additive_schema",scale_schema,true)) {
      TORCH_CHECK(scale_schema.item<int64_t>()==1,"Unknown physical Scale schema");enablePhysicalScaleAdditive();
    } else TORCH_CHECK(!physicalScaleAdditive(),"Physical Scale schema missing");
    torch::Tensor schema;
    if(archive.try_read("geometry_head_schema",schema,true))configureGeometryHeads(schema.item<int64_t>());
    else TORCH_CHECK(geometry_variant_==0,"Geometry variant missing from checkpoint");
    torch::nn::Module::load(archive);syncReady();
  }
  static torch::Tensor physical(const torch::Tensor& xyz,const torch::Tensor& scale,
      const torch::Tensor& rotation,const torch::Tensor& dc,const torch::Tensor& logits,
      const torch::Tensor& anchor_center,const torch::Tensor& anchor_scale,
      const torch::Tensor& world_center,const torch::Tensor& world_spread,bool log_scale_input=false) {
    auto q=rotation.unbind(1);auto w=q[0],x=q[1],y=q[2],z=q[3];
    auto R=torch::stack({1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),
        1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)},1).reshape({-1,3,3});
    torch::Tensor cov;
    if(log_scale_input){
      // Rescale numerator and trace floor together; preserve the original shape.
      auto shift=std::get<0>(scale.max(1,true)).clamp_min(.5*std::log(1e-12)).detach();
      auto variance=(2*(scale-shift)).exp();
      cov=(R*variance.unsqueeze(1)).matmul(R.transpose(1,2));
      auto floor=(std::log(1e-12)-2*shift).exp().reshape({-1,1,1});
      auto trace=cov.diagonal(0,1,2).sum(1).reshape({-1,1,1});
      cov=cov/torch::where(trace>=floor,trace,floor);
    }else{
      cov=(R*scale.square().unsqueeze(1)).matmul(R.transpose(1,2));
      cov=cov/cov.diagonal(0,1,2).sum(1).clamp_min(1e-12).reshape({-1,1,1});
    }
    auto shape=torch::stack({cov.select(1,0).select(1,0),cov.select(1,1).select(1,1),cov.select(1,2).select(1,2),
        cov.select(1,0).select(1,1),cov.select(1,0).select(1,2),cov.select(1,1).select(1,2)},1);
    return torch::cat({(xyz-world_center)/world_spread,torch::asinh((xyz-anchor_center)/anchor_scale),
        log_scale_input?scale:scale.log(),dc.reshape({xyz.size(0),3}),logits,shape},1);
  }
  void initialize(const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& dc,const torch::Tensor& logits,const torch::Tensor& anchor_center,const torch::Tensor& anchor_scale,bool log_scale_input=false) {
    if(ready_)return;
    TORCH_CHECK(xyz.size(0)>0,"Spatial normalization requires a nonempty map");
    torch::NoGradGuard guard;
    world_center.copy_(std::get<0>(xyz.median(0)));
    world_spread.copy_((xyz-world_center).square().mean(0).sqrt().clamp_min(.1));
    auto raw=physical(xyz,scale,rotation,dc,logits,anchor_center,anchor_scale,world_center,world_spread,log_scale_input);
    mean.copy_(raw.mean(0));
    // Same sample std as the frozen reference; define the single-point online boundary.
    stddev.copy_(xyz.size(0)>1?raw.std(std::vector<int64_t>{0},true,false).clamp_min(.1):torch::full_like(mean,.1));
    ready_buffer.fill_(1);ready_=true;
  }
  torch::Tensor input55(const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& dc,const torch::Tensor& logits,const torch::Tensor& center,const torch::Tensor& reference,bool log_scale_input=false) {
    TORCH_CHECK(ready_,"Initialize Spatial normalization from the complete current map first");
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused_&&live_fused_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32){
      std::vector<torch::Tensor> values{xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread,mean,stddev};
      return filterFourier(log_scale_input?mapping_cuda::spatialLiveLogScaleInput55(values):mapping_cuda::spatialLiveInput55(values));
    }
#endif
    auto raw=physical(xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread,log_scale_input);
    auto coords=(xyz-world_center)/world_spread;
    auto freq=torch::pow(2,torch::arange(6,xyz.options()))*M_PI;
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32)
      return filterFourier(mapping_cuda::spatialInput55(raw,coords,mean,stddev,freq));
#endif
    auto angle=coords.unsqueeze(2)*freq;
    return filterFourier(torch::cat({((raw-mean)/stddev).clamp(-15,15),angle.sin().flatten(1),angle.cos().flatten(1)},1));
  }
  torch::Tensor input19(const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& dc,const torch::Tensor& logits,const torch::Tensor& center,const torch::Tensor& reference,bool log_scale_input=false) {
    TORCH_CHECK(ready_,"Initialize physical normalization from the complete current map first");
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused_&&live_fused_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32) {
      std::vector<torch::Tensor> values{xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread,mean,stddev};
      auto packed=log_scale_input?mapping_cuda::spatialLiveLogScaleInput55(values):mapping_cuda::spatialLiveInput55(values);
      return packed.slice(1,0,19);
    }
#endif
    auto raw=physical(xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread,log_scale_input);
    return ((raw-mean)/stddev).clamp(-15,15);
  }
  torch::Tensor forward(const torch::Tensor& xyz,const torch::Tensor& scale,const torch::Tensor& rotation,
      const torch::Tensor& dc,const torch::Tensor& logits,const torch::Tensor& center,const torch::Tensor& reference,double ratio,bool log_scale_input=false,const torch::Tensor& anchor_feature={},const torch::Tensor& packed_residual_input={},const torch::Tensor& view_condition={}) {
    TORCH_CHECK(std::isfinite(ratio)&&ratio>0&&ratio<=1,"Invalid Spatial ratio");
    TORCH_CHECK(!view_condition.defined()||(viewInput()&&view_condition.sizes()==torch::IntArrayRef({xyz.size(0),8})),"Invalid Compensation view condition");
    torch::Tensor input;
#ifdef CARTGS_FUSED_MAPPING_CUDA
    const bool pack_anchor=fuse_anchor_input_&&offset_only_&&ratio_gate&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32&&
        !center.requires_grad()&&!reference.requires_grad()&&!mean.requires_grad()&&!stddev.requires_grad();
#else
    const bool pack_anchor=false;
#endif
    if(packed_residual_input.defined()) {
      TORCH_CHECK(offset_only_&&ratio_feature_width_==48&&ready_&&packed_residual_input.sizes()==torch::IntArrayRef({xyz.size(0),51}),"Invalid fused Residual48 input");
      input=packed_residual_input;
    } else if(offset_only_&&!pack_anchor) {
      mapping_cuda::FusionRange trace("Comp.OffsetNormalize");
      TORCH_CHECK(ready_,"Initialize offset normalization from the complete current map first");
      input=((torch::asinh((xyz-center)/reference)-mean.slice(0,3,6))/stddev.slice(0,3,6)).clamp(-15,15);
    } else if(!pack_anchor)input=ratio_feature_width_?input19(xyz,scale,rotation,dc,logits,center,reference,log_scale_input):input55(xyz,scale,rotation,dc,logits,center,reference,log_scale_input);
    torch::Tensor condition;
    torch::Tensor code;
    if(ratio_projection) {
      const double t=2*(std::log(ratio)-std::log(ratio_min_))/(std::log(ratio_max_)-std::log(ratio_min_))-1;
      uintptr_t stream=0;
#ifdef CARTGS_FUSED_MAPPING_CUDA
      if(xyz.is_cuda())stream=reinterpret_cast<uintptr_t>(c10::cuda::getCurrentCUDAStream(xyz.get_device()).stream());
#endif
      if(cache_ratio_code_)for(const auto& entry:ratio_code_cache_)
        if(entry.ratio==ratio&&entry.low==ratio_min_&&entry.high==ratio_max_&&entry.stream==stream&&
            entry.code.device()==xyz.device()&&entry.code.scalar_type()==xyz.scalar_type()){code=entry.code;break;}
      if(!code.defined()) {
        code=torch::tensor({{t,std::sin(M_PI*t/2),std::cos(M_PI*t/2)}},xyz.options());
        if(cache_ratio_code_) {
          if(ratio_code_cache_.size()==32)ratio_code_cache_.pop_front();
          ratio_code_cache_.push_back({ratio,ratio_min_,ratio_max_,stream,code});
        }
      }
    }
    if(!ratio_feature_width_)condition=torch::full({1,1},2*(std::log(ratio)-std::log(.01))/(std::log(.15)-std::log(.01))-1,xyz.options());
    if(ratio_feature_width_&&!packed_residual_input.defined()) {
      mapping_cuda::FusionRange trace("Comp.FeatureGatePack");
      TORCH_CHECK(anchor_feature.defined()&&anchor_feature.sizes()==torch::IntArrayRef({xyz.size(0),ratio_feature_width_}),"Physical19 requires selected Anchor/ratio32 features");
#ifdef CARTGS_FUSED_MAPPING_CUDA
      if(pack_anchor) {
        TORCH_CHECK(ready_,"Initialize offset normalization from the complete current map first");
        input=mapping_cuda::anchor128Pack(xyz,center,reference,anchor_feature,ratio_gate->forward(code).tanh(),mean,stddev);
      }else
#endif
      input=torch::cat({input,ratio_gate?anchor_feature*(1+ratio_gate->forward(code).tanh()):anchor_feature},1);
    }
    if(direct_ratio_)input=torch::cat({input,condition.expand({xyz.size(0),1})},1);
    auto encoded=[&](){mapping_cuda::FusionRange trace("Comp.Encode");return encode->forward(input);}();
    torch::Tensor first_hidden;
    const bool fuse_heads=fuse_residual_heads_&&!view_condition.defined()&&!torch::GradMode::is_enabled()&&ratio_feature_width_==48&&
        !geometry_variant_&&!bounded_xyz_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32&&
        reference.device()==xyz.device()&&reference.scalar_type()==torch::kFloat32&&reference.is_contiguous()&&
        reference.sizes()==torch::IntArrayRef({xyz.size(0),1});
    const bool parallel_heads=parallel_residual_heads_&&!torch::GradMode::is_enabled()&&ratio_feature_width_==48&&
        !geometry_variant_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32&&xyz.size(0)>=8192;
    if(ratio_projection) {
      auto projected=ratio_projection->forward(code);
      if(fuse_heads&&!early_feature)first_hidden=mapping_cuda::residual48AddReluInference(encoded,projected);
      else encoded=encoded+projected;
    }
    if(early_feature) {
      TORCH_CHECK(anchor_feature.defined()&&anchor_feature.sizes()==torch::IntArrayRef({xyz.size(0),16}),"Early16 requires selected K x 16 Anchor Features");
      encoded=encoded+early_feature->forward(anchor_feature);
    }
    if(view_projection&&(view_condition.defined()||torch::GradMode::is_enabled()))
      encoded=encoded+view_projection->forward(view_condition.defined()?view_condition.detach():torch::zeros({1,8},xyz.options()));
    torch::Tensor hidden;
    if(direct_ratio_||ratio_feature_width_)hidden=torch::relu(body->forward(first_hidden.defined()?first_hidden:torch::relu(encoded)));
    else {
      auto modulation=ratio_net->forward(condition).chunk(4,1);
      auto film=[&](const torch::Tensor& h,const torch::Tensor& g,const torch::Tensor& b){
#ifdef CARTGS_FUSED_MAPPING_CUDA
        if(fused_&&h.is_cuda()&&h.scalar_type()==torch::kFloat32)return mapping_cuda::spatialFilm(h,g,b);
#endif
        return torch::relu((1+g)*h+b);
      };
      hidden=film(encoded,modulation[0],modulation[1]);
      hidden=film(body->forward(hidden),modulation[2],modulation[3]);
    }
    mapping_cuda::FusionRange head_trace("Comp.HeadsOutput");
    std::vector<torch::Tensor> result;
    if(parallel_heads)result=mapping_cuda::residual48HeadsInference(hidden,heads);
    else if(geometry_variant_==2) {
      auto out=geometry_mlp->forward(hidden);result={out.slice(1,0,3),out.slice(1,3,6),out.slice(1,6,10)};
    } else {
      for(const auto& name:{"xyz","scale","rotation"})result.push_back(heads->at<torch::nn::SequentialImpl>(name).forward(hidden));
      if(!geometry_variant_)result.push_back(heads->at<torch::nn::SequentialImpl>("other").forward(hidden));
    }
    if(geometry_variant_)result.push_back(torch::zeros({hidden.size(0),4},hidden.options()));
    if(bounded_xyz_)result[0]=result[0]/torch::sqrt(1+result[0].square().sum(1,true)/4);
    if(fuse_heads)return mapping_cuda::residual48OutputInference(result,reference);
    result[0]=result[0]*reference;
    return torch::cat(result,1);
  }
  torch::nn::Linear encode{nullptr},body{nullptr};
  torch::nn::Linear early_feature{nullptr},ratio_projection{nullptr},view_projection{nullptr};
  torch::nn::Sequential ratio_net{nullptr},ratio_gate{nullptr};torch::nn::ModuleDict heads{nullptr};
  torch::Tensor world_center,world_spread,mean,stddev,ready_buffer,fourier_mask,ratio_bounds;
 private: bool ready_=false,fused_=true,live_fused_=true,direct_ratio_=false,offset_only_=false,bounded_xyz_=false;int64_t fourier_bands_=6,ratio_feature_width_=0;double ratio_min_=.1,ratio_max_=.8;
  int geometry_variant_=0;torch::nn::Sequential geometry_mlp{nullptr};
  struct RatioCode {double ratio,low,high;uintptr_t stream;torch::Tensor code;};
  bool cache_ratio_code_=false;
  bool fuse_anchor_input_=false;
  bool fuse_residual_heads_=false;
  bool parallel_residual_heads_=false;
  std::deque<RatioCode> ratio_code_cache_;
};
TORCH_MODULE(SpatialCompensation);
}
