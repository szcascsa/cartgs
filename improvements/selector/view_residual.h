#pragma once
#include <torch/torch.h>

namespace improvements::selector {

// Camera geometry is computed on the uncompensated Master and detached. One
// seven-channel descriptor is shared by every member of an Anchor.
inline torch::Tensor anchorViewCode(const torch::Tensor& centers,const torch::Tensor& scales,
    const torch::Tensor& camera_center,const torch::Tensor& view,const torch::Tensor& projection,
    int height,int width,float near_plane,const torch::Tensor& scene_scale) {
  torch::NoGradGuard guard;
  TORCH_CHECK(centers.dim()==2&&centers.size(1)==3&&scales.sizes()==torch::IntArrayRef({centers.size(0),1})&&
      height>0&&width>0&&near_plane>0,"Invalid view residual camera/Anchor geometry");
  auto vector=camera_center.reshape({1,3})-centers;
  auto direction=vector/vector.square().sum(1,true).sqrt().clamp_min(1e-6);
  auto homogeneous=torch::cat({centers,torch::ones_like(centers.slice(1,0,1))},1);
  auto camera=homogeneous.matmul(view),clip=homogeneous.matmul(projection);
  auto safe_w=[near_plane](const torch::Tensor& w) {
    return torch::where(w.abs()<near_plane,torch::where(w<0,torch::full_like(w,-near_plane),
        torch::full_like(w,near_plane)),w);
  };
  auto screen=(clip.slice(1,0,2)/safe_w(clip.slice(1,3,4))).clamp(-4,4);
  const auto signs=torch::tensor({{-1.f,-1.f,-1.f},{-1.f,-1.f,1.f},{-1.f,1.f,-1.f},{-1.f,1.f,1.f},
      {1.f,-1.f,-1.f},{1.f,-1.f,1.f},{1.f,1.f,-1.f},{1.f,1.f,1.f}},centers.options());
  auto corners=centers.unsqueeze(1)+scales.unsqueeze(1)*.5*signs;
  auto corners4=torch::cat({corners,torch::ones_like(corners.slice(2,0,1))},2);
  auto projected=corners4.matmul(projection),depth=corners4.matmul(view).select(2,2);
  auto ndc=projected.slice(2,0,2)/safe_w(projected.slice(2,3,4));
  auto extent=std::get<0>(ndc.max(1))-std::get<0>(ndc.min(1));
  auto pixels=extent*torch::tensor({float(width)/height,1.f},centers.options())*.5;
  auto longest=std::get<0>(pixels.max(1)).clamp(0,1e4);
  const auto near=std::get<0>(depth.min(1))<=near_plane,behind=std::get<0>(depth.max(1))<=near_plane;
  longest=torch::where(near,torch::full_like(longest,1e4),longest);
  longest=torch::where(behind,torch::zeros_like(longest),longest);
  return torch::cat({direction,torch::asinh(camera.select(1,2)/scene_scale).unsqueeze(1),
      screen,torch::log1p(longest).unsqueeze(1)},1);
}

class SelectorViewResidualImpl:public torch::nn::Module {
 public:
  explicit SelectorViewResidualImpl(int64_t schema=1) {
    TORCH_CHECK(schema==1||schema==2,"Unknown Selector view residual schema");
    hidden=register_module("hidden",torch::nn::Linear(schema==2?18:39,16));
    output=register_module("output",torch::nn::Linear(16,1));
    torch::NoGradGuard guard;output->weight.zero_();output->bias.zero_();
    scene_scale=register_buffer("scene_scale",torch::ones({}));
    initialized=register_buffer("initialized",torch::zeros({},torch::kLong));
  }
  void initialize(const torch::Tensor& master_xyz) {
    if(ready_)return;
    TORCH_CHECK(master_xyz.dim()==2&&master_xyz.size(1)==3&&master_xyz.size(0)>0,"View normalization needs a nonempty Master");
    torch::NoGradGuard guard;
    scene_scale.copy_((std::get<0>(master_xyz.max(0))-std::get<0>(master_xyz.min(0))).max().clamp_min(1e-4));
    initialized.fill_(1);ready_=true;
  }
  void syncReady() { ready_=initialized.item<int64_t>()!=0; }
  int64_t schema() const { return hidden->weight.size(1)==18?2:1; }
  torch::Tensor forward(const torch::Tensor& input,const torch::Tensor& view_code) {
    TORCH_CHECK(input.defined()&&input.dim()==2&&input.size(1)==(schema()==2?11:32)&&
        view_code.sizes()==torch::IntArrayRef({input.size(0),7}),
        "View residual input must match checkpoint schema (physical11 or hidden32) and view7");
    return output->forward(torch::relu(hidden->forward(torch::cat({input.detach(),view_code.detach()},1))));
  }
  torch::nn::Linear hidden{nullptr},output{nullptr};
  torch::Tensor scene_scale,initialized;
 private:
  bool ready_=false;
};
TORCH_MODULE(SelectorViewResidual);
} // namespace improvements::selector
