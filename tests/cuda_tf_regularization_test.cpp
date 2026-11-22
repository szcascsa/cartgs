#include "include/transform_field.h"
#include "include/transform_field_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
std::ofstream parity;
void compare(const std::string& name,const torch::Tensor& a,const torch::Tensor& b,
             double atol=2e-6,double rtol=2e-5) {
  double error=(a-b).abs().max().item<double>();bool pass=torch::allclose(a,b,rtol,atol,true);
  parity<<name<<','<<error<<','<<atol<<','<<rtol<<','<<pass<<'\n';parity.flush();TORCH_CHECK(pass,name," max_abs=",error);
}
struct Event {cudaEvent_t v;Event(){cudaEventCreate(&v);}~Event(){cudaEventDestroy(v);}
  void record(){cudaEventRecord(v,c10::cuda::getCurrentCUDAStream());}};
void timing(std::ofstream& out,const std::string& backend,const std::string& stage,const std::function<void()>& action) {
  Event a,b;std::vector<float> samples;
  for(int i=0;i<600;++i) {a.record();action();b.record();cudaEventSynchronize(b.v);
    if(i>=100){float ms;cudaEventElapsedTime(&ms,a.v,b.v);samples.push_back(ms);}}
  float mean=std::accumulate(samples.begin(),samples.end(),0.0f)/samples.size();std::sort(samples.begin(),samples.end());
  out<<backend<<','<<stage<<','<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';out.flush();
}
torch::Tensor current(TransformField& tf) {
  const auto& c=tf->config();return tf_cuda::regularization(tf->gridParameters(),c.spatial_smooth_weight,c.ratio_smooth_weight,c.ratio_l1_weight);
}
void check(TransformField& tf,const std::string& label) {
  auto old=tf->regularizationLoss(),cuda=current(tf);auto p=tf->gridParameters();auto upstream=torch::full_like(old,2.5f);
  compare(label+"/loss",cuda,old);
  auto a=torch::autograd::grad({old},p,{upstream},true),floor=torch::autograd::grad({old},p,{upstream}),b=torch::autograd::grad({cuda},p,{upstream});
  for(size_t i=0;i<p.size();++i) {
    compare(label+"/legacy_repeat_"+std::to_string(i),floor[i],a[i],2e-10,2e-4);
    compare(label+"/plane_"+std::to_string(i),b[i],a[i],2e-10,2e-4);
  }
}
}  // namespace
int main(int argc,char** argv) {
  TORCH_CHECK(argc==3,"cuda_tf_regularization_test OUTPUT_DIRECTORY CHECKPOINT");
  torch::set_num_threads(1);torch::manual_seed(20261016);
  parity.open(std::string(argv[1])+"/phase_d_parity.csv");parity<<"case,max_abs,atol,rtol,passed\n";
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  for(int height:{2,3,5,7})for(int mode:{0,1,2}) {
    TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=3;cfg.hexplane.spatial_resolution=height;cfg.hexplane.ratio_resolution=height+1;cfg.hexplane.multires={1,2};
    if(mode==2){cfg.spatial_smooth_weight=0;cfg.ratio_smooth_weight=0;cfg.ratio_l1_weight=0;}
    TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);tf->to(torch::kCUDA);
    {torch::NoGradGuard ng;for(auto p:tf->gridParameters()) {
      if(mode==0)p.uniform_(-2,2);
      else if(mode==1) {
        // Vary width while remaining affine in height: height smoothness is zero.
        auto row=torch::arange(p.size(2),opt).view({1,1,-1,1});
        auto col=torch::arange(p.size(3),opt).square().view({1,1,1,-1});p.copy_(row+col);
      }
    }}
    check(tf,"height="+std::to_string(height)+"/mode="+std::to_string(mode));
  }
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);
  torch::serialize::InputArchive archive;archive.load_from(std::string(argv[2])+"/transform_field.pt");tf->load(archive);tf->to(torch::kCUDA);
  check(tf,"real_checkpoint");
  // With normal eps=1e-15, even a 1e-16 rounding error can alter Adam's
  // first parameter delta. Test the reduction's backward arithmetic at the
  // optimizer boundary, in addition to ordinary gradient atol/rtol.
  {
    auto p=tf->gridParameters();
    auto a=torch::autograd::grad({tf->regularizationLoss()},p);
    auto b=torch::autograd::grad({current(tf)},p);
    std::vector<torch::Tensor> left,right;
    for(size_t i=0;i<p.size();++i) {
      left.push_back(p[i].detach().clone().set_requires_grad(true));
      right.push_back(p[i].detach().clone().set_requires_grad(true));
      left.back().mutable_grad()=a[i];right.back().mutable_grad()=b[i];
    }
    torch::optim::Adam old(left,torch::optim::AdamOptions(.0016).eps(1e-15));
    torch::optim::Adam cuda(right,torch::optim::AdamOptions(.0016).eps(1e-15));
    old.step();cuda.step();
    for(size_t i=0;i<p.size();++i)
      compare("regularization_adam_near_zero/"+std::to_string(i),right[i],left[i],2e-6,2e-4);
  }
  auto x=(torch::rand({137,3},opt)*(tf->aabbMax()-tf->aabbMin())+tf->aabbMin()).detach().set_requires_grad(true);
  auto r=torch::full({137,1},0.05,opt);HexPlaneField grid(tf->aabbMin(),tf->aabbMax(),cfg.hexplane);grid->to(torch::kCUDA);
  {torch::NoGradGuard ng;auto a=grid->gridParameters(),b=tf->gridParameters();for(size_t i=0;i<a.size();++i)a[i].copy_(b[i]);}
  auto old=grid->forward(x,r).sum()+cfg.spatial_smooth_weight*grid->spatialSmoothness()+cfg.ratio_smooth_weight*grid->ratioSmoothness()+cfg.ratio_l1_weight*grid->ratioL1();
  auto cuda=tf_cuda::query(x,r,tf->aabbMin(),tf->aabbMax(),tf->gridParameters()).sum()+current(tf);
  auto a=torch::autograd::grad({old},grid->gridParameters()),b=torch::autograd::grad({cuda},tf->gridParameters());
  for(size_t i=0;i<a.size();++i)compare("query_plus_reg/"+std::to_string(i),b[i],a[i],1e-5,5e-4);
  // Empty TF population does not remove the independent plane regularizer.
  auto empty=tf_cuda::query(torch::empty({0,3},opt),torch::empty({0,1},opt),tf->aabbMin(),tf->aabbMax(),tf->gridParameters());
  TORCH_CHECK(!empty.requires_grad(),"Empty legacy query must have no plane edges");
  current(tf).backward();for(auto p:tf->gridParameters())TORCH_CHECK(p.grad().defined(),"Empty population lost regularization gradient");
  {c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());check(tf,"nondefault_stream");}
  std::ofstream bench(std::string(argv[1])+"/phase_d_benchmark.csv");bench<<"backend,stage,mean_ms,p50_ms,p95_ms\n";
  for(bool custom:{false,true}) {
    auto call=[&](){return custom?current(tf):tf->regularizationLoss();};
    timing(bench,custom?"cuda":"legacy","regularization_forward",[&](){torch::NoGradGuard ng;call();});
    auto y=call();auto p=tf->gridParameters();
    timing(bench,custom?"cuda":"legacy","regularization_backward",[&](){torch::autograd::grad({y},p,{},true);});
    timing(bench,custom?"cuda":"legacy","regularization_forward_backward",[&](){torch::autograd::grad({call()},p);});
  }
  std::cout<<"phase D passed"<<std::endl;
}
