#include "include/hexplane_field.h"
#include "include/transform_field.h"
#include "include/transform_field_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
std::ofstream results;
void compare(const std::string& name,const torch::Tensor& a,const torch::Tensor& b,
             double atol=2e-6,double rtol=2e-4) {
  if(!a.numel()&&!b.numel())return;
  auto d=(a-b).abs();double maximum=d.max().item<double>();
  bool pass=torch::allclose(a,b,rtol,atol,true);
  results<<name<<','<<maximum<<','<<atol<<','<<rtol<<','<<pass<<'\n';results.flush();
  TORCH_CHECK(pass,name," max_abs=",maximum);
}
struct Event {cudaEvent_t v;Event(){cudaEventCreate(&v);}~Event(){cudaEventDestroy(v);}
  void record(){cudaEventRecord(v,c10::cuda::getCurrentCUDAStream());}};
void timing(std::ofstream& out,const std::string& backend,int n,const std::string& stage,
            const std::function<void()>& action) {
  Event a,b;std::vector<float> samples;
  for(int i=0;i<600;++i) {
    a.record();action();b.record();cudaEventSynchronize(b.v);
    if(i>=100){float ms;cudaEventElapsedTime(&ms,a.v,b.v);samples.push_back(ms);}
  }
  float mean=std::accumulate(samples.begin(),samples.end(),0.0f)/samples.size();
  std::sort(samples.begin(),samples.end());
  out<<backend<<','<<n<<','<<stage<<','<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';out.flush();
}
void queryParity(HexPlaneField& field,torch::Tensor xyz,torch::Tensor ratio,const std::string& label) {
  const auto planes=field->gridParameters();
  auto x=xyz.detach().clone().set_requires_grad(true),r=ratio.detach().clone().set_requires_grad(true);
  auto y=field->forward(x,r);auto dy=torch::randn_like(y);
  std::vector<torch::Tensor> args{x,r};args.insert(args.end(),planes.begin(),planes.end());
  auto old=torch::autograd::grad({y},args,{dy},true);
  auto floor=torch::autograd::grad({y},args,{dy});
  auto cx=x.detach().clone().set_requires_grad(true),cr=r.detach().clone().set_requires_grad(true);
  auto cy=tf_cuda::query(cx,cr,field->aabbMin(),field->aabbMax(),planes);
  args[0]=cx;args[1]=cr;
  auto current=torch::autograd::grad({cy},args,{dy});
  compare(label+"/feature",cy,y,2e-6,2e-5);
  for(size_t i=0;i<old.size();++i) {
    compare(label+"/legacy_repeat_"+std::to_string(i),floor[i],old[i],1e-5,5e-4);
    compare(label+"/gradient_"+std::to_string(i),current[i],old[i],i>=2?1e-5:2e-6,i>=2?5e-4:2e-4);
  }
}
}  // namespace
int main(int argc,char** argv) {
  TORCH_CHECK(argc>=2,"cuda_transform_field_test OUTPUT_DIRECTORY [CHECKPOINT]");
  torch::set_num_threads(1);torch::manual_seed(20261013);
  results.open(std::string(argv[1])+"/phase_a_parity.csv");
  results<<"case,max_abs,atol,rtol,passed\n";
  auto options=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  HexPlaneConfig cfg;cfg.feature_dim=3;cfg.spatial_resolution=5;cfg.ratio_resolution=7;cfg.multires={1,2};
  auto lo=torch::tensor({-2.0f,-3.0f,-4.0f},options),hi=torch::tensor({2.0f,3.0f,4.0f},options);
  HexPlaneField small(lo,hi,cfg);small->to(torch::kCUDA);
  {torch::NoGradGuard ng;for(auto& p:small->gridParameters())p.uniform_(-1,1);
    small->gridParameters()[2].select(1,0).zero_();}
  auto random=torch::rand({257,3},options)*(hi-lo)*1.5+lo-(hi-lo)*0.25;
  for(float ratio:{-1.5f,-1.0f,std::nextafter(-1.0f,-2.0f),std::nextafter(-1.0f,0.0f),0.0f,0.01f,0.05f,0.1f,0.15f,std::nextafter(1.0f,0.0f),1.0f,std::nextafter(1.0f,2.0f),1.5f})
    queryParity(small,random,torch::full({257,1},ratio,options),"random_r="+std::to_string(ratio));
  std::vector<float> boundary_values={-2,-3,-4,2,3,4,0,0,0,-2,0,0,0,3,0,-2,3,0,-1,1.5f,2,
      std::nextafter(-2.0f,0.0f),0,0,std::nextafter(-2.0f,-3.0f),0,0,
      std::nextafter(2.0f,0.0f),0,0,std::nextafter(2.0f,3.0f),0,0};
  auto corners=torch::tensor(boundary_values,options).reshape({-1,3});
  queryParity(small,corners,torch::tensor(0.05f,options),"boundaries_scalar");
  queryParity(small,random,torch::rand({257},options)*3-1.5,"per_row_ratio");
  {c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
    queryParity(small,torch::rand({31,3},options),torch::rand({31,1},options),"nondefault_stream");}
  auto empty=tf_cuda::query(torch::empty({0,3},options),torch::empty({0,1},options),lo,hi,small->gridParameters());
  TORCH_CHECK(empty.sizes()==torch::IntArrayRef({0,6}),"Empty query shape");
  TransformFieldConfig fast;fast.hexplane.feature_dim=8;fast.hexplane.multires={1,2};fast.hidden_dim=32;
  TransformField tf(lo,hi,fast);
  if(argc>=3){torch::serialize::InputArchive a;a.load_from(std::string(argv[2])+"/transform_field.pt");tf->load(a);}
  tf->to(torch::kCUDA);
  HexPlaneField grid(tf->aabbMin(),tf->aabbMax(),fast.hexplane);grid->to(torch::kCUDA);
  {torch::NoGradGuard ng;auto p=tf->gridParameters(),g=grid->gridParameters();for(size_t i=0;i<p.size();++i)g[i].copy_(p[i]);}
  queryParity(grid,torch::rand({8491,3},options)*(tf->aabbMax()-tf->aabbMin())+tf->aabbMin(),torch::full({8491,1},0.05,options),"checkpoint");
  std::ofstream bench(std::string(argv[1])+"/phase_a_benchmark.csv");bench<<"backend,N,stage,mean_ms,p50_ms,p95_ms\n";
  for(int n:{1000,5000,8491,10000,20000,50000}) {
    auto x=(torch::rand({n,3},options)*(tf->aabbMax()-tf->aabbMin())+tf->aabbMin()).detach().set_requires_grad(true);
    auto r=torch::full({n,1},0.05,options);auto dy=torch::randn({n,16},options);
    for(bool cuda:{false,true}) {
      auto call=[&](){return cuda?tf_cuda::query(x,r,grid->aabbMin(),grid->aabbMax(),grid->gridParameters()):grid->forward(x,r);};
      timing(bench,cuda?"cuda":"legacy",n,"query_forward",[&](){torch::NoGradGuard ng;call();});
      auto y=call();std::vector<torch::Tensor> inputs{x};auto p=grid->gridParameters();inputs.insert(inputs.end(),p.begin(),p.end());
      timing(bench,cuda?"cuda":"legacy",n,"query_backward",[&](){torch::autograd::grad({y},inputs,{dy},true);});
    }
    std::cout<<"phase A N="<<n<<" passed"<<std::endl;
  }
}
