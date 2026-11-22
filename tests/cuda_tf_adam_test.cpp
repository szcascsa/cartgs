#include "include/transform_field.h"
#include "include/transform_field_cuda_optimizer.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
std::ofstream parity;
void compare(const std::string& name,const torch::Tensor& a,const torch::Tensor& b) {
  double error=a.numel()?(a-b).abs().max().item<double>():0;
  bool pass=torch::allclose(a,b,2e-6,2e-7,true);
  parity<<name<<','<<error<<','<<pass<<'\n';parity.flush();TORCH_CHECK(pass,name," max_abs=",error);
}
void states(const std::string& label,torch::optim::Adam& old,torch::optim::Adam& cuda) {
  for(size_t g=0;g<old.param_groups().size();++g)for(size_t i=0;i<old.param_groups()[g].params().size();++i) {
    auto a=old.param_groups()[g].params()[i],b=cuda.param_groups()[g].params()[i];
    auto name=label+"/g"+std::to_string(g)+"/p"+std::to_string(i);compare(name+"/p",b,a);
    auto x=old.state().find(a.unsafeGetTensorImpl()),y=cuda.state().find(b.unsafeGetTensorImpl());
    TORCH_CHECK((x==old.state().end())==(y==cuda.state().end()),name," state presence");
    if(x==old.state().end())continue;
    const auto& s=static_cast<const torch::optim::AdamParamState&>(*x->second);
    const auto& t=static_cast<const torch::optim::AdamParamState&>(*y->second);
    TORCH_CHECK(s.step()==t.step(),name," per tensor step");
    compare(name+"/m",t.exp_avg(),s.exp_avg());compare(name+"/v",t.exp_avg_sq(),s.exp_avg_sq());
  }
}
void sequence(double eps,double decay,const std::string& label,bool mixed=false) {
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  std::vector<torch::Tensor> a,b;
  for(int n:{17,0,101,1009,32768,3}) {
    auto p=torch::randn({n},opt);a.push_back(p.clone().set_requires_grad(true));b.push_back(p.clone().set_requires_grad(true));
  }
  torch::optim::Adam old({a[0],a[1],a[2]},torch::optim::AdamOptions(0.0016).eps(eps).weight_decay(decay));
  torch::optim::Adam cuda({b[0],b[1],b[2]},torch::optim::AdamOptions(0.0016).eps(eps).weight_decay(decay));
  old.add_param_group(std::vector<torch::Tensor>{a[3],a[4],a[5]});
  cuda.add_param_group(std::vector<torch::Tensor>{b[3],b[4],b[5]});
  for(int step=1;step<=100;++step) {
    double lr=step%9==0?0:0.00016;
    old.param_groups()[1].options().set_lr(lr);cuda.param_groups()[1].options().set_lr(lr);
    if(step==37) {old.state().erase(a[2].unsafeGetTensorImpl());cuda.state().erase(b[2].unsafeGetTensorImpl());}
    for(size_t i=0;i<a.size();++i) {
      // Different clocks, undefined gradients, all-zero gradients, and tiny gradients.
      auto grad=(i==5 || (step+int(i))%4==0)?torch::Tensor():
        (step%7==0?torch::zeros_like(a[i]):torch::randn_like(a[i])*(step%11==0?1e-12f:1.f));
      a[i].mutable_grad()=grad;b[i].mutable_grad()=grad.defined()?grad.clone():torch::Tensor();
    }
    old.step();if(mixed)tf_cuda::tfCudaAdamMixedStep(cuda,{1});else tf_cuda::tfCudaAdamDirect(cuda,{0,1});
    if(step==1 || step==10 || step==37 || step==100)states(label+"/step"+std::to_string(step),old,cuda);
  }
  TORCH_CHECK(old.state().count(a[5].unsafeGetTensorImpl())==0,"No-grad tensor acquired state");
}
struct Event {cudaEvent_t v;Event(){cudaEventCreate(&v);}~Event(){cudaEventDestroy(v);}
  void record(){cudaEventRecord(v,c10::cuda::getCurrentCUDAStream());}};
void timing(std::ofstream& out,const std::string& backend,const std::function<void()>& action) {
  Event a,b;std::vector<float> samples;
  for(int i=0;i<600;++i) {a.record();action();b.record();cudaEventSynchronize(b.v);
    if(i>=100){float ms;cudaEventElapsedTime(&ms,a.v,b.v);samples.push_back(ms);}}
  float mean=std::accumulate(samples.begin(),samples.end(),0.f)/samples.size();std::sort(samples.begin(),samples.end());
  out<<backend<<",adam_direct,"<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';out.flush();
}
}
int main(int argc,char** argv) {
  TORCH_CHECK(argc==3,"cuda_tf_adam_test OUTPUT_DIRECTORY CHECKPOINT");
  torch::set_num_threads(1);torch::manual_seed(20261018);
  parity.open(std::string(argv[1])+"/phase_f1_parity.csv");parity<<"case,max_abs,passed\n";
  for(double eps:{1e-15,1e-8})for(double decay:{0.,0.01})sequence(eps,decay,"eps="+std::to_string(eps)+"/decay="+std::to_string(decay));
  {c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());sequence(1e-15,0.01,"nondefault_stream");}
  sequence(1e-15,0.01,"mixed_gaussian_selector_tf",true);
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto p=torch::ones({5},opt).set_requires_grad(true);p.mutable_grad()=torch::ones_like(p);
  torch::optim::Adam normal({p},torch::optim::AdamOptions(.01).eps(1e-15));
  torch::optim::Adam cd({p},torch::optim::AdamOptions(.01).eps(1e-15));
  torch::optim::Adam dsd({p},torch::optim::AdamOptions(.01).eps(1e-8));
  tf_cuda::tfCudaAdamDirect(normal,{0});auto nm=static_cast<torch::optim::AdamParamState&>(*normal.state().at(p.unsafeGetTensorImpl())).exp_avg().clone();
  tf_cuda::tfCudaAdamDirect(cd,{0});tf_cuda::tfCudaAdamDirect(dsd,{0});
  compare("independent_state",static_cast<torch::optim::AdamParamState&>(*normal.state().at(p.unsafeGetTensorImpl())).exp_avg(),nm);
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);
  torch::serialize::InputArchive archive;archive.load_from(std::string(argv[2])+"/transform_field.pt");tf->load(archive);tf->to(torch::kCUDA);
  for(auto t:tf->parameters())t.mutable_grad()=torch::randn_like(t)*.01;
  std::ofstream bench(std::string(argv[1])+"/phase_f1_benchmark.csv");bench<<"backend,stage,mean_ms,p50_ms,p95_ms\n";
  for(bool custom:{false,true}) {
    torch::optim::Adam optimizer(tf->mlpParameters(),torch::optim::AdamOptions(.00016).eps(1e-15));
    optimizer.add_param_group(tf->gridParameters());optimizer.param_groups()[1].options().set_lr(.0016);
    timing(bench,custom?"cuda":"legacy",[&](){if(custom)tf_cuda::tfCudaAdamDirect(optimizer,{0,1});else optimizer.step();});
  }
  std::cout<<"phase F1 passed"<<std::endl;
}
