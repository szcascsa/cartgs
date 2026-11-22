#include "include/transform_field.h"
#include "include/gradient_constraint.h"
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
  bool pass=torch::allclose(a,b,1e-7,1e-8,true);
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
void sequence(const std::string& label) {
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  std::vector<torch::Tensor> a,b;
  for(int n:{17,101,32768,3}) {
    auto p=torch::randn({n},opt);a.push_back(p.clone().set_requires_grad(true));b.push_back(p.clone().set_requires_grad(true));
  }
  torch::optim::Adam old({a[0]},torch::optim::AdamOptions(.0016).eps(1e-15).weight_decay(.01));
  torch::optim::Adam cuda({b[0]},torch::optim::AdamOptions(.0016).eps(1e-15).weight_decay(.01));
  old.add_param_group(std::vector<torch::Tensor>{a[1],a[2],a[3]});
  cuda.add_param_group(std::vector<torch::Tensor>{b[1],b[2],b[3]});
  for(int step=1;step<=100;++step) {
    double lr=step%9==0?0:.00016;old.param_groups()[1].options().set_lr(lr);cuda.param_groups()[1].options().set_lr(lr);
    if(step==37){old.state().erase(a[2].unsafeGetTensorImpl());cuda.state().erase(b[2].unsafeGetTensorImpl());}
    for(size_t i=0;i<a.size();++i) {
      auto grad=(i==3 || (step+int(i))%4==0)?torch::Tensor():
        (step%7==0?torch::zeros_like(a[i]):torch::randn_like(a[i]));
      a[i].mutable_grad()=grad;b[i].mutable_grad()=grad.defined()?grad.clone():torch::Tensor();
    }
    auto before=b[1].clone();const auto count=cuda.state().size();
    AdamStepPreview x(old),y(cuda,{1});
    TORCH_CHECK(b[1].equal(before) && cuda.state().size()==count,"Prepare mutated live state");
    auto dx=x.delta(),dy=y.delta();
    // Synthetic projection changes direction only; both previews retain original moments.
    if(step%3==0)for(size_t i=0;i<dx.size();++i){dx[i]=dx[i]*.25;dy[i]=dy[i]*.25;}
    double alpha=step%2?.2:1.;auto cx=x.candidate(dx,alpha),cy=y.candidate(dy,alpha);
    if(step==1 || step==10 || step==100)for(size_t i=0;i<cx.size();++i) {
      compare(label+"/delta"+std::to_string(step)+"/"+std::to_string(i),y.delta()[i],x.delta()[i]);
      compare(label+"/candidate"+std::to_string(step)+"/"+std::to_string(i),cy[i],cx[i]);
    }
    if(step%5!=0){x.commit(cx);y.commit(cy);}
    else TORCH_CHECK(b[1].equal(before) && cuda.state().size()==count,"Reject mutated live state");
    if(step==1 || step==10 || step==37 || step==100)states(label+"/step"+std::to_string(step),old,cuda);
  }
  // Validate the whole candidate before writing the non-TF group.
  for(auto p:b)p.mutable_grad()=torch::ones_like(p);
  AdamStepPreview preview(cuda,{1});auto candidate=preview.candidate(preview.delta(),1);
  auto first=b[0].clone();candidate.back()=torch::empty({7},opt);bool caught=false;
  try{preview.commit(candidate);}catch(const c10::Error&){caught=true;}
  TORCH_CHECK(caught && b[0].equal(first),"Invalid TF candidate partially committed non-TF group");
  auto prepared=tf_cuda::tfCudaAdamPrepare(cuda,{1});tf_cuda::tfCudaAdamDiscard(prepared);
  TORCH_CHECK(prepared.empty(),"Discard retained prepared state");
  AdamStepPreview stale(cuda,{1});auto values=stale.candidate(stale.delta(),1);
  {torch::NoGradGuard ng;b[1].add_(1);}
  caught=false;try{stale.commit(values);}catch(const c10::Error&){caught=true;}
  TORCH_CHECK(caught && b[0].equal(first),"Stale TF candidate partially committed");
}
void guard(const std::string& label,double lr,bool projection,bool fallback,bool nonfinite=false) {
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
  auto a=torch::ones({2},opt),b=a.detach().clone().set_requires_grad(true);
  auto c=torch::ones({1},opt),d=c.detach().clone().set_requires_grad(true);
  torch::optim::Adam old({c},torch::optim::AdamOptions(lr).eps(1e-15));
  torch::optim::Adam cuda({d},torch::optim::AdamOptions(lr).eps(1e-15));
  old.add_param_group(std::vector<torch::Tensor>{a});cuda.add_param_group(std::vector<torch::Tensor>{b});
  c.mutable_grad()=torch::zeros_like(c);d.mutable_grad()=torch::zeros_like(d);
  a.mutable_grad()=projection?torch::tensor({-1.f,1.f},opt.requires_grad(false)):torch::ones_like(a);b.mutable_grad()=a.grad().clone();
  auto evaluation=[&](torch::Tensor p) {
    GuardEvaluation e;e.objectives={{"full"},{"lod"}};
    e.evaluate=[p,projection,nonfinite](size_t,const AdamStepPreview& preview,const std::vector<torch::Tensor>* candidate) {
      auto value=candidate?preview.value(p,*candidate):p;
      if(nonfinite && candidate)return torch::full({},std::numeric_limits<float>::quiet_NaN(),p.options().requires_grad(false));
      return projection?value[0].square():value.square().mean();
    };return e;
  };
  GradientConstraintConfig cfg;cfg.cancel_fallback_enable=fallback;
  auto x=GradientConstraintManager(cfg).step(old,evaluation(a)),y=GradientConstraintManager(cfg).step(cuda,evaluation(b),{1});
  TORCH_CHECK(x.status==y.status && x.reason==y.reason && x.committed==y.committed && x.candidates==y.candidates && x.alpha==y.alpha,"Guard decision parity ",label);
  TORCH_CHECK(label!="project" || y.reason=="projected_validated","Projection test missed projection");
  TORCH_CHECK(label=="project" || label=="nonfinite" || y.status==label,"Guard test missed expected branch ",label,"/",y.status);
  states(label,old,cuda);
}
struct Event {cudaEvent_t v;Event(){cudaEventCreate(&v);}~Event(){cudaEventDestroy(v);}
  void record(){cudaEventRecord(v,c10::cuda::getCurrentCUDAStream());}};
}
int main(int argc,char** argv) {
  TORCH_CHECK(argc==3,"cuda_tf_preview_test OUTPUT_DIRECTORY CHECKPOINT");torch::set_num_threads(1);torch::manual_seed(20261019);
  parity.open(std::string(argv[1])+"/phase_f2_parity.csv");parity<<"case,max_abs,passed\n";
  sequence("default_stream");{c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());sequence("nondefault_stream");}
  guard("ACCEPT",.1,false,false);guard("SHRINK",3,false,false);guard("CANCEL",12,false,false);
  guard("project",.1,true,false);guard("FALLBACK",12,false,true);guard("nonfinite",12,false,true,true);
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);
  torch::serialize::InputArchive archive;archive.load_from(std::string(argv[2])+"/transform_field.pt");tf->load(archive);tf->to(torch::kCUDA);
  for(auto t:tf->parameters())t.mutable_grad()=torch::randn_like(t)*.01;
  std::ofstream bench(std::string(argv[1])+"/phase_f2_benchmark.csv");bench<<"backend,stage,mean_ms,p50_ms,p95_ms\n";
  for(bool custom:{false,true}) {
    torch::optim::Adam optimizer(tf->mlpParameters(),torch::optim::AdamOptions(.00016).eps(1e-15));
    optimizer.add_param_group(tf->gridParameters());optimizer.param_groups()[1].options().set_lr(.0016);
    Event a,b;std::vector<float> samples;
    for(int i=0;i<600;++i) {
      a.record();AdamStepPreview preview(optimizer,custom?std::vector<size_t>{0,1}:std::vector<size_t>{});
      preview.commit(preview.candidate(preview.delta(),1));b.record();cudaEventSynchronize(b.v);
      if(i>=100){float ms;cudaEventElapsedTime(&ms,a.v,b.v);samples.push_back(ms);}
    }
    float mean=std::accumulate(samples.begin(),samples.end(),0.f)/samples.size();std::sort(samples.begin(),samples.end());
    bench<<(custom?"cuda":"legacy")<<",adam_preview,"<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';bench.flush();
  }
  std::cout<<"phase F2 passed"<<std::endl;
}
