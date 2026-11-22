#include "include/mapping_cuda_optimizer.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

namespace fs=std::filesystem;
struct Copy {
  std::vector<torch::Tensor> p;
  std::unique_ptr<torch::optim::Adam> adam;
};
Copy fixture(const fs::path& path) {
  torch::serialize::InputArchive a;a.load_from(path.string(),torch::kCUDA);
  Copy c;
  for(int i=0;;++i) {
    torch::Tensor p;if(!a.try_read("parameter_before_"+std::to_string(i),p))break;
    c.p.push_back(p.detach().clone().requires_grad_(true));
    const double lr=i<6?std::vector<double>{.00016,.0025,.000125,.05,.005,.001}[i]:.001;
    auto o=torch::optim::AdamOptions(lr).eps(1e-15);
    if(i==0)c.adam=std::make_unique<torch::optim::Adam>(std::vector<torch::Tensor>{c.p.back()},o);
    else c.adam->add_param_group(torch::optim::OptimizerParamGroup({c.p.back()},std::make_unique<torch::optim::AdamOptions>(o)));
    torch::Tensor grad;if(a.try_read("gradient_"+std::to_string(i),grad))c.p.back().mutable_grad()=grad.clone();
    torch::Tensor m,v,step;
    if(a.try_read("before_"+std::to_string(i)+"_m",m)) {
      a.read("before_"+std::to_string(i)+"_v",v);a.read("before_"+std::to_string(i)+"_step",step);
      auto s=std::make_unique<torch::optim::AdamParamState>();s->step(step.item<int64_t>());
      s->exp_avg(m.clone());s->exp_avg_sq(v.clone());c.adam->state()[c.p.back().unsafeGetTensorImpl()]=std::move(s);
    }
  }
  return c;
}
void compare(std::ofstream& out,const std::string& name,const torch::Tensor& a,const torch::Tensor& b) {
  constexpr double atol=2e-7,rtol=2e-6;
  auto error=(a-b).abs();auto bad=error>atol+rtol*b.abs();
  out<<name<<','<<error.max().item<double>()<<','<<(error/b.abs().clamp_min(1e-30)).max().item<double>()<<','
     <<error.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<','<<atol<<','<<rtol<<'\n';
  TORCH_CHECK(!bad.any().item<bool>(),"Fused Adam parity failed: ",name);
}
void check(Copy& a,Copy& b,std::ofstream& out,const std::string& name) {
  for(size_t i=0;i<a.p.size();++i) {
    const auto key=name+"/"+std::to_string(i);
    compare(out,key+"/parameter",a.p[i],b.p[i]);
    const auto sa=a.adam->state().find(a.p[i].unsafeGetTensorImpl()),sb=b.adam->state().find(b.p[i].unsafeGetTensorImpl());
    TORCH_CHECK((sa==a.adam->state().end())==(sb==b.adam->state().end()),"GradNone/state mismatch");
    if(sa==a.adam->state().end())continue;
    auto& x=static_cast<torch::optim::AdamParamState&>(*sa->second);auto& y=static_cast<torch::optim::AdamParamState&>(*sb->second);
    TORCH_CHECK(x.step()==y.step(),"Independent leaf clock mismatch");
    compare(out,key+"/m",x.exp_avg(),y.exp_avg());compare(out,key+"/v",x.exp_avg_sq(),y.exp_avg_sq());
  }
}
int main(int argc,char** argv) {
 try {
  TORCH_CHECK(argc==3,"fused_mapping_adam_test fixture_dir output_dir");
  torch::set_num_threads(2);torch::manual_seed(2026);torch::NoGradGuard no_grad;
  fs::path input=argv[1],output=argv[2];fs::create_directories(output);
  std::ofstream parity(output/"adam_parity.csv");parity<<"name,max_abs,max_rel,rms,bad_elements,atol,rtol\n";
  std::ofstream timing(output/"adam_timing.csv");timing<<"case,round,implementation,host_ms,cuda_ms\n";
  for(const auto& name:{"Full","lod_0.010000","lod_0.050000","lod_0.100000","lod_0.150000"}) {
    auto a=fixture(input/(std::string(name)+"_parity.pt")),b=fixture(input/(std::string(name)+"_parity.pt"));
    for(int t=0;t<30;++t){a.adam->step();TORCH_CHECK(mapping_cuda::tryFusedDenseAdam(*b.adam),"Supported fixture fell back");check(a,b,parity,std::string(name)+"/step"+std::to_string(t));}
    cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
    for(int repeat=-20;repeat<200;++repeat)for(int order=0;order<2;++order) {
      const bool fused=(repeat+order)%2==0;auto& c=fused?b:a;
      torch::cuda::synchronize();cudaEventRecord(start);auto wall=std::chrono::steady_clock::now();
      if(fused)mapping_cuda::tryFusedDenseAdam(*c.adam);else c.adam->step();
      auto host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count();cudaEventRecord(end);cudaEventSynchronize(end);
      float elapsed;cudaEventElapsedTime(&elapsed,start,end);
      if(repeat>=0)timing<<name<<','<<repeat<<','<<(fused?"fused":"legacy")<<','<<host<<','<<elapsed<<'\n';
    }
    cudaEventDestroy(start);cudaEventDestroy(end);
    std::cout<<"PASS "<<name<<" frozen grad/moments,30 steps,paired timing\n";
  }
  // Non-selected rows with old moment still move; inactive tables never step.
  auto p=torch::zeros({128,14},torch::TensorOptions().device(torch::kCUDA));
  Copy a,b;a.p={p.clone(),p.clone()};b.p={p.clone(),p.clone()};
  a.adam=std::make_unique<torch::optim::Adam>(a.p,torch::optim::AdamOptions(.001).eps(1e-15));
  b.adam=std::make_unique<torch::optim::Adam>(b.p,torch::optim::AdamOptions(.001).eps(1e-15));
  for(int t=0;t<100;++t) {
    auto grad=torch::zeros_like(p);grad[t%128].fill_(1e-10);
    a.p[0].mutable_grad()=grad;b.p[0].mutable_grad()=grad;
    a.adam->step();mapping_cuda::tryFusedDenseAdam(*b.adam);check(a,b,parity,"tiny_dense/"+std::to_string(t));
  }
  TORCH_CHECK(b.p[0][0].abs().sum().item<float>()>0 && b.adam->state().count(b.p[1].unsafeGetTensorImpl())==0,"Dense momentum or inactive clock changed");
  // Unsupported layout/device must return before mutating any parameter.
  auto cpu=torch::ones({4});torch::optim::Adam cpu_adam({cpu},torch::optim::AdamOptions(.001));cpu.mutable_grad()=torch::ones_like(cpu);
  TORCH_CHECK(!mapping_cuda::tryFusedDenseAdam(cpu_adam) && cpu_adam.state().empty(),"CPU fallback mutated state");
  auto strided=torch::ones({3,4},p.options()).transpose(0,1);strided.mutable_grad()=torch::ones_like(strided);
  torch::optim::Adam layout_adam({strided},torch::optim::AdamOptions(.001));
  TORCH_CHECK(!mapping_cuda::tryFusedDenseAdam(layout_adam) && layout_adam.state().empty(),"Layout fallback mutated state");
  Copy options_a,options_b;
  for(int i=0;i<67;++i) {
    auto value=torch::randn({i+1},p.options());options_a.p.push_back(value.clone());options_b.p.push_back(value.clone());
    auto opt=torch::optim::AdamOptions(.001/(i+1)).eps(i%2?1e-15:1e-8).betas(std::make_tuple(.8,.98)).weight_decay(.01);
    if(i==0) {options_a.adam=std::make_unique<torch::optim::Adam>(std::vector<torch::Tensor>{options_a.p.back()},opt);
      options_b.adam=std::make_unique<torch::optim::Adam>(std::vector<torch::Tensor>{options_b.p.back()},opt);}
    else {options_a.adam->add_param_group(torch::optim::OptimizerParamGroup({options_a.p.back()},std::make_unique<torch::optim::AdamOptions>(opt)));
      options_b.adam->add_param_group(torch::optim::OptimizerParamGroup({options_b.p.back()},std::make_unique<torch::optim::AdamOptions>(opt)));}
  }
  for(int t=0;t<30;++t) {
    for(size_t i=0;i<options_a.p.size();++i) {
      auto grad=torch::randn_like(options_a.p[i])*.0001;
      options_a.p[i].mutable_grad()=i%7==size_t(t%7)?torch::Tensor():grad;
      options_b.p[i].mutable_grad()=options_a.p[i].grad();
    }
    options_a.adam->step();TORCH_CHECK(mapping_cuda::tryFusedDenseAdam(*options_b.adam),"Multi-batch options fallback");
    check(options_a,options_b,parity,"options_multibatch/"+std::to_string(t));
  }
  torch::optim::Adam ams({p},torch::optim::AdamOptions(.001).amsgrad(true));
  TORCH_CHECK(!mapping_cuda::tryFusedDenseAdam(ams)&&ams.state().empty(),"AMSGrad must safely fall back");
  auto storage=torch::ones({10},p.options());auto left=storage.slice(0,0,7),right=storage.slice(0,3,10);
  left.mutable_grad()=torch::ones_like(left);right.mutable_grad()=torch::ones_like(right);
  torch::optim::Adam alias({left,right},torch::optim::AdamOptions(.001));
  TORCH_CHECK(!mapping_cuda::tryFusedDenseAdam(alias)&&alias.state().empty()&&torch::equal(storage,torch::ones_like(storage)),
      "Overlapping views must fall back before mutation");
  std::cout<<"PASS fixed Adam tolerances, dense tiny gradients,gradNone,LR,leaf clocks,fallback\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
