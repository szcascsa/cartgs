#include <torch/torch.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>
#include "improvements/selector/gumbel_network.h"
#include "include/transform_field_cuda_optimizer.h"
#include "include/constrained_adam.h"

using improvements::selector::GumbelNetwork;
int main(int argc,char** argv) {
  TORCH_CHECK(argc==2,"optimizer_candidate OUTPUT_DIRECTORY");torch::set_num_threads(1);torch::manual_seed(123);
  std::string dir=argv[1];std::ofstream parity(dir+"/optimizer_candidate_parity.csv");
  parity<<"mode,step,tensor,field,max_abs,exact,within_initial_tolerance\n";
  int failures=0;
  auto compare=[&](const std::string& mode,int step,size_t tensor,const std::string& field,const torch::Tensor& a,const torch::Tensor& b) {
    bool exact=torch::equal(a,b),close=torch::allclose(a,b,2e-4,2e-6);
    parity<<mode<<','<<step<<','<<tensor<<','<<field<<','<<(a-b).abs().max().item<double>()<<','<<exact<<','<<close<<'\n';
    if(!exact)++failures;
  };
  for(std::string mode:{"direct","preview","direct_options","preview_options"}) {
    GumbelNetwork a,b;a->to(torch::kCUDA);b->to(torch::kCUDA);
    auto pa=a->parameters(),pb=b->parameters();{torch::NoGradGuard ng;for(size_t i=0;i<pa.size();++i)pb[i].copy_(pa[i]);}
    const bool alternate=mode.find("options")!=std::string::npos;
    auto options=torch::optim::AdamOptions(.001).eps(alternate?1e-8:1e-15)
        .betas(alternate?std::make_tuple(.8,.99):std::make_tuple(.9,.999)).weight_decay(alternate?.02:0);
    torch::optim::Adam oa(pa,options);
    torch::optim::Adam ob(pb,options);
    for(int step=1;step<=1000;++step) {
      int before=failures;
      const double lr=step%17? .001*std::pow(.999,step):0;
      oa.param_groups()[0].options().set_lr(lr);ob.param_groups()[0].options().set_lr(lr);
      for(size_t i=0;i<pa.size();++i) {
        auto grad=(step+int(i))%7==0?torch::Tensor():
          (step%11==0?torch::zeros_like(pa[i]):torch::randn_like(pa[i])*.01);
        pa[i].mutable_grad()=grad;pb[i].mutable_grad()=grad;
      }
      if(mode.rfind("direct",0)==0){oa.step();tf_cuda::tfCudaAdamDirect(ob,{0});}
      else {
        std::vector<torch::Tensor> before_values;for(auto p:pb)before_values.push_back(p.detach().clone());
        const auto count=ob.state().size();AdamStepPreview x(oa),y(ob,{0});
        for(size_t i=0;i<pb.size();++i)TORCH_CHECK(torch::equal(pb[i],before_values[i]),"prepare mutated parameter");
        TORCH_CHECK(ob.state().size()==count,"prepare mutated live state");
        auto dx=x.delta(),dy=y.delta();
        if(step%3==0)for(size_t i=0;i<dx.size();++i){dx[i]=dx[i]*.25;dy[i]=dy[i]*.25;}
        auto cx=x.candidate(dx,step%2?.2:1.),cy=y.candidate(dy,step%2?.2:1.);
        if(step%5!=0){x.commit(cx);y.commit(cy);}
        else for(size_t i=0;i<pb.size();++i)TORCH_CHECK(torch::equal(pb[i],before_values[i]),"discard mutated parameter");
      }
      for(size_t i=0;i<pa.size();++i) {
        compare(mode,step,i,"parameter",pa[i],pb[i]);
        auto sa=oa.state().find(pa[i].unsafeGetTensorImpl()),sb=ob.state().find(pb[i].unsafeGetTensorImpl());
        TORCH_CHECK((sa==oa.state().end())==(sb==ob.state().end()),"state presence");
        if(sa==oa.state().end())continue;
        const auto& x=static_cast<const torch::optim::AdamParamState&>(*sa->second);
        const auto& y=static_cast<const torch::optim::AdamParamState&>(*sb->second);
        TORCH_CHECK(x.step()==y.step(),"per-tensor step mismatch");
        compare(mode,step,i,"m",x.exp_avg(),y.exp_avg());compare(mode,step,i,"v",x.exp_avg_sq(),y.exp_avg_sq());
      }
      parity.flush();if(failures>before){std::cout<<mode<<" exact failure at step="<<step<<'\n';break;}
    }
  }
  std::ofstream bench(dir+"/optimizer_candidate_benchmark.csv");bench<<"backend,mean_wall_ms,p50_wall_ms,p95_wall_ms\n";
  for(bool fused:{false,true}) {
    GumbelNetwork m;m->to(torch::kCUDA);torch::optim::Adam o(m->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
    for(auto p:m->parameters())p.mutable_grad()=torch::full_like(p,.0001);
    std::vector<double> samples;
    for(int i=0;i<600;++i) {
      cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
      if(fused)tf_cuda::tfCudaAdamDirect(o,{0});else o.step();cudaDeviceSynchronize();
      if(i>=100)samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    auto mean=std::accumulate(samples.begin(),samples.end(),0.)/samples.size();std::sort(samples.begin(),samples.end());
    bench<<(fused?"fused_candidate":"legacy")<<','<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';
  }
  std::cout<<"Selector fused Adam exact failures="<<failures<<"; production optimizer unchanged\n";
  return failures?1:0;
}
