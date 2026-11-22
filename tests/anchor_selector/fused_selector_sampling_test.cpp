#include "include/selector_sampling_cuda.h"
#include "include/selector_handoff.h"
#include "improvements/selector/gumbel_network.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
namespace fs=std::filesystem;using T=torch::Tensor;using L=std::vector<T>;
L legacy(const T& p,const T& u,const T& opacity,const T& inactive){
  auto effective=inactive.defined()?p*(~(inactive&(opacity.reshape({-1})<1.f/255.f))).to(p.scalar_type()):p;
  auto mask=sampleSelectorST(effective,u);return {effective,mask,torch::nonzero(mask.detach()>0).reshape({-1})};
}
T leaf(const T& t){return torch::empty_strided(t.sizes(),t.strides(),t.options()).copy_(t).requires_grad_(true);}
void compare(std::ofstream& out,const std::string& label,const T& a,const T& b,bool exact){
  TORCH_CHECK(a.sizes()==b.sizes(),"sampling shape ",label);auto d=(a.to(torch::kFloat64)-b.to(torch::kFloat64)).abs();
  bool ok=exact?torch::equal(a,b):torch::allclose(a,b,2e-4,2e-5);int64_t i=d.numel()?d.reshape({-1}).argmax().item<int64_t>():-1;
  out<<label<<','<<exact<<','<<(d.numel()?d.max().item<double>():0)<<','<<(d.numel()?(d/a.to(torch::kFloat64).abs().clamp_min(1e-30)).max().item<double>():0)<<','
    <<(d.numel()?d.square().mean().sqrt().item<double>():0)<<','<<i<<','<<ok<<'\n';TORCH_CHECK(ok,"Sampling parity ",label);
}
void parity(const T& p,const T& u,const T& opacity,const T& inactive,float ratio,const std::string& label,std::ofstream& csv){
  auto a=leaf(p),b=leaf(p);auto x=legacy(a,u,opacity,inactive),y=mapping_cuda::selectorSampleCompact(b,u,opacity,inactive);
  for(int i=0;i<3;++i)compare(csv,label+"/output"+std::to_string(i),x[i],y[i],true);
  if(p.numel()){auto weight=torch::randn_like(p);auto lossx=selectorRatioLoss(x[0],ratio),lossy=selectorRatioLoss(y[0],ratio);
    compare(csv,label+"/budget",lossx,lossy,true);
    auto ga=torch::autograd::grad({(x[1]*weight).sum()+.01*lossx},{a})[0];auto gb=torch::autograd::grad({(y[1]*weight).sum()+.01*lossy},{b})[0];
    compare(csv,label+"/photo_budget_vjp",ga,gb,true);
    auto pa=leaf(p),pb=leaf(p);auto xa=legacy(pa,u,opacity,inactive),xb=mapping_cuda::selectorSampleCompact(pb,u,opacity,inactive);
    compare(csv,label+"/budget_only_vjp",torch::autograd::grad({selectorRatioLoss(xa[0],ratio)},{pa})[0],torch::autograd::grad({selectorRatioLoss(xb[0],ratio)},{pb})[0],true);
  }
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==4,"fused_selector_sampling_test checkpoint fixture out");torch::set_num_threads(2);torch::manual_seed(2026);fs::path out=argv[3];fs::create_directories(out);
  std::ofstream csv(out/"sampling_parity.csv");csv<<"name,exact,max_abs,max_rel,rms,worst_index,passed\n";
  std::ofstream timing(out/"sampling_timing.csv");timing<<"ratio,N,repeat,implementation,stage,host_ms,cuda_ms,K\n";
  auto options=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  for(int n:{0,1,7,64,15891,50000,100000}){
    auto storage=torch::rand({n,2},options);auto p=storage.select(1,1);auto u=torch::rand({n,2},options).select(1,0);
    auto opacity=torch::rand({n,1},options),inactive=torch::rand({n},options)>.7;
    for(bool slots:{false,true})parity(p,u,opacity,slots?inactive:T(),.05,"random_stride2/N"+std::to_string(n)+"/slots"+std::to_string(slots),csv);
    if(n){parity(torch::zeros_like(p),torch::zeros_like(p),opacity,{},.01,"equal_K0",csv);
      parity(torch::ones_like(p),torch::zeros_like(p),opacity,{},.15,"KN",csv);
      auto one=torch::zeros_like(p);one[0]=1;parity(one,torch::zeros_like(p),opacity,{},.01,"K1",csv);
      parity(p,p,opacity,inactive,.05,"u_equal_p",csv);
      opacity.fill_(1.f/255.f);parity(p,u,opacity,torch::ones_like(inactive),.05,"opacity_cutoff_equal",csv);
      auto exact=torch::full_like(p,.125);parity(exact,u,opacity,{},.125,"budget_exact_zero",csv);
    }
  }
  fs::path fixtures=fs::path(argv[2]).parent_path();
  cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
  for(float ratio:{.01f,.05f,.1f,.15f}){
    torch::serialize::InputArchive f;f.load_from((fixtures/("lod_"+std::to_string(ratio)+"_parity.pt")).string(),torch::kCUDA);T p,u,logit;
    f.read("probability",p);f.read("uniform",u);f.read("parameter_before_3",logit);auto alpha=logit.sigmoid();
    auto inactive=torch::zeros({p.numel()},options.dtype(torch::kBool));parity(p,u,alpha,inactive,ratio,"real"+std::to_string(ratio),csv);
    auto x=leaf(p),weight=torch::randn_like(p);
    for(int rep=-10;rep<100;++rep)for(int order=0;order<2;++order){bool use=(rep+order)%2==0;L o;T budget;
      torch::cuda::synchronize();nvtxRangePushA(use?"fused/sample_budget_compact":"legacy/sample_budget_compact");cudaEventRecord(start);auto t=std::chrono::steady_clock::now();
      o=use?mapping_cuda::selectorSampleCompact(x,u,alpha,inactive):legacy(x,u,alpha,inactive);budget=selectorRatioLoss(o[0],ratio);
      double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,start,end);nvtxRangePop();
      if(rep>=0)timing<<ratio<<','<<p.numel()<<','<<rep<<','<<(use?"fused":"legacy")<<",forward,"<<host<<','<<ms<<','<<o[2].numel()<<'\n';
      auto loss=(o[1]*weight).sum()+.01*budget;torch::cuda::synchronize();cudaEventRecord(start);t=std::chrono::steady_clock::now();auto grad=torch::autograd::grad({loss},{x});
      host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&ms,start,end);
      if(rep>=0)timing<<ratio<<','<<p.numel()<<','<<rep<<','<<(use?"fused":"legacy")<<",backward,"<<host<<','<<ms<<','<<o[2].numel()<<'\n';
    }
  }
  cudaEventDestroy(start);cudaEventDestroy(end);std::cout<<"PASS stable IDs/K,ST/Budget VJP exact,stride2,boundaries,N0/1/100K,paired timing\n";
  std::ofstream scaling(out/"sampling_scaling.csv");scaling<<"N,ratio,repeat,fused,stage,host_ms,cuda_ms,K\n";cudaEventCreate(&start);cudaEventCreate(&end);
  for(int n:{15891,50000,100000})for(float ratio:{.01f,.05f,.1f,.15f}){
    auto p=leaf(torch::full({n,2},ratio,options).select(1,1)),u=torch::rand({n},options),alpha=torch::ones({n,1},options),inactive=torch::zeros({n},options.dtype(torch::kBool)),w=torch::randn({n},options);
    for(int rep=-5;rep<40;++rep)for(int order=0;order<2;++order){bool use=(rep+order)%2==0;L o;T budget;torch::cuda::synchronize();cudaEventRecord(start);auto t=std::chrono::steady_clock::now();
      o=use?mapping_cuda::selectorSampleCompact(p,u,alpha,inactive):legacy(p,u,alpha,inactive);budget=selectorRatioLoss(o[0],ratio);
      double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,start,end);
      if(rep>=0)scaling<<n<<','<<ratio<<','<<rep<<','<<use<<",forward,"<<host<<','<<ms<<','<<o[2].numel()<<'\n';
      auto loss=(o[1]*w).sum()+.01*budget;torch::cuda::synchronize();cudaEventRecord(start);t=std::chrono::steady_clock::now();auto gradient=torch::autograd::grad({loss},{p});
      host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&ms,start,end);
      if(rep>=0)scaling<<n<<','<<ratio<<','<<rep<<','<<use<<",backward,"<<host<<','<<ms<<','<<o[2].numel()<<'\n';
    }
  }cudaEventDestroy(start);cudaEventDestroy(end);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
