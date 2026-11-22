#include "include/gaussian_rasterizer.h"
#include "cuda_rasterizer/render_workspace.h"
#include <nvToolsExt.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
using T=torch::Tensor;using List=std::vector<T>;namespace fs=std::filesystem;
void compare(std::ofstream& csv,const std::string& name,const T& a,const T& b,const T& baseline_envelope={}){
  TORCH_CHECK(a.defined()==b.defined(),name," gradNone");if(!a.defined()||!a.numel())return;
  auto diff=(a.to(torch::kFloat64)-b.to(torch::kFloat64)).abs();auto bad=diff>2e-5+2e-4*b.abs();
  if(!a.is_floating_point())bad=a!=b;
  auto strict_bad=bad.sum().item<int64_t>();
  if(baseline_envelope.defined())bad=diff>2e-5+2e-4*b.abs()+baseline_envelope;
  csv<<name<<','<<diff.max().item<double>()<<','<<(diff/b.to(torch::kFloat64).abs().clamp_min(1e-30)).max().item<double>()<<','<<diff.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<','<<strict_bad<<'\n';
  TORCH_CHECK(torch::isfinite(a).all().item<bool>()&&!bad.any().item<bool>(),name);
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc>=4,"renderer_workspace_test checkpoint fixture output [stress]");torch::set_num_threads(2);torch::manual_seed(2026);
  fs::path fixtures=fs::path(argv[2]).parent_path(),out=argv[3];fs::create_directories(out);
  std::ofstream csv(out/"workspace_parity.csv"),timing(out/"workspace_timing.csv");csv<<"name,max_abs,max_rel,rms,bad_elements,strict_bad_elements\n";timing<<"case,repeat,reuse,host_ms,cuda_ms,hits,misses,growths,retained_bytes\n";
  for(const auto& name:{"Full","lod_0.010000","lod_0.050000","lod_0.100000","lod_0.150000"}){
    torch::serialize::InputArchive ar;ar.load_from((fixtures/(std::string(name)+"_parity.pt")).string(),torch::kCUDA);
    T x,scale,q,op,dc,sh,ids,view,proj,gt;
    ar.read("xyz",x);ar.read("scale",scale);ar.read("rotation",q);ar.read("opacity",op);ar.read("dc",dc);ar.read("parameter_before_2",sh);
    ar.read("view",view);ar.read("projection",proj);ar.read("gt",gt);if(std::string(name)!="Full"){ar.read("selected_ids",ids);sh=sh.index_select(0,ids);}
    List base{x,dc,sh,op,scale,q};auto bg=torch::zeros({3},x.options()),campos=view.inverse().select(0,3).slice(0,0,3).contiguous();
    GaussianRasterizationSettings settings(gt.size(1),gt.size(2),.6f,.45f,bg,1.f,view,proj,3,campos,false,false,true);
    auto leaves=[&](){List result;for(auto& t:base)result.push_back(t.detach().clone().requires_grad_(true));return result;};
    auto forward=[&](List& a,bool reuse){renderer_workspace::Scope scope(reuse);auto xy=torch::zeros_like(a[0]).requires_grad_(true),empty=torch::empty({0},x.options());
      return rasterizeGaussians(a[0],xy,a[1],a[2],empty,a[3],a[4],a[5],empty,settings);};
    auto a=leaves(),b=leaves();auto y=forward(a,false),z=forward(b,true);
    // A second live Forward must never overwrite the first context.
    auto second=leaves();auto z2=forward(second,true);
    for(size_t i=0;i<y.size();++i)compare(csv,std::string(name)+"/output"+std::to_string(i),y[i],z[i]);
    // Default VJP has the production mean-L1 scale. The optional, separately
    // reported stress probe is ~1000x larger and amplifies rasterizer atomics.
    auto weights=argc>4?torch::randn_like(y[0])*.001:torch::sign(y[0].detach()-gt)/double(gt.numel());
    auto ga=torch::autograd::grad({y[0]},a,{weights},true,false,true);
    List envelope;for(auto& t:ga)envelope.push_back(torch::zeros_like(t));
    // Register the observed per-element atomic envelope from three identical
    // legacy Backwards; do not relax the fixed-gradient optimizer tolerance.
    for(int repeat=0;repeat<3;++repeat){auto gr=torch::autograd::grad({y[0]},a,{weights},true,false,true);
      for(size_t i=0;i<ga.size();++i)envelope[i]=torch::maximum(envelope[i],(ga[i]-gr[i]).abs());}
    auto gb=torch::autograd::grad({z[0]},b,{weights},true,false,true);
    auto gc=torch::autograd::grad({z2[0]},second,{weights},false,false,true);
    for(size_t i=0;i<a.size();++i){compare(csv,std::string(name)+"/legacy_atomic_envelope"+std::to_string(i),envelope[i],torch::zeros_like(envelope[i]),envelope[i]);compare(csv,std::string(name)+"/gradient"+std::to_string(i),ga[i],gb[i],envelope[i]);compare(csv,std::string(name)+"/second_live"+std::to_string(i),ga[i],gc[i],envelope[i]);}
    auto gb2=torch::autograd::grad({z[0]},b,{weights},false,false,true);for(size_t i=0;i<a.size();++i)compare(csv,std::string(name)+"/retained"+std::to_string(i),ga[i],gb2[i],envelope[i]);
    y.clear();z.clear();z2.clear();
    if(std::string(name)=="Full")for(int k:{0,1}){
      List ca,cb;for(auto& value:base){ca.push_back(value.slice(0,0,k).detach().clone().requires_grad_(true));cb.push_back(value.slice(0,0,k).detach().clone().requires_grad_(true));}
      auto ya=forward(ca,false),yb=forward(cb,true);for(size_t i=0;i<ya.size();++i)compare(csv,"K"+std::to_string(k)+"/output"+std::to_string(i),ya[i],yb[i]);
      auto da=torch::autograd::grad({ya[0]},ca,{torch::ones_like(ya[0])/double(gt.numel())},false,false,true),db=torch::autograd::grad({yb[0]},cb,{torch::ones_like(yb[0])/double(gt.numel())},false,false,true);
      for(size_t i=0;i<ca.size();++i)compare(csv,"K"+std::to_string(k)+"/VJP"+std::to_string(i),da[i],db[i]);
    }
    cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
    for(int rep=-10;rep<80;++rep)for(int order=0;order<2;++order){bool reuse=(rep+order)%2==0;auto values=leaves();torch::cuda::synchronize();
      nvtxRangePushA(reuse?"workspace/reuse":"workspace/legacy");cudaEventRecord(begin);auto start=std::chrono::steady_clock::now();
      auto output=forward(values,reuse);auto grads=torch::autograd::grad({output[0]},values,{weights},false,false,true);
      double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();cudaEventRecord(end);nvtxRangePop();cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,begin,end);auto s=renderer_workspace::stats();
      if(rep>=0)timing<<name<<','<<rep<<','<<reuse<<','<<host<<','<<ms<<','<<s.hits<<','<<s.misses<<','<<s.growths<<','<<s.retained_bytes<<'\n';}
    cudaEventDestroy(begin);cudaEventDestroy(end);renderer_workspace::clearIdle();
  }
  std::cout<<"PASS workspace saved contexts, retained graph, Forward/VJP and paired timing\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
