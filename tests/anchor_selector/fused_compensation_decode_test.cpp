#include "include/anchor_selector_cuda.h"
#include "improvements/selector/gumbel_network.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

using Tensor=torch::Tensor;using List=std::vector<Tensor>;
using Net=improvements::selector::GumbelNetworkImpl;
namespace fs=std::filesystem;
List legacy(const List& a) {
  auto geom=Net::applyCompensationResidual({a[0],a[1].log(),a[2]},a[5]);
  auto appearance=Net::applyAppearanceResidual(a[4],a[3],a[5]);
  return {geom.xyz,a[1]*a[5].slice(1,3,6).exp(),geom.rotation,
      appearance.second.sigmoid()*a[6].unsqueeze(1),appearance.first};
}
List clone(const List& a) {List result;for(const auto& x:a)result.push_back(torch::empty_strided(x.sizes(),x.strides(),x.options()).copy_(x.detach()).requires_grad_(true));return result;}
void compare(std::ofstream& out,const std::string& name,const Tensor& a,const Tensor& b) {
  TORCH_CHECK(a.defined()==b.defined(),"Decode gradNone mismatch: ",name);if(!a.defined())return;
  TORCH_CHECK(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type()&&a.device()==b.device(),"Decode layout mismatch");
  if(!a.numel())return;
  auto finite=torch::isfinite(a)&torch::isfinite(b);auto diff=torch::where(finite,(a-b).abs(),torch::zeros_like(a));
  auto bad=(finite&(diff>2e-5+2e-4*b.abs())) | (torch::isnan(a)!=torch::isnan(b)) |
      (torch::isinf(a)!=torch::isinf(b)) | (torch::isinf(a)&torch::isinf(b)&(a!=b));
  out<<name<<','<<diff.max().item<double>()<<','<<(diff/b.abs().clamp_min(1e-30)).nan_to_num().max().item<double>()<<','
     <<diff.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<','<<torch::isnan(a).sum().item<int64_t>()<<'\n';
  TORCH_CHECK(!bad.any().item<bool>(),"Decode parity failed: ",name);
}
List fixture(const fs::path& path) {
  torch::serialize::InputArchive a;a.load_from(path.string(),torch::kCUDA);
  Tensor ids,x,scale,q,dc,op,delta;
  a.read("selected_ids",ids);a.read("parameter_before_0",x);a.read("parameter_before_4",scale);
  a.read("parameter_before_5",q);a.read("parameter_before_1",dc);a.read("parameter_before_3",op);a.read("residual",delta);
  auto rotation=torch::nn::functional::normalize(q);
  return {x.index_select(0,ids),scale.exp().index_select(0,ids),rotation.index_select(0,ids),op.index_select(0,ids),dc.index_select(0,ids),delta,torch::ones({ids.numel()},x.options())};
}
void parity(const List& values,const std::string& name,std::ofstream& log,bool only_xyz=false) {
  auto a=clone(values),b=clone(values);auto x=legacy(a),y=mapping_cuda::decode14(b[0],b[1],b[2],b[3],b[4],b[5],b[6]);
  List gx;for(size_t j=0;j<x.size();++j){compare(log,name+"/forward"+std::to_string(j),x[j],y[j]);gx.push_back(torch::randn_like(x[j])*.001);}
  if(only_xyz){x={x[0]};y={y[0]};gx={gx[0]};}
  auto dx=torch::autograd::grad(x,a,gx,false,false,true),dy=torch::autograd::grad(y,b,gx,false,false,true);
  for(size_t j=0;j<dx.size();++j)compare(log,name+"/backward"+std::to_string(j),dx[j],dy[j]);
}
int main(int argc,char** argv) {
 try {
  TORCH_CHECK(argc==3,"fused_compensation_decode_test fixture_dir output_dir");torch::set_num_threads(2);torch::manual_seed(2026);
  fs::path input=argv[1],output=argv[2];fs::create_directories(output);
  std::ofstream log(output/"decode_parity.csv");log<<"name,max_abs,max_rel,rms,bad_elements,nan_elements\n";
  std::ofstream timing(output/"decode_timing.csv");timing<<"case,K,repeat,implementation,stage,host_ms,cuda_ms\n";
  for(const auto& name:{"lod_0.010000","lod_0.050000","lod_0.100000","lod_0.150000"}) {
    auto values=fixture(input/(std::string(name)+"_parity.pt"));parity(values,name,log);parity(values,std::string(name)+"/unused_outputs",log,true);
    auto a=clone(values);List grads;for(const auto& v:legacy(a))grads.push_back(torch::randn_like(v)*.001);
    cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
    for(int repeat=-10;repeat<100;++repeat)for(int order=0;order<2;++order) {
      bool fused=(repeat+order)%2==0;torch::cuda::synchronize();cudaEventRecord(start);auto begin=std::chrono::steady_clock::now();
      auto out=fused?mapping_cuda::decode14(a[0],a[1],a[2],a[3],a[4],a[5],a[6]):legacy(a);
      double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,start,end);
      if(repeat>=0)timing<<name<<','<<a[0].size(0)<<','<<repeat<<','<<(fused?"fused":"legacy")<<",forward,"<<host<<','<<ms<<'\n';
      cudaEventRecord(start);begin=std::chrono::steady_clock::now();auto grad=torch::autograd::grad(out,a,grads,false,false,true);
      host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&ms,start,end);
      if(repeat>=0)timing<<name<<','<<a[0].size(0)<<','<<repeat<<','<<(fused?"fused":"legacy")<<",backward,"<<host<<','<<ms<<'\n';
    }
    cudaEventDestroy(start);cudaEventDestroy(end);
    std::cout<<"PASS "<<name<<" decode/VJP paired timing\n";
  }
  auto base=fixture(input/"lod_0.050000_parity.pt");
  for(int n:{0,1,7}) {List v;for(auto x:base)v.push_back(x.slice(0,0,n));parity(v,"K"+std::to_string(n),log);}
  auto edge=clone(base);edge[5].data().zero_();parity(edge,"identity",log);
  edge[5].data().slice(1,6,10).fill_(1e-14);edge[5].data().select(1,6).fill_(-1.f);parity(edge,"tiny_quaternion",log);
  edge[5].data().slice(1,6,10).zero_();edge[5].data().select(1,6).fill_(-1.f);parity(edge,"zero_quaternion_legacy_nan",log);
  edge=clone(base);edge[3].data().fill_(-100);edge[5].data().slice(1,3,6).fill_(15);edge[5].data().slice(1,0,3).fill_(10000);parity(edge,"extreme_attributes",log);
  List strided;for(auto x:base)strided.push_back(x.repeat_interleave(2,0).slice(0,0,x.size(0)*2,2));parity(strided,"strided",log);
  auto da=clone(base),db=clone(base);List gathered_a,gathered_b;
  auto duplicate_ids=torch::arange(base[0].size(0),base[0].options().dtype(torch::kLong)).repeat_interleave(2);
  for(size_t j=0;j<base.size();++j){gathered_a.push_back(da[j].index_select(0,duplicate_ids));gathered_b.push_back(db[j].index_select(0,duplicate_ids));}
  auto oa=legacy(gathered_a),ob=mapping_cuda::decode14(gathered_b[0],gathered_b[1],gathered_b[2],gathered_b[3],gathered_b[4],gathered_b[5],gathered_b[6]);
  List upstream;for(auto x:oa)upstream.push_back(torch::randn_like(x)*.001);
  auto ga=torch::autograd::grad(oa,da,upstream),gb=torch::autograd::grad(ob,db,upstream);
  for(size_t j=0;j<ga.size();++j)compare(log,"repeated_ids/scatter"+std::to_string(j),ga[j],gb[j]);
  edge=clone(base);edge[3].data().fill_(100);edge[5].data().slice(1,3,6).fill_(-50);parity(edge,"positive_saturation_tiny_scale",log);
  std::cout<<"PASS Decode14 forward,VJP,gradNone,empty,identity,tiny/zero q,extremes,strides\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
