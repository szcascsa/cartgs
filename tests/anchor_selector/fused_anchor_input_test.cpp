#include "include/anchor_input_cuda.h"
#include "improvements/selector/gumbel_network.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

using Tensor=torch::Tensor;using List=std::vector<Tensor>;namespace fs=std::filesystem;
using Net=improvements::selector::GumbelNetwork;
Tensor legacy(const List& a,double ratio) {
  return torch::cat({a[1].index_select(0,a[2]),(a[0]-a[3].index_select(0,a[2]))/a[4].index_select(0,a[2]),
      torch::cat({a[5],a[6].reshape({-1,3}),a[7]},1),torch::full({a[0].size(0),1},ratio,a[0].options()).log()},1);
}
Tensor fused(const List& a,double ratio){return mapping_cuda::anchorInput27(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],ratio);}
List copy(const List& a){List b;for(int i=0;i<8;++i)b.push_back(a[i].detach().clone().requires_grad_(i==0||i==1||i>=5));return b;}
void compare(std::ofstream& log,const std::string& name,const Tensor& a,const Tensor& b) {
  TORCH_CHECK(a.defined()==b.defined(),"Input27 gradNone mismatch ",name);if(!a.defined()||!a.numel())return;
  TORCH_CHECK(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type()&&a.device()==b.device(),"Input27 shape/type/device mismatch");
  auto d=(a-b).abs(),bad=d>2e-5+2e-4*a.abs();
  log<<name<<','<<d.max().item<double>()<<','<<(d/a.abs().clamp_min(1e-30)).max().item<double>()<<','
      <<d.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<'\n';
  TORCH_CHECK(torch::isfinite(a).all().item<bool>()&&torch::isfinite(b).all().item<bool>()&&!bad.any().item<bool>(),"Input27 parity ",name);
}
void parity(const List& values,double ratio,const std::string& name,std::ofstream& log) {
  auto a=copy(values),b=copy(values);auto x=legacy(a,ratio),y=fused(b,ratio);compare(log,name+"/input",x,y);
  auto g=torch::randn_like(x)*.001;List aa{a[0],a[1],a[5],a[6],a[7]},bb{b[0],b[1],b[5],b[6],b[7]};
  auto gx=torch::autograd::grad({x},aa,{g}),gy=torch::autograd::grad({y},bb,{g});
  for(int j=0;j<5;++j)compare(log,name+"/grad"+std::to_string(j),gx[j],gy[j]);
}
int main(int argc,char** argv) {
 try {
  TORCH_CHECK(argc==4,"fused_anchor_input_test checkpoint fixture_dir output_dir");
  torch::set_num_threads(2);torch::manual_seed(2026);fs::path cp=argv[1],fixture=argv[2],output=argv[3];fs::create_directories(output);
  Net net(.1,true,false,true);net->to(torch::kCUDA);torch::serialize::InputArchive checkpoint;
  checkpoint.load_from((cp/"selector.pt").string(),torch::kCUDA);net->load(checkpoint);
  TORCH_CHECK(net->anchors()->variableMembers(),"Variable Anchor required");
  auto bank=net->anchors();TORCH_CHECK(bank->memberCounts().min().item<int64_t>()==1&&bank->memberCounts().max().item<int64_t>()==10,"Missing 1/10 member fixtures");
  std::ofstream log(output/"pack_parity.csv");log<<"name,max_abs,max_rel,rms,bad_elements\n";
  std::ofstream timing(output/"pack_timing.csv");timing<<"ratio,N,repeat,implementation,stage,host_ms,cuda_ms\n";
  List values;auto names=net->named_parameters();
  for(double ratio:{.01,.05,.10,.15}) {
    torch::serialize::InputArchive archive;archive.load_from((fixture/("lod_"+std::to_string(ratio)+"_parity.pt")).string(),torch::kCUDA);
    Tensor x,s,dc,op,u;archive.read("parameter_before_0",x);archive.read("parameter_before_4",s);
    archive.read("parameter_before_1",dc);archive.read("parameter_before_3",op);archive.read("uniform",u);
    values={x,bank->features(),bank->rows(),bank->centers(),bank->offsetScales(),s.exp(),dc,op.sigmoid()};
    parity(values,ratio,std::to_string(ratio),log);
    auto a=copy(values),b=copy(values);
    auto eval=[&](const Tensor& input){auto h=torch::relu(torch::linear(input,names["local_emd.0.weight"],names["local_emd.0.bias"]));
      auto p=torch::linear(torch::relu(torch::linear(h,names["local_emd.2.weight"],names["local_emd.2.bias"])),names["soft_net.weight"],names["soft_net.bias"]).softmax(-1).select(1,1);
      return List{h,p};};
    auto p=eval(legacy(a,ratio)),q=eval(fused(b,ratio));compare(log,"encoder",p[0],q[0]);compare(log,"probability",p[1],q[1]);
    TORCH_CHECK(torch::equal(torch::nonzero(u<p[1]),torch::nonzero(u<q[1])),"InputPack changed cached selection/K");
    auto ga=torch::autograd::grad({p[1].sum()},{a[0],a[1],a[5],a[6],a[7]});
    auto gb=torch::autograd::grad({q[1].sum()},{b[0],b[1],b[5],b[6],b[7]});
    for(int j=0;j<5;++j)compare(log,"network_grad"+std::to_string(j),ga[j],gb[j]);
    cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);auto g=torch::randn({x.size(0),27},x.options())*.001;
    for(int rep=-10;rep<100;++rep)for(int order=0;order<2;++order) {
      bool use=(rep+order)%2==0;torch::cuda::synchronize();cudaEventRecord(start);auto begin=std::chrono::steady_clock::now();
      auto out=use?fused(a,ratio):legacy(a,ratio);double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
      cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,start,end);
      if(rep>=0)timing<<ratio<<','<<x.size(0)<<','<<rep<<','<<(use?"fused":"legacy")<<",forward,"<<host<<','<<ms<<'\n';
      cudaEventRecord(start);begin=std::chrono::steady_clock::now();auto gradients=torch::autograd::grad({out},{a[0],a[1],a[5],a[6],a[7]},{g});
      host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&ms,start,end);
      if(rep>=0)timing<<ratio<<','<<x.size(0)<<','<<rep<<','<<(use?"fused":"legacy")<<",backward,"<<host<<','<<ms<<'\n';
    }
    cudaEventDestroy(start);cudaEventDestroy(end);
  }
  auto perm=torch::randperm(values[0].size(0),values[2].options());auto reordered=values;
  for(int i:{0,2,5,6,7})reordered[i]=values[i].index_select(0,perm);parity(reordered,.05,"reorder",log);
  auto longtail=copy(values);longtail[0].data().add_(100);parity(longtail,.05,"longtail",log);
  for(int n:{0,1,10}){auto small=values;for(int i:{0,2,5,6,7})small[i]=values[i].slice(0,0,n);parity(small,.05,"N"+std::to_string(n),log);}
  auto strided=values;for(int i=0;i<8;++i)strided[i]=values[i].repeat_interleave(2,0).slice(0,0,values[i].size(0)*2,2);
  parity(strided,.05,"strided",log);
  net->setFusedInputPack(false);auto p=net->forwardSelectionFeatures(values[0],.05,1,{},values[5],values[6],values[7]);
  net->setFusedInputPack(true);auto q=net->forwardSelectionFeatures(values[0],.05,1,{},values[5],values[6],values[7]);
  compare(log,"production_probability",p.probability,q.probability);
  std::cout<<"PASS Input27 Forward/VJP,encoder,probability,exact IDs/K,Variable1..10,reorder,longtail,empty,strides\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
