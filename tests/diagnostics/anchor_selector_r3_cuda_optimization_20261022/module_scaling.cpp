// Artificial size stress, tiled from the same real Variable Anchor fixture.
#include "include/anchor_input_cuda.h"
#include "include/anchor_selector_cuda.h"
#include "include/mapping_cuda_optimizer.h"
#include "improvements/selector/gumbel_network.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
using T=torch::Tensor;using L=std::vector<T>;namespace fs=std::filesystem;
using Net=improvements::selector::GumbelNetwork;
L decode(const L& a,bool use){if(use)return mapping_cuda::decode14(a[0],a[1],a[2],a[3],a[4],a[5],a[6]);
  auto x=Net::Impl::applyCompensationResidual({a[0],a[1].log(),a[2]},a[5]);auto p=Net::Impl::applyAppearanceResidual(a[4],a[3],a[5]);
  return {x.xyz,a[1]*a[5].slice(1,3,6).exp(),x.rotation,p.second.sigmoid()*a[6].unsqueeze(1),p.first};}
T pack(const L& a,bool use){if(use)return mapping_cuda::anchorInput27(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],.05);
  return torch::cat({a[1].index_select(0,a[2]),(a[0]-a[3].index_select(0,a[2]))/a[4].index_select(0,a[2]),
      a[5],a[6].reshape({-1,3}),a[7],torch::full({a[0].size(0),1},.05,a[0].options()).log()},1);}
int main(int argc,char** argv){try {
  TORCH_CHECK(argc==4,"module_scaling checkpoint fixture output");torch::set_num_threads(2);torch::manual_seed(2026);
  Net net(.1,true,false,true);net->to(torch::kCUDA);torch::serialize::InputArchive ar;
  ar.load_from((fs::path(argv[1])/"selector.pt").string(),torch::kCUDA);net->load(ar);auto bank=net->anchors();
  torch::serialize::InputArchive f;f.load_from(argv[2],torch::kCUDA);T x,ls,dc,op,q,d;
  f.read("parameter_before_0",x);f.read("parameter_before_4",ls);f.read("parameter_before_1",dc);f.read("parameter_before_3",op);f.read("parameter_before_5",q);f.read("residual",d);
  std::ofstream csv(argv[3]);csv<<"N,K,module,implementation,repeat,stage,host_ms,cuda_ms,peak_extra_bytes,allocations\n";
  cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
  auto measure=[&](int64_t n,int64_t k,const char* module,bool use,int rep,const char* stage,auto fn){
    torch::cuda::synchronize();auto before=c10::cuda::CUDACachingAllocator::getDeviceStats(0);c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    cudaEventRecord(start);auto t=std::chrono::steady_clock::now();fn();double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
    cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,start,end);auto after=c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    if(rep>=0)csv<<n<<','<<k<<','<<module<<','<<(use?"fused":"legacy")<<','<<rep<<','<<stage<<','<<host<<','<<ms<<','
      <<after.allocated_bytes[0].peak-before.allocated_bytes[0].current<<','<<after.allocation[0].allocated-before.allocation[0].allocated<<'\n';};
  for(int64_t n:{10000,50000,100000}) {
    auto ids=torch::arange(n,bank->rows().options())%x.size(0);L a{x.index_select(0,ids).requires_grad_(true),bank->features().detach().requires_grad_(true),bank->rows().index_select(0,ids),bank->centers(),bank->offsetScales(),
      ls.index_select(0,ids).exp().requires_grad_(true),dc.index_select(0,ids).requires_grad_(true),op.index_select(0,ids).sigmoid().requires_grad_(true)};
    auto g=torch::randn({n,27},x.options())*.001;
    for(int rep=-5;rep<50;++rep)for(int j=0;j<2;++j){bool use=(rep+j)%2==0;T out;
      measure(n,0,"input27",use,rep,"forward",[&]{out=pack(a,use);});
      measure(n,0,"input27",use,rep,"backward",[&]{auto grad=torch::autograd::grad({out},{a[0],a[1],a[5],a[6],a[7]},{g});});}
    // Dense Adam touches N rows irrespective of the 5% selected set.
    L pa{torch::randn({n,14},x.options())},pb{pa[0].clone()};pa[0].mutable_grad()=torch::randn_like(pa[0])*.001;pb[0].mutable_grad()=pa[0].grad();
    torch::optim::Adam oa(pa,torch::optim::AdamOptions(.001).eps(1e-15)),ob(pb,torch::optim::AdamOptions(.001).eps(1e-15));
    for(int rep=-5;rep<50;++rep)for(int j=0;j<2;++j){bool use=(rep+j)%2==0;measure(n,n/20,"dense_r3_adam",use,rep,"update",[&]{if(use)mapping_cuda::tryFusedDenseAdam(ob);else oa.step();});}
    for(double ratio:{.01,.05,.10,.15}) {
      int64_t k=n*ratio;auto ix=torch::arange(k,ids.options())%x.size(0),di=torch::arange(k,ids.options())%d.size(0);
      L b{x.index_select(0,ix),ls.exp().index_select(0,ix),torch::nn::functional::normalize(q).index_select(0,ix),op.index_select(0,ix),dc.index_select(0,ix),d.index_select(0,di),torch::ones({k},x.options())};
      for(auto& t:b)t=t.detach().requires_grad_(true);L grads;for(auto t:decode(b,false))grads.push_back(torch::randn_like(t)*.001);
      for(int rep=-5;rep<50;++rep)for(int j=0;j<2;++j){bool use=(rep+j)%2==0;L out;
        measure(n,k,"decode",use,rep,"forward",[&]{out=decode(b,use);});
        measure(n,k,"decode",use,rep,"backward",[&]{auto grad=torch::autograd::grad(out,b,grads);});}
    }
  }
  cudaEventDestroy(start);cudaEventDestroy(end);std::cout<<"PASS Artificial N scaling and allocator counters\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
