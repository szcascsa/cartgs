#include "include/anchor_context_residual_cuda.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>

using V=std::vector<torch::Tensor>;
double timeStep(const std::function<void()>& f,int repeats) {
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
  cudaEventRecord(s);for(int j=0;j<repeats;++j)f();cudaEventRecord(e);cudaEventSynchronize(e);
  float ms;cudaEventElapsedTime(&ms,s,e);cudaEventDestroy(s);cudaEventDestroy(e);return ms/repeats;
}
int main(int argc,char** argv) {
  if(!torch::cuda::is_available())return 77;
  at::globalContext().setAllowTF32CuBLAS(false);torch::manual_seed(91);torch::set_num_threads(2);
  const bool profile=argc>1 && std::string(argv[1])=="profile";
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  std::cout<<"N,anchor_gap,mode,phase,round,ms\n"<<std::fixed<<std::setprecision(6);
  for(int n:profile?std::vector<int>{10000}:std::vector<int>{10000,50000,100000})for(int gap:{1,8}) {
    const int anchors=(n+3)/4*gap;
    V a{torch::randn({n,11},o).requires_grad_(true),
        (torch::randn({32,11},o)*.02).requires_grad_(true),(torch::randn({32},o)*.02).requires_grad_(true),
        (torch::randn({32,32},o)*.02).requires_grad_(true),(torch::randn({32},o)*.02).requires_grad_(true),
        (torch::randn({2,32},o)*.02).requires_grad_(true),(torch::randn({2},o)*.02).requires_grad_(true),
        (torch::randn({32,17},o)*.02).requires_grad_(true),
        torch::arange(n,o.dtype(torch::kLong)).floor_divide(4)*gap,torch::arange(n,o.dtype(torch::kLong)).remainder(7)!=0};
    V parameters(a.begin(),a.begin()+8);auto gz=torch::randn({n,2},o);
    auto f=[&](int mode,int backward) {
      auto out=mode?mapping_cuda::anchorContextMLPOptimized(a,anchors,mode):mapping_cuda::anchorContextMLP(a,anchors);
      if(backward==2){V private_parameters(parameters.begin()+1,parameters.end());torch::autograd::grad({out[0]},private_parameters,{gz},true);}
      if(backward)torch::autograd::grad({out[0]},parameters,{gz});
    };
    for(int mode:{0,1,3,5,7})for(int j=0;j<10;++j)f(mode,true);
    for(int round=0;round<(profile?1:9);++round)for(int phase=0;phase<4;++phase) {
      std::vector<int> order=round%2?std::vector<int>{7,5,3,1,0}:std::vector<int>{0,1,3,5,7};
      for(int mode:order) {
        const std::string label=phase==0?"inferenceF":(phase==1?"trainingF":(phase==2?"FB":"dualVJP"));
        std::string range="Context.mode"+std::to_string(mode)+".gap"+std::to_string(gap)+"."+label;
        nvtxRangePushA(range.c_str());
        double ms;
        if(phase>=2)ms=timeStep([&](){f(mode,phase==3?2:1);},profile?20:40);
        else if(phase==1)ms=timeStep([&](){f(mode,false);},profile?20:40);
        else {torch::NoGradGuard guard;ms=timeStep([&](){f(mode,false);},profile?20:40);}
        nvtxRangePop();
        std::cout<<n<<','<<gap<<','<<mode<<','<<label<<','<<round<<','<<ms<<std::endl;
      }
    }
  }
}
