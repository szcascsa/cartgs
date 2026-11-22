#include "improvements/selector/gumbel_network.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>

using improvements::selector::AnchorSetEncoder;

double measure(const std::function<void()>& step) {
  for(int j=0;j<5;++j)step();
  cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
  std::vector<double> times;
  for(int round=0;round<5;++round) {
    cudaEventRecord(start);
    for(int j=0;j<20;++j)step();
    cudaEventRecord(end);cudaEventSynchronize(end);float ms=0;cudaEventElapsedTime(&ms,start,end);
    times.push_back(ms/20.);
  }
  cudaEventDestroy(start);cudaEventDestroy(end);std::sort(times.begin(),times.end());return times[2];
}

int main() {
  if(!torch::cuda::is_available())return 77;
  torch::manual_seed(37);torch::set_num_threads(2);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  std::cout<<"N,occupancy,mode,forward_ms,forward_backward_ms\n"<<std::fixed<<std::setprecision(6);
  for(int64_t n:{10000,50000,100000}) {
    const int64_t a=n/4;
    auto owners=torch::arange(n,o.dtype(torch::kLong)).floor_divide(4);
    auto centers=torch::rand({a,3},o),ref=torch::full({a,1},.1,o),ratio=torch::full({1,1},.05,o);
    auto x=(centers.index_select(0,owners)+torch::randn({n,3},o)*.02).requires_grad_(true);
    auto s=torch::rand({n,3},o).requires_grad_(true),dc=torch::rand({n,1,3},o).requires_grad_(true),op=torch::rand({n,1},o).requires_grad_(true);
    auto valid=torch::arange(n,o.dtype(torch::kLong)).remainder(7)!=0;
    AnchorSetEncoder learned(2),statistics(3);learned->to(torch::kCUDA);
    auto w1=torch::randn({32,32},o).mul(.02).requires_grad_(true),b1=torch::randn({32},o).mul(.02).requires_grad_(true);
    auto w2=torch::randn({2,32},o).mul(.02).requires_grad_(true),b2=torch::randn({2},o).mul(.02).requires_grad_(true);
    for(const std::string mode:{"original11_fused","learned50","statistics21_aten","statistics21_fused_input","statistics21_fused"}) {
      const int width=mode=="learned50"?50:(mode=="original11_fused"?11:21);
      auto w0=torch::randn({32,width},o).mul(.02).requires_grad_(true),b0=torch::randn({32},o).mul(.02).requires_grad_(true);
      auto forward=[&]() {
        torch::Tensor input;
        if(mode=="original11_fused")input=mapping_cuda::flexgsInput11(x,s,dc,op,owners,centers,ref,.05);
        else if(mode=="learned50")input=learned->forward(x,s,dc,op,owners,ref,ratio,centers,valid);
        else input=statistics->forward(x,s,dc,op,owners,ref,ratio,centers,valid,mode!="statistics21_aten");
        if(mode=="statistics21_fused" || mode=="original11_fused")return mapping_cuda::flexgsMLP(input,w0,b0,w1,b1,w2,b2)[0];
        return torch::nn::functional::linear(torch::relu(torch::nn::functional::linear(
            torch::relu(torch::nn::functional::linear(input,w0,b0)),w1,b1)),w2,b2);
      };
      auto parameters=torch::autograd::variable_list{x,s,dc,op,w0,b0,w1,b1,w2,b2};
      if(mode=="learned50")for(auto p:learned->parameters())parameters.push_back(p);
      double f=measure([&](){torch::NoGradGuard guard;forward();});
      double fb=measure([&](){auto output=forward();torch::autograd::grad({output.sum()},parameters);});
      std::cout<<n<<",4,"<<mode<<','<<f<<','<<fb<<std::endl;
    }
  }
}
