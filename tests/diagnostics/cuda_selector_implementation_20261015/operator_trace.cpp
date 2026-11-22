#include <torch/torch.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <cuda_runtime.h>
#include <ATen/record_function.h>
#include <fstream>
#include "improvements/selector/gumbel_network.h"

torch::Tensor read(const std::string& path) {
  std::ifstream f(path,std::ios::binary);int64_t n=0,w=0;f.read((char*)&n,8);f.read((char*)&w,8);
  TORCH_CHECK(f && n>0 && w>0,"Invalid fixture");auto t=torch::empty({n,w},torch::kFloat32);
  f.read((char*)t.data_ptr<float>(),n*w*4);TORCH_CHECK(f,"Truncated fixture");return t.cuda();
}
int main(int argc,char** argv) {
  TORCH_CHECK(argc==5,"operator_trace CHECKPOINT FIXTURE legacy|cuda TRACE.json");
  torch::set_num_threads(1);torch::manual_seed(123);
  improvements::selector::GumbelNetwork m;torch::serialize::InputArchive a;a.load_from(argv[1],torch::kCPU);m->load(a);m->to(torch::kCUDA);m->setBackend(argv[3]);
  std::string dir=argv[2];auto data=read(dir+"/input.bin"),meta=read(dir+"/metadata.bin");int64_t mapping=0;std::ifstream(dir+"/spec.txt")>>mapping;
  data=data.index_select(0,torch::nonzero((mapping-meta.select(1,0)>=100)&(meta.select(1,1)>=80)&(meta.select(1,2)>=4)).flatten());
  auto x=data.slice(1,0,3).contiguous().detach().set_requires_grad(true);
  auto q=torch::nn::functional::normalize(data.slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)).detach().set_requires_grad(true);
  auto s=data.slice(1,7,10).exp().detach().set_requires_grad(true);
  for(int i=0;i<110;++i) {
    if(i==100){cudaDeviceSynchronize();using namespace torch::profiler::impl;
      torch::autograd::profiler::prepareProfiler(ProfilerConfig(ProfilerState::KINETO,true,true),{ActivityType::CPU,ActivityType::CUDA});
      torch::autograd::profiler::enableProfiler(ProfilerConfig(ProfilerState::KINETO,true,true),{ActivityType::CPU,ActivityType::CUDA});}
    torch::Tensor p;
    {RECORD_FUNCTION("selector_network_forward",std::vector<c10::IValue>());p=m->forwardProbability(x,q,s,.05);}
    {RECORD_FUNCTION("selector_network_backward",std::vector<c10::IValue>());p.sum().backward();}
    m->zero_grad();x.mutable_grad()=torch::Tensor();q.mutable_grad()=torch::Tensor();s.mutable_grad()=torch::Tensor();
  }
  cudaDeviceSynchronize();torch::autograd::profiler::disableProfiler()->save(argv[4]);
}
