#include <torch/torch.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include "include/model_snapshot.h"
#define private public
#include "improvements/selector/gumbel_network.h"
#undef private
#include "include/selector_cuda.h"

using improvements::selector::GumbelNetwork;
using Clock=std::chrono::steady_clock;
struct Event {
  cudaEvent_t value;
  Event(){TORCH_CHECK(cudaEventCreate(&value)==cudaSuccess,"event create");}
  ~Event(){cudaEventDestroy(value);}
  void record(){TORCH_CHECK(cudaEventRecord(value,c10::cuda::getCurrentCUDAStream())==cudaSuccess,"event record");}
};
void measure(std::ofstream& out,const std::string& backend,int n,const std::string& stage,
             const std::function<void()>& action,bool profile) {
  Event a,b;std::vector<double> event,wall;
  const auto before=c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current;
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  for(int i=0;i<(profile?110:600);++i) {
    TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess,"prior sync");
    if(profile && i==100)TORCH_CHECK(cudaProfilerStart()==cudaSuccess,"profiler start");
    auto start=Clock::now();a.record();nvtxRangePushA(stage.c_str());action();nvtxRangePop();b.record();
    TORCH_CHECK(cudaEventSynchronize(b.value)==cudaSuccess,"end sync");
    if(i>=100){float ms=0;cudaEventElapsedTime(&ms,a.value,b.value);event.push_back(ms);
      wall.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());}
  }
  if(profile)cudaProfilerStop();
  for(auto pair:std::vector<std::pair<std::string,std::vector<double>>>{{"event",event},{"wall",wall}}) {
    auto& v=pair.second;double mean=std::accumulate(v.begin(),v.end(),0.)/v.size();std::sort(v.begin(),v.end());
    out<<backend<<','<<n<<','<<stage<<','<<pair.first<<','<<mean<<','<<v[v.size()/2]<<','<<v[int(.95*(v.size()-1))]<<'\n';out.flush();
  }
  const auto memory=c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  std::ofstream mem("network_memory.csv",std::ios::app);
  mem<<backend<<','<<n<<','<<stage<<','<<before<<','<<memory.allocated_bytes[0].peak<<','
     <<memory.allocated_bytes[0].peak-before<<','<<memory.reserved_bytes[0].peak<<'\n';
  std::cout<<backend<<" M="<<n<<" "<<stage<<" complete\n"<<std::flush;
}
torch::Tensor read(const std::string& path) {
  std::ifstream f(path,std::ios::binary);int64_t n=0,w=0;
  f.read(reinterpret_cast<char*>(&n),8);f.read(reinterpret_cast<char*>(&w),8);
  TORCH_CHECK(f && n>0 && w>0,"fixture ",path);auto t=torch::empty({n,w},torch::kFloat32);
  f.read(reinterpret_cast<char*>(t.data_ptr<float>()),n*w*4);TORCH_CHECK(f,"truncated fixture");return t.cuda();
}
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc>=4,"network_benchmark CHECKPOINT FIXTURE OUTPUT [legacy|cublas|warp|cublas_strict] [profile]");
    torch::set_num_threads(1);torch::manual_seed(123);
    GumbelNetwork model;torch::serialize::InputArchive archive;archive.load_from(argv[1],torch::kCPU);model->load(archive);model->to(torch::kCUDA);
    std::string fixture=argv[2];auto data=read(fixture+"/input.bin"),metadata=read(fixture+"/metadata.bin");
    int64_t mapping=0;std::ifstream(fixture+"/spec.txt")>>mapping;
    auto ids=torch::nonzero((mapping-metadata.select(1,0)>=100)&(metadata.select(1,1)>=80)&(metadata.select(1,2)>=4)).flatten();
    data=data.index_select(0,ids);
    bool profile=argc>5 && std::string(argv[5])=="profile";
    std::ofstream out(argv[3]);out<<"backend,M,stage,metric,mean_ms,p50_ms,p95_ms\n";
    std::vector<std::string> backends=argc>4?std::vector<std::string>{argv[4]}:std::vector<std::string>{"legacy","cublas","warp","cublas_strict"};
    for(int n:profile?std::vector<int>{int(data.size(0))}:std::vector<int>{1000,5000,10000,20000,50000,100000,200000,int(data.size(0))}) {
      TORCH_CHECK(n<=data.size(0),"Fixture too small");
      std::vector<torch::Tensor> input{data.slice(0,0,n).slice(1,0,3).contiguous().detach().set_requires_grad(true),
        torch::nn::functional::normalize(data.slice(0,0,n).slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)).detach().set_requires_grad(true),
        data.slice(0,0,n).slice(1,7,10).exp().detach().set_requires_grad(true),torch::full({1,1},.05,data.options())};
      auto args=std::vector<torch::Tensor>{input[0],input[1],input[2]},parameters=model->parameters();args.insert(args.end(),parameters.begin(),parameters.end());
      auto dy=torch::randn({n,2},data.options());
      for(const auto& backend:backends) {
        TORCH_CHECK(backend=="legacy"||backend=="cublas"||backend=="warp"||backend=="cublas_strict","Invalid backend");
        auto run=[&](){if(backend=="legacy")return model->logitsFor(input[0],input[1],input[2],input[3]);
          return selector_cuda::network(input,parameters,backend=="warp"?selector_cuda::Candidate::Warp:
              (backend=="cublas_strict"?selector_cuda::Candidate::CublasStrict:selector_cuda::Candidate::Cublas));};
        if(profile) {
          measure(out,backend,n,"network_forward_backward",[&](){
            nvtxRangePushA("network_forward");auto y=run();nvtxRangePop();
            nvtxRangePushA("network_backward");auto g=torch::autograd::grad({y},args,{dy});nvtxRangePop();
          },true);
          continue;
        }
        measure(out,backend,n,"network_forward",[&](){torch::NoGradGuard ng;auto z=run();},false);
        auto z=run();
        measure(out,backend,n,"network_backward",[&](){auto g=torch::autograd::grad({z},args,{dy},true);},profile);
        z=torch::Tensor();
        measure(out,backend,n,"network_forward_backward",[&](){auto y=run();auto g=torch::autograd::grad({y},args,{dy});},false);
      }
    }
    std::ofstream environment(std::string(argv[3])+".environment.txt");
    environment<<"allowTF32CuBLAS="<<at::globalContext().allowTF32CuBLAS()<<"\nallowTF32CuDNN="<<at::globalContext().allowTF32CuDNN()<<'\n';
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
