// Same loss/LR/Adam protocol as the original audit; no renderer in this probe.
#include "include/transform_field.h"
#include "include/transform_field_cuda.h"
#include "include/constrained_adam.h"
#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>

struct Event {
  cudaEvent_t value;
  Event(){TORCH_CHECK(cudaEventCreate(&value)==cudaSuccess,"event create");}
  ~Event(){cudaEventDestroy(value);}
  void record(){TORCH_CHECK(cudaEventRecord(value,c10::cuda::getCurrentCUDAStream())==cudaSuccess,"event record");}
};
double elapsed(Event& a,Event& b){float ms;TORCH_CHECK(cudaEventElapsedTime(&ms,a.value,b.value)==cudaSuccess,"event elapsed");return ms;}
void emit(std::ofstream& out,const std::string& backend,const std::string& mode,int n,
          const std::string& stage,std::vector<double> samples) {
  double mean=std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();std::sort(samples.begin(),samples.end());
  out<<backend<<','<<mode<<','<<n<<','<<stage<<','<<mean<<','<<samples[samples.size()/2]<<','<<samples[size_t(0.95*(samples.size()-1))]<<'\n';out.flush();
}
int main(int argc,char** argv) {
  TORCH_CHECK(argc>=4 && argc<=7,"training_benchmark CHECKPOINT INPUT_BIN OUTPUT_PREFIX [cuda_optimizer] [profile BACKEND]");
  torch::set_num_threads(1);torch::manual_seed(20261017);at::globalContext().setAllowTF32CuBLAS(false);
  int argument=4;const bool cuda_optimizer=argc>argument && std::string(argv[argument])=="cuda_optimizer";
  if(cuda_optimizer)++argument;
  const bool profile=argc>argument && (std::string(argv[argument])=="profile" || std::string(argv[argument])=="profile_preview");
  TORCH_CHECK(argc==argument+(profile?2:0),"Invalid benchmark arguments");
  std::string prefix=argv[3];
  std::ifstream binary(argv[2],std::ios::binary);int64_t rows;binary.read(reinterpret_cast<char*>(&rows),8);
  TORCH_CHECK(rows>0 && rows<10000000,"Invalid input count");
  std::vector<float> raw(rows*10);binary.read(reinterpret_cast<char*>(raw.data()),raw.size()*4);TORCH_CHECK(binary,"Invalid input data");
  auto data=torch::from_blob(raw.data(),{rows,10},torch::kFloat32).clone().to(torch::kCUDA);
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  std::ofstream out(prefix+"_benchmark.csv"),memory(prefix+"_memory.csv"),validity(prefix+"_validity.csv"),samples(prefix+"_samples.csv");
  out<<"backend,mode,N,stage,mean_ms,p50_ms,p95_ms\n";
  memory<<"backend,mode,N,before_allocated_bytes,peak_allocated_bytes,peak_reserved_bytes\n";
  validity<<"backend,mode,N,finite\n";
  samples<<"backend,mode,N,iteration,zero_ms,forward_ms,loss_reg_ms,backward_ms,optimizer_ms,total_ms,wall_ms\n";
  std::vector<std::string> backends=profile?std::vector<std::string>{argv[argument+1]}:std::vector<std::string>{"legacy","cublas","warp32"};
  std::vector<int> populations=profile?std::vector<int>{8491}:std::vector<int>{1000,5000,8491,10000,20000,50000};
  std::vector<std::string> modes=profile?std::vector<std::string>{std::string(argv[argument])=="profile_preview"?"regularized_preview":"regularized_adam"}:std::vector<std::string>{"core_adam","regularized_adam","regularized_preview"};
  for(const auto& backend:backends)for(const auto& mode:modes)for(int n:populations) {
    TORCH_CHECK(backend=="legacy" || backend=="cublas" || backend=="warp32","Unknown benchmark backend");
    auto cpu=torch::TensorOptions().dtype(torch::kFloat32);
    TransformField tf(torch::full({3},-1,cpu),torch::full({3},1,cpu),cfg);
    torch::serialize::InputArchive archive;archive.load_from(std::string(argv[1])+"/transform_field.pt",torch::Device(torch::kCPU));tf->load(archive);tf->to(torch::kCUDA);
    auto index=torch::arange(n,data.options().dtype(torch::kLong)).remainder(rows),d=data.index_select(0,index);
    auto x=d.slice(1,0,3).contiguous().detach().set_requires_grad(true),s=d.slice(1,3,6).contiguous().detach().set_requires_grad(true),q=d.slice(1,6,10).contiguous().detach().set_requires_grad(true);
    torch::optim::Adam optimizer(tf->mlpParameters(),torch::optim::AdamOptions(0.00016).eps(1e-15));
    optimizer.add_param_group(tf->gridParameters());optimizer.param_groups()[1].options().set_lr(0.0016);
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    auto before=c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current;
    std::vector<std::vector<double>> metrics(7);Event e[6];
    for(int iteration=0;iteration<(profile?110:600);++iteration) {
      if(profile && iteration==100) {
        TORCH_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream())==cudaSuccess,"profile prior sync");
        TORCH_CHECK(cudaProfilerStart()==cudaSuccess,"profile start");
      }
      auto start=std::chrono::steady_clock::now();nvtxRangePushA("iteration");
      nvtxRangePushA("zero_grad");e[0].record();optimizer.zero_grad(true);
      x.mutable_grad()=torch::Tensor();s.mutable_grad()=torch::Tensor();q.mutable_grad()=torch::Tensor();e[1].record();nvtxRangePop();
      nvtxRangePushA("forward");std::vector<torch::Tensor> y;
      if(backend=="legacy") {auto a=tf->applyResidual(x,s,q,0.05f);y={a.xyz,a.scaling,a.rotation};}
      else {
        auto ratio=torch::full({n,1},0.05,x.options().requires_grad(false));
        auto feature=tf_cuda::query(x,ratio,tf->aabbMin(),tf->aabbMax(),tf->gridParameters());
        auto residual=tf_cuda::neural(feature,tf->mlpParameters(),backend=="cublas"?tf_cuda::MlpBackend::Cublas:tf_cuda::MlpBackend::Warp32);
        y=tf_cuda::output(x,s,q,residual);
      }
      e[2].record();nvtxRangePop();nvtxRangePushA("loss");
      auto loss=(y[0]-x.detach()).mean()+(y[1]/s.detach()).mean()+y[2].mean();nvtxRangePop();
      if(mode!="core_adam") {
        nvtxRangePushA("regularization");
        loss=loss+(backend=="legacy"?tf->regularizationLoss():tf_cuda::regularization(tf->gridParameters(),cfg.spatial_smooth_weight,cfg.ratio_smooth_weight,cfg.ratio_l1_weight));
        nvtxRangePop();
      }
      e[3].record();nvtxRangePushA("backward");loss.backward();e[4].record();nvtxRangePop();
      nvtxRangePushA("optimizer");
      const auto cuda_groups=cuda_optimizer && backend!="legacy"?std::vector<size_t>{0,1}:std::vector<size_t>{};
      if(mode=="regularized_preview") {AdamStepPreview preview(optimizer,cuda_groups);preview.commit(preview.candidate(preview.delta(),1));}
      else if(!cuda_groups.empty())tf_cuda::tfCudaAdamDirect(optimizer,cuda_groups);
      else optimizer.step();
      e[5].record();nvtxRangePop();nvtxRangePushA("synchronize");
      TORCH_CHECK(cudaEventSynchronize(e[5].value)==cudaSuccess,"measurement sync");nvtxRangePop();nvtxRangePop();
      double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(iteration>=100) {
        std::vector<double> row{elapsed(e[0],e[1]),elapsed(e[1],e[2]),elapsed(e[2],e[3]),elapsed(e[3],e[4]),elapsed(e[4],e[5]),elapsed(e[0],e[5]),wall};
        for(size_t i=0;i<row.size();++i)metrics[i].push_back(row[i]);
        samples<<backend<<','<<mode<<','<<n<<','<<iteration-100;for(auto v:row)samples<<','<<v;samples<<'\n';
      }
    }
    if(profile)TORCH_CHECK(cudaProfilerStop()==cudaSuccess,"profile stop");
    const std::vector<std::string> stages{"zero_grad","forward","loss_regularization","backward","optimizer","total","wall"};
    for(size_t i=0;i<stages.size();++i)emit(out,backend,mode,n,stages[i],metrics[i]);
    auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    memory<<backend<<','<<mode<<','<<n<<','<<before<<','<<stats.allocated_bytes[0].peak<<','<<stats.reserved_bytes[0].peak<<'\n';memory.flush();
    std::vector<torch::Tensor> finite;
    for(auto p:tf->parameters()){finite.push_back(torch::isfinite(p).all());if(p.grad().defined())finite.push_back(torch::isfinite(p.grad()).all());}
    for(auto p:{x,s,q})if(p.grad().defined())finite.push_back(torch::isfinite(p.grad()).all());
    for(const auto& item:optimizer.state()) {const auto& st=static_cast<const torch::optim::AdamParamState&>(*item.second);finite.push_back(torch::isfinite(st.exp_avg()).all());finite.push_back(torch::isfinite(st.exp_avg_sq()).all());}
    bool passed=torch::stack(finite).all().item<bool>();validity<<backend<<','<<mode<<','<<n<<','<<passed<<'\n';validity.flush();TORCH_CHECK(passed,"Nonfinite training benchmark");
    std::cout<<backend<<' '<<mode<<" N="<<n<<" passed"<<std::endl;
  }
}
