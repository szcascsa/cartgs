#pragma once
// Diagnostic only: bracket the unchanged TF graph for reverse-only timings.
// This header is compiled into a separate executable, never the production one.
#include "include/transform_field.h"
#include "include/transform_field_cuda_optimizer.h"
#include <torch/csrc/autograd/custom_function.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <deque>
#include <mutex>
#include <iostream>
#include <iomanip>

namespace tf_diagnostic {
enum Stage { Forward, Backward, RegForward, RegBackward, Optimizer, Count };
struct Sample { cudaEvent_t begin=nullptr,end=nullptr; int stage; };
struct Statistics { uint64_t calls=0, samples=0, population=0; double host=0,gpu=0; };
inline std::mutex mutex;
inline std::array<Statistics,Count> statistics;
inline std::deque<Sample> pending;
inline void harvest() {
  for(auto it=pending.begin();it!=pending.end();) {
    auto status=cudaEventQuery(it->end);
    if(status==cudaErrorNotReady){++it;continue;}
    TORCH_CHECK(status==cudaSuccess,"TF diagnostic event failed");
    float ms=0;TORCH_CHECK(cudaEventElapsedTime(&ms,it->begin,it->end)==cudaSuccess,"TF diagnostic elapsed failed");
    statistics[it->stage].gpu+=ms;++statistics[it->stage].samples;
    cudaEventDestroy(it->begin);cudaEventDestroy(it->end);it=pending.erase(it);
  }
}
class Scope {
 public:
  explicit Scope(Stage stage,uint64_t n=0):stage_(stage),start_(std::chrono::steady_clock::now()) {
    std::lock_guard<std::mutex> lock(mutex);harvest();auto& s=statistics[stage];
    ++s.calls;s.population+=n;
    if(s.calls%19==0 && pending.size()<32) {
      sample_.stage=stage;
      TORCH_CHECK(cudaEventCreate(&sample_.begin)==cudaSuccess && cudaEventCreate(&sample_.end)==cudaSuccess,"TF diagnostic event create failed");
      TORCH_CHECK(cudaEventRecord(sample_.begin,c10::cuda::getCurrentCUDAStream())==cudaSuccess,"TF diagnostic begin failed");
    }
  }
  ~Scope() {
    if(sample_.begin)cudaEventRecord(sample_.end,c10::cuda::getCurrentCUDAStream());
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start_).count();
    std::lock_guard<std::mutex> lock(mutex);statistics[stage_].host+=ms;
    if(sample_.begin)pending.push_back(sample_);
  }
 private:
  Stage stage_;Sample sample_{};std::chrono::steady_clock::time_point start_;
};
inline void report() {
  std::lock_guard<std::mutex> lock(mutex);harvest();
  const char* names[]={"forward","backward","regularization_forward","regularization_backward","optimizer"};
  for(int i=0;i<Count;++i){const auto& s=statistics[i];
    std::cout<<std::fixed<<std::setprecision(6)<<"[TF DIAGNOSTIC] stage="<<names[i]<<" calls="<<s.calls
      <<" transformed_N_sum="<<s.population<<" host_mean_ms="<<(s.calls?s.host/s.calls:0)
      <<" cuda_samples="<<s.samples<<" cuda_elapsed_mean_ms="<<(s.samples?s.gpu/s.samples:0)<<std::endl;
  }
  std::cout<<"[TF DIAGNOSTIC] pending_samples="<<pending.size()<<" explicit_synchronize=0 diagnostic_graph_brackets=1"<<std::endl;
}
class GraphFunction : public torch::autograd::Function<GraphFunction> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      at::TensorList inputs,int64_t pointer,double ratio,bool regularization,bool weighted) {
    ctx->set_materialize_grads(false);
    auto* module=reinterpret_cast<TransformFieldImpl*>(pointer);
    auto saved=inputs.vec();
    if(!regularization)for(int i=0;i<3;++i)saved[i]=saved[i].detach().set_requires_grad(inputs[i].requires_grad());
    ctx->saved_data["count"]=static_cast<int64_t>(saved.size());
    ctx->saved_data["reg"]=regularization;
    torch::AutoGradMode enable(true);
    Scope timing(regularization?RegForward:Forward,regularization?0:saved[0].size(0));
    torch::autograd::variable_list outputs;
    if(regularization)outputs={module->regularizationLoss()};
    else {auto y=module->applyResidual(saved[0],saved[1],saved[2],static_cast<float>(ratio),
        weighted?saved[3]:torch::Tensor{});outputs={y.xyz,y.scaling,y.rotation};}
    saved.insert(saved.end(),outputs.begin(),outputs.end());ctx->save_for_backward(saved);
    for(auto& value:outputs)value=value.detach();
    return outputs;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list upstream) {
    Scope timing(ctx->saved_data["reg"].toBool()?RegBackward:Backward);
    auto saved=ctx->get_saved_variables();size_t count=ctx->saved_data["count"].toInt();
    torch::autograd::variable_list targets,outputs,gradients,result(count+4);
    std::vector<size_t> indices;
    for(size_t i=0;i<count;++i)if(saved[i].requires_grad()){targets.push_back(saved[i]);indices.push_back(i);}
    for(size_t i=0;i<upstream.size();++i)if(upstream[i].defined()){outputs.push_back(saved[count+i]);gradients.push_back(upstream[i]);}
    auto derivatives=torch::autograd::grad(outputs,targets,gradients,false,false,true);
    for(size_t i=0;i<indices.size();++i)result[indices[i]]=derivatives[i];
    return result;
  }
};
inline ElasticGaussianAttributes applyResidual(TransformFieldImpl& module,const torch::Tensor& xyz,
    const torch::Tensor& scale,const torch::Tensor& q,float ratio,const torch::Tensor& maturity={}) {
  std::vector<torch::Tensor> inputs{xyz,scale,q};auto parameters=module.parameters();
  if(maturity.defined())inputs.push_back(maturity.detach());
  inputs.insert(inputs.end(),parameters.begin(),parameters.end());
  auto output=GraphFunction::apply(at::TensorList(inputs),reinterpret_cast<int64_t>(&module),static_cast<double>(ratio),false,maturity.defined());
  return {output[0],output[1],output[2]};
}
inline torch::Tensor regularization(TransformFieldImpl& module) {
  auto parameters=module.parameters();
  return GraphFunction::apply(at::TensorList(parameters),reinterpret_cast<int64_t>(&module),0.0,true,false)[0];
}
struct GradRestore {
  std::vector<std::pair<torch::Tensor,torch::Tensor>> values;
  ~GradRestore(){for(auto& value:values)value.first.mutable_grad()=value.second;}
  void clear(const torch::optim::OptimizerParamGroup& group) {
    for(const auto& p:group.params()){values.emplace_back(p,p.grad());p.mutable_grad()=torch::Tensor();}
  }
};
inline void mixedStep(torch::optim::Adam& optimizer,const std::vector<std::size_t>& cuda_groups,
    const std::vector<std::size_t>& tf_groups) {
  if(tf_groups.empty()){optimizer.step();return;}
  {GradRestore restore;for(auto i:tf_groups)restore.clear(optimizer.param_groups()[i]);optimizer.step();}
  Scope timing(Optimizer);
  if(!cuda_groups.empty())tf_cuda::tfCudaAdamDirect(optimizer,cuda_groups);
  else {
    GradRestore restore;
    for(size_t i=0;i<optimizer.param_groups().size();++i)
      if(std::find(tf_groups.begin(),tf_groups.end(),i)==tf_groups.end())restore.clear(optimizer.param_groups()[i]);
    optimizer.step();
  }
}
}  // namespace tf_diagnostic
