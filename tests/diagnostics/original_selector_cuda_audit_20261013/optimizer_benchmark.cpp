#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include "improvements/selector/gumbel_network.h"
#include "include/constrained_adam.h"
void emit(std::ofstream& out,const std::string& mode,const std::string& metric,std::vector<double> v) {double mean=std::accumulate(v.begin(),v.end(),0.0)/v.size();std::sort(v.begin(),v.end());out<<mode<<','<<metric<<','<<mean<<','<<v[v.size()/2]<<','<<v[size_t(.95*(v.size()-1))]<<'\n';out.flush();}
int main(int argc,char** argv) {
  TORCH_CHECK(argc==2,"optimizer_benchmark DIAGNOSTICS");torch::set_num_threads(1);std::string dir=argv[1];
  std::ofstream csv(dir+"/native_optimizer.csv");csv<<"mode,metric,mean,p50,p95\n";
  for(const std::string mode:{"adam_shared_options","preview_commit","preview_discard","zero_grad"}) {
    improvements::selector::GumbelNetwork model;model->to(torch::kCUDA);auto parameters=model->parameters();torch::optim::Adam opt(parameters,torch::optim::AdamOptions(.001).eps(1e-15));
    std::vector<torch::Tensor> grads;for(auto& p:parameters){auto g=torch::full_like(p,1e-4);p.mutable_grad()=g;grads.push_back(g);}
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);std::vector<double> event,wall;
    for(int i=0;i<600;++i) {
      if(mode=="zero_grad")for(size_t j=0;j<parameters.size();++j)parameters[j].mutable_grad()=grads[j];
      cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();cudaEventRecord(a,c10::cuda::getCurrentCUDAStream());
      if(mode=="zero_grad")opt.zero_grad(true);
      else if(mode=="adam_shared_options")opt.step();
      else {AdamStepPreview preview(opt);auto candidate=preview.candidate(preview.delta(),1);if(mode=="preview_commit")preview.commit(candidate);}
      cudaEventRecord(b,c10::cuda::getCurrentCUDAStream());cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);
      if(i>=100){event.push_back(ms);wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
    }
    emit(csv,mode,"cuda_event_ms",event);emit(csv,mode,"wall_ms",wall);cudaEventDestroy(a);cudaEventDestroy(b);std::cout<<mode<<" complete\n";
  }
  improvements::selector::GumbelNetwork model;model->to(torch::kCUDA);auto params=model->parameters();torch::optim::Adam opt(params,torch::optim::AdamOptions(.001).eps(1e-15));std::vector<torch::Tensor> old;for(auto& p:params){old.push_back(p.detach().clone());p.mutable_grad()=torch::ones_like(p);}
  {AdamStepPreview preview(opt);auto candidate=preview.candidate(preview.delta(),1);}
  bool same=opt.state().empty();for(size_t j=0;j<params.size();++j)same &=torch::equal(params[j],old[j]);TORCH_CHECK(same,"preview discard mutated state");
  {AdamStepPreview preview(opt);preview.commit(preview.candidate(preview.delta(),1));}
  bool steps=true;for(auto& item:opt.state())steps &=static_cast<torch::optim::AdamParamState&>(*item.second).step()==1;TORCH_CHECK(steps,"preview commit clock");
  std::ofstream(dir+"/native_preview_validity.json")<<"{\"discard_unchanged\":true,\"commit_state_step_1\":true,\"state_tensors\":"<<opt.state().size()<<"}\n";
}
