#include <torch/torch.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include "include/model_snapshot.h"
// Diagnostic access only. Production forwardHard is the final baseline;
// private logitsFor is exposed to time its unchanged constituent operations.
#define private public
#include "improvements/selector/gumbel_network.h"
#undef private
#include "include/elastic_budget.h"
#include "include/gi_teacher.h"

using improvements::selector::GumbelNetwork;
using Clock = std::chrono::steady_clock;
struct Event {
  cudaEvent_t e;
  Event() { TORCH_CHECK(cudaEventCreate(&e)==cudaSuccess,"event create"); }
  ~Event() { cudaEventDestroy(e); }
  void record() { TORCH_CHECK(cudaEventRecord(e,c10::cuda::getCurrentCUDAStream())==cudaSuccess,"event record"); }
};
double elapsed(Event& a, Event& b) { float x; TORCH_CHECK(cudaEventElapsedTime(&x,a.e,b.e)==cudaSuccess,"event elapsed");return x; }
torch::Tensor readBinary(const std::string& path) {
  std::ifstream f(path,std::ios::binary);int64_t n,w;
  f.read(reinterpret_cast<char*>(&n),8);f.read(reinterpret_cast<char*>(&w),8);
  std::vector<float> v(n*w);f.read(reinterpret_cast<char*>(v.data()),v.size()*4);
  TORCH_CHECK(f && n>0 && w>0,"binary input");
  return torch::from_blob(v.data(),{n,w},torch::kFloat32).clone().to(torch::kCUDA);
}
void emit(std::ofstream& out,const std::string& mode,int64_t n,int64_t m,float ratio,const std::string& stage,const std::string& metric,std::vector<double> v) {
  const double mean=std::accumulate(v.begin(),v.end(),0.0)/v.size();
  std::sort(v.begin(),v.end());out<<mode<<','<<n<<','<<m<<','<<ratio<<','<<stage<<','<<metric<<','<<mean<<','<<v[v.size()/2]<<','<<v[size_t(.95*(v.size()-1))]<<'\n';out.flush();
}
int main(int argc,char** argv) {
  TORCH_CHECK(argc>=2,"native_benchmark DIAGNOSTICS [profile]");
  torch::set_num_threads(1);std::string dir=argv[1];bool operators=argc>2 && std::string(argv[2])=="operators";bool profile=argc>2 && (std::string(argv[2])=="profile" || operators);bool gradients=argc>2 && std::string(argv[2])=="gradients";
  auto values=readBinary(dir+"/input.bin"),metadata=readBinary(dir+"/metadata.bin");
  int64_t mapping;std::ifstream(dir+"/spec.txt")>>mapping;
  auto makeModel=[&] {GumbelNetwork model;torch::serialize::InputArchive a;
    a.load_from("/home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt",torch::Device(torch::kCPU));
    model->load(a);model->to(torch::kCUDA);return model;};
  std::ofstream csv,samples,valid;
  if(!profile && !gradients) {csv.open(dir+"/native_benchmark.csv");samples.open(dir+"/native_samples.csv");valid.open(dir+"/native_validity.csv");csv<<"mode,N_total,N_network,ratio,stage,metric,mean,p50,p95\n";samples<<"mode,N_total,N_network,ratio,sample,K,total_event_ms,wall_ms\n";valid<<"mode,N,ratio,finite\n";}
  // Native/Python probability and staged/production fixed-RNG parity.
  if(!profile) {
    auto model=makeModel();auto mature=(mapping-metadata.select(1,0)>=100)&(metadata.select(1,1)>=80)&(metadata.select(1,2)>=4);
    auto ids=torch::nonzero(mature).flatten().slice(0,0,257);
    auto x=values.slice(1,0,3).index_select(0,ids),q=torch::nn::functional::normalize(values.slice(1,3,7).index_select(0,ids),torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto s=values.slice(1,7,10).index_select(0,ids).exp();
    std::vector<float> expected(257);std::ifstream f(dir+"/python_probability.bin",std::ios::binary);f.read(reinterpret_cast<char*>(expected.data()),expected.size()*4);
    auto probability=model->forwardProbability(x,q,s,.05f);double error=(probability.cpu()-torch::tensor(expected)).abs().max().item<double>();
    torch::manual_seed(123);auto actual=model->forwardHard(x,q,s,.05f);
    torch::manual_seed(123);auto logits=model->logitsFor(x,q,s,torch::full({1,1},.05f,x.options()));
    auto noise=-torch::empty_like(logits).exponential_().log();auto soft=(logits+noise).softmax(-1);
    auto hard=torch::zeros_like(logits).scatter_(-1,std::get<1>(soft.max(-1,true)),1.0);auto split=(hard-soft.detach()+soft).select(1,1);
    std::ofstream p(dir+"/native_parity.json");p<<"{\"python_probability_max_abs\":"<<error<<",\"staged_hard_exact\":"<<(torch::equal(actual,split)?"true":"false")<<"}\n";
    TORCH_CHECK(error<2e-6 && torch::equal(actual,split),"diagnostic parity");
    if(gradients) {
      model->zero_grad();x=x.detach().set_requires_grad(true);q=q.detach().set_requires_grad(true);s=s.detach().set_requires_grad(true);
      logits=model->logitsFor(x,q,s,torch::full({1,1},.05f,x.options()));soft=(logits+noise.detach()).softmax(-1);
      hard=torch::zeros_like(logits).scatter_(-1,std::get<1>(soft.max(-1,true)),1.0);split=(hard-soft.detach()+soft).select(1,1);
      auto compact=torch::nonzero(split.detach()>0).flatten();auto alpha=values.slice(1,10,11).index_select(0,ids).sigmoid().detach().set_requires_grad(true);
      auto selected_opacity=split.index_select(0,compact).unsqueeze(1)*alpha.index_select(0,compact);
      auto loss=selected_opacity.mean()+.01*torch::abs(logits.softmax(-1).select(1,1).sum()-.05*257)/(.05*257);
      loss.backward();torch::serialize::OutputArchive a;
      for(auto entry:std::vector<std::pair<std::string,torch::Tensor>>{{"x",x},{"q",q},{"s",s},{"alpha",alpha},{"noise",noise},{"hard",split},{"x_grad",x.grad()},{"q_grad",q.grad()},{"s_grad",s.grad()},{"alpha_grad",alpha.grad()}})a.write(entry.first,entry.second.detach().cpu());
      int i=0;for(auto& parameter:model->parameters())a.write("grad_"+std::to_string(i++),parameter.grad().cpu());
      a.save_to(dir+"/native_gradient_oracle.pt");std::cout<<"Native structural gradient oracle complete"<<std::endl;return 0;
    }
  }
  const std::vector<std::string> stages={"protection","features","network","gumbel","probability","row_argmax","mask","compact_indices","gather","loss_vjp","backward","optimizer","zero_grad"};
  std::vector<int64_t> sizes={1000,5000,10000,20000,50000,100000,200000,values.size(0)};
  if(profile) sizes={values.size(0)};
  for(const auto& mode:profile?std::vector<std::string>{"staged_train"}:std::vector<std::string>{"cpp_inference","caller_train","staged_train"}) {
    for(int64_t n:sizes) for(float ratio:profile?std::vector<float>{.05f}:std::vector<float>{.01f,.05f,.1f,.15f}) {
      auto model=makeModel();std::vector<torch::Tensor> original;for(auto& p:model->parameters()) original.push_back(p.detach().clone());
      auto rows=torch::arange(n,torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong)).remainder(values.size(0));
      auto data=values.index_select(0,rows),meta=metadata.index_select(0,rows);
      auto x=data.slice(1,0,3).contiguous().detach().set_requires_grad(true);
      auto qr=data.slice(1,3,7).contiguous().detach().set_requires_grad(true);
      auto sr=data.slice(1,7,10).contiguous().detach().set_requires_grad(true);
      auto op=data.slice(1,10,11).contiguous().detach().set_requires_grad(true);
      auto dc=data.slice(1,11,14).contiguous().reshape({n,1,3}).detach().set_requires_grad(true);
      auto rest=data.slice(1,14).contiguous().reshape({n,3,-1}).transpose(1,2).contiguous().detach().set_requires_grad(true);
      GaussianProtection protection;protection.config.performance_logging=false;protection.mapping_iteration=mapping;
      protection.birth_iter=meta.select(1,0).to(torch::kLong);protection.optimized_count=meta.select(1,1).to(torch::kInt32);protection.unique_keyframe_count=meta.select(1,2).to(torch::kInt32);protection.cached_uniform=meta.select(1,3).contiguous();
      ElasticBudgetManager manager;manager.performance_logging=false;manager.useUniform(protection.cached_uniform);
      auto mature_initial=protection.matureMask();const int64_t m=mature_initial.sum().item<int64_t>();
      torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
      auto generator=at::cuda::detail::createCUDAGenerator();
      std::vector<std::vector<double>> gpu(stages.size()),cpu(stages.size());std::vector<double> total,wall,ks;
      Event a,b,ev[14];
      const int iterations=profile?110:600;
      for(int iteration=0;iteration<iterations;++iteration) {
        if(profile && iteration==100) {cudaDeviceSynchronize();if(operators){using namespace torch::profiler::impl;torch::autograd::profiler::prepareProfiler(ProfilerConfig(ProfilerState::KINETO,true,true),{ActivityType::CPU,ActivityType::CUDA});torch::autograd::profiler::enableProfiler(ProfilerConfig(ProfilerState::KINETO,true,true),{ActivityType::CPU,ActivityType::CUDA});}else TORCH_CHECK(cudaProfilerStart()==cudaSuccess,"profile start");}
        std::unique_ptr<torch::NoGradGuard> inference_guard;if(mode=="cpp_inference")inference_guard=std::make_unique<torch::NoGradGuard>();
        TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess,"prior sync");
        const bool detailed=mode=="staged_train";
        const auto start=Clock::now();if(profile)nvtxRangePushA("iteration");a.record();
        int stage=0;auto phase=[&](const std::string& name,auto fn) {if(!detailed){fn();return;}if(profile)nvtxRangePushA(name.c_str());ev[stage].record();auto t=Clock::now();fn();double ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();if(profile)nvtxRangePop();if(iteration>=100)cpu[stage].push_back(ms);++stage;};
        torch::Tensor mature,ids,q,s,alpha,xm,qm,sm,logits,noise,soft,prob,hard,mask,selected,opacity_selected,photo,loss;
        ElasticBudgetPlan budget;bool all=false;
        phase("protection",[&]{mature=protection.matureMask();ids=torch::nonzero(mature).reshape({-1});all=ids.numel()==n;budget=manager.plan(manager.population(mature,protection.classify().need,ids.numel()),ratio,protection.config);});
        phase("features",[&]{q=torch::nn::functional::normalize(qr,torch::nn::functional::NormalizeFuncOptions().dim(1));s=sr.exp();alpha=op.sigmoid();xm=all?x:x.index_select(0,ids);qm=all?q:q.index_select(0,ids);sm=all?s:s.index_select(0,ids);});
        phase("network",[&]{if(m==0){hard=torch::empty({0},x.options());prob=hard;return;}if(mode=="caller_train")hard=model->forwardHard(xm,qm,sm,ratio,1.0,&prob);else if(mode=="cpp_inference")prob=model->forwardProbability(xm,qm,sm,ratio);else logits=model->logitsFor(xm,qm,sm,torch::full({1,1},ratio,x.options()));});
        phase("gumbel",[&]{if(mode=="staged_train" && m)noise=-torch::empty_like(logits).exponential_().log();else if(mode=="cpp_inference"){generator.set_current_seed(0);noise=at::rand({n},generator,x.options());}});
        phase("probability",[&]{if(mode=="staged_train" && m){prob=logits.softmax(-1).select(1,1);soft=((logits+noise)/1.0).softmax(-1);}});
        phase("row_argmax",[&]{if(mode=="staged_train" && m)selected=std::get<1>(soft.max(-1,true));});
        phase("mask",[&]{if(mode=="staged_train" && m)hard=(torch::zeros_like(logits).scatter_(-1,selected,1.0)-soft.detach()+soft).select(1,1);else if(mode=="cpp_inference")hard=(noise.index_select(0,ids)<prob).to(torch::kFloat32);auto full=all?hard:torch::zeros({n},x.options()).index_copy(0,ids,hard);auto admitted=budget.admitted.to(torch::kFloat32);mask=admitted+(1-admitted)*full;});
        phase("compact_indices",[&]{selected=torch::nonzero(mask.detach()>0).reshape({-1});});
        phase("gather",[&]{auto xs=x.index_select(0,selected),ss=s.index_select(0,selected),qs=q.index_select(0,selected),dcs=dc.index_select(0,selected),rss=rest.index_select(0,selected);opacity_selected=mask.index_select(0,selected).unsqueeze(1)*alpha.index_select(0,selected);});
        phase("loss_vjp",[&]{if(mode!="cpp_inference"){auto expected=m? (all?prob.index_select(0,ids):prob).sum():torch::zeros({},x.options());auto ratio_loss=torch::abs(expected-budget.target.mature)/std::max(1.0,budget.target.total);photo=opacity_selected.numel()?opacity_selected.mean():torch::zeros({},x.options());loss=photo+.01*ratio_loss;}});
        phase("backward",[&]{if(mode!="cpp_inference" && loss.requires_grad())loss.backward();});
        phase("optimizer",[&]{if(mode!="cpp_inference")optimizer.step();});
        phase("zero_grad",[&]{if(mode!="cpp_inference"){optimizer.zero_grad(true);for(auto* t:{&x,&qr,&sr,&op,&dc,&rest})t->mutable_grad()=torch::Tensor();}});
        if(detailed)ev[stage].record();b.record();TORCH_CHECK(cudaEventSynchronize(b.e)==cudaSuccess,"end sync");if(profile)nvtxRangePop();
        if(iteration>=100 && !profile){double t=elapsed(a,b),w=std::chrono::duration<double,std::milli>(Clock::now()-start).count();total.push_back(t);wall.push_back(w);ks.push_back(selected.numel());if(detailed)for(size_t j=0;j<stages.size();++j)gpu[j].push_back(elapsed(ev[j],ev[j+1]));samples<<mode<<','<<n<<','<<m<<','<<ratio<<','<<iteration-100<<','<<selected.numel()<<','<<t<<','<<w<<'\n';}
        // Keep the checkpoint selection distribution fixed. Discard all private
        // diagnostic updates after the timed region; no production state is used.
        {torch::NoGradGuard no_grad;auto params=model->parameters();for(size_t j=0;j<params.size();++j)params[j].copy_(original[j]);}
      }
      if(profile){cudaDeviceSynchronize();if(operators)torch::autograd::profiler::disableProfiler()->save(dir+"/native_operator_trace.json");else cudaProfilerStop();std::cout<<"PROFILE complete N="<<n<<" M="<<m<<std::endl;return 0;}
      emit(csv,mode,n,m,ratio,"total","cuda_event_ms",total);emit(csv,mode,n,m,ratio,"total","wall_ms",wall);emit(csv,mode,n,m,ratio,"selected","count",ks);
      if(mode=="staged_train")for(size_t j=0;j<stages.size();++j){emit(csv,mode,n,m,ratio,stages[j],"cuda_event_ms",gpu[j]);emit(csv,mode,n,m,ratio,stages[j],"dispatch_ms",cpu[j]);}
      bool finite=true;for(auto& p:model->parameters())finite &= torch::isfinite(p).all().item<bool>();for(auto& item:optimizer.state()){auto& st=static_cast<torch::optim::AdamParamState&>(*item.second);finite &= torch::isfinite(st.exp_avg()).all().item<bool>() && torch::isfinite(st.exp_avg_sq()).all().item<bool>();}
      valid<<mode<<','<<n<<','<<ratio<<','<<finite<<'\n';valid.flush();std::cout<<mode<<" N="<<n<<" M="<<m<<" ratio="<<ratio<<" complete"<<std::endl;
    }
  }
}
