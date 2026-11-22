#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include "improvements/selector/gumbel_network.h"
#include "include/elastic_budget.h"
#include "include/gi_teacher.h"
torch::Tensor read(const std::string& path) {std::ifstream f(path,std::ios::binary);int64_t n,w;f.read(reinterpret_cast<char*>(&n),8);f.read(reinterpret_cast<char*>(&w),8);std::vector<float> v(n*w);f.read(reinterpret_cast<char*>(v.data()),v.size()*4);TORCH_CHECK(f,"input");return torch::from_blob(v.data(),{n,w}).clone().cuda();}
void emit(std::ofstream& out,const std::string& stage,int64_t n,float ratio,const std::string& metric,std::vector<double> v) {double mean=std::accumulate(v.begin(),v.end(),0.0)/v.size();std::sort(v.begin(),v.end());out<<stage<<','<<n<<','<<ratio<<','<<metric<<','<<mean<<','<<v[v.size()/2]<<','<<v[size_t(.95*(v.size()-1))]<<'\n';out.flush();}
int main(int argc,char** argv) {
  TORCH_CHECK(argc==2,"isolated_costs DIAGNOSTICS");torch::set_num_threads(1);std::string dir=argv[1];auto data=read(dir+"/input.bin"),meta=read(dir+"/metadata.bin");
  std::vector<float> gi_raw(data.size(0));std::ifstream gi_file(dir+"/frame_gi.bin",std::ios::binary);gi_file.read(reinterpret_cast<char*>(gi_raw.data()),gi_raw.size()*4);TORCH_CHECK(gi_file,"real GI fixture");auto gi_all=torch::tensor(gi_raw).cuda();
  std::ofstream out(dir+"/native_isolated_costs.csv");out<<"stage,N,ratio,metric,mean,p50,p95\n";
  for(int64_t n:{1000,5000,10000,20000,50000,52181,100000,108688,200000,219969}) {
    auto x=data.slice(0,0,n).slice(1,0,3).contiguous().detach().set_requires_grad(true),qr=data.slice(0,0,n).slice(1,3,7).contiguous(),sr=data.slice(0,0,n).slice(1,7,10).contiguous();auto q=torch::nn::functional::normalize(qr),s=sr.exp();
    improvements::selector::GumbelNetwork model;torch::serialize::InputArchive archive;archive.load_from("/home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt",torch::kCPU);model->load(archive);model->to(torch::kCUDA);
    auto mature=(3400-meta.slice(0,0,n).select(1,0)>=100)&(meta.slice(0,0,n).select(1,1)>=80)&(meta.slice(0,0,n).select(1,2)>=4);auto uniform=meta.slice(0,0,n).select(1,3).contiguous();
    auto need=1-torch::minimum(torch::minimum(((3400-meta.slice(0,0,n).select(1,0))/100).clamp(0,1),(meta.slice(0,0,n).select(1,1)/80).clamp(0,1)),(meta.slice(0,0,n).select(1,2)/4).clamp(0,1));
    ElasticBudgetManager manager;manager.performance_logging=false;manager.useUniform(uniform);ProtectionConfig cfg;cfg.performance_logging=false;auto population=manager.population(mature,need,mature.sum().item<int64_t>());
    auto gi=gi_all.slice(0,0,n);auto gi_observed=gi>=0;
    for(float ratio:{.01f,.05f,.1f,.15f})for(const std::string stage:{"inference_network","training_forward","training_backward","feature_activations","selection_mask_index","gi_teacher","protection_budget"}) {
      std::vector<double> gpu,wall;cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);torch::Tensor loss,output,logits;
      auto precomputed_probability=model->forwardProbability(x,q,s,ratio).detach();logits=torch::stack({torch::zeros_like(precomputed_probability),torch::logit(precomputed_probability.clamp(1e-7,1-1e-7))},1).detach();
      for(int i=0;i<600;++i) {
        model->zero_grad();x.mutable_grad()=torch::Tensor();
        if(stage=="training_backward") {torch::Tensor p;auto h=model->forwardHard(x,q,s,ratio,1,&p);loss=h.mean()+.01*torch::abs(p.mean()-ratio);}
        cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();cudaEventRecord(a,c10::cuda::getCurrentCUDAStream());
        if(stage=="inference_network"){torch::NoGradGuard no_grad;output=model->forwardProbability(x,q,s,ratio);}
        else if(stage=="training_forward"){torch::Tensor p;output=model->forwardHard(x,q,s,ratio,1,&p);}
        else if(stage=="training_backward")loss.backward();
        else if(stage=="feature_activations"){output=torch::cat({sr.exp(),torch::nn::functional::normalize(qr)},1);}
        else if(stage=="selection_mask_index"){torch::NoGradGuard no_grad;auto noise=-torch::empty_like(logits).exponential_().log();auto soft=((logits+noise)/1.0).softmax(-1);auto onehot=torch::zeros_like(logits).scatter_(-1,std::get<1>(soft.max(-1,true)),1.0);auto hard=(onehot-soft.detach()+soft).select(1,1);output=torch::nonzero(hard.detach()>0).reshape({-1});}
        else if(stage=="gi_teacher"){auto teacher=GITeacherBuilder().build(gi,mature,~mature,ratio,1,1,population.mature_count,gi_observed);output=teacher.label;}
        else {auto plan=manager.plan(population,ratio,cfg);output=plan.admitted;}
        cudaEventRecord(b,c10::cuda::getCurrentCUDAStream());cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);if(i>=100){gpu.push_back(ms);wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
      }
      emit(out,stage,n,ratio,"cuda_event_ms",gpu);emit(out,stage,n,ratio,"wall_ms",wall);cudaEventDestroy(a);cudaEventDestroy(b);
    }
    std::cout<<"N="<<n<<" complete"<<std::endl;
  }
}
