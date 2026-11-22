#include "improvements/selector/spatial_compensation.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>
using Tensor=torch::Tensor;using List=std::vector<Tensor>;
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==3||argc==4,"spatial_live_benchmark real_fixture.pt output_dir [live]");torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);
  torch::serialize::InputArchive archive;archive.load_from(argv[1],torch::kCUDA);List full(7);
  for(int j=0;j<7;++j)archive.read("input_"+std::to_string(j),full[j],true);
  improvements::selector::SpatialCompensation model;model->to(torch::kCUDA);model->initialize(full[0],full[1],full[2],full[3],full[4],full[5],full[6]);
  {torch::NoGradGuard guard;for(const auto& entry:model->named_parameters()){
    auto name=entry.key();size_t pos=0;while((pos=name.find('.',pos))!=std::string::npos){name.replace(pos,1,"__");pos+=2;}
    Tensor value;archive.read("param__"+name,value,true);entry.value().copy_(value);}}
  const bool live=argc==4;model->setLiveFused(live);
  std::filesystem::create_directories(argv[2]);std::ofstream out(std::filesystem::path(argv[2])/(live?"live_fused_field_timing.csv":"live_field_timing.csv"));
  out<<"ratio,K,block,fused,stage,repeat,cuda_ms,host_submit_ms\n";
  std::ofstream parity(std::filesystem::path(argv[2])/(live?"live_fused_historical_output_parity.csv":"historical_output_parity.csv"));parity<<"ratio,K,max_abs,rms,bad\n";
  const std::vector<double> ratios{.01,.025,.05,.075,.1,.125,.15};
  for(size_t r=0;r<ratios.size();++r){Tensor ids,expected;archive.read("ids_"+std::to_string(r),ids,true);archive.read("expected_"+std::to_string(r),expected,true);
    List a;for(int j=0;j<7;++j)a.push_back(full[j].index_select(0,ids).detach().clone().requires_grad_(j<5));
    auto forward=[&](){return model->forward(a[0],a[1],a[2],a[3],a[4],a[5],a[6],ratios[r]);};
    {torch::NoGradGuard guard;auto actual=forward();auto diff=(actual-expected).abs();auto bad=diff>1e-6+1e-5*expected.abs();
      parity<<ratios[r]<<','<<ids.numel()<<','<<diff.max().item<double>()<<','<<diff.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<'\n';
      TORCH_CHECK(!bad.any().item<bool>(),"Frozen reference output mismatch");}
    auto vjp=torch::ones({ids.numel(),14},full[0].options())*.001;
    for(int block=0;block<3;++block)for(int order=0;order<2;++order){bool fused=(block+order)%2;model->setFused(fused);
      for(int stage=0;stage<2;++stage){auto run=[&](){if(stage==0){torch::NoGradGuard guard;forward();}else{model->zero_grad(true);for(int j=0;j<5;++j)a[j].mutable_grad()=torch::Tensor();forward().backward(vjp);}};
        for(int j=0;j<10;++j)run();cudaDeviceSynchronize();std::vector<cudaEvent_t> start(35),end(35);std::vector<double> cpu;
        for(int j=0;j<35;++j){cudaEventCreate(&start[j]);cudaEventCreate(&end[j]);auto t=std::chrono::steady_clock::now();cudaEventRecord(start[j]);run();cudaEventRecord(end[j]);cpu.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
        cudaDeviceSynchronize();for(int j=0;j<35;++j){float ms;cudaEventElapsedTime(&ms,start[j],end[j]);out<<ratios[r]<<','<<ids.numel()<<','<<block<<','<<fused<<','<<(stage?"forward_backward":"forward")<<','<<j<<','<<ms<<','<<cpu[j]<<'\n';cudaEventDestroy(start[j]);cudaEventDestroy(end[j]);}
      }
    }
  }std::cout<<"LIVE_FIELD_BENCHMARK_PASS\n";return 0;
}catch(const c10::Error& e){std::cerr<<e.what()<<'\n';return 1;}}
