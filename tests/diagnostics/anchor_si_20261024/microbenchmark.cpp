#include "include/anchor_si.h"
#include "include/gaussian_rasterizer.h"
#include "cuda_rasterizer/anchor_si_capture.h"
#include "cuda_rasterizer/render_workspace.h"
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) { try {
  TORCH_CHECK(argc==4,"anchor_si_microbenchmark checkpoint native_fixture output");torch::set_num_threads(2);torch::manual_seed(2026);
  namespace fs=std::filesystem;fs::path out=argv[3];fs::create_directories(out);renderer_workspace::Scope workspace(true);
  auto net=improvements::selector::GumbelNetwork(.1,true,false,true);net->to(torch::kCUDA);
  torch::serialize::InputArchive model;model.load_from((fs::path(argv[1])/"selector.pt").string(),torch::kCUDA);net->load(model);
  for(auto& item:net->named_parameters())item.value().requires_grad_(item.key().rfind("local_emd.2.",0)==0 || item.key().rfind("soft_net.",0)==0);
  auto bank=net->anchors();bank->setFusedFullStats(true);bank->assertBlocks(bank->rows().numel());
  torch::serialize::InputArchive fixture;fixture.load_from(argv[2],torch::kCUDA);
  torch::Tensor x,s,q,op,dc,sh,view,proj,gt;fixture.read("xyz",x);fixture.read("scale",s);fixture.read("rotation",q);fixture.read("opacity",op);
  fixture.read("dc",dc);fixture.read("parameter_before_2",sh);fixture.read("view",view);fixture.read("projection",proj);fixture.read("gt",gt);
  auto opts=x.options().requires_grad(false);
  auto bg=torch::zeros({3},opts),cam=view.inverse().select(0,3).slice(0,0,3).contiguous(),empty=torch::empty({0},opts);
  GaussianRasterizationSettings settings(gt.size(1),gt.size(2),.6,.45,bg,1,view,proj,3,cam,false,false,false);
  auto render=[&](bool capture,bool gradients){anchor_si::CaptureScope scope(capture);std::vector<torch::Tensor> inputs;
    for(auto tensor:{x,dc,sh,op,s,q})inputs.push_back(gradients?tensor.detach().clone().requires_grad_(true):tensor.detach());
    auto xy=torch::zeros_like(x);return std::make_pair(rasterizeGaussians(inputs[0],xy,inputs[1],inputs[2],empty,inputs[3],inputs[4],inputs[5],empty,settings),inputs);};
  auto legacy=render(false,true),candidate=render(true,true);TORCH_CHECK(torch::equal(legacy.first[0],candidate.first[0])&&torch::equal(legacy.first[1],candidate.first[1]),"Real Full RGB/radii changed");
  auto weight=torch::sign(legacy.first[0].detach()-gt)/double(gt.numel());
  auto ga=torch::autograd::grad({legacy.first[0]},legacy.second,{weight}),gb=torch::autograd::grad({candidate.first[0]},candidate.second,{weight});
  std::ofstream parity(out/"real_renderer_parity.csv");parity<<"group,max_abs,max_rel,rms,bad\n";
  for(size_t i=0;i<ga.size();++i) {
    auto difference=(ga[i]-gb[i]).abs(),bad=difference>2e-5+2e-4*ga[i].abs();
    parity<<i<<','<<difference.max().item<double>()<<','<<(difference/ga[i].abs().clamp_min(1e-30)).max().item<double>()<<','<<difference.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<'\n';
    TORCH_CHECK(!bad.any().item<bool>(),"Real Renderer VJP gate failed");
  }
  anchor_si::Config config;config.enabled=true;anchor_si::State state(config);
  auto contribution=candidate.first[2].detach();for(int i=0;i<16;++i)if(state.captureNextFull())state.observe(net,contribution,op,gt.size(1)*gt.size(2),1,{.01,.05,.1,.15});
  std::ofstream timing(out/"timing.csv");timing<<"stage,repeat,capture,host_ms,cuda_ms,N,A\n";
  cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
  auto measure=[&](const std::string& stage,int repeat,bool capture,auto operation) {
    torch::cuda::synchronize();cudaEventRecord(begin);auto start=std::chrono::steady_clock::now();operation();
    double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,begin,end);
    if(repeat>=0)timing<<stage<<','<<repeat<<','<<capture<<','<<host<<','<<ms<<','<<x.size(0)<<','<<bank->size()<<'\n';
  };
  for(int repeat=-20;repeat<160;++repeat)for(int order=0;order<2;++order) {
    bool capture=(repeat+order)%2==0;
    measure("Full_forward",repeat,capture,[&]{torch::NoGradGuard guard;auto value=render(capture,false);});
  }
  for(int repeat=-10;repeat<100;++repeat) {
    measure("observe_and_targets",repeat,true,[&]{state.observe(net,contribution,op,gt.size(1)*gt.size(2),1,{.01,.05,.1,.15});});
  }
  for(float ratio:{.01f,.05f,.1f,.15f}) {
    auto p=net->forwardSelectionFeatures(x,ratio,1,{},s,dc,op).probability;
    int index=ratio<.02?0:(ratio<.075?1:(ratio<.125?2:3));
    for(int repeat=-10;repeat<100;++repeat) {
      net->zero_grad(true);
      measure("L2_and_head_VJP_"+std::to_string(ratio),repeat,true,[&]{auto loss=state.loss(net,p,op,index,1);state.accumulateHeadGradients(loss,net);});
    }
  }
  cudaEventDestroy(begin);cudaEventDestroy(end);std::cout<<"PASS real Full Forward/VJP and paired SI overhead benchmark\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;} }
