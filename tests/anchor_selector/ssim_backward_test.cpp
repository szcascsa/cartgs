#include "include/loss_utils.h"
#include "cuda_rasterizer/ssim_backend.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==4,"ssim_backward_test checkpoint fixture output");torch::set_num_threads(2);torch::manual_seed(2026);std::filesystem::path out=argv[3];std::filesystem::create_directories(out);
  std::ofstream parity(out/"ssim_parity.csv"),timing(out/"ssim_timing.csv"),attrs(out/"ssim_attributes.csv");
  parity<<"H,W,max_abs,max_rel,rms,bad_elements\n";timing<<"H,W,repeat,optimized,host_ms,cuda_ms\n";attrs<<"optimized,registers,shared_bytes,max_threads,resident_blocks_per_sm\n";
  for(bool split:{false,true}){auto a=ssim_backend::attributes(split);attrs<<split;for(auto x:a)attrs<<','<<x;attrs<<'\n';}
  torch::serialize::InputArchive ar;ar.load_from(argv[2],torch::kCUDA);torch::Tensor render,gt;ar.read("render",render);ar.read("gt",gt);
  for(auto hw:std::vector<std::pair<int,int>>{{480,640},{240,320},{17,31},{1,1},{61,83}}){
    auto x=torch::nn::functional::interpolate(render.unsqueeze(0),torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{hw.first,hw.second}).mode(torch::kBilinear).align_corners(false)).squeeze(0).contiguous();
    auto y=torch::nn::functional::interpolate(gt.unsqueeze(0),torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{hw.first,hw.second}).mode(torch::kBilinear).align_corners(false)).squeeze(0).contiguous();
    auto w=torch::randn_like(x);torch::Tensor a,b;
    {ssim_backend::Scope scope(false);a=fusedssim_backward(.01f*.01f,.03f*.03f,x,y,w);}
    {ssim_backend::Scope scope(true);b=fusedssim_backward(.01f*.01f,.03f*.03f,x,y,w);}
    auto diff=(a-b).abs(),bad=diff>2e-5+2e-4*a.abs();parity<<hw.first<<','<<hw.second<<','<<diff.max().item<double>()<<','<<(diff/a.abs().clamp_min(1e-30)).max().item<double>()<<','<<diff.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<'\n';
    TORCH_CHECK(torch::isfinite(b).all().item<bool>()&&!bad.any().item<bool>(),"SSIM VJP mismatch");
    auto xa=x.detach().clone().requires_grad_(true),xb=x.detach().clone().requires_grad_(true);torch::Tensor la,lb;
    {ssim_backend::Scope scope(false);la=loss_utils::fast_ssim(xa,y);}
    {ssim_backend::Scope scope(true);lb=loss_utils::fast_ssim(xb,y);}
    auto ga=torch::autograd::grad({la},{xa})[0],gb=torch::autograd::grad({lb},{xb})[0];TORCH_CHECK(torch::equal(ga,gb),"SSIM saved backend/autograd parity");
    cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
    for(int rep=-20;rep<150;++rep)for(int order=0;order<2;++order){bool split=(rep+order)%2==0;ssim_backend::Scope scope(split);torch::cuda::synchronize();cudaEventRecord(begin);auto start=std::chrono::steady_clock::now();nvtxRangePushA(split?"ssim/channel_blocks":"ssim/channel_loop");
      auto result=fusedssim_backward(.01f*.01f,.03f*.03f,x,y,w);double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();cudaEventRecord(end);nvtxRangePop();cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,begin,end);
      if(rep>=0)timing<<hw.first<<','<<hw.second<<','<<rep<<','<<split<<','<<host<<','<<ms<<'\n';}
    cudaEventDestroy(begin);cudaEventDestroy(end);
  }
  std::cout<<"PASS unchanged SSIM arithmetic, channel-block VJP and paired timing\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
