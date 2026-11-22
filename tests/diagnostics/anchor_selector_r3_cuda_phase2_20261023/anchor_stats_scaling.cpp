// 人工规模压力测试；真实拓扑/窗口正确性由 fused_anchor_stats_test 验收。
#include "include/anchor_stats_cuda.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <fstream>
#include <chrono>
#include <iostream>
int main(int argc,char** argv){TORCH_CHECK(argc==2,"anchor_stats_scaling output.csv");torch::set_num_threads(2);torch::manual_seed(2026);std::ofstream csv(argv[1]);csv<<"N,A,repeat,fused,stage,host_ms,cuda_ms\n";
  auto opts=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
  for(int64_t n:{15891,50000,100000}){int64_t a=(n+9)/10;auto owner=torch::arange(n,opts.dtype(torch::kLong))/10;owner=owner.to(torch::kLong);
    auto counts=torch::full({a},10,owner.options());counts[-1].fill_(n-10*(a-1));auto levels=torch::arange(a,owner.options())%3;
    auto centers=torch::randn({a,3},opts),scales=torch::full({a,1},.1,opts),alpha=torch::rand({n,1},opts),visible=torch::arange(0,n,3,owner.options());
    auto members=mapping_cuda::validateAnchors(owner,counts,levels,centers,scales);
    std::vector<torch::Tensor> s{torch::zeros({a},opts.dtype(torch::kFloat64)),torch::zeros({a},owner.options()),torch::zeros({n},opts.dtype(torch::kFloat64)),torch::zeros_like(owner),torch::zeros_like(owner)};
    for(int rep=-10;rep<60;++rep)for(int order=0;order<2;++order)for(bool stats:{false,true}){bool use=(rep+order)%2==0;torch::cuda::synchronize();cudaEventRecord(begin);auto start=std::chrono::steady_clock::now();
      if(!stats){if(use)members=mapping_cuda::validateAnchors(owner,counts,levels,centers,scales);else{
        TORCH_CHECK((owner>=0).all().item<bool>()&&(owner<a).all().item<bool>());auto actual=torch::zeros_like(counts);actual.index_add_(0,owner,torch::ones_like(owner));
        TORCH_CHECK(torch::equal(actual,counts)&&((actual>=1)&(actual<=10)).all().item<bool>()&&((levels>=0)&(levels<=2)).all().item<bool>()&&torch::isfinite(centers).all().item<bool>()&&torch::isfinite(scales).all().item<bool>()&&(scales>0).all().item<bool>());
      }}else if(use)mapping_cuda::anchorFullStats(owner,members,counts,visible,alpha,s);else{
        auto mark=torch::zeros({a},opts.dtype(torch::kBool));mark.index_fill_(0,owner.index_select(0,visible),true);auto values=alpha.reshape({-1}).to(torch::kFloat64),active=mark.index_select(0,owner);
        auto observed=torch::where(active,values,torch::zeros_like(values));s[0].index_add_(0,owner,observed);s[1]+=mark.to(torch::kLong);s[2]+=observed;s[3]+=((values<.02)&active).to(torch::kLong);s[4]+=((values<1./255)&active).to(torch::kLong);
      }
      auto host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,begin,end);if(rep>=0)csv<<n<<','<<a<<','<<rep<<','<<use<<','<<(stats?"stats_only":"validate")<<','<<host<<','<<ms<<'\n';
    }
  }cudaEventDestroy(begin);cudaEventDestroy(end);std::cout<<"PASS synthetic scale benchmark\n";
}
