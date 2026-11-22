#include "probe_common.h"
int main(int argc,char** argv) {
  torch::set_num_threads(2);const std::string root=argv[1];
  for(int j:{0,5}) {
    torch::serialize::InputArchive ar;ar.load_from(root+"/snapshots/d0_r"+std::to_string(j)+"_fused.pt",torch::Device(torch::kCUDA));
    torch::Tensor owners,g,table;ar.read("owners",owners);ar.read("incoming_a_0",g);ar.read("scatter_a_0",table);
    for(int deterministic:{0,1})for(int rep=0;rep<4;++rep) {
      at::globalContext().setDeterministicAlgorithms(deterministic,false);
      auto p=torch::zeros_like(table).requires_grad_(true);auto f=p.index_select(0,owners);
      const auto label="ScatterOnly.d"+std::to_string(deterministic)+".r"+std::to_string(j);nvtxRangePushA(label.c_str());f.backward(g);nvtxRangePop();
    }
  }
  torch::cuda::synchronize();std::cout<<"SCATTER_KERNEL_IDENTITY_CAPTURE_COMPLETE"<<std::endl;
}
