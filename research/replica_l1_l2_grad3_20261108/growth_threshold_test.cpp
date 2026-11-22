#include "improvements/selector/anchor_bank.h"
#include <iostream>
#include <limits>

int main(int argc,char** argv) {
  const auto device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  const auto options=torch::TensorOptions().device(device).dtype(torch::kFloat32);
  for(const auto& entry:std::vector<std::array<double,3>>{
      {0,1.5,0},{0,2.,1},{1,1.9,-1},{1,2.,1},{1,2.999,1},{1,3.,2},{1,3.5,2},{1,4.,2}}) {
    improvements::selector::AnchorBank bank(.1,0);
    bank->enableExplicitBlocks();bank->enableVariableMembers();
    bank->to(device);
    auto observations=torch::tensor({{.011f,.012f,.013f}},options);
    auto initial=bank->observeBlocks(observations,torch::full_like(observations,.4));
    auto xyz=initial.xyz;
    const auto level=int64_t(entry[0]);
    const auto owner=torch::nonzero(bank->levels()==level).reshape({-1})[0].item<int64_t>();
    const auto row=torch::nonzero(bank->rows()==owner).reshape({-1})[0].item<int64_t>();
    xyz[row][0]=.711f;
    auto gradient=torch::zeros({xyz.size(0),1},options.dtype(torch::kFloat64));gradient[row]=entry[1];
    auto grown=bank->growBlocks(xyz,initial.color,gradient,torch::ones_like(gradient),1.,2.);
    if(entry[2]<0) {
      TORCH_CHECK(grown.xyz.numel()==0,"Below-threshold member grew");
    } else {
      TORCH_CHECK(grown.xyz.size(0)==1&&bank->levels()[-1].item<int64_t>()==int64_t(entry[2]),
                  "Incorrect growth level for source ",level," gradient ",entry[1]);
    }
    bank->assertBlocks(xyz.size(0)+grown.xyz.size(0));
  }
  std::cout<<"PASS: actual Anchor growth; L0 unchanged; L1-to-L2 opens at 3*tau0\n";
}
