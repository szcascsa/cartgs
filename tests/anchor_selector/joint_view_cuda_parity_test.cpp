#include <torch/torch.h>
#include "include/flexgs_selector_cuda.h"
#include <filesystem>
#include <iostream>

// Run the same binary with old/new libgaussian_mapper.so, then compare archives.
int main(int argc,char**argv) {
  torch::set_num_threads(2);torch::manual_seed(71);at::globalContext().setAllowTF32CuBLAS(false);
  auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  torch::serialize::OutputArchive archive;int count=0;
  for(int width:{11,21})for(int n:{0,1,31,2048,13298}) {
    std::vector<torch::Tensor> inputs;
    for(auto shape:std::vector<std::vector<int64_t>>{{n,width},{32,width},{32},{32,32},{32},{2,32},{2}})
      inputs.push_back(torch::randn(shape,o).requires_grad_(true));
    auto out=mapping_cuda::flexgsMLP(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6]);
    auto a=torch::randn_like(out[0]),b=torch::randn_like(out[1]);
    auto gradients=torch::autograd::grad({out[0],out[1]},inputs,{a,b});
    for(size_t j=0;j<out.size();++j)archive.write("c"+std::to_string(count)+"_out"+std::to_string(j),out[j].detach(),true);
    for(size_t j=0;j<gradients.size();++j)archive.write("c"+std::to_string(count)+"_grad"+std::to_string(j),gradients[j].detach(),true);
    ++count;
  }
  std::filesystem::path path=argc>1?argv[1]:"/tmp/joint_view_cuda_parity.pt";
  archive.save_to(path.string());std::cout<<"PASS legacy CUDA dump "<<count<<" cases\n";
}
