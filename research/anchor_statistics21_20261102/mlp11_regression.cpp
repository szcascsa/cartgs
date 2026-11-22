#include "include/flexgs_selector_cuda.h"
#include <filesystem>

int main(int argc,char** argv) {
  TORCH_CHECK(argc==2,"mlp11_regression output archive");
  torch::manual_seed(19);torch::set_num_threads(2);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::serialize::OutputArchive saved;
  for(int64_t n:{1,1031,20000}) {
    auto x=torch::randn({n,11},o).requires_grad_(true);
    auto w0=torch::randn({32,11},o).requires_grad_(true),b0=torch::randn({32},o).requires_grad_(true);
    auto w1=torch::randn({32,32},o).requires_grad_(true),b1=torch::randn({32},o).requires_grad_(true);
    auto w2=torch::randn({2,32},o).requires_grad_(true),b2=torch::randn({2},o).requires_grad_(true);
    auto result=mapping_cuda::flexgsMLP(x,w0,b0,w1,b1,w2,b2);
    auto g=torch::autograd::grad({result[0],result[1]},{x,w0,b0,w1,b1,w2,b2},
        {torch::randn_like(result[0]),torch::randn_like(result[1])});
    auto prefix="n"+std::to_string(n)+"_";
    saved.write(prefix+"input",x.detach(),true);
    saved.write(prefix+"logits",result[0].detach(),true);saved.write(prefix+"shared",result[1].detach(),true);
    for(size_t j=0;j<g.size();++j)saved.write(prefix+"gradient"+std::to_string(j),g[j],true);
  }
  saved.save_to(argv[1]);
}
