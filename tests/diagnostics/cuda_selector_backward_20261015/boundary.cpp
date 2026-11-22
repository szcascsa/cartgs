#define main previous_benchmark_main
#include "../cuda_selector_implementation_20261015/network_benchmark.cpp"
#undef main
int main(int argc,char** argv) {
  torch::set_num_threads(1);torch::manual_seed(2026);torch::NoGradGuard ng;
  at::globalContext().setAllowTF32CuBLAS(false);GumbelNetwork model;model->to(torch::kCUDA);
  std::ofstream out(argv[1]);out<<"M,ratio_rows,gradient,exact,max_abs\n";int failures=0;
  auto options=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  for(int n:{0,1,2,3,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,31,32,33,63,64,65,127,128,129,257,1000,5000,10000})for(bool row:{false,true}) {
    std::vector<torch::Tensor> x{torch::randn({n,3},options),torch::randn({n,4},options),
      torch::randn({n,3},options).exp(),torch::full({row?n:1,1},.05,options)};
    auto p=model->parameters(),h=selector_cuda::forward(x,p,selector_cuda::Candidate::CublasStrict);
    h.erase(h.begin()+8);auto dy=torch::randn({n,2},options);
    auto reference=static_cast<selector_cuda::BackwardStrategy>(argc>3?std::stoi(argv[3]):6);
    auto a=selector_cuda::backward(x,p,h,dy,reference);
    auto b=selector_cuda::backward(x,p,h,dy,static_cast<selector_cuda::BackwardStrategy>(std::stoi(argv[2])));
    for(int i=0;i<22;++i){bool exact=torch::equal(a[i],b[i]);failures+=!exact;
      out<<n<<','<<x[3].size(0)<<','<<i<<','<<exact<<','<<(a[i].numel()?(a[i]-b[i]).abs().max().item<float>():0)<<'\n';}
    out.flush();
  }
  std::cout<<"nonexact="<<failures<<'\n';return failures?1:0;
}
