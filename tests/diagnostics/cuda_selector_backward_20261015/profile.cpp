#define main previous_benchmark_main
#include "../cuda_selector_implementation_20261015/network_benchmark.cpp"
#undef main
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc>=3,"backward_profile CHECKPOINT FIXTURE [M] [scalar|row]");
    torch::set_num_threads(1);torch::manual_seed(123);
    at::globalContext().setAllowTF32CuBLAS(false);
    GumbelNetwork model;torch::serialize::InputArchive archive;
    archive.load_from(argv[1],torch::kCPU);model->load(archive);model->to(torch::kCUDA);
    std::string fixture=argv[2];auto data=read(fixture+"/input.bin"),metadata=read(fixture+"/metadata.bin");
    int64_t mapping=0;std::ifstream(fixture+"/spec.txt")>>mapping;
    auto ids=torch::nonzero((mapping-metadata.select(1,0)>=100)&(metadata.select(1,1)>=80)&(metadata.select(1,2)>=4)).flatten();
    data=data.index_select(0,ids);int n=argc>3?std::stoi(argv[3]):219969;
    bool row=argc>4&&std::string(argv[4])=="row";
    std::vector<torch::Tensor> x{data.slice(0,0,n).slice(1,0,3).contiguous().detach().set_requires_grad(true),
      torch::nn::functional::normalize(data.slice(0,0,n).slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)).detach().set_requires_grad(true),
      data.slice(0,0,n).slice(1,7,10).exp().detach().set_requires_grad(true),
      torch::full({row?n:1,1},.05,data.options()).set_requires_grad(true)};
    auto p=model->parameters(),args=x;args.insert(args.end(),p.begin(),p.end());
    auto strategy=static_cast<selector_cuda::BackwardStrategy>(argc>5?std::stoi(argv[5]):1);
    auto y=selector_cuda::network(x,p,selector_cuda::Candidate::CublasStrict,strategy);
    auto dy=torch::randn({n,2},data.options());
    for(int i=0;i<100;++i){auto g=torch::autograd::grad({y},args,{dy},true);}
    cudaDeviceSynchronize();cudaProfilerStart();
    for(int i=0;i<10;++i){nvtxRangePushA("backward");auto g=torch::autograd::grad({y},args,{dy},true);nvtxRangePop();}
    cudaDeviceSynchronize();cudaProfilerStop();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
