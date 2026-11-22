#define main previous_benchmark_main
#include "../cuda_selector_implementation_20261015/network_benchmark.cpp"
#undef main
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc>=5,"benchmark CHECKPOINT FIXTURE OUTPUT STRATEGY [M]");
    torch::set_num_threads(1);torch::manual_seed(123);at::globalContext().setAllowTF32CuBLAS(false);
    GumbelNetwork model;torch::serialize::InputArchive archive;
    archive.load_from(argv[1],torch::kCPU);model->load(archive);model->to(torch::kCUDA);
    std::string fixture=argv[2];auto data=read(fixture+"/input.bin"),meta=read(fixture+"/metadata.bin");
    int64_t mapping=0;std::ifstream(fixture+"/spec.txt")>>mapping;
    data=data.index_select(0,torch::nonzero((mapping-meta.select(1,0)>=100)&(meta.select(1,1)>=80)&(meta.select(1,2)>=4)).flatten());
    int mode=std::stoi(argv[4]);std::vector<int> grid{1000,5000,10000,20000,50000,100000,200000,219969};
    if(argc>5)grid={std::stoi(argv[5])};
    std::ofstream out(argv[3]);out<<"backend,M,stage,metric,mean_ms,p50_ms,p95_ms\n";
    for(int n:grid) {
      std::vector<torch::Tensor> x{data.slice(0,0,n).slice(1,0,3).contiguous().detach().set_requires_grad(true),
        torch::nn::functional::normalize(data.slice(0,0,n).slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)).detach().set_requires_grad(true),
        data.slice(0,0,n).slice(1,7,10).exp().detach().set_requires_grad(true),torch::full({1,1},.05,data.options())};
      auto p=model->parameters(),args=std::vector<torch::Tensor>{x[0],x[1],x[2]};args.insert(args.end(),p.begin(),p.end());
      auto dy=torch::randn({n,2},data.options());
      auto run=[&](){return mode<0?model->logitsFor(x[0],x[1],x[2],x[3]):selector_cuda::network(x,p,
        selector_cuda::Candidate::CublasStrict,static_cast<selector_cuda::BackwardStrategy>(mode));};
      std::string name=mode<0?"legacy":"strategy"+std::to_string(mode);
      measure(out,name,n,"network_forward",[&](){torch::NoGradGuard ng;auto y=run();},false);
      auto y=run();measure(out,name,n,"network_backward",[&](){auto g=torch::autograd::grad({y},args,{dy},true);},false);
      y=torch::Tensor();measure(out,name,n,"network_forward_backward",[&](){auto z=run();auto g=torch::autograd::grad({z},args,{dy});},false);
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
