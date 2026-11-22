#define main previous_network_benchmark_main
#include "../cuda_selector_implementation_20261015/network_benchmark.cpp"
#undef main

int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==4,"backward_benchmark CHECKPOINT AUDIT_FIXTURE OUTPUT");
    torch::set_num_threads(1);torch::manual_seed(123);at::globalContext().setAllowTF32CuBLAS(false);
    GumbelNetwork model;torch::serialize::InputArchive archive;archive.load_from(argv[1],torch::kCPU);model->load(archive);model->to(torch::kCUDA);
    auto data=read(std::string(argv[2])+"/input.bin"),meta=read(std::string(argv[2])+"/metadata.bin");
    int64_t mapping=0;std::ifstream(std::string(argv[2])+"/spec.txt")>>mapping;
    data=data.index_select(0,torch::nonzero((mapping-meta.select(1,0)>=100)&(meta.select(1,1)>=80)&(meta.select(1,2)>=4)).flatten());
    std::ofstream out(argv[3]);out<<"backend,M,stage,metric,mean_ms,p50_ms,p95_ms\n";
    std::ofstream parity(std::string(argv[3])+".parity.csv");parity<<"M,backend,gradient,exact,max_abs,rms,relative_l2,cosine,passed\n";
    for(int n:{1000,50000,219969}) {
      std::vector<torch::Tensor> input{data.slice(0,0,n).slice(1,0,3).contiguous().detach().set_requires_grad(true),
        torch::nn::functional::normalize(data.slice(0,0,n).slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)).detach().set_requires_grad(true),
        data.slice(0,0,n).slice(1,7,10).exp().detach().set_requires_grad(true),torch::full({1,1},.05,data.options())};
      auto parameters=model->parameters(),targets=std::vector<torch::Tensor>{input[0],input[1],input[2]};targets.insert(targets.end(),parameters.begin(),parameters.end());
      auto dy=torch::randn({n,2},data.options());
      const std::vector<std::string> names{"legacy","current","fused_activation","batched_weight","fused_bias","grouped_branch"};
      for(int mode=-1;mode<5;++mode) {
        std::string name=names[mode+1];
        auto run=[&](){return mode<0?model->logitsFor(input[0],input[1],input[2],input[3]):
          selector_cuda::network(input,parameters,selector_cuda::Candidate::CublasStrict,static_cast<selector_cuda::BackwardStrategy>(mode));};
        {
          auto a=model->logitsFor(input[0],input[1],input[2],input[3]),b=run();
          auto ga=torch::autograd::grad({a},targets,{dy}),gb=torch::autograd::grad({b},targets,{dy});
          for(size_t i=0;i<ga.size();++i){
            bool exact=torch::equal(ga[i],gb[i]);double maximum=0,rms=0,relative=0,cosine=1;
            if(!exact){auto x=ga[i].to(torch::kDouble).flatten(),y=gb[i].to(torch::kDouble).flatten(),d=x-y;
              double nx=x.norm().item<double>(),ny=y.norm().item<double>();maximum=d.abs().max().item<double>();rms=d.square().mean().sqrt().item<double>();
              relative=d.norm().item<double>()/std::max(nx,1e-12);cosine=nx&&ny?torch::dot(x,y).item<double>()/(nx*ny):double(nx==ny);}
            parity<<n<<','<<name<<','<<i<<','<<exact<<','<<maximum<<','<<rms<<','<<relative<<','<<cosine<<','<<(relative<=1e-4 && cosine>=.99999)<<'\n';parity.flush();
          }
        }
        measure(out,name,n,"network_forward",[&](){torch::NoGradGuard ng;auto y=run();},false);
        auto y=run();measure(out,name,n,"network_backward",[&](){auto g=torch::autograd::grad({y},targets,{dy},true);},false);
        y=torch::Tensor();measure(out,name,n,"network_forward_backward",[&](){auto y=run();auto g=torch::autograd::grad({y},targets,{dy});},false);
      }
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
