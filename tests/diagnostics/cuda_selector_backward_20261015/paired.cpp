#define main previous_benchmark_main
#include "../cuda_selector_implementation_20261015/network_benchmark.cpp"
#undef main
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc>=4,"paired CHECKPOINT FIXTURE OUTPUT [profile]");
    torch::set_num_threads(1);torch::manual_seed(123);at::globalContext().setAllowTF32CuBLAS(false);
    GumbelNetwork model;torch::serialize::InputArchive archive;archive.load_from(argv[1],torch::kCPU);
    model->load(archive);model->to(torch::kCUDA);
    std::string fixture=argv[2];auto data=read(fixture+"/input.bin"),meta=read(fixture+"/metadata.bin");
    int64_t mapping=0;std::ifstream(fixture+"/spec.txt")>>mapping;
    data=data.index_select(0,torch::nonzero((mapping-meta.select(1,0)>=100)&(meta.select(1,1)>=80)&(meta.select(1,2)>=4)).flatten());
    bool profile=argc>4;
    std::ofstream out(argv[3]);out<<"strategy,M,stage,metric,mean_ms,p50_ms,p95_ms\n";
    std::vector<int> grid=profile?std::vector<int>{219969}:std::vector<int>{1000,5000,10000,20000,50000,100000,200000,219969};
    for(int n:grid) {
      std::vector<torch::Tensor> x{data.slice(0,0,n).slice(1,0,3).contiguous().detach().set_requires_grad(true),
        torch::nn::functional::normalize(data.slice(0,0,n).slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)).detach().set_requires_grad(true),
        data.slice(0,0,n).slice(1,7,10).exp().detach().set_requires_grad(true),torch::full({1,1},.05,data.options())};
      auto p=model->parameters(),args=std::vector<torch::Tensor>{x[0],x[1],x[2]};args.insert(args.end(),p.begin(),p.end());
      auto dy=torch::randn({n,2},data.options());Event a,mid,b;
      std::map<int,std::map<std::string,std::vector<double>>> samples;
      for(int i=0;i<(profile?110:600);++i) {
        if(profile && i==100)cudaProfilerStart();
        for(int mode:i%2?std::vector<int>{6,1}:std::vector<int>{1,6}) {
          cudaDeviceSynchronize();auto start=Clock::now();a.record();
          std::string label="strategy"+std::to_string(mode);nvtxRangePushA(label.c_str());
          {
            nvtxRangePushA("network_forward");
            auto y=selector_cuda::network(x,p,selector_cuda::Candidate::CublasStrict,static_cast<selector_cuda::BackwardStrategy>(mode));
            nvtxRangePop();mid.record();nvtxRangePushA("network_backward");
            auto g=torch::autograd::grad({y},args,{dy});nvtxRangePop();
          }
          nvtxRangePop();b.record();cudaEventSynchronize(b.value);
          if(i>=100) {
            float total=0,forward=0,backward=0;cudaEventElapsedTime(&total,a.value,b.value);
            cudaEventElapsedTime(&forward,a.value,mid.value);cudaEventElapsedTime(&backward,mid.value,b.value);
            samples[mode]["fb_event"].push_back(total);samples[mode]["forward_part_event"].push_back(forward);
            samples[mode]["backward_part_event"].push_back(backward);
            samples[mode]["fb_wall"].push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
          }
        }
      }
      if(profile)cudaProfilerStop();
      for(auto& mode:samples)for(auto& sample:mode.second) {
        auto v=sample.second;auto mean=std::accumulate(v.begin(),v.end(),0.)/v.size();std::sort(v.begin(),v.end());
        out<<mode.first<<','<<n<<','<<sample.first<<",paired,"<<mean<<','<<v[v.size()/2]<<','<<v[int(.95*(v.size()-1))]<<'\n';out.flush();
      }
      std::cout<<"paired M="<<n<<" complete\n"<<std::flush;
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
