// Separate reverse-only MLP timing. Forward context is prepared before timing.
#include "include/transform_field.h"
#include "include/transform_field_cuda.h"
#include <torch/nn/functional/linear.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <numeric>

int main(int argc,char** argv) {
  TORCH_CHECK(argc==3,"mlp_backward_benchmark CHECKPOINT OUTPUT_CSV");
  torch::set_num_threads(1);torch::manual_seed(20261013);
  at::globalContext().setAllowTF32CuBLAS(false);
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);
  torch::serialize::InputArchive archive;archive.load_from(std::string(argv[1])+"/transform_field.pt");
  tf->load(archive);tf->to(torch::kCUDA);
  auto p=tf->mlpParameters();
  std::ofstream out(argv[2]);out<<"backend,N,stage,mean_ms,p50_ms,p95_ms\n";
  cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
  for(int n:{1000,5000,8491,10000,20000,50000}) {
    auto feature=torch::randn({n,16},opt).set_requires_grad(true);
    auto linear=[&](int i,const torch::Tensor& x){return torch::nn::functional::linear(x,p[2*i],p[2*i+1]);};
    auto h=linear(0,feature);std::vector<torch::Tensor> y,dy,args{feature};
    args.insert(args.end(),p.begin(),p.end());
    for(int i=0;i<3;++i){y.push_back(linear(2+2*i,torch::relu(linear(1+2*i,torch::relu(h)))));dy.push_back(torch::randn_like(y.back()));}
    std::vector<torch::Tensor> context;
    {torch::NoGradGuard ng;context=tf_cuda::mlpForward(feature,p);}
    // Numerical equality is checked here in addition to the existing parity suite.
    auto expected=torch::autograd::grad(y,args,dy,true);
    auto actual=tf_cuda::mlpBackward(context,p,dy);
    for(size_t i=0;i<expected.size();++i)TORCH_CHECK(torch::allclose(actual[i],expected[i],2e-4,2e-6),"MLP gradient parity ",i);
    for(bool cuda:{false,true}) {
      std::vector<float> samples;
      for(int iteration=0;iteration<600;++iteration) {
        cudaEventRecord(begin,c10::cuda::getCurrentCUDAStream());
        if(cuda){torch::NoGradGuard ng;tf_cuda::mlpBackward(context,p,dy);}
        else torch::autograd::grad(y,args,dy,true);
        cudaEventRecord(end,c10::cuda::getCurrentCUDAStream());cudaEventSynchronize(end);
        if(iteration>=100){float ms;cudaEventElapsedTime(&ms,begin,end);samples.push_back(ms);}
      }
      float mean=std::accumulate(samples.begin(),samples.end(),0.0f)/samples.size();
      std::sort(samples.begin(),samples.end());
      out<<(cuda?"cuda":"legacy")<<','<<n<<",mlp_backward,"<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';out.flush();
    }
  }
  cudaEventDestroy(begin);cudaEventDestroy(end);
}
