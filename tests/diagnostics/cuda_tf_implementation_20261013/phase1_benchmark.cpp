// Phase 1 only: fused query with the unchanged ATen MLP/output/regularizer/Adam.
#include "include/transform_field.h"
#include "include/transform_field_cuda.h"
#include "include/quaternion_utils.h"
#include <torch/nn/functional/linear.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

struct Event {cudaEvent_t v;Event(){cudaEventCreate(&v);}~Event(){cudaEventDestroy(v);}
  void record(){cudaEventRecord(v,c10::cuda::getCurrentCUDAStream());}};
int main(int argc,char** argv) {
  TORCH_CHECK(argc==4,"phase1_benchmark CHECKPOINT INPUT_BIN OUTPUT_CSV");
  torch::set_num_threads(1);
  std::ifstream f(argv[2],std::ios::binary);int64_t count;f.read(reinterpret_cast<char*>(&count),8);
  std::vector<float> raw(count*10);f.read(reinterpret_cast<char*>(raw.data()),raw.size()*4);
  TORCH_CHECK(f,"Invalid input");
  auto data=torch::from_blob(raw.data(),{count,10},torch::kFloat32).clone().to(torch::kCUDA);
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  std::ofstream out(argv[3]);out<<"backend,N,stage,mean_ms,p50_ms,p95_ms\n";
  for(int n:{1000,5000,8491,10000,20000,50000})for(bool cuda:{false,true}) {
    auto cpu=torch::TensorOptions().dtype(torch::kFloat32);
    TransformField field(torch::full({3},-1,cpu),torch::full({3},1,cpu),cfg);
    torch::serialize::InputArchive a;a.load_from(std::string(argv[1])+"/transform_field.pt");field->load(a);field->to(torch::kCUDA);
    auto idx=torch::arange(n,data.options().dtype(torch::kLong)).remainder(count);
    auto d=data.index_select(0,idx);auto x=d.slice(1,0,3).contiguous().detach().set_requires_grad(true);
    auto s=d.slice(1,3,6).contiguous().detach().set_requires_grad(true);
    auto q=d.slice(1,6,10).contiguous().detach().set_requires_grad(true);
    auto ratio=torch::full({n,1},0.05,data.options());
    auto parameters=field->mlpParameters();
    torch::optim::Adam adam(parameters,torch::optim::AdamOptions(0.00016).eps(1e-15));
    adam.add_param_group(field->gridParameters());adam.param_groups()[1].options().set_lr(0.0016);
    auto linear=[&](int i,const torch::Tensor& z){return torch::nn::functional::linear(z,parameters[2*i],parameters[2*i+1]);};
    Event start,end;std::vector<float> samples;
    for(int i=0;i<600;++i) {
      start.record();adam.zero_grad(true);x.mutable_grad()=torch::Tensor();s.mutable_grad()=torch::Tensor();q.mutable_grad()=torch::Tensor();
      ElasticGaussianAttributes y;
      if(cuda) {
        auto feature=tf_cuda::query(x,ratio,field->aabbMin(),field->aabbMax(),field->gridParameters());
        auto h=linear(0,feature);
        auto dx=linear(2,torch::relu(linear(1,torch::relu(h))));
        auto ds=linear(4,torch::relu(linear(3,torch::relu(h))));
        auto dq=linear(6,torch::relu(linear(5,torch::relu(h))));
        auto identity=torch::zeros_like(dq);identity.select(1,0).fill_(1);
        y={x+dx,s*ds.exp(),quaternion_utils::normalize(quaternion_utils::multiply(quaternion_utils::normalize(identity+dq),q))};
      } else y=field->applyResidual(x,s,q,0.05f);
      auto loss=(y.xyz-x.detach()).mean()+(y.scaling/s.detach()).mean()+y.rotation.mean()+field->regularizationLoss();
      loss.backward();adam.step();end.record();cudaEventSynchronize(end.v);
      if(i>=100){float ms;cudaEventElapsedTime(&ms,start.v,end.v);samples.push_back(ms);}
    }
    auto mean=std::accumulate(samples.begin(),samples.end(),0.0f)/samples.size();std::sort(samples.begin(),samples.end());
    out<<(cuda?"cuda_query_old_mlp":"legacy")<<','<<n<<",total_regularized,"<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';out.flush();
    std::cout<<n<<" cuda_query="<<cuda<<" ms="<<mean<<std::endl;
  }
}
