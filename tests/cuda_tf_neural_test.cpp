#include "include/transform_field.h"
#include "include/transform_field_cuda.h"
#include "include/quaternion_utils.h"
#include <torch/nn/functional/linear.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
std::ofstream parity;
void compare(const std::string& name,const torch::Tensor& a,const torch::Tensor& b,
             double atol=2e-6,double rtol=2e-5) {
  double error=a.numel()?(a-b).abs().max().item<double>():0;
  bool pass=torch::allclose(a,b,rtol,atol,true);
  parity<<name<<','<<error<<','<<atol<<','<<rtol<<','<<pass<<'\n';parity.flush();
  TORCH_CHECK(pass,name," max_abs=",error);
}
struct Event {cudaEvent_t v;Event(){cudaEventCreate(&v);}~Event(){cudaEventDestroy(v);}
  void record(){cudaEventRecord(v,c10::cuda::getCurrentCUDAStream());}};
void timing(std::ofstream& out,const std::string& backend,int n,const std::string& stage,
            const std::function<void()>& action) {
  Event a,b;std::vector<float> samples;
  for(int i=0;i<600;++i) {
    a.record();action();b.record();cudaEventSynchronize(b.v);
    if(i>=100){float ms;cudaEventElapsedTime(&ms,a.v,b.v);samples.push_back(ms);}
  }
  float mean=std::accumulate(samples.begin(),samples.end(),0.0f)/samples.size();
  std::sort(samples.begin(),samples.end());
  out<<backend<<','<<n<<','<<stage<<','<<mean<<','<<samples[250]<<','<<samples[474]<<'\n';out.flush();
}
std::vector<torch::Tensor> oldMlp(const torch::Tensor& f,const std::vector<torch::Tensor>& p) {
  auto linear=[&](int i,const torch::Tensor& x){return torch::nn::functional::linear(x,p[2*i],p[2*i+1]);};
  auto h=torch::relu(linear(0,f));std::vector<torch::Tensor> context{f,h};
  for(int i=0;i<3;++i)context.push_back(torch::relu(linear(1+2*i,h)));
  for(int i=0;i<3;++i)context.push_back(linear(2+2*i,context[2+i]));
  return context;
}
std::vector<torch::Tensor> heads(const std::vector<torch::Tensor>& c) {return {c[5],c[6],c[7]};}
std::vector<torch::Tensor> oldOutput(torch::Tensor x,torch::Tensor s,torch::Tensor q,
                                  const std::vector<torch::Tensor>& r,bool log=false) {
  auto id=torch::zeros_like(r[2]);id.select(1,0).fill_(1);
  return {x+r[0],log?s+r[1]:s*r[1].exp(),quaternion_utils::normalize(
      quaternion_utils::multiply(quaternion_utils::normalize(id+r[2]),q))};
}
void mlpParity(const torch::Tensor& f,const std::vector<torch::Tensor>& p,const std::string& label) {
  auto old=oldMlp(f,p);
  for(auto backend:{tf_cuda::MlpBackend::Cublas,tf_cuda::MlpBackend::Warp32}) {
    if(backend==tf_cuda::MlpBackend::Warp32 && (p[0].size(0)!=32 || f.size(1)!=16))continue;
    auto current=tf_cuda::mlpForward(f,p,backend);
    for(int i=1;i<8;++i)compare(label+(backend==tf_cuda::MlpBackend::Cublas?"/cublas/":"/warp/")+std::to_string(i),current[i],old[i]);
  }
}
}  // namespace
int main(int argc,char** argv) {
  TORCH_CHECK(argc==4,"cuda_tf_neural_test OUTPUT_DIRECTORY CHECKPOINT INPUT_BIN");
  torch::set_num_threads(1);torch::manual_seed(20261014);
  at::globalContext().setAllowTF32CuBLAS(false);
  parity.open(std::string(argv[1])+"/phase_b_parity.csv");parity<<"case,max_abs,atol,rtol,passed\n";
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);tf->to(torch::kCUDA);
  {torch::NoGradGuard ng;for(auto p:tf->mlpParameters())p.uniform_(-0.5,0.5);}
  mlpParity(torch::randn({2037,16},opt),tf->mlpParameters(),"random");
  mlpParity(torch::zeros({37,16},opt),tf->mlpParameters(),"zeros");
  mlpParity(torch::empty({0,16},opt),tf->mlpParameters(),"empty");
  TransformFieldConfig defaults;
  TransformField normal(torch::full({3},-1,opt),torch::full({3},1,opt),defaults);normal->to(torch::kCUDA);
  mlpParity(torch::randn({137,normal->mlpParameters()[0].size(1)},opt),normal->mlpParameters(),"default_h64");
  for(bool log:{false,true}) {
    auto x=torch::randn({1007,3},opt),s=torch::rand({1007,3},opt),q=torch::randn({1007,4},opt);
    auto r=std::vector<torch::Tensor>{torch::randn_like(x),torch::randn_like(s),torch::randn_like(q)};
    {torch::NoGradGuard ng;
      q[0].zero_();q[1].fill_(1e-14f);q[2].fill_(5e-13f);q[3].fill_(1e-12f);
      q[4].zero_();q[4][0].fill_(1);q[5].copy_(-q[4]);
      r[2][6].zero_();r[2][6][0].fill_(-1); // exact zero residual quaternion
      r[2][7].zero_();r[2][7][0].fill_(-1);r[2][7][1].fill_(1e-13f);
    }
    auto a=oldOutput(x,s,q,r,log),b=tf_cuda::outputForward(x,s,q,r,log);
    for(int i=0;i<3;++i)compare("output/log="+std::to_string(log)+"/"+std::to_string(i),b[i],a[i]);
  }
  torch::serialize::InputArchive archive;archive.load_from(std::string(argv[2])+"/transform_field.pt");tf->load(archive);tf->to(torch::kCUDA);
  std::ifstream input(argv[3],std::ios::binary);int64_t count;input.read(reinterpret_cast<char*>(&count),8);
  std::vector<float> raw(count*10);input.read(reinterpret_cast<char*>(raw.data()),raw.size()*4);TORCH_CHECK(input,"Invalid saved input");
  auto data=torch::from_blob(raw.data(),{count,10},torch::kFloat32).clone().to(torch::kCUDA);
  std::ofstream bench(std::string(argv[1])+"/phase_b_benchmark.csv");bench<<"backend,N,stage,mean_ms,p50_ms,p95_ms\n";
  torch::NoGradGuard ng;
  for(int n:{1000,5000,8491,10000,20000,50000}) {
    auto index=torch::arange(n,data.options().dtype(torch::kLong)).remainder(count);auto d=data.index_select(0,index);
    auto x=d.slice(1,0,3),s=d.slice(1,3,6),q=d.slice(1,6,10),r=torch::full({n,1},0.05,opt);
    auto feature=tf_cuda::queryForward([&](){std::vector<torch::Tensor> a{x,r,tf->aabbMin(),tf->aabbMax()};auto p=tf->gridParameters();a.insert(a.end(),p.begin(),p.end());return a;}());
    mlpParity(feature,tf->mlpParameters(),"checkpoint_N="+std::to_string(n));
    for(int backend=0;backend<3;++backend) {
      auto name=backend==0?"legacy":(backend==1?"cublas":"warp32");
      auto call=[&](){return backend==0?oldMlp(feature,tf->mlpParameters()):tf_cuda::mlpForward(feature,tf->mlpParameters(),backend==1?tf_cuda::MlpBackend::Cublas:tf_cuda::MlpBackend::Warp32);};
      auto context=call();auto residual=heads(context),expected=oldOutput(x,s,q,heads(oldMlp(feature,tf->mlpParameters())));
      auto result=backend==0?oldOutput(x,s,q,residual):tf_cuda::outputForward(x,s,q,residual);
      for(int i=0;i<3;++i)compare("checkpoint/"+std::string(name)+"/N="+std::to_string(n)+"/out"+std::to_string(i),result[i],expected[i]);
      timing(bench,name,n,"mlp_forward",[&](){call();});
      timing(bench,name,n,"output_forward",[&](){if(backend==0)oldOutput(x,s,q,residual);else tf_cuda::outputForward(x,s,q,residual);});
      timing(bench,name,n,"total_forward",[&](){
        if(backend==0)tf->applyResidual(x,s,q,0.05f);
        else {
          auto f=tf_cuda::query(x,r,tf->aabbMin(),tf->aabbMax(),tf->gridParameters());
          auto c=tf_cuda::mlpForward(f,tf->mlpParameters(),backend==1?tf_cuda::MlpBackend::Cublas:tf_cuda::MlpBackend::Warp32);
          tf_cuda::outputForward(x,s,q,heads(c));
        }
      });
    }
    std::cout<<"phase B N="<<n<<" passed"<<std::endl;
  }
  {c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());mlpParity(torch::randn({67,16},opt),tf->mlpParameters(),"nondefault_stream");}
}
