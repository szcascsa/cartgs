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
             double atol=2e-6,double rtol=2e-4) {
  double error=a.numel()?(a-b).abs().max().item<double>():0;
  bool pass=torch::allclose(a,b,rtol,atol,true);
  parity<<name<<','<<error<<','<<atol<<','<<rtol<<','<<pass<<'\n';parity.flush();
  if(!pass) {
    auto bad=~torch::isclose(a,b,rtol,atol,true);
    auto index=torch::nonzero(bad).slice(0,0,20).cpu();
    std::cerr<<name<<" mismatch indices "<<index<<"\nCUDA first rows "<<a.slice(0,0,std::min<int64_t>(10,a.size(0))).cpu()
             <<"\nlegacy first rows "<<b.slice(0,0,std::min<int64_t>(10,b.size(0))).cpu()<<std::endl;
  }
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
  auto h=linear(0,f);std::vector<torch::Tensor> out;
  for(int i=0;i<3;++i)out.push_back(linear(2+2*i,torch::relu(linear(1+2*i,torch::relu(h)))));
  return out;
}
std::vector<torch::Tensor> oldOutput(const std::vector<torch::Tensor>& in,bool log) {
  auto id=torch::zeros_like(in[5]);id.select(1,0).fill_(1);
  return {in[0]+in[3],log?in[1]+in[4]:in[1]*in[4].exp(),quaternion_utils::normalize(
      quaternion_utils::multiply(quaternion_utils::normalize(id+in[5]),in[2]))};
}
std::vector<torch::Tensor> cudaFull(TransformField& tf,torch::Tensor x,torch::Tensor s,torch::Tensor q,float ratio,tf_cuda::MlpBackend backend) {
  auto r=torch::full({x.size(0),1},ratio,x.options().requires_grad(false));
  auto f=tf_cuda::query(x,r,tf->aabbMin(),tf->aabbMax(),tf->gridParameters());
  auto residual=tf_cuda::neural(f,tf->mlpParameters(),backend);
  return tf_cuda::output(x,s,q,residual);
}
void outputParity(const std::vector<torch::Tensor>& in,bool log,const std::string& label) {
  auto old=oldOutput(in,log),current=tf_cuda::output(in[0],in[1],in[2],{in[3],in[4],in[5]},log);
  std::vector<torch::Tensor> dy;for(auto t:old)dy.push_back(torch::randn_like(t));
  auto a=torch::autograd::grad(old,in,dy),b=torch::autograd::grad(current,in,dy);
  for(int i=0;i<6;++i)compare(label+"/gradient_"+std::to_string(i),b[i],a[i]);
}
void mlpParity(TransformField& tf,torch::Tensor feature,const std::string& label) {
  auto parameters=tf->mlpParameters(),old=oldMlp(feature,parameters);
  std::vector<torch::Tensor> dy;for(auto t:old)dy.push_back(torch::randn_like(t));
  std::vector<torch::Tensor> args{feature};args.insert(args.end(),parameters.begin(),parameters.end());
  auto a=torch::autograd::grad(old,args,dy,true),floor=torch::autograd::grad(old,args,dy);
  for(int i=0;i<15;++i)compare(label+"/legacy_repeat_"+std::to_string(i),floor[i],a[i]);
  for(auto backend:{tf_cuda::MlpBackend::Cublas,tf_cuda::MlpBackend::Warp32}) {
    if(backend==tf_cuda::MlpBackend::Warp32 && (feature.size(1)!=16 || parameters[0].size(0)!=32))continue;
    auto y=tf_cuda::neural(feature,parameters,backend),b=torch::autograd::grad(y,args,dy);
    for(int i=0;i<15;++i)compare(label+(backend==tf_cuda::MlpBackend::Cublas?"/cublas/":"/warp/")+std::to_string(i),b[i],a[i]);
  }
  auto partial_old=oldMlp(feature,parameters),partial_cuda=tf_cuda::neural(feature,parameters);
  auto pa=torch::autograd::grad({partial_old[0]},args,{dy[0]},false,false,true);
  auto pb=torch::autograd::grad({partial_cuda[0]},args,{dy[0]},false,false,true);
  for(int i=0;i<15;++i) {
    TORCH_CHECK(pa[i].defined()==pb[i].defined(),"Unused head gradient materialized: ",label," ",i);
    if(pa[i].defined())compare(label+"/partial_head/"+std::to_string(i),pb[i],pa[i]);
  }
}
void fullParity(TransformField& tf,torch::Tensor x,torch::Tensor s,torch::Tensor q,const std::string& label) {
  auto old=tf->applyResidual(x,s,q,0.05f);
  std::vector<torch::Tensor> dy{torch::randn_like(x),torch::randn_like(s),torch::randn_like(q)},args{x,s,q};
  auto planes=tf->gridParameters(),parameters=tf->mlpParameters();args.insert(args.end(),planes.begin(),planes.end());args.insert(args.end(),parameters.begin(),parameters.end());
  auto a=torch::autograd::grad({old.xyz,old.scaling,old.rotation},args,dy,true),floor=torch::autograd::grad({old.xyz,old.scaling,old.rotation},args,dy);
  for(size_t i=0;i<a.size();++i)compare(label+"/legacy_repeat_"+std::to_string(i),floor[i],a[i],i>=3 && i<3+planes.size()?1e-5:2e-6,i>=3 && i<3+planes.size()?5e-4:2e-4);
  for(auto backend:{tf_cuda::MlpBackend::Cublas,tf_cuda::MlpBackend::Warp32}) {
    auto y=cudaFull(tf,x,s,q,0.05f,backend),b=torch::autograd::grad(y,args,dy);
    for(size_t i=0;i<a.size();++i)compare(label+(backend==tf_cuda::MlpBackend::Cublas?"/cublas/":"/warp/")+std::to_string(i),b[i],a[i],i>=3 && i<3+planes.size()?1e-5:2e-6,i>=3 && i<3+planes.size()?5e-4:2e-4);
  }
  auto y=cudaFull(tf,x,s,q,0.05f,tf_cuda::MlpBackend::Cublas);
  x.mutable_grad()=torch::ones_like(x); // pre-existing Full loss contribution
  y[0].sum().backward();auto direct=torch::autograd::grad({tf->applyResidual(x,s,q,0.05f).xyz},{x},{torch::ones_like(x)})[0];
  compare(label+"/full_lod_accumulation",x.grad(),1+direct);
  for(auto& p:args)p.mutable_grad()=torch::Tensor();
}
}  // namespace
int main(int argc,char** argv) {
  TORCH_CHECK(argc==4,"cuda_tf_backward_test OUTPUT_DIRECTORY CHECKPOINT INPUT_BIN");
  torch::set_num_threads(1);torch::manual_seed(20261015);at::globalContext().setAllowTF32CuBLAS(false);
  parity.open(std::string(argv[1])+"/phase_c_parity.csv");parity<<"case,max_abs,atol,rtol,passed\n";
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  for(bool log:{false,true}) {
    std::vector<torch::Tensor> in;for(int i=0;i<6;++i)in.push_back(torch::randn({1067,(i==2||i==5)?4:3},opt));
    in[2][0].zero_();in[2][1].fill_(1e-14f);in[2][2].fill_(5e-13f);in[2][3].fill_(1e-12f);
    in[2][4].zero_();in[2][4][0].fill_(1);in[2][5].copy_(-in[2][4]);
    in[5][6].zero_();in[5][6][0].fill_(-1);in[5][7].zero_();in[5][7][0].fill_(-1);in[5][7][1].fill_(1e-13f);
    for(auto& t:in)t.set_requires_grad(true);
    outputParity(in,log,"output/log="+std::to_string(log));
  }
  TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
  TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);tf->to(torch::kCUDA);
  {torch::NoGradGuard ng;for(auto p:tf->mlpParameters())p.uniform_(-0.5,0.5);}
  mlpParity(tf,torch::randn({2037,16},opt).set_requires_grad(true),"random");
  {torch::NoGradGuard ng;for(auto p:tf->mlpParameters())p.zero_();}
  mlpParity(tf,torch::zeros({37,16},opt).set_requires_grad(true),"relu_zero");
  mlpParity(tf,torch::empty({0,16},opt).set_requires_grad(true),"empty");
  TransformFieldConfig defaults;TransformField normal(torch::full({3},-1,opt),torch::full({3},1,opt),defaults);normal->to(torch::kCUDA);
  mlpParity(normal,torch::randn({137,normal->mlpParameters()[0].size(1)},opt).set_requires_grad(true),"default_h64");
  torch::serialize::InputArchive archive;archive.load_from(std::string(argv[2])+"/transform_field.pt");tf->load(archive);tf->to(torch::kCUDA);
  std::ifstream input(argv[3],std::ios::binary);int64_t count;input.read(reinterpret_cast<char*>(&count),8);
  std::vector<float> raw(count*10);input.read(reinterpret_cast<char*>(raw.data()),raw.size()*4);TORCH_CHECK(input,"Invalid saved input");
  auto data=torch::from_blob(raw.data(),{count,10},torch::kFloat32).clone().to(torch::kCUDA);
  auto x=data.slice(1,0,3).contiguous().detach().set_requires_grad(true),s=data.slice(1,3,6).contiguous().detach().set_requires_grad(true),q=data.slice(1,6,10).contiguous().detach().set_requires_grad(true);
  fullParity(tf,x,s,q,"real_checkpoint");
  {c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
    fullParity(tf,torch::rand({43,3},opt).set_requires_grad(true),torch::rand({43,3},opt).set_requires_grad(true),torch::randn({43,4},opt).set_requires_grad(true),"nondefault_stream");}
  {auto p=tf->mlpParameters();auto saved=p[0].detach().clone();auto y=tf_cuda::neural(torch::randn({7,16},opt),p);
    {torch::NoGradGuard ng;p[0].add_(0.1);}
    bool caught=false;try{y[0].sum().backward();}catch(const c10::Error&){caught=true;}
    TORCH_CHECK(caught,"Saved parameter revision check missing");
    {torch::NoGradGuard ng;p[0].copy_(saved);}
  }
  std::ofstream bench(std::string(argv[1])+"/phase_c_benchmark.csv");bench<<"backend,N,stage,mean_ms,p50_ms,p95_ms\n";
  for(int n:{1000,5000,8491,10000,20000,50000}) {
    auto idx=torch::arange(n,data.options().dtype(torch::kLong)).remainder(count);auto d=data.index_select(0,idx);
    auto bx=d.slice(1,0,3).contiguous().detach().set_requires_grad(true),bs=d.slice(1,3,6).contiguous().detach().set_requires_grad(true),bq=d.slice(1,6,10).contiguous().detach().set_requires_grad(true);
    for(int backend=0;backend<3;++backend) {
      const std::string name=backend==0?"legacy":(backend==1?"cublas":"warp32");
      std::vector<torch::Tensor> y;
      if(!backend){auto a=tf->applyResidual(bx,bs,bq,0.05f);y={a.xyz,a.scaling,a.rotation};}
      else y=cudaFull(tf,bx,bs,bq,0.05f,backend==1?tf_cuda::MlpBackend::Cublas:tf_cuda::MlpBackend::Warp32);
      std::vector<torch::Tensor> args{bx,bs,bq},dy;auto p=tf->parameters();args.insert(args.end(),p.begin(),p.end());for(auto t:y)dy.push_back(torch::randn_like(t));
      timing(bench,name,n,"total_backward",[&](){torch::autograd::grad(y,args,dy,true);});
    }
    std::cout<<"phase C N="<<n<<" passed"<<std::endl;
  }
}
