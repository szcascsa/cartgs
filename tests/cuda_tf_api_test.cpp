#include "include/transform_field.h"
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <fstream>
#include <iostream>

namespace {
std::ofstream parity;
void compare(const std::string& name,const torch::Tensor& a,const torch::Tensor& b,double atol=2e-6,double rtol=2e-5) {
  TORCH_CHECK(a.defined()==b.defined(),name," gradient presence");
  if(!a.defined())return;
  double error=a.numel()?(a-b).abs().max().item<double>():0;bool pass=torch::allclose(a,b,rtol,atol,true);
  parity<<name<<','<<error<<','<<pass<<'\n';parity.flush();TORCH_CHECK(pass,name," max_abs=",error);
}
void check(TransformField& old,TransformField& cuda,int n,float ratio,bool logscale,bool duplicates,const std::string& label) {
  auto opt=old->aabbMin().options();auto x=torch::randn({n,3},opt),s=torch::rand({n,3},opt)+.1,q=torch::randn({n,4},opt);
  std::vector<torch::Tensor> a,b;
  for(const auto& t:{x,s,q}){a.push_back(t.clone().set_requires_grad(true));b.push_back(t.clone().set_requires_grad(true));}
  auto indices=torch::arange(n,opt.dtype(torch::kLong));
  if(duplicates && n)indices=torch::tensor({0,0,n-1,0,n-1},opt.dtype(torch::kLong));
  auto call=[&](TransformField& tf,std::vector<torch::Tensor>& in) {
    auto xx=in[0].index_select(0,indices),ss=in[1].index_select(0,indices),qq=in[2].index_select(0,indices);
    if(logscale){auto y=tf->applyLogResidual({xx,ss,qq},ratio);return std::vector<torch::Tensor>{y.xyz,y.log_scaling,y.rotation};}
    auto y=tf->applyResidual(xx,ss,qq,ratio);return std::vector<torch::Tensor>{y.xyz,y.scaling,y.rotation};
  };
  auto ya=call(old,a),yb=call(cuda,b);std::vector<torch::Tensor> upstream;
  for(size_t i=0;i<ya.size();++i){compare(label+"/output"+std::to_string(i),yb[i],ya[i]);upstream.push_back(torch::randn_like(ya[i]));}
  auto pa=old->parameters(),pb=cuda->parameters();a.insert(a.end(),pa.begin(),pa.end());b.insert(b.end(),pb.begin(),pb.end());
  auto ga=torch::autograd::grad(ya,a,upstream,false,false,true),gb=torch::autograd::grad(yb,b,upstream,false,false,true);
  for(size_t i=0;i<ga.size();++i)compare(label+"/grad"+std::to_string(i),gb[i],ga[i],i>=3 && pa[i-3].dim()==4?1e-5:2e-6,i>=3 && pa[i-3].dim()==4?5e-4:2e-4);
}
}
int main(int argc,char** argv) {
  TORCH_CHECK(argc==3,"cuda_tf_api_test OUTPUT_DIRECTORY CHECKPOINT");torch::set_num_threads(1);torch::manual_seed(20261020);at::globalContext().setAllowTF32CuBLAS(false);
  parity.open(std::string(argv[1])+"/phase_api_parity.csv");parity<<"case,max_abs,passed\n";
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  for(bool outside:{false,true}) {
    TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};cfg.outside_identity=outside;
    TransformField old(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);cfg.backend="cuda";
    TransformField cuda(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);
    for(auto* tf:{&old,&cuda}){torch::serialize::InputArchive archive;archive.load_from(std::string(argv[2])+"/transform_field.pt");(*tf)->load(archive);(*tf)->to(torch::kCUDA);}
    auto a=old->named_parameters(),b=cuda->named_parameters();TORCH_CHECK(a.size()==b.size(),"Parameter count changed");
    for(const auto& p:a){TORCH_CHECK(b.contains(p.key()) && b[p.key()].sizes()==p.value().sizes(),"Checkpoint parameter changed");compare("checkpoint/"+p.key(),b[p.key()],p.value(),0,0);}
    for(int n:{0,17})for(float ratio:{-1.f,0.f,.01f,.05f,.1f,.15f,1.f,-1.1f,1.1f})for(bool logs:{false,true})
      check(old,cuda,n,ratio,logs,false,"outside="+std::to_string(outside)+"/N="+std::to_string(n)+"/r="+std::to_string(ratio)+"/log="+std::to_string(logs));
    check(old,cuda,17,.05,true,true,"duplicate_scatter");
    auto teacher=cuda->cloneFrozen();for(auto p:teacher->parameters())TORCH_CHECK(!p.requires_grad(),"Teacher is not frozen");
    auto xyz=torch::zeros({17,3},opt),scaling=torch::ones_like(xyz),rotation=torch::zeros({17,4},opt);rotation.select(1,0).fill_(1);
    auto expected=cuda->applyResidual(xyz,scaling,rotation,.05),actual=teacher->applyResidual(xyz,scaling,rotation,.05);
    compare("frozen_teacher/xyz",actual.xyz,expected.xyz);compare("frozen_teacher/scale",actual.scaling,expected.scaling);compare("frozen_teacher/q",actual.rotation,expected.rotation);
    const auto file=std::string(argv[1])+"/cuda_api_checkpoint.pt";torch::serialize::OutputArchive output;cuda->save(output);output.save_to(file);
    torch::serialize::InputArchive input;input.load_from(file);old->load(input);
    auto saved=old->applyResidual(xyz,scaling,rotation,.05);compare("cuda_save_legacy_load/xyz",saved.xyz,expected.xyz);compare("cuda_save_legacy_load/q",saved.rotation,expected.rotation);
    {c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());check(old,cuda,17,.1,true,false,"nondefault_stream");}
  }
  std::cout<<"phase API passed"<<std::endl;
}
