#include "improvements/selector/spatial_compensation.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <filesystem>
#include <iostream>
using Tensor=torch::Tensor;using List=std::vector<Tensor>;
using Spatial=improvements::selector::SpatialCompensation;
void compare(std::ofstream& log,const std::string& name,const Tensor& a,const Tensor& b,bool grad=false){
  TORCH_CHECK(a.sizes()==b.sizes(),name);if(!a.numel())return;
  auto d=(a-b).abs();auto bad=~torch::isfinite(a)|~torch::isfinite(b)|(d>((grad?1e-5:1e-6)+(grad?1e-4:1e-5)*a.abs()));
  log<<name<<','<<d.max().item<double>()<<','<<d.square().mean().sqrt().item<double>()<<','<<bad.sum().item<int64_t>()<<'\n';
  TORCH_CHECK(!bad.any().item<bool>(),name," parity failed");
}
int main(int argc,char** argv){try{
  torch::set_num_threads(2);torch::manual_seed(5100);at::globalContext().setAllowTF32CuBLAS(false);
  std::filesystem::path dir=argc>1?argv[1]:"spatial_test";std::filesystem::create_directories(dir);
  std::ofstream log(dir/"parity.csv");log<<"tensor,max_abs,rms,bad\n";
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);int64_t N=1025;
  auto xyz=torch::randn({N,3},o),scale=torch::rand({N,3},o)*.2+.001;
  auto rotation=torch::nn::functional::normalize(torch::randn({N,4},o));
  List values{xyz,scale,rotation,torch::randn({N,1,3},o),torch::randn({N,1},o),xyz.floor()*.1,torch::full({N,1},.1,o)};
  Spatial model;model->to(torch::kCUDA);model->initialize(values[0],values[1],values[2],values[3],values[4],values[5],values[6]);
  {torch::NoGradGuard guard;for(const auto& name:{"xyz","scale","rotation","other"})model->heads->at<torch::nn::SequentialImpl>(name).at<torch::nn::LinearImpl>(2).weight.normal_(0,.01);}
  for(auto K:{0,1,17,68,1025})for(double ratio:{.01,.025,.05,.075,.1,.125,.15}){
    List inputs[2];Tensor out[2],input[2];List grads[2],input_grads[2];auto ids=torch::arange(K,xyz.options().dtype(torch::kLong));
    auto vjp=torch::randn({K,14},o)*.001,input_vjp=torch::randn({K,55},o)*.001;
    for(int j=0;j<2;++j){for(auto value:values)inputs[j].push_back(value.index_select(0,ids).detach().clone().requires_grad_(true));
      model->setFused(j==1);model->setLiveFused(j==1);auto& a=inputs[j];input[j]=model->input55(a[0],a[1],a[2],a[3],a[4],a[5],a[6]);
      out[j]=model->forward(a[0],a[1],a[2],a[3],a[4],a[5],a[6],ratio);
      List params=model->parameters();params.insert(params.end(),a.begin(),a.end());
      grads[j]=torch::autograd::grad({out[j]},params,{vjp},true);
      input_grads[j]=torch::autograd::grad({input[j]},a,{input_vjp});}
    auto name=std::to_string(K)+"/"+std::to_string(ratio);compare(log,name+"/input55",input[0],input[1]);compare(log,name+"/output",out[0],out[1]);
    for(size_t k=0;k<grads[0].size();++k)compare(log,name+"/gradient"+std::to_string(k),grads[0][k],grads[1][k],true);
    for(size_t k=0;k<input_grads[0].size();++k)compare(log,name+"/direct_input_gradient"+std::to_string(k),input_grads[0][k],input_grads[1][k],true);
  }
  model->setFused(true);torch::save(model,(dir/"model.pt").string());Spatial loaded;loaded->to(torch::kCUDA);torch::load(loaded,(dir/"model.pt").string());
  auto infer=[&](Spatial& m){return m->forward(values[0],values[1],values[2],values[3],values[4],values[5],values[6],.075);};
  compare(log,"checkpoint",infer(model),infer(loaded));TORCH_CHECK(loaded->ready(),"Normalization ready flag missing");
  // Same explicit live-input VJP and Adam clock, without Renderer nondeterminism.
  loaded->setFused(false);torch::optim::Adam oa(model->parameters(),torch::optim::AdamOptions(.001));
  torch::optim::Adam ob(loaded->parameters(),torch::optim::AdamOptions(.001));
  for(int step=0;step<12;++step){double ratio=std::vector<double>{.01,.05,.1,.15}[step%4];
    auto vjp=torch::randn({N,14},o)*.001;oa.zero_grad(true);ob.zero_grad(true);
    model->forward(values[0],values[1],values[2],values[3],values[4],values[5],values[6],ratio).backward(vjp);
    loaded->forward(values[0],values[1],values[2],values[3],values[4],values[5],values[6],ratio).backward(vjp);oa.step();ob.step();
    auto aa=model->parameters(),bb=loaded->parameters();
    for(size_t j=0;j<aa.size();++j){compare(log,"step"+std::to_string(step)+"/parameter"+std::to_string(j),aa[j],bb[j]);
      auto& sa=static_cast<torch::optim::AdamParamState&>(*oa.state().at(aa[j].unsafeGetTensorImpl()));
      auto& sb=static_cast<torch::optim::AdamParamState&>(*ob.state().at(bb[j].unsafeGetTensorImpl()));
      TORCH_CHECK(sa.step()==sb.step(),"Adam clock mismatch");compare(log,"step/exp_avg",sa.exp_avg(),sb.exp_avg(),true);compare(log,"step/exp_avg_sq",sa.exp_avg_sq(),sb.exp_avg_sq(),true);
    }
  }
  // Restore the exported model so Python comparison uses the matching output.
  torch::save(model,(dir/"model.pt").string());
  torch::serialize::OutputArchive fixture;for(size_t i=0;i<values.size();++i)fixture.write("input_"+std::to_string(i),values[i],true);fixture.write("expected",infer(model),true);fixture.save_to((dir/"fixture.pt").string());
  // Geometry/row changes must read current attributes; normalization remains fixed.
  auto before=infer(model).detach();values[0]=values[0]+.01;TORCH_CHECK(!torch::equal(before,infer(model)),"Stale geometry");
  values[0]=values[0].flip(0);auto refmean=model->mean.clone();model->initialize(values[0],values[1],values[2],values[3],values[4],values[5],values[6]);TORCH_CHECK(torch::equal(refmean,model->mean),"Statistics changed after initialization");
  std::cout<<"SPATIAL_PARITY_PASS parameters=";int64_t count=0;for(const auto& p:model->parameters())count+=p.numel();std::cout<<count<<'\n';return 0;
}catch(const c10::Error& e){std::cerr<<e.what()<<'\n';return 1;}}
