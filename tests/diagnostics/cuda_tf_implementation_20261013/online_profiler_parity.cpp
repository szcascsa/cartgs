#include "online_tf_profiler.h"
#include <fstream>

int main(int argc,char** argv) {
  TORCH_CHECK(argc==3,"online_profiler_parity CHECKPOINT OUTPUT_CSV");
  torch::set_num_threads(1);torch::manual_seed(20261014);
  at::globalContext().setAllowTF32CuBLAS(false);
  auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  std::ofstream out(argv[2]);out<<"backend,case,index,max_abs,passed\n";
  for(const std::string backend:{"legacy","cuda"}) {
    TransformFieldConfig cfg;cfg.backend=backend;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
    TransformField tf(torch::full({3},-1,opt),torch::full({3},1,opt),cfg);
    torch::serialize::InputArchive archive;archive.load_from(std::string(argv[1])+"/transform_field.pt");
    tf->load(archive);tf->to(torch::kCUDA);
    auto x=torch::rand({1071,3},opt).set_requires_grad(true),s=torch::rand({1071,3},opt).set_requires_grad(true),q=torch::randn({1071,4},opt).set_requires_grad(true);
    auto p=tf->parameters();std::vector<torch::Tensor> targets{x,s,q};targets.insert(targets.end(),p.begin(),p.end());
    auto a=tf->applyResidual(x,s,q,.05f),b=tf_diagnostic::applyResidual(*tf,x,s,q,.05f);
    std::vector<torch::Tensor> ya{a.xyz,a.scaling,a.rotation},yb{b.xyz,b.scaling,b.rotation},dy;
    for(auto y:ya)dy.push_back(torch::randn_like(y));
    auto ga=torch::autograd::grad(ya,targets,dy),gb=torch::autograd::grad(yb,targets,dy);
    for(size_t i=0;i<ga.size();++i) {
      double error=(ga[i]-gb[i]).abs().max().item<double>();
      bool pass=torch::allclose(ga[i],gb[i],5e-4,1e-5);
      out<<backend<<",graph_bracket,"<<i<<','<<error<<','<<pass<<'\n';TORCH_CHECK(pass,"Diagnostic graph changed gradients ",backend," ",i);
    }
    auto ra=tf->regularizationLoss(),rb=tf_diagnostic::regularization(*tf);
    auto rga=torch::autograd::grad({ra},p,{},false,false,true),rgb=torch::autograd::grad({rb},p,{},false,false,true);
    for(size_t i=0;i<p.size();++i){TORCH_CHECK(rga[i].defined()==rgb[i].defined(),"Regularization edge changed");
      if(rga[i].defined()){bool pass=torch::equal(rga[i],rgb[i]);out<<backend<<",reg_bracket,"<<i<<",0,"<<pass<<'\n';TORCH_CHECK(pass,"Regularization bracket changed gradient");}}
    std::vector<torch::Tensor> left{torch::randn({19},opt)},right{left[0].clone()};
    for(auto v:p){left.push_back(v.detach().clone());right.push_back(v.detach().clone());}
    for(auto& v:left)v.set_requires_grad(true);for(auto& v:right)v.set_requires_grad(true);
    torch::optim::Adam old(left,torch::optim::AdamOptions(.001).eps(1e-15));
    torch::optim::Adam split(right,torch::optim::AdamOptions(.001).eps(1e-15));
    // Move TF tensors into a distinct logical group, leaving Gaussian stand-in.
    old.param_groups()[0].params()={left[0]};old.add_param_group(std::vector<torch::Tensor>{left.begin()+1,left.end()});
    split.param_groups()[0].params()={right[0]};split.add_param_group(std::vector<torch::Tensor>{right.begin()+1,right.end()});
    std::vector<std::size_t> cuda_groups=backend=="cuda"?std::vector<std::size_t>{1}:std::vector<std::size_t>{};
    for(int step=0;step<10;++step){
      for(size_t i=0;i<left.size();++i){auto g=torch::randn_like(left[i]);left[i].mutable_grad()=g;right[i].mutable_grad()=g.clone();}
      tf_cuda::tfCudaAdamMixedStep(old,cuda_groups);tf_diagnostic::mixedStep(split,cuda_groups,{1});
      for(size_t i=0;i<left.size();++i){bool pass=torch::equal(left[i],right[i]);
        const auto& astate=static_cast<const torch::optim::AdamParamState&>(*old.state().at(left[i].unsafeGetTensorImpl()));
        const auto& bstate=static_cast<const torch::optim::AdamParamState&>(*split.state().at(right[i].unsafeGetTensorImpl()));
        pass=pass && astate.step()==bstate.step() && torch::equal(astate.exp_avg(),bstate.exp_avg()) && torch::equal(astate.exp_avg_sq(),bstate.exp_avg_sq());
        out<<backend<<",split_optimizer_step"<<step<<','<<i<<",0,"<<pass<<'\n';TORCH_CHECK(pass,"Diagnostic optimizer changed state");}
    }
  }
  tf_diagnostic::report();
}
