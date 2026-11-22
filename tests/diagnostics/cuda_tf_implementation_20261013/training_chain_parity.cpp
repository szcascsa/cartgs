// Fixed real inputs isolate backend errors from online SLAM/Mapping scheduling.
#include "include/transform_field.h"
#include "include/transform_field_cuda_optimizer.h"
#include "include/transform_field_cuda.h"
#include <ATen/Context.h>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
  TORCH_CHECK(argc==4,"training_chain_parity CHECKPOINT INPUT_BIN OUTPUT_CSV");
  torch::set_num_threads(1);torch::manual_seed(20261014);
  at::globalContext().setAllowTF32CuBLAS(false);
  std::ifstream binary(argv[2],std::ios::binary);int64_t n=0;
  binary.read(reinterpret_cast<char*>(&n),8);TORCH_CHECK(n>0 && n<1000000,"input size");
  std::vector<float> raw(n*10);binary.read(reinterpret_cast<char*>(raw.data()),raw.size()*4);TORCH_CHECK(binary,"input data");
  auto d=torch::from_blob(raw.data(),{n,10},torch::kFloat32).clone().to(torch::kCUDA);
  auto x=d.slice(1,0,3).contiguous(),s=d.slice(1,3,6).contiguous(),q=d.slice(1,6,10).contiguous();
  std::ofstream out(argv[3]);out<<"mode,step,case,max_abs,within_starting_tolerance\n";
  int failed=0;
  auto compare=[&](const std::string& mode,int step,const std::string& name,const torch::Tensor& a,const torch::Tensor& b,double atol,double rtol) {
    TORCH_CHECK(a.defined()==b.defined(),name," edge presence");if(!a.defined())return;
    TORCH_CHECK(torch::isfinite(a).all().item<bool>() && torch::isfinite(b).all().item<bool>(),name," nonfinite");
    bool pass=torch::allclose(a,b,rtol,atol);
    out<<mode<<','<<step<<','<<name<<','<<(a-b).abs().max().item<double>()<<','<<pass<<'\n';
    if(!pass)++failed;
  };
  for(const std::string mode:{"teacher_forced","free_rollout","native_teacher_forced","native_free_rollout","cublas_teacher_forced","cublas_free_rollout"}) {
    const bool native=mode.find("native_")==0,cublas=mode.find("cublas_")==0;
    const bool teacher=mode.find("teacher_forced")!=std::string::npos;
    TransformFieldConfig cfg;cfg.hidden_dim=32;cfg.hexplane.feature_dim=8;cfg.hexplane.multires={1,2};
    TransformField old(torch::full({3},-1,x.options()),torch::full({3},1,x.options()),cfg);
    cfg.backend=native?"legacy":"cuda";TransformField cuda(torch::full({3},-1,x.options()),torch::full({3},1,x.options()),cfg);
    for(auto* tf:{&old,&cuda}){torch::serialize::InputArchive a;a.load_from(std::string(argv[1])+"/transform_field.pt");(*tf)->load(a);(*tf)->to(torch::kCUDA);}
    auto pa=old->parameters(),pb=cuda->parameters();
    torch::optim::Adam oa(old->mlpParameters(),torch::optim::AdamOptions(.00016).eps(1e-15));oa.add_param_group(old->gridParameters());oa.param_groups()[1].options().set_lr(.0016);
    torch::optim::Adam ob(cuda->mlpParameters(),torch::optim::AdamOptions(.00016).eps(1e-15));ob.add_param_group(cuda->gridParameters());ob.param_groups()[1].options().set_lr(.0016);
    for(int step=1;step<=100;++step) {
      if(teacher) {
        torch::NoGradGuard guard;
        for(size_t i=0;i<pa.size();++i) {
          pb[i].copy_(pa[i]);
          auto state=oa.state().find(pa[i].unsafeGetTensorImpl());
          if(state!=oa.state().end()) {
            const auto& a=static_cast<const torch::optim::AdamParamState&>(*state->second);
            auto b=std::make_unique<torch::optim::AdamParamState>();b->step(a.step());b->exp_avg(a.exp_avg().clone());b->exp_avg_sq(a.exp_avg_sq().clone());
            ob.state()[pb[i].unsafeGetTensorImpl()]=std::move(b);
          }
        }
      }
      oa.zero_grad(true);ob.zero_grad(true);
      std::vector<torch::Tensor> xa{x.clone().set_requires_grad(true),s.clone().set_requires_grad(true),q.clone().set_requires_grad(true)},xb;
      for(auto t:xa)xb.push_back(t.detach().clone().set_requires_grad(true));
      float ratio=std::vector<float>{.01f,.05f,.10f,.15f}[(step-1)%4];
      auto a=old->applyResidual(xa[0],xa[1],xa[2],ratio);
      ElasticGaussianAttributes b;
      if(cublas) {
        auto feature=tf_cuda::query(xb[0],torch::full({n,1},ratio,x.options()),cuda->aabbMin(),cuda->aabbMax(),cuda->gridParameters());
        auto residual=tf_cuda::neural(feature,cuda->mlpParameters(),tf_cuda::MlpBackend::Cublas);
        auto y=tf_cuda::output(xb[0],xb[1],xb[2],residual);b={y[0],y[1],y[2]};
      } else b=cuda->applyResidual(xb[0],xb[1],xb[2],ratio);
      // Full input loss is accumulated before LoD + regularization backward.
      auto full=[](const std::vector<torch::Tensor>& in){return in[0].square().mean()*.01+in[1].mean()*.02+in[2].square().mean()*.01;};
      full(xa).backward();full(xb).backward();
      auto loss=[&](const ElasticGaussianAttributes& y,TransformField& tf){return (y.xyz-x).mean()+(y.scaling/s).mean()+y.rotation.mean()+tf->regularizationLoss();};
      auto la=loss(a,old),lb=loss(b,cuda);la.backward();lb.backward();
      bool inspect=teacher || step==1 || step==10 || step==100;
      if(inspect) {
        compare(mode,step,"loss",la,lb,2e-6,2e-5);
        compare(mode,step,"xyz",a.xyz,b.xyz,2e-6,2e-5);compare(mode,step,"scale",a.scaling,b.scaling,2e-6,2e-5);compare(mode,step,"rotation",a.rotation,b.rotation,2e-6,2e-5);
        for(size_t i=0;i<3;++i)compare(mode,step,"input_grad"+std::to_string(i),xa[i].grad(),xb[i].grad(),2e-6,2e-4);
        for(size_t i=0;i<pa.size();++i)compare(mode,step,"grad"+std::to_string(i),pa[i].grad(),pb[i].grad(),pa[i].dim()==4?1e-5:2e-6,pa[i].dim()==4?5e-4:2e-4);
      }
      oa.step();if(native)ob.step();else tf_cuda::tfCudaAdamDirect(ob,{0,1});
      if(inspect)for(size_t i=0;i<pa.size();++i) {
        compare(mode,step,"p"+std::to_string(i),pa[i],pb[i],2e-6,2e-4);
        if(step==1 && i==6) {
          auto index=(pa[i]-pb[i]).abs().reshape({-1}).argmax().item<int64_t>();
          std::cout<<"plane6_first_step mode="<<mode<<" name="<<old->named_parameters().keys()[i]<<" index="<<index
            <<" old_p="<<pa[i].reshape({-1})[index].item<float>()<<" other_p="<<pb[i].reshape({-1})[index].item<float>()
            <<" old_grad="<<pa[i].grad().reshape({-1})[index].item<float>()<<" other_grad="<<pb[i].grad().reshape({-1})[index].item<float>()<<std::endl;
        }
        const auto& astate=static_cast<const torch::optim::AdamParamState&>(*oa.state().at(pa[i].unsafeGetTensorImpl()));
        const auto& bstate=static_cast<const torch::optim::AdamParamState&>(*ob.state().at(pb[i].unsafeGetTensorImpl()));
        TORCH_CHECK(astate.step()==bstate.step(),"step mismatch");
        compare(mode,step,"m"+std::to_string(i),astate.exp_avg(),bstate.exp_avg(),2e-6,2e-4);
        compare(mode,step,"v"+std::to_string(i),astate.exp_avg_sq(),bstate.exp_avg_sq(),2e-6,2e-4);
      }
      out.flush();
    }
  }
  std::cout<<"fixed_input_training_chain starting_tolerance_failures="<<failed<<std::endl;
  return failed?1:0;
}
