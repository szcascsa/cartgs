#include "probe_common.h"
#include <memory>
#include <iomanip>
struct AdamBundle {
  std::vector<torch::Tensor> parameter;std::unique_ptr<torch::optim::Adam> optimizer;
  AdamBundle(const std::vector<torch::Tensor>& original) {
    for(auto p:original)parameter.push_back(p.detach().clone().requires_grad_(true));
    optimizer=std::make_unique<torch::optim::Adam>(parameter,torch::optim::AdamOptions(.001).eps(1e-15));
  }
  void assign(const std::vector<torch::Tensor>& gradient) {
    optimizer->zero_grad(true);for(size_t j=0;j<gradient.size();++j)parameter[j].mutable_grad()=gradient[j].defined()?gradient[j].clone():torch::Tensor();
  }
  void step(bool fused=true) {if(fused){TORCH_CHECK(mapping_cuda::tryFusedDenseAdam(*optimizer),"Production dense Adam fallback");}else optimizer->step();}
  void copy(const AdamBundle& other) {
    torch::NoGradGuard guard;
    for(size_t j=0;j<parameter.size();++j) {
      parameter[j].copy_(other.parameter[j]);auto found=other.optimizer->state().find(other.parameter[j].unsafeGetTensorImpl());if(found==other.optimizer->state().end())continue;
      const auto& s=static_cast<torch::optim::AdamParamState&>(*found->second);auto state=std::make_unique<torch::optim::AdamParamState>();
      state->step(s.step());state->exp_avg(s.exp_avg().clone());state->exp_avg_sq(s.exp_avg_sq().clone());optimizer->state()[parameter[j].unsafeGetTensorImpl()]=std::move(state);
    }
  }
};
void compare(std::ofstream& csv,const std::string& prefix,AdamBundle& a,AdamBundle& b,const AdamBundle& initial) {
  for(size_t j=0;j<a.parameter.size();++j) {
    std::string name="table"+std::to_string(j);log_difference(csv,prefix,name+"_gradient",a.parameter[j].grad(),b.parameter[j].grad());
    log_difference(csv,prefix,name+"_parameter",a.parameter[j],b.parameter[j]);
    log_difference(csv,prefix,name+"_update",a.parameter[j]-initial.parameter[j],b.parameter[j]-initial.parameter[j]);
    auto ia=a.optimizer->state().find(a.parameter[j].unsafeGetTensorImpl()),ib=b.optimizer->state().find(b.parameter[j].unsafeGetTensorImpl());
    TORCH_CHECK((ia==a.optimizer->state().end())==(ib==b.optimizer->state().end()),"Independent table Adam state existence");if(ia==a.optimizer->state().end())continue;
    auto& sa=static_cast<torch::optim::AdamParamState&>(*ia->second);auto& sb=static_cast<torch::optim::AdamParamState&>(*ib->second);
    TORCH_CHECK(sa.step()==sb.step(),"Independent table clock changed");
    log_difference(csv,prefix,name+"_first_moment",sa.exp_avg(),sb.exp_avg());log_difference(csv,prefix,name+"_second_moment",sa.exp_avg_sq(),sb.exp_avg_sq());
  }
}
int main(int argc,char** argv) {
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);const std::string root=argv[1];Scene scene(root,argv[2]);
  std::vector<torch::Tensor> warm_gradient;
  {torch::serialize::InputArchive ar;ar.load_from(root+"/snapshots/d1_r5_fused.pt",torch::Device(torch::kCUDA));for(int j=0;j<6;++j){torch::Tensor g;ar.read("leaf_a_"+std::to_string(j),g);warm_gradient.push_back(g);}}
  std::ofstream csv(root+"/adam_nodes.csv");csv<<"deterministic,ratio_index,warm,comparison,node,bits,total,different,max_abs,relative_l2\n";
  std::ofstream examples(root+"/adam_amplification.csv");examples<<"deterministic,ratio_index,warm,table,anchor,channel,gradient_a,gradient_b,parameter_before,parameter_a,parameter_b,parameter_delta,first_moment_before,second_moment_before\n";examples<<std::setprecision(12);
  for(int deterministic:{0,1})for(int j=0;j<11;++j) {
    torch::serialize::InputArchive repeat,fused;std::string base=root+"/snapshots/d"+std::to_string(deterministic)+"_r"+std::to_string(j);
    repeat.load_from(base+"_repeat.pt",torch::Device(torch::kCUDA));fused.load_from(base+"_fused.pt",torch::Device(torch::kCUDA));
    size_t count=weights(std::vector<double>{.01,.05,.1,.2,.3,.4,.025,.075,.15,.25,.35}[j]).size();
    std::vector<torch::Tensor> ga(6),gr(6),gb(6);
    for(size_t k=0;k<count;++k){repeat.read("leaf_a_"+std::to_string(k),ga[k]);repeat.read("leaf_b_"+std::to_string(k),gr[k]);fused.read("leaf_b_"+std::to_string(k),gb[k]);}
    for(int warm:{0,1}) {
      AdamBundle initial(scene.tables);
      if(warm)for(int step=0;step<18;++step){std::vector<torch::Tensor> g(6);for(int k=0;k<=step%6;++k)g[k]=warm_gradient[k]*(1.+.125*(step%3));initial.assign(g);initial.step();}
      AdamBundle a(scene.tables),r(scene.tables),b(scene.tables),rescued(scene.tables),native(scene.tables);
      for(auto* p:{&a,&r,&b,&rescued,&native})p->copy(initial);
      a.assign(ga);r.assign(gr);b.assign(gb);rescued.assign(ga);native.assign(ga);
      a.step();r.step();b.step();rescued.step();native.step(false);
      auto prefix=std::to_string(deterministic)+","+std::to_string(j)+","+std::to_string(warm)+",";
      compare(csv,prefix+"baseline_repeat",a,r,initial);compare(csv,prefix+"baseline_vs_fused",a,b,initial);
      compare(csv,prefix+"same_gradient_rescue",a,rescued,initial);compare(csv,prefix+"native_vs_production_Adam",a,native,initial);
      for(size_t k=0;k<count;++k) {
        auto flat=(a.parameter[k]-b.parameter[k]).abs().flatten().argmax().item<int64_t>();int64_t row=flat/48,col=flat%48;
        double m=0,v=0;auto state=initial.optimizer->state().find(initial.parameter[k].unsafeGetTensorImpl());
        if(state!=initial.optimizer->state().end()){auto& s=static_cast<torch::optim::AdamParamState&>(*state->second);m=s.exp_avg()[row][col].item<double>();v=s.exp_avg_sq()[row][col].item<double>();}
        examples<<deterministic<<','<<j<<','<<warm<<','<<k<<','<<row<<','<<col<<','<<ga[k][row][col].item<double>()<<','<<gb[k][row][col].item<double>()<<','
          <<initial.parameter[k][row][col].item<double>()<<','<<a.parameter[k][row][col].item<double>()<<','<<b.parameter[k][row][col].item<double>()<<','
          <<(a.parameter[k]-b.parameter[k])[row][col].item<double>()<<','<<m<<','<<v<<'\n';
      }
      examples.flush();
      if(deterministic==0&&warm==0&&j==5) {
        torch::serialize::OutputArchive out;
        for(size_t k=0;k<count;++k) {
          out.write("gradient_a_"+std::to_string(k),ga[k].cpu());out.write("gradient_b_"+std::to_string(k),gb[k].cpu());
          out.write("parameter_before_"+std::to_string(k),initial.parameter[k].cpu());out.write("parameter_a_"+std::to_string(k),a.parameter[k].cpu());out.write("parameter_b_"+std::to_string(k),b.parameter[k].cpu());
        }
        out.save_to(root+"/snapshots/cold_adam_high_ratio.pt");
      }
      csv.flush();
    }
    std::cout<<"ADAM_NODE_COMPLETE deterministic="<<deterministic<<" ratio_index="<<j<<std::endl;
  }
}
