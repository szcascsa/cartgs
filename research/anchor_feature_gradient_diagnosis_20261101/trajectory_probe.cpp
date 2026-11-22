#include "probe_common.h"
#include <memory>
struct Branch {
  Scene scene;std::unique_ptr<torch::optim::Adam> optimizer;
  explicit Branch(const std::string& root,const std::string& checkpoint):scene(root,checkpoint) {
    optimizer=std::make_unique<torch::optim::Adam>(scene.net->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
  }
};
struct StepCapture {torch::Tensor output,input_gradient;std::vector<torch::Tensor> gathered_gradient;};
StepCapture step_backward(Branch& branch,const torch::Tensor& ids,double ratio,const torch::Tensor& up,bool fused) {
  branch.optimizer->zero_grad(true);auto a=branch.scene.net->anchors();auto m=branch.scene.net->spatialCompensation();
  auto owners=a->rows().index_select(0,ids),x=branch.scene.xyz.index_select(0,ids),c=a->centers().index_select(0,owners),r=a->offsetScales().index_select(0,owners);
  auto w=a->compensationResidualWeights(ratio);std::vector<torch::Tensor> f;StepCapture capture;capture.gathered_gradient.resize(w.size());
  for(size_t j=0;j<w.size();++j){f.push_back(a->compensationFeatureTables()[j].index_select(0,owners));f.back().register_hook([&capture,j](torch::Tensor g){capture.gathered_gradient[j]=g.clone();return g;});}
  auto input=fused?diagnostic_cuda::residual48Pack(x,c,r,m->mean,m->stddev,f,w):legacy_pack(x,c,r,m->mean,m->stddev,f,w);
  input.register_hook([&capture](torch::Tensor g){capture.input_gradient=g.clone();return g;});
  auto output=m->forward(x,{},{},{},{},c,r,ratio,true,{},input);output.backward(up);capture.output=output.detach();return capture;
}
void group(std::ofstream& csv,const std::string& prefix,const std::string& node,const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& b) {
  Difference all;for(size_t j=0;j<a.size();++j){auto d=difference(a[j],b[j]);all.equal&=d.equal;all.total+=d.total;all.different+=d.different;all.absolute=std::max(all.absolute,d.absolute);all.relative_l2=std::max(all.relative_l2,d.relative_l2);}
  csv<<prefix<<','<<node<<','<<all.equal<<','<<all.total<<','<<all.different<<','<<all.absolute<<','<<all.relative_l2<<'\n';
}
std::vector<torch::Tensor> gradients(const std::vector<torch::Tensor>& p) {std::vector<torch::Tensor> g;for(auto t:p)g.push_back(t.grad());return g;}
std::vector<torch::Tensor> moment(Branch& b,int kind) {
  std::vector<torch::Tensor> out;for(auto t:b.scene.net->anchors()->compensationFeatureTables()) {
    auto found=b.optimizer->state().find(t.unsafeGetTensorImpl());if(found==b.optimizer->state().end()){out.push_back({});continue;}
    auto& s=static_cast<torch::optim::AdamParamState&>(*found->second);out.push_back(kind==0?s.exp_avg():s.exp_avg_sq());
  }return out;
}
int main(int argc,char** argv) {
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);const std::string root=argv[1],checkpoint=argv[2];
  const int steps=argc>3?std::stoi(argv[3]):72;const std::string name=argc>4?argv[4]:"trajectories";std::ofstream csv(root+"/"+name+".csv");
  csv<<"deterministic,seed,step,ratio,comparison,node,bits,total,different,max_abs,max_tensor_relative_l2\n";
  for(int deterministic:{0,1})for(int seed=0;seed<2;++seed) {
    if(argc>5&&std::stoi(argv[5])==0&&deterministic)continue;
    at::globalContext().setDeterministicAlgorithms(deterministic,false);Branch a(root,checkpoint),repeat(root,checkpoint),b(root,checkpoint);
    const std::vector<double> ratios{.01,.05,.1,.2,.3,.4,.025,.075,.15,.25,.35};
    for(int step=0;step<steps;++step) {
      const int j=step%ratios.size();double ratio=ratios[j];auto ids=a.scene.ids(ratio,j);torch::manual_seed(224500+seed*1000+step);auto up=torch::randn({ids.numel(),14},a.scene.xyz.options());
      auto ca=step_backward(a,ids,ratio,up,false),cr=step_backward(repeat,ids,ratio,up,false),cb=step_backward(b,ids,ratio,up,true);
      auto prefix=[&](bool fused){std::ostringstream s;s<<deterministic<<','<<seed<<','<<step<<','<<ratio<<','<<(fused?"baseline_vs_fused":"baseline_repeat");return s.str();};
      for(bool fused:{false,true}) {
        auto& other=fused?b:repeat;auto& c=fused?cb:cr;auto pre=prefix(fused);
        log_difference(csv,pre,"forward",ca.output,c.output);log_difference(csv,pre,"packed_VJP",ca.input_gradient,c.input_gradient);
        group(csv,pre,"gather_input_VJP",ca.gathered_gradient,c.gathered_gradient);
        group(csv,pre,"table_gradient",gradients(a.scene.net->anchors()->compensationFeatureTables()),gradients(other.scene.net->anchors()->compensationFeatureTables()));
        group(csv,pre,"network_gradient",gradients(a.scene.net->spatialCompensation()->parameters()),gradients(other.scene.net->spatialCompensation()->parameters()));
      }
      // Causal diagnostic only: force exactly the same Scatter result before all Adam updates.
      if(argc>6&&std::stoi(argv[6])) {
        for(auto* other:{&repeat,&b})for(size_t k=0;k<6;++k) {
          auto source=a.scene.net->anchors()->compensationFeatureTables()[k].grad();
          other->scene.net->anchors()->compensationFeatureTables()[k].mutable_grad()=source.defined()?source.clone():torch::Tensor();
        }
      }
      for(auto* branch:{&a,&repeat,&b})TORCH_CHECK(mapping_cuda::tryFusedDenseAdam(*branch->optimizer),"Expected production FP32 Adam");
      for(bool fused:{false,true}) {
        auto& other=fused?b:repeat;auto pre=prefix(fused);
        group(csv,pre,"table_parameter",a.scene.net->anchors()->compensationFeatureTables(),other.scene.net->anchors()->compensationFeatureTables());
        group(csv,pre,"first_moment",moment(a,0),moment(other,0));group(csv,pre,"second_moment",moment(a,1),moment(other,1));
        group(csv,pre,"network_parameter",a.scene.net->spatialCompensation()->parameters(),other.scene.net->spatialCompensation()->parameters());
        for(size_t k=0;k<6;++k) {
          auto ta=a.scene.net->anchors()->compensationFeatureTables()[k],tb=other.scene.net->anchors()->compensationFeatureTables()[k];
          auto ia=a.optimizer->state().find(ta.unsafeGetTensorImpl()),ib=other.optimizer->state().find(tb.unsafeGetTensorImpl());
          TORCH_CHECK((ia==a.optimizer->state().end())==(ib==other.optimizer->state().end()),"Feature Adam missing state");
          if(ia!=a.optimizer->state().end())TORCH_CHECK(static_cast<torch::optim::AdamParamState&>(*ia->second).step()==static_cast<torch::optim::AdamParamState&>(*ib->second).step(),"Feature Adam clock mismatch");
        }
      }
      csv.flush();if(step%12==0)std::cout<<"TRAJECTORY det="<<deterministic<<" seed="<<seed<<" step="<<step<<std::endl;
    }
  }
  std::cout<<"MATCHED_VJP_TRAJECTORY_COMPLETE"<<std::endl;
}
