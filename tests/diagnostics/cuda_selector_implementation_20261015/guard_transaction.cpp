#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include "improvements/selector/gumbel_network.h"
#include "include/gradient_constraint.h"

using improvements::selector::GumbelNetwork;
void exact(const torch::Tensor& a,const torch::Tensor& b) {
  TORCH_CHECK(a.defined()==b.defined(),"Gradient presence differs");
  if(a.defined())TORCH_CHECK(torch::equal(a,b),"Transaction tensor differs: ",(a-b).abs().max().item<double>());
}
bool sameValues(const std::vector<double>& a,const std::vector<double>& b) {
  if(a.size()!=b.size())return false;
  for(size_t i=0;i<a.size();++i)if(a[i]!=b[i] && !(std::isnan(a[i])&&std::isnan(b[i])))return false;
  return true;
}
struct State {
  GumbelNetwork network;
  torch::Tensor gaussian,field;
  std::unique_ptr<torch::optim::Adam> optimizer;
  State(bool cuda,double lr,bool projection) {
    network->to(torch::kCUDA);network->setBackend(cuda?"cuda":"legacy");
    auto o=torch::TensorOptions().device(torch::kCUDA).requires_grad(true);
    gaussian=torch::ones({projection?2:1},o);field=torch::ones({2},o);
    optimizer=std::make_unique<torch::optim::Adam>(std::vector<torch::Tensor>{gaussian},torch::optim::AdamOptions(lr).eps(1e-15));
    optimizer->add_param_group(network->parameters());optimizer->param_groups()[1].options().set_lr(.001);
    optimizer->add_param_group(std::vector<torch::Tensor>{field});optimizer->param_groups()[2].options().set_lr(.001);
  }
  std::vector<torch::Tensor> parameters() {
    auto p=std::vector<torch::Tensor>{gaussian};auto n=network->parameters();p.insert(p.end(),n.begin(),n.end());p.push_back(field);return p;
  }
};
void compare(State& a,State& b) {
  auto pa=a.parameters(),pb=b.parameters();
  for(size_t i=0;i<pa.size();++i) {
    exact(pa[i],pb[i]);exact(pa[i].grad(),pb[i].grad());
    auto x=a.optimizer->state().find(pa[i].unsafeGetTensorImpl()),y=b.optimizer->state().find(pb[i].unsafeGetTensorImpl());
    TORCH_CHECK((x==a.optimizer->state().end())==(y==b.optimizer->state().end()),"State presence differs");
    if(x==a.optimizer->state().end())continue;
    const auto& sx=static_cast<const torch::optim::AdamParamState&>(*x->second);
    const auto& sy=static_cast<const torch::optim::AdamParamState&>(*y->second);
    TORCH_CHECK(sx.step()==sy.step(),"Step differs");exact(sx.exp_avg(),sy.exp_avg());exact(sx.exp_avg_sq(),sy.exp_avg_sq());
  }
}
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==2,"guard_transaction OUTPUT.csv");torch::set_num_threads(1);torch::manual_seed(361);
    std::ofstream out(argv[1]);out<<"case,status,reason,committed,all_parameters_gradients_states_exact\n";
    for(const std::string name:{"accept","shrink","projection","reject","fallback","nonfinite"}) {
      const bool projection=name=="projection";double lr=name=="shrink"?3:(name=="reject"||name=="fallback"?12:.1);
      State a(false,lr,projection),b(true,lr,projection);
      {torch::NoGradGuard ng;auto pa=a.parameters(),pb=b.parameters();for(size_t i=0;i<pa.size();++i)pb[i].copy_(pa[i]);}
      auto x=torch::randn({257,3},torch::kCUDA),q=torch::randn({257,4},torch::kCUDA),s=torch::randn({257,3},torch::kCUDA);
      auto la=a.network->forwardProbability(x,q,s,.05),lb=b.network->forwardProbability(x,q,s,.05);exact(la,lb);
      la.square().mean().backward();lb.square().mean().backward();
      // Keep an undefined gradient and an allocated zero gradient distinct.
      a.network->parameters()[0].mutable_grad()=torch::Tensor();b.network->parameters()[0].mutable_grad()=torch::Tensor();
      for(auto* state:{&a,&b}) {
        state->network->parameters()[1].mutable_grad()=torch::zeros_like(state->network->parameters()[1]);
        state->gaussian.mutable_grad()=projection?torch::tensor({-1.f,1.f},x.options()):torch::ones_like(state->gaussian);
        state->field.mutable_grad()=torch::ones_like(state->field);
      }
      compare(a,b);auto before=a.parameters();for(auto& p:before)p=p.detach().clone();
      auto run=[&](State& state,bool fused) {
        GuardEvaluation evaluation;evaluation.objectives={{"full"},{"lod"}};
        evaluation.evaluate=[&](std::size_t,const AdamStepPreview& preview,const std::vector<torch::Tensor>* candidate) {
          if(name=="nonfinite")return torch::full({},std::numeric_limits<float>::quiet_NaN(),x.options());
          auto p=candidate?preview.value(state.gaussian,*candidate):state.gaussian;
          return projection?p[0].square():p.square().mean();
        };
        GradientConstraintConfig config;config.cancel_fallback_enable=name=="fallback";
        return GradientConstraintManager(config).step(*state.optimizer,evaluation,fused?std::vector<std::size_t>{1,2}:std::vector<std::size_t>{});
      };
      auto ra=run(a,false),rb=run(b,true);
      TORCH_CHECK(ra.status==rb.status && ra.reason==rb.reason && ra.committed==rb.committed && ra.alpha==rb.alpha &&
          sameValues(ra.before,rb.before) && sameValues(ra.after,rb.after) && ra.update_ratio_by_group==rb.update_ratio_by_group,"Guard decisions differ for ",name);
      const std::string expected=name=="accept"||projection?"ACCEPT":name=="shrink"?"SHRINK":name=="fallback"?"FALLBACK":"CANCEL";
      TORCH_CHECK(ra.status==expected,"Unexpected status for ",name,": ",ra.status);
      compare(a,b);
      if(!ra.committed){auto p=a.parameters();for(size_t i=0;i<p.size();++i)exact(p[i],before[i]);TORCH_CHECK(a.optimizer->state().empty(),"Rejected joint transaction mutated Adam");}
      out<<name<<','<<ra.status<<','<<ra.reason<<','<<ra.committed<<",1\n";out.flush();
    }
    std::cout<<"Six joint Gaussian/Selector/TF Guard transaction cases exact\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
