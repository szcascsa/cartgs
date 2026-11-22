#include "include/transform_field_cuda_optimizer.h"
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <cmath>
#include <unordered_set>

namespace tf_cuda {
namespace {
void validateGroups(torch::optim::Adam& optimizer,const std::vector<std::size_t>& groups) {
  std::unordered_set<void*> seen;
  // Validate the entire TF population before any state or parameter mutation.
  for(auto index:groups) {
    TORCH_CHECK(index<optimizer.param_groups().size(),"Invalid TF Adam group");
    const auto& group=optimizer.param_groups()[index];
    const auto& o=static_cast<const torch::optim::AdamOptions&>(group.options());
    TORCH_CHECK(!o.amsgrad(),"TF CUDA Adam does not support AMSGrad");
    for(const auto& p:group.params()) {
      TORCH_CHECK(seen.insert(p.unsafeGetTensorImpl()).second,"Duplicate TF Adam parameter");
      if(!p.grad().defined())continue;
      TORCH_CHECK(p.is_cuda() && p.scalar_type()==torch::kFloat32 && p.is_contiguous() &&
          !p.grad().is_sparse() && p.grad().device()==p.device() && p.grad().scalar_type()==torch::kFloat32 &&
          p.grad().sizes()==p.sizes() && p.grad().is_contiguous(),"TF Adam requires contiguous CUDA FP32 parameters/gradients");
      auto old=optimizer.state().find(p.unsafeGetTensorImpl());
      if(old!=optimizer.state().end()) {
        const auto& s=static_cast<const torch::optim::AdamParamState&>(*old->second);
        for(auto t:{s.exp_avg(),s.exp_avg_sq()})TORCH_CHECK(t.device()==p.device() && t.scalar_type()==p.scalar_type() && t.sizes()==p.sizes() && t.is_contiguous(),"Invalid TF Adam state");
      }
    }
  }
}
void options(AdamUpdate& u,const torch::optim::AdamOptions& o,int64_t step) {
  const double b1=std::get<0>(o.betas()),b2=std::get<1>(o.betas());
  u.beta1=b1;u.beta2=b2;u.alpha1=1-b1;u.alpha2=1-b2;
  u.inverse_bias2=1.0f/static_cast<float>(std::sqrt(1-std::pow(b2,step)));
  u.step_size=-o.lr()/(1-std::pow(b1,step));u.eps=o.eps();u.weight_decay=o.weight_decay();
}
}
void tfCudaAdamDirect(torch::optim::Adam& optimizer,const std::vector<std::size_t>& groups) {
  torch::NoGradGuard no_grad;validateGroups(optimizer,groups);
  std::vector<AdamUpdate> updates;
  std::vector<torch::optim::AdamParamState*> states;
  std::vector<std::unique_ptr<torch::optim::AdamParamState>> fresh;
  std::vector<int64_t> steps;
  for(auto index:groups) {
    const auto& group=optimizer.param_groups()[index];
    const auto& o=static_cast<const torch::optim::AdamOptions&>(group.options());
    for(const auto& p:group.params()) {
      if(!p.grad().defined())continue;
      AdamUpdate u{};u.parameter=p;u.gradient=p.grad();
      auto old=optimizer.state().find(p.unsafeGetTensorImpl());
      torch::optim::AdamParamState* state;
      if(old!=optimizer.state().end()) {
        state=&static_cast<torch::optim::AdamParamState&>(*old->second);
        fresh.push_back(nullptr);u.old_mean=state->exp_avg();u.old_variance=state->exp_avg_sq();
      } else {
        fresh.push_back(std::make_unique<torch::optim::AdamParamState>());state=fresh.back().get();
        state->exp_avg(torch::empty_like(p));state->exp_avg_sq(torch::empty_like(p));
      }
      const int64_t step=state->step()+1;
      u.mean=state->exp_avg();u.variance=state->exp_avg_sq();u.proposed=p;
      // ATen div by a CPU scalar uses an FP32 reciprocal multiplication.
      options(u,o,step);
      updates.push_back(u);states.push_back(state);steps.push_back(step);
    }
  }
  adamUpdate(updates);
  for(size_t i=0;i<updates.size();++i) {
    states[i]->step(steps[i]);
    if(fresh[i])optimizer.state()[updates[i].parameter.unsafeGetTensorImpl()]=std::move(fresh[i]);
    torch::autograd::increment_version(updates[i].parameter);
    torch::autograd::increment_version(updates[i].mean);
    torch::autograd::increment_version(updates[i].variance);
  }
}
AdamPreparation tfCudaAdamPrepare(torch::optim::Adam& optimizer,const std::vector<std::size_t>& groups) {
  torch::NoGradGuard no_grad;validateGroups(optimizer,groups);
  AdamPreparation result;std::vector<AdamUpdate> updates;
  for(auto index:groups) {
    const auto& group=optimizer.param_groups()[index];
    const auto& o=static_cast<const torch::optim::AdamOptions&>(group.options());
    for(const auto& p:group.params()) {
      if(!p.grad().defined())continue;
      AdamPreparedTensor next{};next.parameter=p;next.group=index;next.parameter_version=p._version();
      next.state=std::make_unique<torch::optim::AdamParamState>();
      AdamUpdate u{};u.parameter=p;u.gradient=p.grad();
      auto old=optimizer.state().find(p.unsafeGetTensorImpl());int64_t step=1;
      if(old!=optimizer.state().end()) {
        next.previous_state=old->second.get();
        const auto& s=static_cast<const torch::optim::AdamParamState&>(*old->second);
        u.old_mean=s.exp_avg();u.old_variance=s.exp_avg_sq();step=s.step()+1;
      }
      next.state->step(step);next.state->exp_avg(torch::empty_like(p));next.state->exp_avg_sq(torch::empty_like(p));
      next.proposed=torch::empty_like(p);next.delta=torch::empty_like(p);
      u.mean=next.state->exp_avg();u.variance=next.state->exp_avg_sq();u.proposed=next.proposed;u.delta=next.delta;
      options(u,o,step);updates.push_back(u);result.push_back(std::move(next));
    }
  }
  adamUpdate(updates);return result;
}
void tfCudaAdamMixedStep(torch::optim::Adam& optimizer,const std::vector<std::size_t>& groups,
                        ProtectionProfiler* profiler) {
  if (groups.empty()) { optimizer.step(); return; }
  validateGroups(optimizer,groups);
  // Preserve the ordinary optimizer for Gaussian/Selector groups and restore
  // TF gradient handles even if it throws. No gradient values are modified.
  struct GradRestore {
    std::vector<std::pair<torch::Tensor,torch::Tensor>> values;
    ~GradRestore(){for(auto& value:values)value.first.mutable_grad()=value.second;}
  };
  {
    GradRestore restore;
    for(auto index:groups)for(const auto& p:optimizer.param_groups()[index].params()) {
      restore.values.emplace_back(p,p.grad());p.mutable_grad()=torch::Tensor();
    }
    optimizer.step();
  }
  ProtectionProfiler::Scope timing(profiler, ProtectionProfiler::TFOptimizer, true);
  tfCudaAdamDirect(optimizer,groups);
}
void tfCudaAdamValidate(torch::optim::Adam& optimizer,const AdamPreparation& prepared,
                        const std::vector<torch::Tensor>& candidate) {
  TORCH_CHECK(prepared.size()==candidate.size(),"CUDA Adam candidate count mismatch");
  for(size_t i=0;i<prepared.size();++i) {
    const auto& t=prepared[i];const auto& c=candidate[i];
    TORCH_CHECK(t.state && t.parameter._version()==t.parameter_version,"Stale CUDA Adam preparation");
    TORCH_CHECK(t.group<optimizer.param_groups().size(),"Stale CUDA Adam group");
    bool found=false;for(const auto& p:optimizer.param_groups()[t.group].params())found|=p.is_same(t.parameter);
    TORCH_CHECK(found,"CUDA Adam population changed during transaction");
    auto old=optimizer.state().find(t.parameter.unsafeGetTensorImpl());
    TORCH_CHECK((old==optimizer.state().end()?nullptr:old->second.get())==t.previous_state,"CUDA Adam state replaced during transaction");
    TORCH_CHECK(c.defined() && c.device()==t.parameter.device() && c.scalar_type()==torch::kFloat32 &&
        c.sizes()==t.parameter.sizes() && c.is_contiguous(),"Invalid CUDA Adam candidate");
  }
}
void tfCudaAdamCommit(torch::optim::Adam& optimizer,AdamPreparation& prepared,
                      const std::vector<torch::Tensor>& candidate) {
  torch::NoGradGuard no_grad;tfCudaAdamValidate(optimizer,prepared,candidate);
  std::vector<torch::Tensor> parameters;for(const auto& t:prepared)parameters.push_back(t.parameter);
  adamCopy(parameters,candidate);
  for(auto& t:prepared) {
    torch::autograd::increment_version(t.parameter);
    optimizer.state()[t.parameter.unsafeGetTensorImpl()]=std::move(t.state);
  }
  prepared.clear();
}
void tfCudaAdamDiscard(AdamPreparation& prepared) {prepared.clear();}
}  // namespace tf_cuda
