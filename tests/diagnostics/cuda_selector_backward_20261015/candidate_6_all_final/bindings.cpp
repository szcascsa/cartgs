#include <torch/extension.h>
#include "include/selector_cuda.h"
#include "include/constrained_adam.h"
#include "include/transform_field_cuda_optimizer.h"
#include "include/gi_teacher.h"

class DirectAdam {
 public:
  DirectAdam(std::vector<torch::Tensor> parameters,bool fused,double eps,double beta1,double beta2,double decay)
      : parameters_(std::move(parameters)),fused_(fused),
        optimizer_(parameters_,torch::optim::AdamOptions(.001).eps(eps).betas({beta1,beta2}).weight_decay(decay)) {}
  void gradients(const std::vector<c10::optional<torch::Tensor>>& grads,double lr) {
    TORCH_CHECK(grads.size()==parameters_.size(),"Gradient count");
    optimizer_.param_groups()[0].options().set_lr(lr);
    for(size_t i=0;i<grads.size();++i)parameters_[i].mutable_grad()=grads[i].has_value()?*grads[i]:torch::Tensor();
  }
  void step(const std::vector<c10::optional<torch::Tensor>>& grads,double lr) {
    gradients(grads,lr);
    if(fused_)tf_cuda::tfCudaAdamDirect(optimizer_,{0});else optimizer_.step();
  }
  std::vector<torch::Tensor> preview(const std::vector<c10::optional<torch::Tensor>>& grads,
                                   double lr,double alpha,double direction_scale,bool commit) {
    gradients(grads,lr);AdamStepPreview preparation(optimizer_,fused_?std::vector<size_t>{0}:std::vector<size_t>{});
    auto direction=preparation.delta();for(auto& value:direction)value=value*direction_scale;
    auto candidate=preparation.candidate(direction,alpha);
    if(commit)preparation.commit(candidate);
    return candidate;
  }
  std::tuple<std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<int64_t>> state() {
    std::vector<torch::Tensor> mean,variance;std::vector<int64_t> step;
    for(auto p:parameters_) {
      auto it=optimizer_.state().find(p.unsafeGetTensorImpl());
      if(it==optimizer_.state().end()){mean.push_back(torch::zeros_like(p));variance.push_back(torch::zeros_like(p));step.push_back(-1);}
      else {const auto& state=static_cast<const torch::optim::AdamParamState&>(*it->second);
        mean.push_back(state.exp_avg());variance.push_back(state.exp_avg_sq());step.push_back(state.step());}
    }
    return {mean,variance,step};
  }
 private:
  std::vector<torch::Tensor> parameters_;bool fused_;torch::optim::Adam optimizer_;
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("network",[](const std::vector<torch::Tensor>& x,const std::vector<torch::Tensor>& p){
    return selector_cuda::network(x,p,selector_cuda::Candidate::CublasStrict,static_cast<selector_cuda::BackwardStrategy>(6));
  });
  m.def("stages",[](const std::vector<torch::Tensor>& x,const std::vector<torch::Tensor>& p){
    return selector_cuda::forward(x,p,selector_cuda::Candidate::CublasStrict);
  });
  m.def("gi_targets",[](torch::Tensor importance,torch::Tensor observed,torch::Tensor maturity,float ratio,float entry){
    auto teacher=GITeacherBuilder().build(importance,torch::isfinite(importance),observed,maturity,ratio,1,1,entry);
    return std::make_pair(teacher.label,teacher.weight);
  });
  m.def("gi_loss",[](torch::Tensor prediction,torch::Tensor label,torch::Tensor weight,bool smooth){
    GITeacher teacher;teacher.label=label;teacher.weight=weight;return teacher.loss(prediction,smooth);
  });
  pybind11::class_<DirectAdam>(m,"Adam")
    .def(pybind11::init<std::vector<torch::Tensor>,bool,double,double,double,double>(),
      pybind11::arg("parameters"),pybind11::arg("fused"),pybind11::arg("eps")=1e-15,
      pybind11::arg("beta1")=.9,pybind11::arg("beta2")=.999,pybind11::arg("decay")=0.)
    .def("step",&DirectAdam::step).def("state",&DirectAdam::state).def("preview",&DirectAdam::preview);
}
