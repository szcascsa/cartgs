#include "include/transform_field_cuda.h"
#include <torch/csrc/autograd/custom_function.h>

namespace tf_cuda {
thread_local ProtectionProfiler* training_profiler = nullptr;
namespace {
void captureProfile(torch::autograd::AutogradContext* ctx) {
  ctx->saved_data["training_profiler"] = reinterpret_cast<int64_t>(training_profiler);
}
ProtectionProfiler* capturedProfile(torch::autograd::AutogradContext* ctx) {
  return reinterpret_cast<ProtectionProfiler*>(ctx->saved_data["training_profiler"].toInt());
}
class QueryFunction : public torch::autograd::Function<QueryFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               at::TensorList inputs) {
    ctx->save_for_backward(inputs.vec());
    captureProfile(ctx);
    return queryForward(inputs.vec());
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list gradients) {
    // SavedVariable checks Tensor version counters, including AABB and planes.
    ProtectionProfiler::Scope timing(capturedProfile(ctx), ProtectionProfiler::TFBackward, true);
    return queryBackward(ctx->get_saved_variables(), gradients.at(0).contiguous());
  }
};
class NeuralFunction : public torch::autograd::Function<NeuralFunction> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                               at::TensorList inputs,int64_t backend) {
    auto in=inputs.vec();
    ctx->set_materialize_grads(false);
    auto context=mlpForward(in[0],{in.begin()+1,in.end()},static_cast<MlpBackend>(backend));
    // Save feature/parameters and only four ReLU activations. Tensor version
    // checks protect both the parameters and inputs until backward completes.
    in.insert(in.end(),context.begin()+1,context.begin()+5);
    ctx->save_for_backward(in);
    captureProfile(ctx);
    return {context[5],context[6],context[7]};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list gradients) {
    auto saved=ctx->get_saved_variables();
    ProtectionProfiler::Scope timing(capturedProfile(ctx), ProtectionProfiler::TFBackward, true);
    for(auto& g:gradients)if(g.defined())g=g.contiguous();
    std::vector<torch::Tensor> context{saved[0]};
    context.insert(context.end(),saved.begin()+15,saved.end());
    auto result=mlpBackward(context,{saved.begin()+1,saved.begin()+15},gradients);
    result.push_back(torch::Tensor()); // backend enum
    return result;
  }
};
class OutputFunction : public torch::autograd::Function<OutputFunction> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                               at::TensorList inputs,bool log_scale) {
    auto in=inputs.vec();ctx->set_materialize_grads(false);
    ctx->save_for_backward(in);ctx->saved_data["log_scale"]=log_scale;
    captureProfile(ctx);
    return outputForward(in[0],in[1],in[2],{in[3],in[4],in[5]},log_scale);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list gradients) {
    for(auto& g:gradients)if(g.defined())g=g.contiguous();
    ProtectionProfiler::Scope timing(capturedProfile(ctx), ProtectionProfiler::TFBackward, true);
    auto result=outputBackward(ctx->get_saved_variables(),gradients,ctx->saved_data["log_scale"].toBool());
    result.push_back(torch::Tensor());
    return result;
  }
};
class RegularizationFunction : public torch::autograd::Function<RegularizationFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,at::TensorList planes,
                               double spatial,double ratio,double l1) {
    ctx->save_for_backward(planes.vec());
    captureProfile(ctx);
    ctx->saved_data["spatial"]=spatial;ctx->saved_data["ratio"]=ratio;ctx->saved_data["l1"]=l1;
    return regularizationForward(planes.vec(),spatial,ratio,l1);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list gradients) {
    ProtectionProfiler::Scope timing(capturedProfile(ctx), ProtectionProfiler::TFBackward, true);
    auto result=regularizationBackward(ctx->get_saved_variables(),gradients[0],
        ctx->saved_data["spatial"].toDouble(),ctx->saved_data["ratio"].toDouble(),ctx->saved_data["l1"].toDouble());
    for(int i=0;i<3;++i)result.push_back(torch::Tensor());return result;
  }
};
}  // namespace

torch::Tensor query(const torch::Tensor& xyz, const torch::Tensor& ratio,
                    const torch::Tensor& lo, const torch::Tensor& hi,
                    const std::vector<torch::Tensor>& planes) {
  std::vector<torch::Tensor> inputs{xyz, ratio, lo, hi};
  inputs.insert(inputs.end(), planes.begin(), planes.end());
  // Legacy HexPlane's empty early return has no input/plane autograd edges.
  if(xyz.size(0)==0)return queryForward(inputs);
  return QueryFunction::apply(at::TensorList(inputs));
}
std::vector<torch::Tensor> neural(const torch::Tensor& feature,
    const std::vector<torch::Tensor>& parameters,MlpBackend backend) {
  std::vector<torch::Tensor> in{feature};in.insert(in.end(),parameters.begin(),parameters.end());
  return NeuralFunction::apply(at::TensorList(in),static_cast<int64_t>(backend));
}
std::vector<torch::Tensor> output(const torch::Tensor& xyz,const torch::Tensor& scale,
    const torch::Tensor& quaternion,const std::vector<torch::Tensor>& residual,bool log_scale) {
  TORCH_CHECK(residual.size()==3,"TF requires three residual heads");
  std::vector<torch::Tensor> in{xyz,scale,quaternion};in.insert(in.end(),residual.begin(),residual.end());
  return OutputFunction::apply(at::TensorList(in),log_scale);
}
torch::Tensor regularization(const std::vector<torch::Tensor>& planes,
    float spatial,float ratio,float l1) {
  return RegularizationFunction::apply(at::TensorList(planes),double(spatial),double(ratio),double(l1));
}
}  // namespace tf_cuda
