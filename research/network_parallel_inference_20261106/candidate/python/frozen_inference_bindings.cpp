#include <torch/extension.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/ops/_addmm_activation.h>
#include <vector>

torch::Tensor selector_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor selector_view_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, bool, bool);
torch::Tensor condition_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, bool);
torch::Tensor view_code_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, double, double);
void display_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor compensation_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const std::vector<torch::Tensor>&, const torch::Tensor&);
torch::Tensor add_relu(const torch::Tensor&, const torch::Tensor&);
torch::Tensor add_view_relu(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor head_pack(const std::vector<torch::Tensor>&, const torch::Tensor&);
torch::Tensor publish_render(const std::vector<torch::Tensor>&);
torch::Tensor publish_selected(const std::vector<torch::Tensor>&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, bool);
torch::Tensor exact_select(const torch::Tensor&, const torch::Tensor&, int64_t,
    const std::vector<torch::Tensor>&, bool);
std::vector<torch::Tensor> visible_candidates(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);

torch::Tensor frame_counts(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);

torch::Tensor quaternion_identity(const torch::Tensor&);
torch::Tensor quaternion_divide(const torch::Tensor&, const torch::Tensor&);
torch::Tensor quaternion_multiply(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);

torch::Tensor quaternion_decode(const torch::Tensor& residual, const torch::Tensor& rows,
    const torch::Tensor& master) {
  TORCH_CHECK(!torch::GradMode::is_enabled(), "Frozen quaternion only");
  c10::cuda::CUDAGuard guard(residual.device());
  auto q=quaternion_identity(residual);
  q=quaternion_divide(q,at::linalg_vector_norm(q,2,at::IntArrayRef{-1},true));
  q=quaternion_multiply(q,rows,master);
  q=quaternion_divide(q,at::linalg_vector_norm(q,2,at::IntArrayRef{-1},true));
  // Keep the original Renderer activation's second normalization.
  return quaternion_divide(q,at::linalg_vector_norm(q,2,at::IntArrayRef{-1},true));
}

torch::Tensor frustum_planes(const torch::Tensor&);
torch::Tensor frustum_test(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor selection_scores(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);

torch::Tensor frustum(const torch::Tensor& centers, const torch::Tensor& scales,
    const torch::Tensor& projection, const torch::Tensor& occupied) {
  TORCH_CHECK(!torch::GradMode::is_enabled(), "Frozen frustum only");
  // Keep the original GEMM and its strided plane operand. Only fuse its perimeter.
  auto planes=frustum_planes(projection);
  auto distance=at::matmul(centers,planes.slice(1,0,3).t());
  return frustum_test(distance,planes,scales,occupied);
}

torch::Tensor linear_relu(const torch::Tensor& x,const torch::Tensor& w,
    const torch::Tensor& b,bool fused) {
  return fused?at::_addmm_activation(b,x,w.t(),1,1,false):at::relu(at::linear(x,w,b));
}

torch::Tensor selector_logits(const torch::Tensor& input, const std::vector<torch::Tensor>& p,bool fused_relu) {
  TORCH_CHECK(!torch::GradMode::is_enabled() && p.size()==6, "Frozen Selector inference only");
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kFloat && input.dim()==2 &&
      (input.size(1)==11 || input.size(1)==19) && p[0].size(1)==input.size(1),
      "Selector input must match CUDA FP32 Nx11/Nx19 weights");
  c10::cuda::CUDAGuard guard(input.device());
  auto h=linear_relu(input,p[0],p[1],fused_relu);
  h=linear_relu(h,p[2],p[3],fused_relu);
  return at::linear(h,p[4],p[5]);
}

torch::Tensor compensation_network(const torch::Tensor& input, const torch::Tensor& code,
    const torch::Tensor& reference, const std::vector<torch::Tensor>& p,
    const torch::Tensor& condition, const torch::Tensor& view_weight,bool fused_relu,
    const std::vector<int64_t>& streams,const std::vector<int64_t>& events) {
  TORCH_CHECK(!torch::GradMode::is_enabled() && p.size()==21, "Frozen Residual48 inference only");
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kFloat && input.dim()==2 && input.size(1)==51 &&
      code.sizes()==at::IntArrayRef({1,3}) && reference.numel()==input.size(0), "Residual48 input layout");
  c10::cuda::CUDAGuard guard(input.device());
  auto encoded=at::linear(input,p[0],p[1]);
  auto projected=at::linear(code,p[4]);
  torch::Tensor activation;
  if(condition.defined() && condition.numel()) {
    TORCH_CHECK(condition.sizes()==at::IntArrayRef({input.size(0),8}) &&
        view_weight.sizes()==at::IntArrayRef({64,8}), "Compensation view layout");
    activation=add_view_relu(encoded,projected,at::linear(condition,view_weight));
  } else activation=add_relu(encoded,projected);
  auto hidden=linear_relu(activation,p[2],p[3],fused_relu);
  std::vector<torch::Tensor> parts;
  const auto current=c10::cuda::getCurrentCUDAStream();
  const bool parallel=!streams.empty();
  TORCH_CHECK(!parallel || (streams.size()==4 && events.size()==5), "Four Head stream resources");
  if(parallel)C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(events[0]),current));
  for(int j=0;j<4;++j) {
    const int k=5+j*4;
    if(parallel) {
      auto side=c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(streams[j]),input.get_device());
      C10_CUDA_CHECK(cudaStreamWaitEvent(side,reinterpret_cast<cudaEvent_t>(events[0]),0));
      c10::cuda::CUDACachingAllocator::recordStream(hidden.storage().data_ptr(),side);
      for(int t=0;t<4;++t)c10::cuda::CUDACachingAllocator::recordStream(p[k+t].storage().data_ptr(),side);
      c10::cuda::CUDAStreamGuard side_guard(side);
      parts.push_back(at::linear(linear_relu(hidden,p[k],p[k+1],fused_relu),p[k+2],p[k+3]));
      C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(events[j+1]),side));
    } else parts.push_back(at::linear(linear_relu(hidden,p[k],p[k+1],fused_relu),p[k+2],p[k+3]));
  }
  if(parallel)for(int j=0;j<4;++j) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(current,reinterpret_cast<cudaEvent_t>(events[j+1]),0));
    c10::cuda::CUDACachingAllocator::recordStream(parts[j].storage().data_ptr(),current);
  }
  return head_pack(parts,reference);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("quaternion_identity", &quaternion_identity);
  module.def("quaternion_divide", &quaternion_divide);
  module.def("quaternion_multiply", &quaternion_multiply);
  module.def("quaternion_decode", &quaternion_decode);
  module.def("frame_counts", &frame_counts);
  module.def("frustum", &frustum);
  module.def("selection_scores", &selection_scores);
  module.def("selector_pack", &selector_pack);
  module.def("selector_view_pack", &selector_view_pack);
  module.def("condition_pack", &condition_pack);
  module.def("view_code_pack", &view_code_pack);
  module.def("display_pack", &display_pack);
  module.def("selector_logits", &selector_logits,pybind11::arg("input"),
      pybind11::arg("parameters"),pybind11::arg("fused_relu")=false);
  module.def("compensation_pack", &compensation_pack);
  module.def("compensation_network", &compensation_network, pybind11::arg("input"),
      pybind11::arg("code"), pybind11::arg("reference"), pybind11::arg("parameters"),
      pybind11::arg("condition")=torch::Tensor(), pybind11::arg("view_weight")=torch::Tensor(),pybind11::arg("fused_relu")=false,
      pybind11::arg("streams")=std::vector<int64_t>{},pybind11::arg("events")=std::vector<int64_t>{});
  module.def("publish_render", &publish_render);
  module.def("publish_selected", &publish_selected);
  module.def("exact_select", &exact_select, pybind11::arg("scores"), pybind11::arg("eligible"),
      pybind11::arg("k"), pybind11::arg("workspace")=std::vector<torch::Tensor>{}, pybind11::arg("narrow_indices")=true);
  module.def("visible_candidates", &visible_candidates);
}
