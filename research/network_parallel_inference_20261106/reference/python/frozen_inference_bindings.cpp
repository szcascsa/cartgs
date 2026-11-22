#include <torch/extension.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
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

torch::Tensor selector_logits(const torch::Tensor& input, const std::vector<torch::Tensor>& p) {
  TORCH_CHECK(!torch::GradMode::is_enabled() && p.size()==6, "Frozen Selector inference only");
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kFloat && input.dim()==2 &&
      (input.size(1)==11 || input.size(1)==19) && p[0].size(1)==input.size(1),
      "Selector input must match CUDA FP32 Nx11/Nx19 weights");
  c10::cuda::CUDAGuard guard(input.device());
  auto h=at::relu(at::linear(input,p[0],p[1]));
  h=at::relu(at::linear(h,p[2],p[3]));
  return at::linear(h,p[4],p[5]);
}

torch::Tensor compensation_network(const torch::Tensor& input, const torch::Tensor& code,
    const torch::Tensor& reference, const std::vector<torch::Tensor>& p,
    const torch::Tensor& condition, const torch::Tensor& view_weight) {
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
  auto hidden=at::relu(at::linear(activation,p[2],p[3]));
  std::vector<torch::Tensor> parts;
  for(int j=0;j<4;++j) {
    const int k=5+j*4;
    parts.push_back(at::linear(at::relu(at::linear(hidden,p[k],p[k+1])),p[k+2],p[k+3]));
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
  module.def("selector_logits", &selector_logits);
  module.def("compensation_pack", &compensation_pack);
  module.def("compensation_network", &compensation_network, pybind11::arg("input"),
      pybind11::arg("code"), pybind11::arg("reference"), pybind11::arg("parameters"),
      pybind11::arg("condition")=torch::Tensor(), pybind11::arg("view_weight")=torch::Tensor());
  module.def("publish_render", &publish_render);
  module.def("publish_selected", &publish_selected);
  module.def("exact_select", &exact_select, pybind11::arg("scores"), pybind11::arg("eligible"),
      pybind11::arg("k"), pybind11::arg("workspace")=std::vector<torch::Tensor>{}, pybind11::arg("narrow_indices")=true);
  module.def("visible_candidates", &visible_candidates);
}
