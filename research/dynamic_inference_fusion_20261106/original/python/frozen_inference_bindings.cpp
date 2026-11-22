#include <torch/extension.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

torch::Tensor selector_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor compensation_pack(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const std::vector<torch::Tensor>&, const torch::Tensor&);
torch::Tensor add_relu(const torch::Tensor&, const torch::Tensor&);
torch::Tensor head_pack(const std::vector<torch::Tensor>&, const torch::Tensor&);
torch::Tensor publish_render(const std::vector<torch::Tensor>&);
torch::Tensor publish_selected(const std::vector<torch::Tensor>&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, bool);
torch::Tensor exact_select(const torch::Tensor&, const torch::Tensor&, int64_t,
    const std::vector<torch::Tensor>&);
std::vector<torch::Tensor> visible_candidates(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);

torch::Tensor selector_logits(const torch::Tensor& input, const std::vector<torch::Tensor>& p) {
  TORCH_CHECK(!torch::GradMode::is_enabled() && p.size()==6, "Frozen Selector inference only");
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kFloat && input.dim()==2 && input.size(1)==11,
      "Selector input must be CUDA FP32 Nx11");
  c10::cuda::CUDAGuard guard(input.device());
  auto h=at::relu(at::linear(input,p[0],p[1]));
  h=at::relu(at::linear(h,p[2],p[3]));
  return at::linear(h,p[4],p[5]);
}

torch::Tensor compensation_network(const torch::Tensor& input, const torch::Tensor& code,
    const torch::Tensor& reference, const std::vector<torch::Tensor>& p) {
  TORCH_CHECK(!torch::GradMode::is_enabled() && p.size()==21, "Frozen Residual48 inference only");
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kFloat && input.dim()==2 && input.size(1)==51 &&
      code.sizes()==at::IntArrayRef({1,3}) && reference.numel()==input.size(0), "Residual48 input layout");
  c10::cuda::CUDAGuard guard(input.device());
  auto encoded=at::linear(input,p[0],p[1]);
  auto projected=at::linear(code,p[4]);
  auto hidden=at::relu(at::linear(add_relu(encoded,projected),p[2],p[3]));
  std::vector<torch::Tensor> parts;
  for(int j=0;j<4;++j) {
    const int k=5+j*4;
    parts.push_back(at::linear(at::relu(at::linear(hidden,p[k],p[k+1])),p[k+2],p[k+3]));
  }
  return head_pack(parts,reference);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("selector_pack", &selector_pack);
  module.def("selector_logits", &selector_logits);
  module.def("compensation_pack", &compensation_pack);
  module.def("compensation_network", &compensation_network);
  module.def("publish_render", &publish_render);
  module.def("publish_selected", &publish_selected);
  module.def("exact_select", &exact_select, pybind11::arg("scores"), pybind11::arg("eligible"),
      pybind11::arg("k"), pybind11::arg("workspace")=std::vector<torch::Tensor>{});
  module.def("visible_candidates", &visible_candidates);
}
