#include <torch/extension.h>

torch::Tensor deployment_view(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, double, double);
std::vector<torch::Tensor> deployment_selector(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const std::vector<torch::Tensor>&, double, bool, int64_t);
std::vector<torch::Tensor> deployment_compensate(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    const torch::Tensor&, const torch::Tensor&, const std::vector<double>&,
    const std::vector<double>&, bool, int64_t, bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("view", &deployment_view);
  module.def("selector", &deployment_selector);
  module.def("compensate", &deployment_compensate);
}
