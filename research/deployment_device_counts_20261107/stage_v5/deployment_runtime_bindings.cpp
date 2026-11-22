#include <torch/extension.h>
void bind_deployment_prepare(pybind11::module_&);
int64_t deployment_select_scan_bytes(int64_t);
void deployment_select_device(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
torch::Tensor deployment_select(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const torch::Tensor&,const std::vector<torch::Tensor>&);

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
  bind_deployment_prepare(module);
  module.def("view", &deployment_view);
  module.def("selector", &deployment_selector);
  module.def("compensate", &deployment_compensate);
  module.def("select", &deployment_select);
  module.def("_select_device", &deployment_select_device);
  module.def("_scan_bytes", &deployment_select_scan_bytes);
}
