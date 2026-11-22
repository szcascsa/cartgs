#include <torch/extension.h>
void bind_deployment_prepare(pybind11::module_&);
std::vector<torch::Tensor> deployment_fused_candidates(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
void deployment_finish_counts(const torch::Tensor&,const torch::Tensor&);
torch::Tensor frustum_planes(const torch::Tensor&);
torch::Tensor frustum_test(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
std::vector<torch::Tensor> visible_candidates(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
torch::Tensor deployment_radix_select(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const std::vector<torch::Tensor>&,const torch::Tensor&);
torch::Tensor deployment_radix_single(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const std::vector<torch::Tensor>&,const torch::Tensor&);
torch::Tensor deployment_radix_fused(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const std::vector<torch::Tensor>&,const torch::Tensor&);
void deployment_select_device(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
torch::Tensor deployment_select(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const torch::Tensor&,const std::vector<torch::Tensor>&);

torch::Tensor deployment_view(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, double, double);
std::vector<torch::Tensor> deployment_selector(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const std::vector<torch::Tensor>&, double, bool, int64_t);
std::vector<torch::Tensor> deployment_selector_tiled(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const std::vector<torch::Tensor>&,double,bool,int64_t);
std::vector<torch::Tensor> deployment_compensate(const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    const torch::Tensor&, const torch::Tensor&, const std::vector<double>&,
    const std::vector<double>&, bool, int64_t, bool);
std::vector<torch::Tensor> deployment_compensate_tiled(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<double>&,
    const std::vector<double>&,bool,int64_t,bool,int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  bind_deployment_prepare(module);
  module.def("view", &deployment_view);
  module.def("selector", &deployment_selector);
  module.def("_selector_tiled", &deployment_selector_tiled);
  module.def("compensate", &deployment_compensate);
  module.def("_compensate_tiled", &deployment_compensate_tiled);
  module.def("select", &deployment_select);
  module.def("_select_device", &deployment_select_device);
  module.def("_radix_select", &deployment_radix_select);
  module.def("_radix_single", &deployment_radix_single);
  module.def("_radix_fused", &deployment_radix_fused);
  module.def("_fused_candidates", &deployment_fused_candidates);
  module.def("_finish_counts", &deployment_finish_counts);
  module.def("_frustum_planes", &frustum_planes);
  module.def("_frustum_test", &frustum_test);
  module.def("_visible_candidates", &visible_candidates);
}
