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

torch::Tensor deployment_selector_prefix(const torch::Tensor&,const torch::Tensor&);
std::vector<torch::Tensor> deployment_selector_impl(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,
    double,bool,int64_t,bool,const torch::Tensor&);
torch::Tensor deployment_view_masked(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double,const torch::Tensor&);

std::vector<torch::Tensor> deployment_tensor_selector(const torch::Tensor& physical,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& view,const torch::Tensor& noise,
    const std::vector<torch::Tensor>& weights,double ratio,bool local,int64_t schema,
    const torch::Tensor& prefix,const std::vector<torch::Tensor>& packed,int64_t mode,int64_t warps);

std::vector<torch::Tensor> deployment_tensor_compensate(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus,
    const std::vector<torch::Tensor>& packed,int64_t mode,int64_t warps,const torch::Tensor& shared_status,int64_t projection_mode);

std::vector<torch::Tensor> deployment_tensor_pack(const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,const torch::Tensor&,int64_t,int64_t);
std::vector<torch::Tensor> preparation_frustum_debug(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t);
torch::Tensor preparation_camera(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
std::vector<torch::Tensor> preparation_camera_parts(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  bind_deployment_prepare(module);
  module.def("_preparation_frustum",&preparation_frustum_debug);
  module.def("_preparation_camera",&preparation_camera,pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("_preparation_camera_parts",&preparation_camera_parts,pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("_tensor_pack",&deployment_tensor_pack);
  module.def("_tensor_selector",&deployment_tensor_selector);
  module.def("_tensor_compensate",&deployment_tensor_compensate);
  module.def("_selector_prefix", &deployment_selector_prefix);
  module.def("_selector_prefixed", &deployment_selector_impl);
  module.def("_view_masked", &deployment_view_masked);
  module.def("view", &deployment_view);
  module.def("selector", &deployment_selector);
  module.def("_selector_tiled", &deployment_selector_tiled);
  module.def("compensate", &deployment_compensate);
  module.def("_compensate_tiled", &deployment_compensate_tiled);
  module.def("select", &deployment_select);
  module.def("_select_device", &deployment_select_device);
  module.def("_radix_select", &deployment_radix_select);
  module.def("_radix_fused", &deployment_radix_fused);
  module.def("_fused_candidates", &deployment_fused_candidates);
  module.def("_finish_counts", &deployment_finish_counts);
  module.def("_frustum_planes", &frustum_planes);
  module.def("_frustum_test", &frustum_test);
  module.def("_visible_candidates", &visible_candidates);
}
