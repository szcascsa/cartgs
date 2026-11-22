#include <torch/extension.h>
#include "include/selector_cuda.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("network",[](const std::vector<torch::Tensor>& x,const std::vector<torch::Tensor>& p,int strategy){
    return selector_cuda::network(x,p,selector_cuda::Candidate::CublasStrict,static_cast<selector_cuda::BackwardStrategy>(strategy));
  },pybind11::arg("inputs"),pybind11::arg("parameters"),pybind11::arg("strategy")=0);
  m.def("stages",[](const std::vector<torch::Tensor>& x,const std::vector<torch::Tensor>& p){
    return selector_cuda::forward(x,p,selector_cuda::Candidate::CublasStrict);
  });
}
