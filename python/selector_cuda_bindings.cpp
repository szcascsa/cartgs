#include <torch/extension.h>
#include "include/selector_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("network", [](const std::vector<torch::Tensor>& inputs,
                            const std::vector<torch::Tensor>& parameters) {
    return selector_cuda::network(inputs, parameters, selector_cuda::Candidate::CublasStrict);
  });
}
