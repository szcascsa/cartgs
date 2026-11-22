#include <torch/extension.h>
#include "include/selector_cuda.h"
#include <ATen/cuda/CUDAContext.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("stages", [](const std::vector<torch::Tensor>& inputs,
                          const std::vector<torch::Tensor>& parameters) {
    return selector_cuda::forward(inputs, parameters, selector_cuda::Candidate::CublasStrict);
  });
  module.def("ratio_first_gemm", [](const torch::Tensor& x, const torch::Tensor& w,
                                    const torch::Tensor& bias, bool transpose) {
    auto y=bias.expand({x.size(0),32}).clone();
    const float alpha=1,beta=1;
    TORCH_CUDABLAS_CHECK(cublasSgemm(at::cuda::getCurrentCUDABlasHandle(),
        transpose?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,32,x.size(0),1,&alpha,
        w.data_ptr<float>(),transpose?1:32,x.data_ptr<float>(),1,&beta,y.data_ptr<float>(),32));
    return y.relu();
  });
}
