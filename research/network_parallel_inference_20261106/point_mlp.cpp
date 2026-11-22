#include <torch/extension.h>
torch::Tensor selector(torch::Tensor,std::vector<torch::Tensor>);
torch::Tensor compensator(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,std::vector<torch::Tensor>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("selector",&selector);m.def("compensator",&compensator);}
