#include <torch/extension.h>
void register_pending_raster(pybind11::module_&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){register_pending_raster(m);}
