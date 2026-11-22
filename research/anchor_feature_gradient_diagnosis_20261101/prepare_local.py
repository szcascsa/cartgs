"""只恢复上一轮被拒绝的研究候选；不改生产源码。"""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
old=R.parent/'residual48_cuda_optimization_20261101'
cpp=(old/'rejected/residual48_training_vjp.cpp').read_text().replace('include/residual48_fusion.h','diagnostic_pack.h').replace('namespace mapping_cuda','namespace diagnostic_cuda')
(R/'diagnostic_pack.cpp').write_text(cpp)
source=(old/'rejected/residual48_training_and_grouped_relu.cu').read_text()
kernels=source[source.index('__global__ void pack('):source.index('torch::Tensor residual48AddReluInference')]
functions=source[source.index('std::vector<torch::Tensor> residual48PackForward'):source.index('torch::Tensor residual48InferencePack')]
cuda='#include "diagnostic_pack.h"\n#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDAException.h>\n#include <c10/cuda/CUDAStream.h>\nnamespace diagnostic_cuda {\nnamespace {\nstruct Features {const float* table[6];float weight[6];int count;};\n'+kernels+functions+'}\n'
(R/'diagnostic_pack.cu').write_text(cuda)
files=['diagnostic_pack.h','diagnostic_pack.cpp','diagnostic_pack.cu']
(R/'recovered_candidate_sha.json').write_text(json.dumps({n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files},indent=2))
