"""只记录环境、构建输入和指定生产文件哈希，不保存凭据或完整环境变量。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

dest = Path(__file__).resolve().parent
root = dest.parents[2]
production = Path('/home/prmi/zwj/mycartgs')


def command(args):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


files = ['improvements/selector/gumbel_network.h', 'include/model_snapshot.h',
         'include/selector_cuda.h', 'src/selector_cuda.cpp', 'src/selector_cuda.cu',
         'tests/diagnostics/cuda_selector_implementation_20261015/CMakeLists.txt',
         'tests/diagnostics/cuda_selector_implementation_20261015/selector_parity.cpp',
         'tests/diagnostics/cuda_selector_implementation_20261015/network_benchmark.cpp']
production_files = ['improvements/selector/gumbel_network.h', 'include/gaussian_mapper.h',
                    'src/gaussian_mapper.cpp', 'cfg/selector/selector_config.yaml',
                    'cfg/selector/fast.yaml', 'bin/tum_rgbd', 'lib/libgaussian_mapper.so']
payload = dict(
    gpu=command(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap,memory.used,utilization.gpu', '--format=csv,noheader']),
    processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']),
    nvcc=command(['/usr/local/cuda-11.8/bin/nvcc', '--version']),
    gcc=command(['/usr/bin/g++-9', '--version']),
    nsys=command(['nsys', '--version']), python=sys.version,
    python_torch=torch.__version__, python_torch_cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    native_torch_version=(production/'third_party/libtorch-cu118/libtorch/include/torch/csrc/api/include/torch/version.h').read_text(),
    native_build_flags=(dest/'build-candidates/CMakeFiles/selector_parity.dir/flags.make').read_text(),
    allow_tf32_cublas=torch.backends.cuda.matmul.allow_tf32,
    environment={name: os.environ.get(name) for name in
                 ['CUDA_LAUNCH_BLOCKING', 'CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'DISABLE_ADDMM_CUDA_LT']},
    compiled_sources={name: digest(root/name) for name in files},
    production_files={name: digest(production/name) for name in production_files},
    binaries={name: digest(dest/'build-candidates'/name) for name in ['selector_parity', 'network_benchmark']},
)
name = sys.argv[1] if len(sys.argv) > 1 else 'environment.json'
(dest/name).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'output': name, 'gpu': payload['gpu'], 'processes': payload['processes']}, ensure_ascii=False))
