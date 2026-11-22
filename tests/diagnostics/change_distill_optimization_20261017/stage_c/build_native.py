"""在服务器独立目录编译冻结的本地实现，不覆盖项目 build/lib。"""
from pathlib import Path
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import torch

D = Path(__file__).resolve().parent
CUDA = Path('/usr/local/cuda-11.8')
T = Path('/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch')
JSONCPP = Path('/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5')
OBJECTS = D / 'objects_abi1'
OBJECTS.mkdir(exist_ok=True)
commands = []
includes = ['-I' + str(T / 'include'), '-I' + str(T / 'include/torch/csrc/api/include'),
            '-I' + str(CUDA / 'include'), '-I' + str(JSONCPP / 'include'), '-I' + str(D)]
common = ['-O3', '-DNDEBUG', '-std=c++17', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=1',
          '-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA']


def execute(args):
    commands.append(args)
    print(shlex.join(args), flush=True)
    subprocess.run(args, check=True, cwd=D)


cuda_files = ['selector_cuda.cu', 'transform_field_cuda.cu', 'transform_field_cuda_neural.cu',
              'transform_field_cuda_regularization.cu', 'transform_field_cuda_optimizer.cu']


def build_cuda(name):
    output = OBJECTS / (name + '.o')
    if output.exists():
        return output
    execute([str(CUDA / 'bin/nvcc'), *[x for x in common if x != '-fPIC'], '-gencode=arch=compute_86,code=sm_86',
             '-Xcompiler=-fPIC', *includes, '-I' + str(D / 'source'),
             '-c', str(D / 'source/src' / name), '-o', str(output)])
    return output


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    cuda_objects = list(pool.map(build_cuda, cuda_files))
cpp_files = ['change_management.cpp', 'change_distill.cpp', 'transform_field.cpp', 'hexplane_field.cpp',
             'selector_cuda.cpp', 'transform_field_cuda.cpp', 'transform_field_cuda_optimizer.cpp', 'protection_profiler.cpp']
versions=[('native', 'source')]
if sys.argv[1:]==['instrumented']:
    versions=versions[1:]
for version, folder in versions:
    combined = D / ('combined_' + version + '.cpp')
    source = '#include "perf_scope.h"\n'
    for name in cpp_files:
        source += '#include "' + str(D / folder / 'src' / name) + '"\n'
    source += '#include "' + str(D / 'benchmark.cpp') + '"\n'
    combined.write_text(source)
    obj = OBJECTS / (version + '.o')
    execute(['/usr/bin/g++', *common, *includes, '-I' + str(D / folder),
             '-c', str(combined), '-o', str(obj)])
    execute(['/usr/bin/g++', '-Wl,--export-dynamic', str(obj), *map(str, cuda_objects),
             '-L' + str(T / 'lib'), '-L' + str(CUDA / 'lib64'),
             '-L' + str(JSONCPP / 'lib'), '-Wl,-rpath,' + str(JSONCPP / 'lib'),
             '-Wl,-rpath,' + str(T / 'lib'), '-Wl,-rpath,' + str(CUDA / 'lib64'),
             '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10', '-lc10_cuda', '-lcudart',
             '-lcublas', '-lcublasLt', '-lnvToolsExt', '-ljsoncpp', '-o', str(D / ('benchmark_' + version + '.tmp'))])
    os.replace(D / ('benchmark_' + version + '.tmp'), D / ('benchmark_' + version))
(D / 'raw/build_commands.json').write_text(json.dumps(commands, indent=2))
print('BUILD COMPLETE', flush=True)
