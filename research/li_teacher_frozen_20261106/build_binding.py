"""在独立目录原样编译CUDA源码，适配Python Torch ABI，不修改Kernel。"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import torch
os.environ.setdefault('CUDA_HOME','/usr/local/cuda-11.8')
from torch.utils.cpp_extension import load

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
started=time.monotonic();os.environ['MAX_JOBS']='2'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
if Path('/usr/bin/gcc-9').exists():os.environ['CC']='/usr/bin/gcc-9'
if Path('/usr/bin/g++-9').exists():os.environ['CXX']='/usr/bin/g++-9'
# Native LibTorch uses ABI1, installed Python Torch ABI0: recompile unchanged
# rasterizer sources for this interpreter, rather than mixing incompatible libs.
cuda=['rasterizer_impl.cu','forward.cu','backward.cu','rasterize_points.cu','adam.cu','conv.cu']
extension=load(name='li_teacher_frozen',sources=[str(HERE/'teacher_binding.cpp')]+[str(ROOT/'source/cuda_rasterizer'/n) for n in cuda],
    extra_include_paths=[str(ROOT/'source'),str(ROOT/'source/cuda_rasterizer'),'/usr/include/eigen3'],
    extra_cflags=['-O1'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr'],
    build_directory=str(ROOT/'build'),verbose=True)
value=dict(passed=True,seconds=time.monotonic()-started,path=extension.__file__,
    sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
    cuda_sources_unchanged=True,torch_cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    cuda_source_hashes={n:hashlib.sha256((ROOT/'source/cuda_rasterizer'/n).read_bytes()).hexdigest() for n in cuda})
(ROOT/'reports/BUILD.json').write_text(json.dumps(value,indent=2));print(value,flush=True)
