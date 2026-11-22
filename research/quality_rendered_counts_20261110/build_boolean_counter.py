"""独立诊断构建：只统计是否贡献，保留原像素合成；不改生产源码或动态库。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'dual_view_li_best_full_20261107/source/cuda_rasterizer'
COPY=ROOT/'boolean_counter_source'
BUILD=ROOT/'boolean_counter_build'
COPY.mkdir(exist_ok=True);BUILD.mkdir(exist_ok=True)
for p in SOURCE.iterdir():
    if p.is_file():shutil.copy2(p,COPY/p.name)
p=COPY/'forward.cu'
text=p.read_text()
old='''      if (collect_importance) {
        const int gaussian_id = collected_id[j];
        atomicAdd(frame_importance + gaussian_id, (collect_importance == 2 ? alpha : con_o.w) * T);
        if (collect_importance == 1) atomicAdd(contribution_count + gaussian_id, 1);
      }'''
new='''      if (collect_importance == 1) {
        const int gaussian_id = collected_id[j];
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(contribution_count[gaussian_id]);
        if (flag.load(cuda::memory_order_relaxed) == 0)
          flag.store(1, cuda::memory_order_relaxed);
      }'''
assert text.count(old)==1
p.write_text('#include <cuda/atomic>\n'+text.replace(old,new))
binding=ROOT/'boolean_counter_binding.cpp'
binding.write_text('''#include <torch/extension.h>
#include "rasterize_points.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 pybind11::enum_<RasterizeMode>(m,"Mode").value("Inference",RasterizeMode::Inference);
 m.def("render_with_counts",&RasterizeGaussiansCUDAWithImportance);
}
''')
os.environ.update(CUDA_HOME='/usr/local/cuda-11.8',TORCH_CUDA_ARCH_LIST='8.6',
                  CXX='/usr/bin/g++-9',CC='/usr/bin/gcc-9',MAX_JOBS='2')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
from torch.utils.cpp_extension import load
flags=['-DRasterizeMode=BooleanCountMode','-DCudaRasterizer=BooleanCountRasterizer',
       '-DFORWARD=BOOLEAN_COUNT_FORWARD','-DBACKWARD=BOOLEAN_COUNT_BACKWARD',
       '-Danchor_si=boolean_count_anchor_si','-Drenderer_workspace=boolean_count_workspace']
extension=load(name='boolean_render_counts',
    sources=[str(COPY/n) for n in ['rasterizer_impl.cu','forward.cu','backward.cu',
                                  'rasterize_points.cu','adam.cu','conv.cu']]+[str(binding)],
    extra_include_paths=[str(COPY)],extra_cflags=flags,
    extra_cuda_cflags=flags,build_directory=str(BUILD),verbose=True)
(ROOT/'BOOLEAN_COUNTER_BUILD.json').write_text(json.dumps(dict(path=extension.__file__,
    source=str(SOURCE),source_forward_sha256=hashlib.sha256((SOURCE/'forward.cu').read_bytes()).hexdigest(),
    diagnostic_forward_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    change='GI浮点累加及像素次数替换为原位置的单调原子0/1标记，仅用于统计独立贡献点数'),ensure_ascii=False,indent=2))
print('BOOLEAN_COUNTER_READY',extension.__file__,flush=True)
