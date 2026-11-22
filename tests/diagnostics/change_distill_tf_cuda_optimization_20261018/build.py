"""独立 TU，链接冻结的正式 CUDA archives，保持 --fmad=false。"""
from pathlib import Path
import json
import hashlib
import os
import shutil
import subprocess
import sys

D = Path(__file__).resolve().parent
ROOT = Path('/home/prmi/zwj/mycartgs')
stage = D / sys.argv[1]
version = sys.argv[2] if len(sys.argv) > 2 else 'native'
folder = {'native': 'source', 'profile': 'instrumented', 'verify': 'verify'}[version]
T = ROOT / 'third_party/libtorch-cu118/libtorch'
J = ROOT / 'third_party/install/jsoncpp-1.9.5'
C = Path('/usr/local/cuda-11.8')
common = ['-O3', '-DNDEBUG', '-std=c++17', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=1', '-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA']
inc = ['-I'+str(p) for p in [stage, stage/folder, T/'include', T/'include/torch/csrc/api/include', J/'include', C/'include']]
commands = []
def run(cmd):
    commands.append(cmd)
    subprocess.run(cmd, check=True)
libs = D / 'formal_libraries'
if not libs.exists():
    libs.mkdir()
    for name in ['libselector_cuda.a', 'libtransform_field_cuda.a']:
        shutil.copy2(ROOT / 'build-cuda118-sm86-gcc9' / name, libs / name)
extras = []
for name in ['change_distill_transport.cu']:
    path = stage/'source/src'/name
    if path.exists():
        key=hashlib.sha256(path.read_bytes()+(stage/'source/include/change_distill_transport_cuda.h').read_bytes()+b'CUDA11.8_sm86_ABI1_O3_fmad_false').hexdigest()
        cache=D/'cuda_objects';cache.mkdir(exist_ok=True)
        obj=cache/(key+'.o')
        if not obj.exists():
            run([str(C/'bin/nvcc'), *[x for x in common if x != '-fPIC'], '--fmad=false',
                 '-gencode=arch=compute_86,code=sm_86', '-Xcompiler=-fPIC', *inc,
                 '-c', str(path), '-o', str(obj)])
        extras.append(str(obj))
files = ['change_management.cpp', 'change_distill.cpp', 'transform_field.cpp', 'hexplane_field.cpp',
         'selector_cuda.cpp', 'transform_field_cuda.cpp', 'transform_field_cuda_optimizer.cpp', 'protection_profiler.cpp']
combined = stage / ('combined_'+version+'.cpp')
text = '#include "perf_scope.h"\n'
if version == 'verify': text += '#include "verify_scope.h"\n'
for name in files: text += '#include "'+str(stage/folder/'src'/name)+'"\n'
text += '#include "'+str(stage/('benchmark_verify.cpp' if version=='verify' else 'benchmark.cpp'))+'"\n'
combined.write_text(text)
obj = stage / (version+'.o')
run(['/usr/bin/g++', *common, *inc, '-c', str(combined), '-o', str(obj)])
cmd = ['/usr/bin/g++', '-Wl,--export-dynamic', str(obj), *extras,
       str(libs/'libtransform_field_cuda.a'), str(libs/'libselector_cuda.a')]
for p in [T/'lib', C/'lib64', J/'lib']: cmd += ['-L'+str(p), '-Wl,-rpath,'+str(p)]
cmd += ['-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10', '-lc10_cuda', '-lcudart', '-lcublas', '-lcublasLt', '-lnvToolsExt', '-ljsoncpp', '-o', str(stage/('benchmark_'+version))]
run(cmd)
(stage/'raw'/('build_'+version+'.json')).write_text(json.dumps(commands, indent=2))
print('BUILD PASS', stage.name, version, flush=True)
