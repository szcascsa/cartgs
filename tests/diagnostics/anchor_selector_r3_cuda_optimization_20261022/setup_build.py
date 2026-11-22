"""隔离源码/构建，复用审计runtime，不覆盖Legacy或审计证据。"""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022')
OLD=Path('/home/prmi/zwj/private-compensation-online-20261022')
SOURCE=ROOT/'source';BUILD=ROOT/'build';ROOT.mkdir(exist_ok=True)
if not SOURCE.exists():
    shutil.copytree(OLD/'source',SOURCE,symlinks=True)
    shutil.copytree(OLD/'build',BUILD,symlinks=True)
    for p in BUILD.rglob('*'):
        if not p.is_file() or p.is_symlink() or p.suffix not in ['.txt','.cmake','.make','.ninja','.internal','.ts','.d']:continue
        try:text=p.read_text()
        except UnicodeDecodeError:continue
        if str(OLD) in text:p.write_text(text.replace(str(OLD),str(ROOT)))
patch=ROOT/'patch'
for p in patch.rglob('*'):
    if p.is_file():
        target=SOURCE/p.relative_to(patch);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
env=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'])
targets=json.loads((ROOT/'targets.json').read_text()) if (ROOT/'targets.json').exists() else ['fused_mapping_adam_test','tum_rgbd']
commands=[['/home/prmi/.local/bin/cmake','-S',str(SOURCE),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86',
    '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
    '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch',
    '-DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4',
    '-Djsoncpp_DIR=/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp',
    '-DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/eigen-3.3.7/build-cmake','-DCARTGS_BUILD_TESTS=ON','-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON'],
    ['/home/prmi/.local/bin/cmake','--build',str(BUILD),'-j2','--target',*targets]]
for i,cmd in enumerate(commands):
    with (ROOT/('configure.log' if i==0 else 'compile.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(ROOT/'build_commands.json').write_text(json.dumps(commands,indent=2))
print('BUILD_PASS',targets,flush=True)
