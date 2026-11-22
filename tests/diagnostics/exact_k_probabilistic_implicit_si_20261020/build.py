from pathlib import Path
import subprocess,os
ROOT=Path(__file__).resolve().parent
cmake='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
env=dict(os.environ,CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',CUDACXX='/usr/local/cuda-11.8/bin/nvcc',CUDA_HOME='/usr/local/cuda-11.8')
for name,cmd in [('configure',[cmake,'-S',str(ROOT),'-B',str(ROOT/'build'),'-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=86','-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8','-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch']),('compile',[cmake,'--build',str(ROOT/'build'),'-j1'])]:
    with open(ROOT/(name+'.log'),'w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
