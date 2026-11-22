from pathlib import Path
import subprocess
D=Path(__file__).resolve().parent
T=Path('/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch')
C=Path('/usr/local/cuda-11.8')
subprocess.run([str(C/'bin/nvcc'),'-O3','-std=c++17','--fmad=false','-D_GLIBCXX_USE_CXX11_ABI=1',
 '-gencode=arch=compute_86,code=sm_86','-I'+str(D/'baseline/source'),'-I'+str(T/'include'),'-I'+str(T/'include/torch/csrc/api/include'),
 str(D/'quaternion_probe.cu'),'-L'+str(T/'lib'),'-Xlinker=-rpath,'+str(T/'lib'),'-Xlinker=--no-as-needed','-ltorch','-ltorch_cpu','-ltorch_cuda','-lc10','-lc10_cuda','-o',str(D/'quaternion_probe')],check=True)
with (D/'quaternion_probe.csv').open('w') as f: subprocess.run([str(D/'quaternion_probe')],stdout=f,check=True)
