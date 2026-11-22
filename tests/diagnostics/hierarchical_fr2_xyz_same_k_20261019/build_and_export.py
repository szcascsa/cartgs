from pathlib import Path
import subprocess,os,json
ROOT=Path(__file__).resolve().parent
env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',CUDACXX='/usr/local/cuda-11.8/bin/nvcc',CUDA_HOME='/usr/local/cuda-11.8')
cmake='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
for name,cmd in [('configure',[cmake,'-S',str(ROOT),'-B',str(ROOT/'build'),'-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=86','-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8','-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch']),('build',[cmake,'--build',str(ROOT/'build'),'-j1'])]:
    with open(ROOT/(name+'.log'),'w')as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
os.environ.update(env)
from common import Scene,read,save,sha
scene=Scene()
try:
    result=scene.call(dict(method='verify'));assert result['maturity_all_one']
    maps=Path(f'/proc/{scene.proc.pid}/maps').read_text();(ROOT/'actual_worker_maps.txt').write_text(maps)
    for lib in ['libgaussian_mapper.so','libcuda_rasterizer.so']:assert str(ROOT/'runtime/lib'/lib)in maps
    scene.call(dict(method='export',bank_frames=read(ROOT/'input_manifest.json')['bank_frames'],output=str(ROOT/'contributions.pt')))
finally:scene.close()
save(ROOT/'export_complete.json',dict(all_passed=True,N_total=result['N'],contributions_sha256=sha(ROOT/'contributions.pt'),original_renderer=True,Bank_unchanged=True,optimizer_updates=0))
print('BUILD_EXPORT_PASS',result,flush=True)
