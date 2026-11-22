"""单阶段先独立编译/验收，再整合生产构建。"""
from pathlib import Path
import shlex
import subprocess
import sys
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023');O=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');B=O/'build'
phase=sys.argv[1] if len(sys.argv)>1 else '4'
def flags(path):return {k:shlex.split(v) for line in path.read_text().splitlines() if ' = ' in line for k,v in [line.split(' = ',1)]}
C=flags(B/'CMakeFiles/fused_anchor_input_test.dir/flags.make');U=flags(B/'CMakeFiles/transform_field_cuda.dir/flags.make')
V=flags(B/'CMakeFiles/cuda_rasterizer.dir/flags.make')
files={'4':['src/anchor_stats_cuda.cu','tests/anchor_selector/fused_anchor_stats_test.cpp'],
       '5':['src/selector_sampling_cuda.cpp','src/selector_sampling_cuda.cu','tests/anchor_selector/fused_selector_sampling_test.cpp'],
       '6':['cuda_rasterizer/rasterize_points.cu','tests/anchor_selector/renderer_workspace_test.cpp'],
       '7':['cuda_rasterizer/conv.cu','tests/anchor_selector/ssim_backward_test.cpp'],
       '8':['tests/diagnostics/anchor_selector_r3_cuda_phase2_20261023/anchor_stats_threshold_probe.cpp']}[phase]
objects=[]
with (R/('phase'+phase+'_compile.log')).open('w') as log:
    for name in files:
        obj=R/(Path(name).name+'.o');objects.append(str(obj))
        if obj.exists() and obj.stat().st_mtime>(R/'patch'/name).stat().st_mtime:continue
        if name.endswith('.cu'):
            options=V if phase in ['6','7'] else U
            cmd=['/usr/local/cuda-11.8/bin/nvcc','-I'+str(R/'patch'),'-I'+str(R/'patch/cuda_rasterizer'),'-I'+str(O/'source/cuda_rasterizer')]+options['CUDA_DEFINES']+options['CUDA_INCLUDES']+options['CUDA_FLAGS']
        else:cmd=['/usr/bin/g++-9','-I'+str(R/'patch'),'-I'+str(O/'source/improvements/selector')]+C['CXX_DEFINES']+C['CXX_INCLUDES']+C['CXX_FLAGS']
        subprocess.run(cmd+['-c',str(R/'patch'/name),'-o',str(obj)],cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    if phase=='8':objects.append(str(R/'anchor_stats_cuda.cu.o'))
    link=shlex.split((B/'CMakeFiles/fused_anchor_input_test.dir/link.txt').read_text());idx=next(i for i,s in enumerate(link) if s.endswith('fused_anchor_input_test.cpp.o'));link[idx:idx+1]=objects
    link[link.index('-o')+1]=str(R/('phase'+phase+'_probe'));link.append('/usr/local/cuda-11.8/lib64/libnvToolsExt.so')
    subprocess.run(link,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    cmd=[str(R/('phase'+phase+'_probe')),'/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply',
        '/home/prmi/zwj/anchor-selector-r3-performance-20261022/native1/Full_parity.pt',str(R/('phase'+phase))]
    subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PHASE'+phase+'_PASS',flush=True)
