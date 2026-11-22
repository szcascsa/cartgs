"""独立Decode/VJP正确性及成对计时，尚不接入生产Mapper。"""
from pathlib import Path
import shlex
import subprocess

R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');B=Path('/home/prmi/zwj/private-compensation-online-20261022/build')
def flags(path):
    result={}
    for line in path.read_text().splitlines():
        if ' = ' in line:
            k,v=line.split(' = ',1);result[k]=shlex.split(v)
    return result
C=flags(B/'CMakeFiles/anchor_checkpoint_probe.dir/flags.make');U=flags(B/'CMakeFiles/transform_field_cuda.dir/flags.make')
objects=[]
with (R/'phase2_probe_build.log').open('w') as log:
    for name in ['src/anchor_selector_cuda.cpp','tests/anchor_selector/fused_compensation_decode_test.cpp']:
        obj=R/(Path(name).stem+'.o');objects.append(str(obj))
        cmd=['/usr/bin/g++-9','-I'+str(R/'patch')]+C['CXX_DEFINES']+C['CXX_INCLUDES']+C['CXX_FLAGS']+['-c',str(R/'patch'/name),'-o',str(obj)]
        subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    obj=R/'anchor_selector_cuda.cu.o';objects.append(str(obj))
    cmd=['/usr/local/cuda-11.8/bin/nvcc','-I'+str(R/'patch')]+U['CUDA_DEFINES']+U['CUDA_INCLUDES']+U['CUDA_FLAGS']+['-c',str(R/'patch/src/anchor_selector_cuda.cu'),'-o',str(obj)]
    subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    cmd=shlex.split((B/'CMakeFiles/anchor_checkpoint_probe.dir/link.txt').read_text())
    i=cmd.index('CMakeFiles/anchor_checkpoint_probe.dir/tests/anchor_selector/checkpoint_probe.cpp.o');cmd[i:i+1]=objects
    cmd[cmd.index('-o')+1]=str(R/'phase2_probe');subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(R/'phase2_probe'),'/home/prmi/zwj/anchor-selector-r3-performance-20261022/native1',str(R/'phase2')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('PHASE2_PROBE_PASS',flush=True)
