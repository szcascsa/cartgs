"""Phase1先独立链接审计库，生产大目标在另一构建进程中编译。"""
from pathlib import Path
import shlex
import subprocess

R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022')
B=Path('/home/prmi/zwj/private-compensation-online-20261022/build')
T=B/'CMakeFiles/anchor_checkpoint_probe.dir';flags={}
for line in (T/'flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
objects=[]
with (R/'phase1_probe_build.log').open('w') as log:
    for name in ['src/mapping_cuda_optimizer.cpp','tests/anchor_selector/fused_mapping_adam_test.cpp']:
        obj=R/(Path(name).stem+'.o');objects.append(str(obj))
        cmd=['/usr/bin/g++-9','-I'+str(R/'patch')]+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-c',str(R/'patch'/name),'-o',str(obj)]
        subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    cmd=shlex.split((T/'link.txt').read_text())
    i=cmd.index('CMakeFiles/anchor_checkpoint_probe.dir/tests/anchor_selector/checkpoint_probe.cpp.o');cmd[i:i+1]=objects
    cmd[cmd.index('-o')+1]=str(R/'phase1_probe')
    subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(R/'phase1_probe'),'/home/prmi/zwj/anchor-selector-r3-performance-20261022/native1',str(R/'phase1')],
                   stdout=log,stderr=subprocess.STDOUT,check=True)
print('PHASE1_PROBE_PASS',flush=True)
