"""复制上一轮最终源码/构建；不覆盖上一轮证据。"""
import json
from pathlib import Path
import shutil
import subprocess
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023');O=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022')
if not (R/'source').exists():
    shutil.copytree(O/'source',R/'source',symlinks=True);shutil.copytree(O/'build',R/'build',symlinks=True)
    for p in (R/'build').rglob('*'):
        if not p.is_file() or p.is_symlink() or p.suffix not in ['.txt','.cmake','.make','.ninja','.internal','.ts','.d']:continue
        try:t=p.read_text()
        except UnicodeDecodeError:continue
        if str(O) in t:p.write_text(t.replace(str(O),str(R)))
for p in (R/'patch').rglob('*'):
    if p.is_file():
        dest=R/'source'/p.relative_to(R/'patch');dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or dest.read_bytes()!=p.read_bytes():shutil.copy2(p,dest)
commands=[['/home/prmi/.local/bin/cmake','-S',str(R/'source'),'-B',str(R/'build')],
    ['/home/prmi/.local/bin/cmake','--build',str(R/'build'),'-j2','--target',*json.loads((R/'targets.json').read_text())]]
for i,c in enumerate(commands):
    with (R/('configure.log' if i==0 else 'compile.log')).open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
print('BUILD_PASS',flush=True)
