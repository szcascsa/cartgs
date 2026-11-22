"""使用已验证构建的flags/link编译独立benchmark，不改CMake或生产对象。"""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
BUILD = Path('/home/prmi/zwj/private-compensation-online-20261022/build')
target = BUILD/'CMakeFiles/anchor_checkpoint_probe.dir'
flags = {}
for line in (target/'flags.make').read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
cmd = ['/usr/bin/g++-9'] + flags['CXX_DEFINES'] + flags['CXX_INCLUDES'] + flags['CXX_FLAGS']
cmd += ['-c', str(ROOT/'native_benchmark.cpp'), '-o', str(ROOT/'native_benchmark.o')]
with (ROOT/'build.log').open('w') as log:
    subprocess.run(cmd, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
    link = shlex.split((target/'link.txt').read_text())
    link[link.index('CMakeFiles/anchor_checkpoint_probe.dir/tests/anchor_selector/checkpoint_probe.cpp.o')] = str(ROOT/'native_benchmark.o')
    link[link.index('-o')+1] = str(ROOT/'native_benchmark')
    subprocess.run(link, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
(ROOT/'build_commands.json').write_text(json.dumps(dict(compile=cmd, link=link), indent=2))
print('BUILT', flush=True)
