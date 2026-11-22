"""在隔离源码上编译现有生命周期/epoch与 cache 测试。"""
from pathlib import Path
import json
import subprocess
import sys

D = Path(__file__).resolve().parent
stage = D / sys.argv[1]
commands = json.loads((stage / 'raw/build_commands.json').read_text())
compile_command = next(c for c in commands if '-c' in c and c[0].endswith('g++'))
link_command = commands[-1]
combined = (stage / 'combined_native.cpp').read_text()
combined = combined[:combined.index('#include "' + str(stage / 'benchmark.cpp'))]
for backend, definitions in [('cpu', []), ('cuda', ['-DCARTGS_TEST_CUDA_BACKEND', '-DCARTGS_TEST_SELECTOR_CUDA'])]:
    target = stage / ('lifecycle_' + backend + '.cpp')
    target.write_text(combined + '#include "' + str(D / 'tests/change_distill_test.cpp') + '"\n')
    obj = stage / ('lifecycle_' + backend + '.o')
    command = compile_command.copy()
    command[command.index('-c') + 1] = str(target)
    command[command.index('-o') + 1] = str(obj)
    command += definitions + ['-I' + str(D / 'tests')]
    subprocess.run(command, check=True)
    command = link_command.copy()
    command[2] = str(obj)
    command[-1] = str(stage / ('lifecycle_' + backend))
    subprocess.run(command, check=True)
    subprocess.run([command[-1]], check=True)
    print(backend + ' lifecycle PASS', flush=True)
target = stage / 'cache_test.cpp'
target.write_text('#include "' + str(D / 'tests/frozen_teacher_target_cache_test.cpp') + '"\n')
command = compile_command.copy()
command[command.index('-c') + 1] = str(target)
command[command.index('-o') + 1] = str(stage / 'cache_test.o')
subprocess.run(command, check=True)
command = link_command.copy()
command[2] = str(stage / 'cache_test.o')
command[-1] = str(stage / 'cache_test')
subprocess.run(command, check=True)
subprocess.run([command[-1]], check=True)
print('cache PASS', flush=True)
