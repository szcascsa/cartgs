"""串行测量，避免本轮自身任务争用 GPU；保留两轮原始结果。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
options = ['--fused-decode'] if '--fused-decode' in sys.argv else []
tag = 'fused_' if options else ''
if '--prefix-cache' in sys.argv:options += ['--prefix-cache'];tag = 'prefix_'


def run(command, log):
    with (root/log).open('w') as out:
        subprocess.run([str(value) for value in command], stdout=out, stderr=subprocess.STDOUT,
                       check=True, env=os.environ.copy())


run([sys.executable, root/'pipeline.py', 'check']+options, tag+'pipeline_check.log')
run([sys.executable, root/'pipeline.py', 'bench']+options, tag+'benchmark_final1.log')
for suffix in ('json', 'csv'):shutil.copyfile(root/('BENCHMARK.'+suffix), root/('BENCHMARK_'+tag+'final1.'+suffix))
run([sys.executable, root/'pipeline.py', 'bench']+options, tag+'benchmark_final2.log')
for suffix in ('json', 'csv'):shutil.copyfile(root/('BENCHMARK.'+suffix), root/('BENCHMARK_'+tag+'final2.'+suffix))
run([sys.executable, root/'microbench.py']+options, tag+'microbenchmark.log')
if options:shutil.copyfile(root/'MICROBENCHMARK.json', root/'FUSED_MICROBENCHMARK.json')
nsys = '/usr/local/cuda-11.8/bin/nsys'
run([nsys, 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
    '--cuda-graph-trace=node', '--force-overwrite=true', '-o', root/(tag+'final_nodes'),
    sys.executable, root/'pipeline.py', 'profile']+options, tag+'nsight_final.log')
run([nsys, 'export', '--type=sqlite', '--force-overwrite=true', '-o', root/(tag+'final_nodes.sqlite'),
    root/(tag+'final_nodes.nsys-rep')], tag+'nsight_final_export.log')
run([sys.executable, root/'analyze_nsight.py', root/(tag+'final_nodes.sqlite')], tag+'nsight_final_analysis.log')
(root/(tag+'PERFORMANCE_COMPLETE.json')).write_text(json.dumps(dict(passed=True))+'\n')
