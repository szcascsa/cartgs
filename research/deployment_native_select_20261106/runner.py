"""独立试验串行执行；任一步失败立即停止，不与自身其他GPU测试并发。"""
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def run(name,*args):
    print('START',name,*args,flush=True)
    subprocess.run([sys.executable,str(HERE/(name+'.py')),*map(str,args)],check=True)
    print('PASS',name,*args,flush=True)


def main():
    assert json.loads((ROOT/'reports/SELECTION_VALIDATION.json').read_text())['passed']
    run('selection_benchmark')
    run('validate')
    run('extra_checks')
    run('trajectory')
    for repeat in [1,2]:run('benchmark',repeat)
    for backend in ['previous_cpp','native_select']:run('memory',backend)
    print('VALIDATION_BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':main()
