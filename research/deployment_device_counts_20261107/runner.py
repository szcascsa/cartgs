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
    run('device_selection')
    run('validate')
    run('large_pipeline')
    run('extra_checks')
    run('trajectory')
    for repeat in [1,2]:run('benchmark',repeat)
    for backend in ['previous_cpp','device_counts']:run('memory',backend)
    print('VALIDATION_BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':main()
