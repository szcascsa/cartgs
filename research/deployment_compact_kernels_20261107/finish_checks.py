from pathlib import Path
import subprocess
import json
import time


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'
CUDA=Path('/usr/local/cuda-11.8/bin')


def main():
    identity=json.loads((ROOT/'reports/FINAL_PROCESS.json').read_text())
    process=Path(f'/proc/{identity["pid"]}/stat')
    while process.exists() and process.read_text().split()[2]!='Z' and process.read_text().split()[21]==identity['start_ticks']:
        time.sleep(5)
    states=json.loads((ROOT/'reports/RUNNER_STATUS.json').read_text())
    assert len(states)==20 and all(x['state']=='complete' for x in states),states[-1]
    tasks=[
        ('cli',[PYTHON,'-u',str(HERE/'cli_smoke.py')]),
        ('sanitizer_tum',[str(CUDA/'compute-sanitizer'),'--tool','memcheck','--error-exitcode','97',PYTHON,'-u',str(HERE/'smoke.py'),'tum']),
        ('sanitizer_replica',[str(CUDA/'compute-sanitizer'),'--tool','memcheck','--error-exitcode','97',PYTHON,'-u',str(HERE/'smoke.py'),'replica']),
        ('nsight',[str(CUDA/'nsys'),'profile','--force-overwrite=true','--trace=cuda,nvtx,osrt',
                    '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                    '--output='+str(ROOT/'reports/PREPARE_PROFILE'),PYTHON,'-u',str(HERE/'benchmark.py'),'1','--profile']),
        ('export',[str(CUDA/'nsys'),'export','--type=sqlite','--force-overwrite=true',
                    '--output='+str(ROOT/'reports/PREPARE_PROFILE.sqlite'),str(ROOT/'reports/PREPARE_PROFILE.nsys-rep')]),
        ('analyze',[PYTHON,'-u',str(HERE/'profile_graph.py')])]
    states=[]
    for name,command in tasks:
        row=dict(task=name,state='running',start=time.time())
        states.append(row)
        print('START',name,flush=True)
        with (ROOT/'logs'/f'{name}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=process.pid
            (ROOT/'reports/FINISH_STATUS.json').write_text(json.dumps(states,indent=2))
            code=process.wait()
        row.update(state='complete' if code==0 else 'failed',exit=code,end=time.time())
        (ROOT/'reports/FINISH_STATUS.json').write_text(json.dumps(states,indent=2))
        print('END',name,code,flush=True)
        if code:raise SystemExit(code)
    print('ALL_PASS',flush=True)


if __name__=='__main__':main()
