from pathlib import Path
import subprocess
import json
import time
import threading
import csv


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'
CUDA='/usr/local/cuda-11.8/bin/'
stop=threading.Event()


def monitor():
    with (ROOT/'reports/GPU_MONITOR.jsonl').open('w') as f:
        while not stop.is_set():
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits'],text=True).strip()
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits'],text=True).strip()
            f.write(json.dumps(dict(utc=time.time(),gpu=gpu,processes=processes))+'\n');f.flush()
            stop.wait(2)


def main():
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    tasks=[('validate',[PYTHON,'-u',str(HERE/'benchmark.py'),'validate'])]
    tasks += [('benchmark_'+str(i),[PYTHON,'-u',str(HERE/'benchmark.py'),str(i)]) for i in [1,2,3]]
    tasks += [
        ('nsight',[CUDA+'nsys','profile','--force-overwrite=true','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
            '--cuda-graph-trace=node','--output='+str(ROOT/'reports/PREPARE_PROFILE'),PYTHON,'-u',str(HERE/'benchmark.py'),'1','--profile']),
        ('export',[CUDA+'nsys','export','--type=sqlite','--force-overwrite=true','--output='+str(ROOT/'reports/PREPARE_PROFILE.sqlite'),str(ROOT/'reports/PREPARE_PROFILE.nsys-rep')]),
        ('analyze',[PYTHON,'-u',str(HERE/'profile_graph.py')])]
    states=[]
    try:
        for name,command in tasks:
            row=dict(task=name,state='running',start=time.time());states.append(row)
            print('START',name,flush=True)
            with (ROOT/'logs'/f'{name}.log').open('w') as f:
                child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                row.update(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
                (ROOT/'reports/STATUS.json').write_text(json.dumps(states,indent=2))
                code=child.wait()
            row.update(state='complete' if code==0 else 'failed',exit=code,end=time.time())
            (ROOT/'reports/STATUS.json').write_text(json.dumps(states,indent=2))
            print('END',name,code,flush=True)
            if code:raise SystemExit(code)
        print('ALL_PASS',flush=True)
    finally:
        stop.set();worker.join(timeout=10)


if __name__=='__main__':main()
