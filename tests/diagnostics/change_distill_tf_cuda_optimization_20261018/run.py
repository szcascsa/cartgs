"""固定真实事件，空闲 GPU gate，保存外部进程监测。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
EVENT = '/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1'
def processes():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
for spec in sys.argv[1:]:
    stage, branch, folder, version, repeats = spec.split(':')
    assert not processes(), 'External GPU process: '+processes()
    output=D/stage/'raw'/folder
    assert not output.exists(), output
    output.mkdir(parents=True)
    mode={'native':'0','verify':'0','profile':'1','nsys':'3'}[version]
    command=[str(D/stage/('benchmark_profile' if version=='nsys' else 'benchmark_'+version)),EVENT,str(output),branch,'5','8192','4',repeats,mode,'2' if version!='verify' else '0']
    if version=='nsys':
        command=['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx,cublas,osrt','--sample=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(output/'trace'),*command]
    (output/'command.json').write_text(json.dumps(command,indent=2))
    records=[]
    env=os.environ.copy()
    if stage=='baseline_production':
        env['LD_LIBRARY_PATH']=str(D/'deployment/backup/lib')+':'+env.get('LD_LIBRARY_PATH','')
    with (output/'stdout.log').open('w') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        while p.poll() is None:
            records.append({'time':time.time(),'pid':p.pid,'gpu':processes()})
            time.sleep(.25)
    (output/'monitor.json').write_text(json.dumps(records,indent=2))
    assert p.returncode==0, output
    # nsys spawns its own child benchmark; keep the monitor for manual PID verification.
    if version!='nsys':
        assert not [r for r in records if any(line.split(',')[0].strip()!=str(p.pid) for line in r['gpu'].splitlines())], 'contaminated'
    if version=='nsys':
        with (output/'stats.csv').open('w') as log:
            subprocess.run(['/usr/local/cuda-11.8/bin/nsys','stats','--report','gpukernsum,cudaapisum,nvtxsum','--format','csv','--force-export=true',str(output/'trace.nsys-rep')],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS',spec,flush=True)
