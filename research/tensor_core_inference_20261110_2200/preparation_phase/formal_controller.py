"""有身份保护的顺序配对；只对明确外部负载整批重测，不重试正确性失败。"""
from pathlib import Path
import json,subprocess,sys,time,os

HERE=Path(__file__).resolve().parent;R=HERE/'reports';identity=R/'FORMAL_CONTROLLER.json'
ticks=lambda pid:Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[19]
if identity.is_file():
    prior=json.loads(identity.read_text());proc=Path(f'/proc/{prior["pid"]}/stat')
    assert not(proc.is_file() and ticks(prior['pid'])==prior['start_ticks'] and proc.read_text().rpartition(')')[2].split()[0]!='Z'),'Existing formal controller'
state=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state='running',accepted=[],rejected=[])
save=lambda:identity.write_text(json.dumps(state,indent=2))
save()
try:
    for rnd in [1,2]:
        for group in list(range(8)) if rnd==1 else list(reversed(range(8))):
            for attempt in range(1,5):
                name=f'formal_v2_r{rnd}_g{group}_a{attempt}'
                subprocess.run([sys.executable,'-B','launch.py','--measure',name,sys.executable,'-B','-u','benchmark_preparation.py',
                    '--output',name,'--round',str(rnd),'--slice',str(group)],cwd=HERE,check=True)
                while True:
                    job=json.loads((R/'jobs'/(name+'.json')).read_text())
                    if job.get('state') in ['failed','complete']:break
                    time.sleep(.3)
                if job['returncode']!=0:raise RuntimeError((name,'Runtime/correctness failed; no blind retry'))
                if job['measurement_valid']:
                    state['accepted'].append(name);save();print('PREPARATION_FORMAL_ACCEPTED',name,flush=True);break
                state['rejected'].append(name);save();print('EXTERNAL_LOAD_REJECTED',name,flush=True)
            else:raise RuntimeError(('Repeated external load',rnd,group))
    state['state']='complete'
except Exception as error:
    state['state']='failed';state['error']=str(error);raise
finally:save()
