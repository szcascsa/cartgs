"""相同工作量拆成2场景短批次；外部负载整批拒绝，最多4次，不删单个慢样本。"""
from pathlib import Path
import json,subprocess,sys,time,os
HERE=Path(__file__).resolve().parent;R=HERE/'reports'
identity=R/'BATCH_CONTROLLER.json'
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[19]
if identity.exists():
    old=json.loads(identity.read_text());proc=Path(f'/proc/{old["pid"]}/stat')
    if proc.exists():assert ticks(old['pid'])!=old['start_ticks'] or proc.read_text().rpartition(')')[2].split()[0]=='Z','Existing batch controller'
state=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state='running',completed=[])
def save():identity.write_text(json.dumps(state,indent=2))
save();accepted=[];rejected=[]
try:
    for round_index in [1,2]:
        groups=list(range(8)) if round_index==1 else list(reversed(range(8)))
        for group in groups:
            for attempt in range(1,5):
                name=f'paired_c2_r{round_index}_g{group}_a{attempt}'
                cmd=[sys.executable,'launch.py','--measure',name,sys.executable,'-B','-u','benchmark.py','--output',name,'--round',str(round_index),'--all-scenes','--slice',str(group)]
                subprocess.run(cmd,cwd=HERE,check=True)
                path=R/'jobs'/(name+'.json')
                while True:
                    job=json.loads(path.read_text())
                    if job.get('state') in ['complete','failed']:break
                    time.sleep(.3)
                if job['returncode']!=0:raise RuntimeError((name,'Correctness/runtime failure; no blind retry'))
                if job['measurement_valid']:
                    accepted.append(name);state['completed'].append(name);save();print('PAIRED_BATCH_VALID',name,flush=True);break
                rejected.append(name);print('PAIRED_BATCH_EXTERNAL_LOAD',name,flush=True)
            else:raise RuntimeError(('Repeated external load',round_index,group))
    (R/'MEASUREMENT_RUNS.json').write_text(json.dumps(dict(formal=accepted,rejected=rejected,earlier_excluded=['formal01','formal02','formal03'],earlier_candidate='reports/C1_EVALUATION',protocol='S4/C2组合两轮均2场景短批，6重复，原完整工作量；外部负载整批拒绝，无单样本删减'),ensure_ascii=False,indent=2))
    state['state']='complete'
except Exception as error:
    state['state']='failed';state['error']=str(error);raise
finally:save()
