"""只读检查配对重跑的真实PID、最新步数、评价进度和长尾。"""
from pathlib import Path
import csv
import json
import time

R=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())


def live(identity):
    try:
        fields=Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and fields[19]==identity['start_ticks'] and identity['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except (FileNotFoundError,KeyError):return False


def main():
    state=read(R/'LOD_PIPELINE.json');jobs=[]
    for job in state['jobs']:
        result=dict(id=job['id'],state=job['state'],target_steps=job['target_steps'],
             health_failures=job.get('health_failures',0),oom_failures=job.get('oom_failures',0),
             evaluations_complete=sum(e['state']=='complete' for e in job['evaluations']))
        if job.get('attempt'):
            folder=Path(job['attempt']);trace=folder/'model/real_lod_steps.csv'
            if trace.exists():
                rows=list(csv.DictReader(trace.open()));result['trace_rows']=len(rows)
                result['last_trace']=rows[-1] if rows else None
            health=folder/'model/lod_health.jsonl'
            if health.exists():
                lines=health.read_text().splitlines();result['last_health']=json.loads(lines[-1]) if lines else None
        for item in ([job] if job['state']=='running' else [])+[e for e in job['evaluations'] if e['state']=='running']:
            output=Path(item['attempt']);log=output/('train.log' if item is job else 'evaluate.log')
            result['active']=dict(process=item['process'],live=live(item['process']),output=str(output),
                   log_age_seconds=time.time()-log.stat().st_mtime,log_tail=log.read_text()[-200:])
        jobs.append(result)
    value=dict(phase=state['phase'],error=state.get('error'),jobs=jobs,active_validation=state.get('active_validation'))
    if value['active_validation']:
        v=value['active_validation'];value['active_validation']=dict(id=v['id'],state=v['state'],
                evaluations_complete=sum(e['state']=='complete' for e in v.get('evaluations',[])))
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
