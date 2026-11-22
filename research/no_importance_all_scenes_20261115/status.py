"""只读交叉核查队列、进程身份、日志和训练CSV实际增量。"""
from pathlib import Path
import json
import shutil
import time
from prepare import R,read
from pipeline import alive


state=read(R/'LOD_PIPELINE.json');active=[]
for j in state['jobs']:
    for scope,item in [('train',j)]+[(e['scope']+'_'+str(e['ratio']),e) for e in j['evaluations']]:
        if item.get('state')!='running':continue
        proc=item.get('process',{});out=Path(item['attempt']);log=out/('train.log' if scope=='train' else 'evaluate.log')
        entry=dict(id=j['id'],scope=scope,process=proc,identity_matches=alive(proc),step=item.get('step'),
            target_steps=j['target_steps'],log_bytes=log.stat().st_size,log_age_seconds=round(time.time()-log.stat().st_mtime,1),
            no_progress_seconds=item.get('no_progress_seconds'))
        trace=out/'model/real_lod_steps.csv'
        if trace.exists():entry.update(trace_bytes=trace.stat().st_size,trace_age_seconds=round(time.time()-trace.stat().st_mtime,1))
        if not entry['identity_matches']:entry['log_tail']=log.read_text(errors='replace')[-1500:]
        active.append(entry)
print(json.dumps(dict(phase=state['phase'],counts=state.get('counts'),active=active,
    evaluations_complete=sum(e['state']=='complete' for j in state['jobs'] for e in j['evaluations']),
    free_GiB=round(shutil.disk_usage(R).free/2**30,2),finished=(R/'FINISHED.json').exists()),ensure_ascii=False))
