"""核对队列身份、逐帧进度与已完成数值结果。"""
from pathlib import Path
import collections
import json
import time
from common import ROOT,load


def alive(identity):
    path=Path(f"/proc/{identity['pid']}/stat")
    if not path.exists():return False
    fields=path.read_text().rsplit(')',1)[1].split()
    return (fields[0]!='Z' and fields[19]==identity['start_ticks'] and
            identity['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def main():
    state=load(ROOT/'EXPERIMENTS.json')
    value=dict(phase=state['phase'],error=state.get('error'),
               groups=dict(collections.Counter(g['state'] for g in state['groups'])),
               groups_completed_on_disk=len(list((ROOT/'runs').rglob('COMPLETE.json'))),
               completed_evaluations=len(list((ROOT/'runs').rglob('metric_summary.json'))),
               controller_alive=alive(load(ROOT/'CONTROLLER.json')))
    if (ROOT/'ACTIVE.json').exists():
        active=load(ROOT/'ACTIVE.json')
        value['active']=dict(group=active['group'],smoke=active['smoke'],
                             worker_alive=alive(active['process']),pid=active['process']['pid'])
        progress_path=ROOT/'runs'/active['group']/'PROGRESS.json'
        if not progress_path.exists():progress_path=ROOT/'PROGRESS.json'
        if progress_path.exists():
            progress=load(progress_path)
            value['progress']=dict(progress,age_seconds=time.time()-progress['updated_unix'])
    if (ROOT/'PARALLEL_ACTIVE.json').exists():
        active=load(ROOT/'PARALLEL_ACTIVE.json')
        value['parallel']=dict(group=active['group'],worker_alive=alive(active['process']),pid=active['process']['pid'])
        progress_path=ROOT/'runs'/active['group']/'PROGRESS.json'
        if progress_path.exists():
            progress=load(progress_path)
            value['parallel']['progress']=dict(progress,age_seconds=time.time()-progress['updated_unix'])
    if (ROOT/'PARALLEL_ERROR.json').exists():value['parallel_error']=load(ROOT/'PARALLEL_ERROR.json')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
