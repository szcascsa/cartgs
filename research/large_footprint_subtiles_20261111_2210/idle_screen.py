"""等待已识别的外部测速结束，再独占配对筛选；不停止任何外部进程。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OTHER='/home/prmi/zwj/experiments/office2_f2_single_frame_20261111'
PYTHON=sys.executable


def other_jobs():
    result=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            value=path.read_bytes().replace(b'\x00',b' ').decode()
            if OTHER in value and 'python' in value:
                fields=(path.parent/'stat').read_text().rpartition(')')[2].split()
                if fields[0]!='Z':result.append(dict(pid=int(path.parent.name),start_ticks=fields[19],command=value[:500]))
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeDecodeError):pass
    return result


def main():
    status=HERE/'reports/IDLE_SCREEN.json'
    state=dict(pid=os.getpid(),state='waiting',started_at=time.time(),attempts=[])
    def save():status.write_text(json.dumps(state,indent=2))
    while True:
        jobs=other_jobs()
        state['external_jobs']=jobs;state['checked_at']=time.time();save()
        if not jobs:break
        time.sleep(10)
    # 先核对编译/正确性均已结束，再执行约半分钟的候选配对。
    for path in (HERE/'reports/jobs').glob('*.json'):
        j=json.loads(path.read_text())
        assert j['state'] in ('complete','failed'),(path,j)
    valid=json.loads((HERE/'reports/validation06/SUMMARY.json').read_text())
    assert set(valid['valid'])=={'direct_tile128','direct_tile64'},valid
    for attempt in range(1,4):
        name=f'screen_clean{attempt:02d}'
        state.update(state='measuring',current=name);save()
        cmd=[PYTHON,'-B','launch.py','--measure',name,PYTHON,'-u','-B','screen.py','--output',name,
             '--modes','0,1,2,3,5,6,7,11,12,13,14,15,16,17,18,19,20,21','--repeats','24']
        subprocess.run(cmd,cwd=HERE,check=True)
        job=HERE/'reports/jobs'/(name+'.json')
        while True:
            j=json.loads(job.read_text())
            if j['state'] in ('complete','failed'):break
            time.sleep(1)
        state['attempts'].append(name);save()
        assert j['returncode']==0,j
        if j['measurement_valid']:
            state.update(state='complete',accepted=name,ended_at=time.time());save()
            print('CLEAN_SCREEN_COMPLETE',name,flush=True);return
    state.update(state='blocked',reason='3 consecutive external-load rejections');save()
    raise RuntimeError(state)


if __name__=='__main__':main()
