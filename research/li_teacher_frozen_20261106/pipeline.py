"""等待独立构建，顺序执行smoke、完整冻结评价与汇总。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def identity(pid):
    return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def main():
    lock=(ROOT/'reports/pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'reports/PIPELINE.json';assert not path.exists()
    state=dict(state='waiting_build',controller=identity(os.getpid()),completed=[])
    path.write_text(json.dumps(state,indent=2))
    build=json.loads((ROOT/'reports/BUILD_PROCESS_ABI0.json').read_text())
    while not (ROOT/'reports/BUILD.json').exists():
        f=Path(f'/proc/{build["pid"]}/stat')
        assert f.exists() and f.read_text().split()[21]==build['start_ticks'] and f.read_text().split()[2]!='Z','构建退出但未成功，检查日志'
        time.sleep(2)
    for name,args in [('smoke',['evaluate.py','--smoke']),('evaluation',['evaluate.py']),('summary',['summarize.py'])]:
        state.update(state='running',stage=name,updated_unix=time.time());path.write_text(json.dumps(state,indent=2))
        with (ROOT/'logs'/f'{name}.log').open('w') as f:
            c=subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(HERE/args[0])]+args[1:],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            state['active']=identity(c.pid);path.write_text(json.dumps(state,indent=2));code=c.wait()
        assert code==0,f'{name}失败，保留现场并检查原因，禁止盲目重试'
        state['completed'].append(name)
    state.update(state='evaluation_complete',active=None,updated_unix=time.time());path.write_text(json.dumps(state,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as e:
        p=ROOT/'reports/PIPELINE.json'
        if p.exists():
            s=json.loads(p.read_text());s.update(state='failed',error=repr(e));p.write_text(json.dumps(s,indent=2))
        raise
