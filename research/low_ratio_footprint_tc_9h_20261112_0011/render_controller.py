"""冻结后全覆盖验收。无精度/速度结果反馈调参，受扰段完整保留。"""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    lock=(HERE/'reports/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'reports/RENDER_CONTROLLER.json';assert not path.exists()
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time(),accepted=[],rejected=[])
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    def result(name):
        while True:
            j=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
            if j['state'] in ('complete','failed'):return j
            time.sleep(1)
    def job(label,cmd,measure=False):
        protocol=json.loads((HERE/'reports/PROTOCOL.json').read_text())
        assert time.time()<datetime.fromisoformat(protocol['final_validation_reserve']).timestamp(),'预算已进入终验预留期'
        assert shutil.disk_usage(HERE).free>=5*(1<<30),'磁盘低于5GiB'
        for attempt in range(1,5):
            name=f'{label}_a{attempt}'
            subprocess.run([sys.executable,'-B','launch.py']+(['--measure'] if measure else [])+
                [name,*cmd(name)],cwd=HERE,check=True)
            value=result(name);assert value['returncode']==0,(name,value)
            if value['measurement_valid']:
                state['accepted'].append(name);save();return
            state['rejected'].append(name);save()
        raise RuntimeError((label,'外部负载持续，无有效测量'))
    save()
    try:
        assert result('edges01')['returncode']==0
        job('diagnose_smoke01',lambda name:[sys.executable,'-u','-B','diagnose_smoke.py','--output',name])
        subprocess.run([sys.executable,'-B','freeze_render.py'],cwd=HERE,check=True)
        for workload in ['lod','nested']:
            for group in range(16):
                label=f'quality_{workload}_g{group}'
                def command(name):return [sys.executable,'-u','-B','quality.py','--output',name,'--modes','37,38,39,40','--group',str(group)]+(['--nested'] if workload=='nested' else [])
                job(label,command)
        state['quality_completed_at']=time.time();save()
        for iteration in [1,2]:
            for group in (range(16) if iteration==1 else reversed(range(16))):
                label=f'formal_r{iteration}_g{group}'
                def command(name):return [sys.executable,'-u','-B','formal_render.py','--output',name,'--round',str(iteration),'--group',str(group)]+(['--diagnostics'] if iteration==1 else [])
                job(label,command,True)
        for tool in ['memcheck','initcheck','synccheck']:
            job(tool+'01',lambda name:['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','87',
                sys.executable,'-u','-B','edges.py','--output',name,'--modes','37,38,39,40','--sanitizer'])
            assert 'ERROR SUMMARY: 0 errors' in (HERE/'logs'/(state['accepted'][-1]+'.log')).read_text()
        state.update(state='complete',ended_at=time.time());save()
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_at=time.time());save();raise
    print('RENDER_ACCEPTANCE_COMPLETE',flush=True)


if __name__=='__main__':main()
