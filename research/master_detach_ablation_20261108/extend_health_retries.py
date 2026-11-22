"""将本批健康失败额外重试上限设为5，接管活评价worker而不重跑评价。"""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import run_batch as b
import supervisor

R=b.R;p=b.p


def update_state(state):
    assert state['protocol']['max_health_retries']==2
    assert state['protocol']['max_oom_retries']==2
    state['protocol']['max_health_retries']=5
    restored=[]
    for job in state['jobs']:
        if job['state']=='skipped_health' and b.health_decision(job,5):
            assert not job.get('trained_model')
            job.update(state='retry_pending',health_retry_extension_unix=time.time())
            restored.append(job['id'])
    return restored


def process_state(info):
    return (Path('/proc')/str(info['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]


def main():
    assert not (R/'HOLD').exists()
    original=p.read(R/'LOD_PIPELINE.json')
    assert original['phase']=='parallel_running' and 'train' not in original['parallel_workers']
    identities={name:p.read(R/name) for name in ['SUPERVISOR.json','LOD_CONTROLLER.json']}
    for name,info in identities.items():
        assert supervisor.alive(info)
        command=(Path('/proc')/str(info['pid'])/'cmdline').read_bytes().split(b'\0')
        expected='supervisor.py' if name=='SUPERVISOR.json' else 'parallel_queue.py'
        assert str(R/expected).encode() in command
    backup=R/'reports'/('health_retries5_backup_'+str(time.time_ns()));backup.mkdir()
    for name in ['LOD_PIPELINE.json','SUPERVISOR.json','LOD_CONTROLLER.json']:
        p.write(backup/name,p.read(R/name))
    stopped=[];committed=False
    try:
        for info in identities.values():
            os.kill(info['pid'],signal.SIGSTOP);stopped.append(info)
            for _ in range(100):
                if process_state(info) in ['T','t']:break
                time.sleep(.01)
            assert process_state(info) in ['T','t']
        state=p.read(R/'LOD_PIPELINE.json')
        assert state['phase']=='parallel_running' and 'train' not in state['parallel_workers']
        restored=update_state(state)
        assert len(restored)==7
        state.update(phase='resume_requested',error='')
        p.save(state);committed=True
        receipt=dict(changed_unix=time.time(),old_extra_retries=2,new_extra_retries=5,
                     max_total_health_attempts=6,already_used_preserved=True,
                     max_oom_retries=2,restored_jobs=restored,old_controllers=identities,
                     adopted_workers=state['parallel_workers'],backup=str(backup))
        p.write(R/'reports/HEALTH_RETRIES5.json',receipt)
        # 只结束本批两个调度进程；评价worker及原生评价进程不接收信号。
        for info in identities.values():
            assert supervisor.alive(info)
            os.kill(info['pid'],signal.SIGTERM);os.kill(info['pid'],signal.SIGCONT)
        for _ in range(100):
            if not any(supervisor.alive(info) for info in identities.values()):break
            time.sleep(.02)
        assert not any(supervisor.alive(info) for info in identities.values())
        with (R/'logs/supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(R/'supervisor.py')],cwd=R,
                env=b.environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['new_supervisor']=p.identity(child.pid)
        p.write(R/'reports/HEALTH_RETRIES5.json',receipt)
        print(restored,receipt['new_supervisor'],flush=True)
    finally:
        if not committed:
            for info in stopped:
                if supervisor.alive(info):os.kill(info['pid'],signal.SIGCONT)


if __name__=='__main__':main()
