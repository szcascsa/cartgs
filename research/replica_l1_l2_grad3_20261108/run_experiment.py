"""运行16组Full＋tail和全轨迹评价；本次不追加LoD，不触碰其他队列。"""
from pathlib import Path
from collections import Counter
import fcntl
import importlib.util
import os
import subprocess
import sys
import time
from prepare_experiment import R,read,write,sha


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    assert Path(__file__).resolve().parent==R
    lock=(R/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=load('full_training_queue',R/'queue.py')
    assert read(R/'NO_TRAIN_RUNTIME_VERIFICATION.json')['passed']
    assert read(R/'START_AUTHORIZATION.json')['approved_by_user']
    write(R/'COORDINATOR.json',dict(**q.process_identity(os.getpid()),started_unix=time.time()))
    assert (R/'HOLD').exists();(R/'HOLD').rename(R/'HOLD_RELEASED')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',
             PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    # Full在线Mapping的训练预算由异步SLAM决定，沿用原Full批次的训练/评价时序。
    for script,action,identity in [('queue.py','start','QUEUE_CONTROLLER.json'),
                                   ('evaluate_trajectory.py','run','TRAJECTORY_EVALUATION_LAUNCH.json')]:
        with (R/'logs'/(script+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(R/script),action],cwd=R,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(R/identity,q.process_identity(child.pid));code=child.wait()
        write(R/'reports'/(script+'.exit.json'),dict(returncode=code,finished_unix=time.time()))
        assert code==0,script
        assert not (R/'HOLD').exists(),'新实验被暂停，保留任务'
        assert not any(j['state'] in ['held','running','retry_pending'] for j in read(R/'QUEUE.json'))
    before=read(R/'PROTECTED_BEFORE.json');after={p:sha(Path(p)) for p in before}
    assert before==after,'原Full源码或动态库改变';write(R/'PROTECTED_AFTER.json',after)
    jobs=read(R/'QUEUE.json');evaluations=read(R/'TRAJECTORY_EVALUATION.json')
    write(R/'FINISHED.json',dict(passed=all(j['state']=='complete' for j in jobs)
          and all(j['state']=='complete' for j in evaluations['jobs']),
          training=dict(Counter(j['state'] for j in jobs)),evaluation=evaluations['counts'],
          protected_files_unchanged=len(after),saved_images=0,finished_unix=time.time()))
    print('EXPERIMENT_FINISHED',read(R/'FINISHED.json'),flush=True)


if __name__=='__main__':main()
