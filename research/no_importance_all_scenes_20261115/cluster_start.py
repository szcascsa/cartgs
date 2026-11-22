"""云节点唯一入口：验收通过后才能启动分派的正式队列。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT='/home/prmi/zwj/experiments/no_importance_all_scenes_20261115'
worker=sys.argv[1];assert worker in ['cloud_a','cloud_b']
R=Path(ROOT);lock=(R/'cloud_session.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,CARTGS_CLUSTER_WORKER=worker,CARTGS_CLUSTER_ROOT=ROOT,TORCH_HOME='/home/prmi/.cache/torch',PYTHONDONTWRITEBYTECODE='1')
py='/home/prmi/anaconda3/envs/zwj/bin/python'
(R/'logs').mkdir(exist_ok=True)
for script,args,phase in [('cluster_validate.py',[worker],'validating'),('run.py',[],'training')]:
    with (R/'logs'/('cloud_'+phase+'.log')).open('w') as log:
        p=subprocess.Popen([py,'-u',str(R/script),*args],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
        (R/'reports/CLOUD_SESSION.json').write_text(json.dumps(dict(worker=worker,phase=phase,parent_pid=os.getpid(),child_pid=p.pid,started_unix=time.time()),indent=2))
        code=p.wait()
    if code:
        (R/'reports/CLOUD_SESSION.json').write_text(json.dumps(dict(worker=worker,phase='failed_'+phase,returncode=code,log=str(R/'logs'/('cloud_'+phase+'.log'))),indent=2))
        sys.exit(code)
(R/'reports/CLOUD_SESSION.json').write_text(json.dumps(dict(worker=worker,phase='local_finished',finished_unix=time.time()),indent=2))
