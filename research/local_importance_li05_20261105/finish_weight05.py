"""只启动一次后处理；训练失败或控制器异常退出时保留现场。"""
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
lock=(ROOT/'reports/postprocess.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    state=json.loads((ROOT/'reports/EXPERIMENT_STATUS.json').read_text())
    if state['state']=='failed':raise SystemExit('训练或验证失败，停止后续评价')
    if state['state']=='training_complete':break
    controller=state['controller'];p=Path(f'/proc/{controller["pid"]}/stat')
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==controller['boot_id']
    if not p.exists() or p.read_text().split()[21]!=controller['start_ticks']:
        raise SystemExit('控制器退出且训练未完成，停止后续评价')
    time.sleep(5)
for name in ['evaluate_weight05.py','summarize_weight05.py']:
    command=['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(ROOT/'source/research/local_importance_li05_20261105'/name)]
    with (ROOT/'logs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record=dict(stage=name,pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (ROOT/'reports/POSTPROCESS_PROCESS.json').write_text(json.dumps(record,indent=2))
        code=child.wait()
    if code:raise SystemExit(f'{name}失败，查看日志；不自动重复评价')
print('TRAIN_EVALUATION_SUMMARY_COMPLETE',flush=True)
