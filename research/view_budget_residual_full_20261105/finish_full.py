"""完整训练通过后只执行一次评价和汇总；失败保留现场。"""
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
lock=(ROOT/'reports/postprocess.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    state=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
    assert state['state']!='failed','训练失败，停止评价并检查现场'
    if state['state']=='training_complete':break
    process=state['controller'];p=Path(f'/proc/{process["pid"]}/stat')
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']
    assert p.exists() and p.read_text().split()[21]==process['start_ticks'],'训练控制器异常退出'
    time.sleep(5)
for name in ['evaluate_full.py','summarize_full.py']:
    command=['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(ROOT/'source/research/view_budget_residual_full_20261105'/name)]
    with (ROOT/'logs'/f'{name}.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        v=dict(stage=name,pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (ROOT/'reports/POSTPROCESS_PROCESS.json').write_text(json.dumps(v,indent=2))
        code=child.wait()
    assert code==0,f'{name}失败，不自动重复；检查日志'
print('TRAIN_EVALUATE_SUMMARIZE_COMPLETE',flush=True)
