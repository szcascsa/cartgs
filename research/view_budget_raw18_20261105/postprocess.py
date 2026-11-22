"""完整训练后执行一次评价和汇总，不重复已完成任务。"""
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
lock=(ROOT/'reports/postprocess.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    p=ROOT/'reports/TRAINING_STATUS.json'
    if p.exists():
        state=json.loads(p.read_text())
        assert state['state']!='failed','训练失败，停止评价并检查日志'
        if state['state']=='training_complete':break
        process=state['controller'];stat=Path(f'/proc/{process["pid"]}/stat')
        assert stat.exists() and stat.read_text().split()[21]==process['start_ticks'],'训练控制器异常退出'
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']
    time.sleep(5)
for name in ['evaluate_full.py','summarize_full.py']:
    with (ROOT/'logs'/f'{name}.log').open('w') as f:
        child=subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(HERE/name)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        process=dict(stage=name,pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (ROOT/'reports/POSTPROCESS_PROCESS.json').write_text(json.dumps(process,indent=2))
        code=child.wait()
    assert code==0,f'{name}失败，检查日志，禁止覆盖已有输出'
print('TRAIN_EVALUATE_SUMMARIZE_COMPLETE',flush=True)
