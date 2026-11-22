"""训练成功后启动评价与汇总；检测控制器退出，不覆盖已有评价。"""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
while True:
    state=json.loads((ROOT/'reports/EXPERIMENT_STATUS.json').read_text())
    if state['state']=='failed':raise SystemExit('训练或验证失败，停止后续评价')
    if state['state']=='training_complete':break
    controller=state['controller'];p=Path(f'/proc/{controller["pid"]}/stat')
    if not p.exists() or p.read_text().split()[21]!=controller['start_ticks']:
        raise SystemExit('控制器退出且训练未完成，停止后续评价')
    time.sleep(5)
for name in ['evaluate_experiment.py','summarize_experiment.py']:
    command=['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(ROOT/'source/research/local_importance_li_20261105'/name)]
    with (ROOT/'logs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record=dict(stage=name,pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
        (ROOT/'reports/POSTPROCESS_PROCESS.json').write_text(json.dumps(record,indent=2))
        code=child.wait()
    if code:raise SystemExit(f'{name}失败，查看日志；不自动重复评价')
print('TRAIN_EVALUATION_SUMMARY_COMPLETE',flush=True)
