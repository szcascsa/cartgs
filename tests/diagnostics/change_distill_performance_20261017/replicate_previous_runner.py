"""在当前空闲机器上复跑上一轮 runner，区分历史机器状态与新 harness 边界。"""
from pathlib import Path
import json
import subprocess
from run_matrix import RAW, EVENT, wait_idle, command

PRIOR = Path('/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038')
for epochs in [5, 20]:
    for repeat in range(3):
        wait_idle()
        output = RAW / 'previous_runner_replication' / f'e{epochs}_rep{repeat}'
        output.mkdir(parents=True, exist_ok=True)
        command([PRIOR / 'paired_runner', 'benchmark', EVENT, output, epochs, 0, 0, 1], output / 'stdout.log')
(RAW / 'previous_runner_replication_complete.json').write_text(json.dumps({'complete': True, 'repeats': 3}))
