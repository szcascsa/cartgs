from pathlib import Path
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
with (ROOT/'logs/GPU_MONITOR.jsonl').open('a',buffering=1) as out:
    while not (ROOT/'reports/STOP_MONITOR').exists():
        def query(fields,flag):
            return subprocess.check_output(['nvidia-smi',flag+'='+fields,'--format=csv,noheader'],text=True).strip()
        row=dict(time=time.time(),gpu=query('utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu','--query-gpu'),
            processes=query('pid,used_memory','--query-compute-apps'))
        out.write(json.dumps(row)+'\n');time.sleep(3)
