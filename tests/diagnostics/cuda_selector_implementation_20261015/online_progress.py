"""只读在线队列进展，不导入 Torch 或占用 GPU。"""
import json
from pathlib import Path
import re
import time

root = Path(__file__).resolve().parents[3]
output = root/'results/selector_online_ab_final_20261015'
state = json.loads((output/'status.json').read_text())
print(f"completed={sum(bool(r.get('complete')) for r in state['runs'])}/24 active={state.get('active')} error={state.get('error')}")
if state.get('active'):
    run = output/state['active']
    path = run/'evaluation.log' if (run/'evaluation.log').exists() else run/'train.log'
    text = path.read_text(errors='replace') if path.exists() else ''
    progress = re.findall(r'rendering\s+(\S+):\s+([0-9]+)%[^\r\n]*?([0-9]+)/([0-9]+)', text)
    if progress:
        print('evaluation ratio=%s percent=%s frames=%s/%s' % progress[-1])
    else:
        print(path.name+': '+(text.splitlines()[-1][:140] if text.strip() else 'starting'))
    if path.exists():
        print('log_age_seconds=%.1f' % (time.time()-path.stat().st_mtime))
