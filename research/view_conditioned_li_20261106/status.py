"""当前任务只读进度与进程/文件交叉核验。"""
from pathlib import Path
import json
import time
from common import ROOT,identity

result={}
for name in ['PIPELINE.json','FINISH.json','POSTPROCESS.json']:
    p=ROOT/'reports'/name
    if p.exists():
        v=json.loads(p.read_text());result[name]={k:v[k] for k in ['state','stage','error'] if k in v}
        process=v.get('controller')
        if process:
            pid=process['pid'];stat=Path(f'/proc/{pid}/stat')
            result[name]['live_identity_match']=stat.exists() and stat.read_text().split()[2]!='Z' and identity(pid)==process
p=ROOT/'evaluation/STATUS.json'
if p.exists():result['evaluation']=json.loads(p.read_text())
p=ROOT/'evaluation/reports/PER_VIEW.csv'
if p.exists():result['csv']=dict(bytes=p.stat().st_size,seconds_since_write=round(time.time()-p.stat().st_mtime,2))
if any(v.get('state')=='failed' for v in result.values() if isinstance(v,dict)):
    for p in sorted((ROOT/'logs').glob('*.log'),key=lambda p:p.stat().st_mtime)[-3:]:result[p.name]=p.read_text()[-2300:]
print(json.dumps(result,ensure_ascii=False),flush=True)
