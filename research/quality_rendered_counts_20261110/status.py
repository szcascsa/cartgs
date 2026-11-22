"""只读核查本次计数进度和进程身份。"""
from pathlib import Path
import json

r=Path(__file__).resolve().parent
groups=list((r/'counts').glob('**/COMPLETE.json'))
records=list((r/'counts').glob('**/summary.json'))
active=[]
errors=[]
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for p in (r/'counts').glob('**/PROCESS.json'):
    if (p.parent/'COMPLETE.json').exists():continue
    identity=json.loads(p.read_text());stat=Path(f"/proc/{identity['pid']}/stat")
    alive=False
    if stat.exists():
        fields=stat.read_text().rsplit(')',1)[1].split()
        alive=fields[0]!='Z' and fields[19]==identity['start_ticks'] and identity['boot_id']==boot
    progress=[json.loads(q.read_text()) for q in p.parent.glob('*/PROGRESS.json')
              if not (q.parent/'summary.json').exists()]
    active.append(dict(group=str(p.parent.relative_to(r/'counts')),alive=alive,progress=progress))
    text=(p.parent/'worker.log').read_text()
    if 'Traceback' in text:errors.append(dict(group=str(p.parent),tail=text[-1500:]))
print(json.dumps(dict(groups=len(groups),records=len(records),active=active,errors=errors,
                      complete=(r/'COMPLETE.json').exists()),ensure_ascii=False))
