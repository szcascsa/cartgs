from pathlib import Path
import sqlite3,json,statistics
from collections import defaultdict
p=Path(__file__).resolve().parents[3]/'reports/PROFILE.sqlite'
db=sqlite3.connect(p);db.row_factory=sqlite3.Row
strings=dict(db.execute('select id,value from StringIds'))
apis=list(db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'))
kernels=defaultdict(list);copies=defaultdict(list)
for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):kernels[r['correlationId']].append(dict(r))
for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):copies[r['correlationId']].append(dict(r))
scopes=[]
for s in db.execute('select * from NVTX_EVENTS where end is not null'):
 label=s['text'] or strings.get(s['textId'],'')
 if label.startswith(('prepare/','render/','node/')):scopes.append((label,dict(s)))
groups=defaultdict(list)
for label,s in scopes:
 issued=[a for a in apis if a['globalTid']==s['globalTid'] and s['start']<=a['start'] and a['end']<=s['end']]
 work=[k for a in issued for k in kernels[a['correlationId']]]
 mem=[c for a in issued for c in copies[a['correlationId']]]
 g=defaultdict(lambda:dict(calls=0,ns=0))
 for k in work:
  name=strings[k['demangledName']];g[name]['calls']+=1;g[name]['ns']+=k['end']-k['start']
 groups[label].append(dict(wall_us=(s['end']-s['start'])/1000,kernels=len(work),kernel_us=sum(k['end']-k['start'] for k in work)/1000,
  launches=sum('Launch' in strings[a['nameId']] for a in issued),syncs=sum('Synchronize' in strings[a['nameId']] for a in issued),
  sync_us=sum(a['end']-a['start'] for a in issued if 'Synchronize' in strings[a['nameId']])/1000,dtoh=sum(c['copyKind']==2 for c in mem),by_name=g))
summary=[]
for label,rows in groups.items():
 names=defaultdict(lambda:dict(calls=0,gpu_us=0))
 for row in rows:
  for k,v in row['by_name'].items():names[k]['calls']+=v['calls']/len(rows);names[k]['gpu_us']+=v['ns']/1000/len(rows)
 summary.append(dict(label=label,samples=len(rows),**{k:statistics.median(r[k] for r in rows) for k in ['wall_us','kernels','kernel_us','launches','syncs','sync_us','dtoh']},top=sorted([dict(name=k,**v) for k,v in names.items()],key=lambda r:r['gpu_us'],reverse=True)))
(p.parent/'PROFILE_ANALYSIS.json').write_text(json.dumps(dict(summary=summary),indent=2))
print(json.dumps([{k:v for k,v in s.items() if k!='top'} for s in summary if s['label'].startswith('prepare/')],indent=2))
