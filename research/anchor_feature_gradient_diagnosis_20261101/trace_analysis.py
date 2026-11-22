"""基于实际Nsight correlation确定默认/确定性Scatter采用的kernel。"""
from common import *
import sqlite3,csv,collections,bisect,urllib.request

c=sqlite3.connect(str(R/'scatter_only.sqlite'));c.row_factory=sqlite3.Row
strings={x['id']:x['value'] for x in c.execute('select id,value from StringIds')}
ranges=[]
for x in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null'):
    text=x['text'] or strings.get(x['textId'],'')
    if text.startswith('ScatterOnly.'):ranges.append((x['start'],x['end'],text))
ranges.sort();starts=[x[0] for x in ranges]
api={x['correlationId']:x for x in c.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
g=collections.defaultdict(lambda:[0,0])
for x in c.execute('select start,end,demangledName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
    a=api.get(x['correlationId'])
    if a is None:continue
    j=bisect.bisect_right(starts,a['start'])-1
    if j<0 or a['start']>ranges[j][1]:continue
    z=g[ranges[j][2],strings[x['demangledName']]];z[0]+=1;z[1]+=x['end']-x['start']
with (R/'SCATTER_KERNEL_PATHS.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,['range','kernel','calls','gpu_total_us']);w.writeheader()
    for (label,name),(n,t) in sorted(g.items()):w.writerow(dict(range=label,kernel=name,calls=n,gpu_total_us=t/1000))
# Official source at the exact deployed PyTorch tag; source-only diagnostic evidence.
url='https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/aten/src/ATen/native/cuda/Indexing.cu'
try:
    data=urllib.request.urlopen(url,timeout=20).read();p=R/'pytorch_v2.3.1_Indexing.cu';p.write_bytes(data)
    save('pytorch_source.json',dict(url=url,sha256=sha(p),downloaded=True))
except Exception as e:save('pytorch_source.json',dict(url=url,downloaded=False,error=str(e)))
print('SCATTER_KERNEL_PATHS_EXPORTED',len(g),flush=True)
