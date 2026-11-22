from pathlib import Path
import collections
import csv
import hashlib
import json
import sqlite3
import statistics
import sys
import torch
D=Path(__file__).resolve().parent
result={}
for stage in sys.argv[1:]:
    root=D/stage
    records={}
    for p in (root/'raw').glob('*/runs.json'):
        rows=[r for r in json.loads(p.read_text()) if r['label'].startswith('warm_')]
        if not rows: continue
        rec={'n':len(rows)}
        for field in ['cpu_wall_ms','cuda_elapsed_ms','peak_allocated_bytes','additional_allocated_bytes']:
            v=[r[field] for r in rows]
            rec[field]={'median':statistics.median(v),'min':min(v),'max':max(v),'std':statistics.pstdev(v)}
        rec['steps']=[rows[0]['selector_steps'],rows[0]['tf_steps']]
        records[p.parent.name]=rec
    v=root/'raw/verify_tf/verification.pt'
    if v.exists():
        a=torch.jit.load(str(v),map_location='cpu').state_dict()
        batches=[t for k,t in a.items() if k.startswith('ids_')]
        groups=[]
        if batches:
            counts=collections.Counter(hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in batches[:20])
            records['layouts']={'misses':len(batches[:20]),'unique_ordered_id_sets':len(counts),'reuse':list(counts.values()),'sizes':[t.numel() for t in batches[:20]]}
    for p in (root/'raw').glob('*/*/components.csv'):
        if p.parent.name!='warm_0': continue
        c=collections.defaultdict(lambda:[0,0,0])
        for row in csv.DictReader(p.open()):
            c[row['component']][0]+=1;c[row['component']][1]+=float(row['cpu_us']);c[row['component']][2]+=float(row['cuda_span_us'])
        records[p.parent.parent.name+'_components']={k:{'calls':v[0],'cpu_ms':v[1]/1000,'cuda_span_ms':v[2]/1000} for k,v in c.items()}
    result[stage]=records
(D/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
