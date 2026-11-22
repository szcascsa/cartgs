"""汇总同相机／预算／轮次的配对中位数；保留全部退化项。"""
from pathlib import Path
from collections import defaultdict
import csv
import math
import statistics
import sys

root=Path(__file__).resolve().parents[1]
e=root/'evidence/cpu_optimization'
paths=[Path(x) for x in sys.argv[1:]] or sorted(e.glob('cpu_opt_*.csv'))
aggregate=[]
paired=[]
for path in paths:
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        key=(int(r['round']),int(r['camera']),round(float(r['ratio']),4),int(r['fixed_k']))
        assert all(math.isfinite(float(r[k])) and float(r[k])>=0 for k in ['cpu_total_ms','prepare_ms','selector_ms','topk_ms','student_ms','decode_ms'])
        groups[key][r['variant']].append(r)
    summaries=defaultdict(list)
    for key,variants in sorted(groups.items()):
        control=next(name for name in variants if name!='optimized')
        a,b=variants[control],variants['optimized']
        assert a and len(a)==len(b)
        assert len({(r['N'],r['eligible'],r['K']) for r in a+b})==1
        item=dict(source=path.name,control=control,round=key[0],camera=key[1],ratio=key[2],fixed_k=key[3],N=int(a[0]['N']),
            eligible=int(a[0]['eligible']),K=int(a[0]['K']),samples_per_variant=len(a),flags=int(b[0]['flags']),threads=int(b[0]['threads']))
        for variant,data in [('original',a),('optimized',b)]:
            for k in ['cpu_total_ms','prepare_ms','selector_ms','topk_ms','student_ms','decode_ms']:
                values=sorted(float(r[k]) for r in data)
                item[f'{variant}_{k}']=statistics.median(values)
                if k=='cpu_total_ms':item[f'{variant}_p95_ms']=values[math.ceil(.95*len(values))-1]
        item['speedup']=item['original_cpu_total_ms']/item['optimized_cpu_total_ms']
        paired.append(item);summaries[(key[2],key[3])].append(item)
    for (ratio,k),values in sorted(summaries.items()):
        speed=math.exp(statistics.mean(math.log(r['speedup']) for r in values))
        item=dict(source=path.name,ratio=ratio,fixed_k=k,flags=values[0]['flags'],threads=values[0]['threads'],
            original_ms=statistics.median(r['original_cpu_total_ms'] for r in values),
            optimized_ms=statistics.median(r['optimized_cpu_total_ms'] for r in values),speedup=speed,
            reduction_percent=100*(1-1/speed),slower_pairs=sum(r['speedup']<1 for r in values),pairs=len(values))
        aggregate.append(item)
    main=[r for r in paired if r['source']==path.name and not r['fixed_k']]
    speed=math.exp(statistics.mean(math.log(r['speedup']) for r in main))
    print(f'{path.name}: flags={main[0]["flags"]} threads={main[0]["threads"]} ratio10-50={speed:.4f}x slower={sum(r["speedup"]<1 for r in main)}/{len(main)}')
for name,rows in [('PAIRED_RESULTS.csv',paired),('AGGREGATE_RESULTS.csv',aggregate),('SLOWER_PAIRS.csv',[r for r in paired if r['speedup']<1])]:
    with (e/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0] if rows else paired[0]));writer.writeheader();writer.writerows(rows)
