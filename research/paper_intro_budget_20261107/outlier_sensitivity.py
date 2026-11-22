"""按用户指定离群点重算固定4000点耗时，保留原始数据。"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent/'results/reports';OUT=ROOT/'outlier_sensitivity';OUT.mkdir(exist_ok=True)


def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(rows):
    x=np.asarray([float(v['total_ms']) for v in rows]);assert np.isfinite(x).all()
    return dict(views=len(x),mean_ms=float(x.mean()),std_ms=float(x.std(ddof=0)),cv_percent=float(100*x.std(ddof=0)/x.mean()),max_ms=float(x.max()))


inputs=[ROOT/'PAPER_INTRO_PER_VIEW.csv',ROOT/'TIMINGS.csv',ROOT/'PAPER_INTRO_SUMMARY.json']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
views=[v for v in read(inputs[0]) if v['method']=='fixed_4000'];raw=[v for v in read(inputs[1]) if v['method']=='fixed_4000']
assert len(views)==792 and len(raw)==2376
peak=max(views,key=lambda v:float(v['total_ms']));frame=int(peak['frame'])
repeats=[v for v in raw if int(v['frame'])==frame];assert len(repeats)==3
bad=max(repeats,key=lambda v:float(v['total_ms']))
key=lambda v:(int(v['frame']),int(v['repeat']),int(v['attempt']))
filtered_raw=[v for v in raw if key(v)!=key(bad)];assert len(filtered_raw)==2375
groups=defaultdict(list)
for v in filtered_raw:groups[int(v['frame'])].append(v)
reaggregated=[]
for v in views:
    t=groups[int(v['frame'])];reaggregated.append(dict(v,total_ms=float(np.mean([float(x['total_ms']) for x in t])),timing_repeats_used=len(t)))
assert len(reaggregated)==792 and sum(v['timing_repeats_used']==2 for v in reaggregated)==1
removed_view=[v for v in views if int(v['frame'])!=frame];assert len(removed_view)==791
result=dict(method='fixed_4000',std_ddof=0,view_weights='equal',selection='user requested exclusion of the largest plotted latency point',
    excluded_view=peak,raw_repeats_at_excluded_view=repeats,excluded_single_repeat=bad,
    full_data=stats(views),remove_entire_view=stats(removed_view),remove_one_repeat_keep_all_views=stats(reaggregated),
    raw_input_hashes=hashes,original_data_unchanged=True)
write(OUT/'FIXED4000_REMOVE_VIEW.csv',removed_view);write(OUT/'FIXED4000_REMOVE_ONE_REPEAT_RAW.csv',filtered_raw)
write(OUT/'FIXED4000_REAGGREGATED_PER_VIEW.csv',reaggregated)
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in hashes.items())
(OUT/'OUTLIER_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 固定4000点：指定离群点剔除后重算\n',
    f'离群视角为帧{frame}；原逐视角三轮均值{float(peak["total_ms"]):.7f}ms。三次原始测量为'+ '、'.join(f'{float(v["total_ms"]):.7f}' for v in repeats)+'ms。\n',
    '| 口径 | 视角数 | 平均耗时ms | 标准差ms | CV |','|---|---:|---:|---:|---:|']
for label,name in [('原始全量','full_data'),('删除该视角','remove_entire_view'),('仅删除异常的一次测量，重算该视角','remove_one_repeat_keep_all_views')]:
    v=result[name];lines.append(f'| {label} | {v["views"]} | {v["mean_ms"]:.7f} | {v["std_ms"]:.7f} | {v["cv_percent"]:.5f}% |')
lines+=['\n统计沿用先前口径：每个视角等权，以重复测量均值作为T，标准差ddof=0；没有跨帧平滑。',
    '保留全部792个视角时，仅帧0的重复次数从3改为2；其他791帧不变。不能将2375条剩余重复测量直接混合求标准差来替代逐视角统计。',
    '本次为查看结果后按用户指定进行的离群点敏感性分析，原始图、原始主表和全量CSV均保留；若用于论文，应同时注明剔除规则及全量结果。']
(OUT/'OUTLIER_RECALCULATION.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['full_data','remove_entire_view','remove_one_repeat_keep_all_views']},ensure_ascii=False))
