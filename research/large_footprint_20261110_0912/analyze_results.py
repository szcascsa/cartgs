"""全部场景/失败项保留，区分旧入口收益与本轮新增收益。"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='*');parser.add_argument('--manifest')
parser.add_argument('--workloads');args=parser.parse_args()
if args.manifest:
    manifest=json.loads((HERE/'reports'/args.manifest).read_text());assert manifest['status']=='complete'
    args.runs=manifest['runs']
assert len(args.runs)>=2
frames=[];checks=[];events=[];memory=[];workloads=[]
for name in args.runs:
    job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
    summary=json.loads((HERE/'reports'/name/'SUMMARY.json').read_text())
    assert job['measurement_valid'] and job['foreign_active_samples']==0 and summary['passed'] and summary['scenes'] in [1,24]
    root=HERE/'reports'/name
    frames.append(pd.read_csv(root/'samples.csv'));checks.append(pd.read_csv(root/'correctness.csv'))
    events.append(pd.read_csv(root/'events.csv'));memory.append(pd.read_csv(root/'memory.csv'))
    if (root/'workloads.csv').stat().st_size:workloads.append(pd.read_csv(root/'workloads.csv'))
if args.workloads:
    # 外部负载会污染计时，不影响已与A逐位核对的静态整数统计；不导入该轮耗时。
    root=HERE/'reports'/args.workloads;assert json.loads((root/'SUMMARY.json').read_text())['passed']
    workloads.append(pd.read_csv(root/'workloads.csv'))
d=pd.concat(frames,ignore_index=True);c=pd.concat(checks,ignore_index=True)
assert len(d['round'].unique())==2 and d.scene.nunique()==24
assert c[['rgb_bitwise','radii_bitwise','ids_bitwise','camera_bitwise']].all().all()
keys=['scene','split','condition','round','scope']
s=d.groupby(keys+['variant']).agg(median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda x:x.quantile(.95)),
    std_ms=('wall_ms','std'),samples=('wall_ms','size'),K_median=('K','median'),K_min=('K','min'),K_max=('K','max'),N=('N','first')).reset_index()
for name in ['A','G']:
    ref=s[s.variant==name][keys+['median_ms','p95_ms']].rename(columns={'median_ms':name+'_ms','p95_ms':name+'_p95_ms'})
    s=s.merge(ref,on=keys,how='left')
    s['speedup_vs_'+name]=s[name+'_ms']/s.median_ms
    s['latency_change_pct_vs_'+name]=(s.median_ms/s[name+'_ms']-1)*100
    s['saved_us_vs_'+name]=(s[name+'_ms']-s.median_ms)*1000
s.to_csv(HERE/'reports/RASTER_AND_PIPELINE_RESULTS.csv',index=False)
v=d.groupby(keys+['frame','motion','variant']).agg(median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda x:x.quantile(.95)),K=('K','first'),N=('N','first')).reset_index()
for name in ['A','G']:
    ref=v[v.variant==name][keys+['frame','motion','median_ms']].rename(columns={'median_ms':name+'_ms'})
    v=v.merge(ref,on=keys+['frame','motion'],how='left');v['speedup_vs_'+name]=v[name+'_ms']/v.median_ms
v.to_csv(HERE/'reports/PER_VIEW_RESULTS.csv',index=False)
pd.concat(events,ignore_index=True).to_csv(HERE/'reports/CUDA_EVENTS.csv',index=False)
pd.concat(memory,ignore_index=True).to_csv(HERE/'reports/MEMORY.csv',index=False)
if workloads:pd.concat(workloads,ignore_index=True).to_csv(HERE/'reports/WORKLOADS.csv',index=False)
full=s[(s.condition=='full')&s.variant.isin(['C','D'])].copy()
full.to_csv(HERE/'reports/FULL_COST.csv',index=False)
loss=s[s.variant.isin(['C','D'])&(s.speedup_vs_G<1)]
loss.to_csv(HERE/'reports/ALL_SLOWER_CASES.csv',index=False)
ratios=['local0.01','local0.03','local0.05','local0.1','local0.3','local0.7','full'];aggregate=[]
for (scope,name,condition),g in s[s.variant.isin(['C','D'])].groupby(['scope','variant','condition']):
    if scope=='prepare':continue
    aggregate.append(dict(scope=scope,variant=name,condition=condition,scene_rounds=len(g),
        geometric_speedup_G=float(np.exp(np.log(g.speedup_vs_G).mean())),
        geometric_speedup_A=float(np.exp(np.log(g.speedup_vs_A).mean())),
        min_speedup_G=float(g.speedup_vs_G.min()),max_speedup_G=float(g.speedup_vs_G.max()),
        saved_us_G_median=float(g.saved_us_vs_G.median()),saved_us_G_min=float(g.saved_us_vs_G.min()),
        saved_us_G_max=float(g.saved_us_vs_G.max()),slower_G=int((g.speedup_vs_G<1).sum()),
        median_C_ms=float(g.median_ms.median()),median_G_ms=float(g.G_ms.median())))
aggregate=pd.DataFrame(aggregate);aggregate.to_csv(HERE/'reports/AGGREGATE.csv',index=False)
agreement=[]
for (scene,split,condition,scope,name),g in s[s.variant.isin(['C','D'])].groupby(['scene','split','condition','scope','variant']):
    if scope=='prepare':continue
    assert len(g)==2
    agreement.append(dict(scene=scene,split=split,condition=condition,scope=scope,variant=name,
        both_faster_G=bool((g.speedup_vs_G>=1).all()),both_slower_G=bool((g.speedup_vs_G<1).all()),
        speedup_min=float(g.speedup_vs_G.min()),speedup_max=float(g.speedup_vs_G.max()),
        round_median_variation_pct=float((g.median_ms.max()/g.median_ms.min()-1)*100)))
agreement=pd.DataFrame(agreement);agreement.to_csv(HERE/'reports/ROUND_AGREEMENT.csv',index=False)
count=v[v.variant.isin(['C','D'])].copy();k=['scene','round','scope','frame','motion','variant']
reference=count[count.condition=='full'][k+['median_ms']].rename(columns={'median_ms':'own_full_ms'})
count=count.merge(reference,on=k);count['speedup_vs_own_full']=count.own_full_ms/count.median_ms
count.to_csv(HERE/'reports/COUNT_VS_FULL.csv',index=False)
jobs=[];environment=[]
for path in sorted((HERE/'reports/jobs').glob('*.json')):
    job=json.loads(path.read_text());jobs.append({k:job.get(k) for k in ['name','state','returncode','elapsed_s','measured','foreign_active_samples','observations','measurement_valid']})
    if path.stem in args.runs:
        for line in (HERE/'logs'/(path.stem+'_MONITOR.jsonl')).read_text().splitlines():
            row=json.loads(line);values=row['gpu'].split(',');assert len(values)==5
            environment.append(dict(run=path.stem,time=row['time'],utilization=float(values[0].strip().split()[0]),
                memory_MiB=float(values[1].strip().split()[0]),clock_MHz=float(values[2].strip().split()[0]),
                temperature_C=float(values[3].strip().split()[0]),power_W=float(values[4].strip().split()[0]),foreign_active=bool(row['foreign_active'])))
pd.DataFrame(jobs).to_csv(HERE/'reports/JOBS.csv',index=False);pd.DataFrame(environment).to_csv(HERE/'reports/GPU_ENVIRONMENT.csv',index=False)
result=dict(passed=True,runs=args.runs,samples=len(d),correctness_rows=len(c),scenes=d.scene.nunique(),
    heldout_scenes=d[d.split=='heldout'].scene.nunique(),RGB_radii_ID_camera_bitwise=True,
    aggregate=aggregate.to_dict('records'),full_cost=full[['scene','round','scope','variant','median_ms','G_ms','speedup_vs_G','latency_change_pct_vs_G']].to_dict('records'))
(HERE/'reports/RESULTS_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(aggregate[aggregate.condition.isin(['local0.01','local0.05','local0.1','full'])].round(5).to_string(index=False))
print('TOTAL',len(d),'CHECKS',len(c),'SCENES',d.scene.nunique())
