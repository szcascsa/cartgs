"""本轮只计算相对原学生的新增收益，全部退化与质量差异保留。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
PRIMARY=['local0.1','local0.2','local0.3','local0.4','local0.5']
def read(n):return json.loads((HERE/'reports'/n).read_text())
def geo(x):return float(np.exp(np.log(x).mean()))
def valid(n):
    j=read('jobs/'+n+'.json');assert j['measurement_valid'] and not j['foreign_active_samples'] and not j['foreign_cpu_active_samples'],j
frames=[];components=[];checks=[];memory=[]
for n in ['formal03','formal02']:
    valid(n);assert read(n+'/SUMMARY.json')['passed'] and not read(n+'/SUMMARY.json')['cuda_initialized']
    for sink,name in [(frames,'samples'),(components,'stages'),(checks,'checks'),(memory,'memory')]:sink.append(pd.read_csv(HERE/'reports'/n/(name+'.csv')))
d=pd.concat(frames);st=pd.concat(components);c=pd.concat(checks);mem=pd.concat(memory);keys=['scene','round','condition']
groups=c.groupby(keys+['frame']);assert groups.ids_sha256.nunique().eq(1).all() and groups.score_sha256.nunique().eq(1).all() and c.finite.all()
def pair(rows,keys,col):
    a=rows.groupby(keys+['variant']).agg(ms=(col,'median'),p95=(col,lambda x:x.quantile(.95)),maximum_ms=(col,'max'),samples=(col,'size')).reset_index();base=a[a.variant=='baseline'][keys+['ms','p95']].rename(columns={'ms':'baseline_ms','p95':'baseline_p95'})
    a=a[a.variant!='baseline'].merge(base,on=keys);a['speedup']=a.baseline_ms/a.ms;a['reduction_percent']=100*(1-a.ms/a.baseline_ms);return a
cpu=pair(d,keys,'ms');cpu=cpu.merge(d.groupby(keys).agg(N=('N','first'),M=('M','median'),K=('K','median')).reset_index(),on=keys)
comp=pair(st,keys+['stage'],'ms')
for name,table in [('CPU_PAIRED_RESULTS',cpu),('COMPONENT_PAIRED_RESULTS',comp),('CPU_CORRECTNESS',c),('CPU_MEMORY',mem)]:table.to_csv(HERE/'reports'/(name+'.csv'),index=False)
agg=[]
for (variant,condition),g in cpu.groupby(['variant','condition']):
    cc=comp[(comp.variant==variant)&(comp.condition==condition)&(comp.stage==4)]
    agg.append(dict(variant=variant,condition=condition,geometric_speedup=geo(g.speedup),baseline_ms=g.baseline_ms.median(),optimized_ms=g.ms.median(),slower_rounds=int((g.speedup<1).sum()),scene_rounds=len(g),
        compensator_speedup=geo(cc.speedup) if len(cc) else None,comp_baseline_ms=cc.baseline_ms.median() if len(cc) else None,comp_optimized_ms=cc.ms.median() if len(cc) else None))
pd.DataFrame(agg).to_csv(HERE/'reports/CPU_AGGREGATE.csv',index=False)
handoff=[];raster=[]
for run in ['handoff01','handoff02']:
    valid(run);assert read(run+'/SUMMARY.json')['passed'];handoff.append(pd.read_csv(HERE/'reports'/run/'samples.csv'));raster.append(pd.read_csv(HERE/'reports'/run/'raster.csv'))
hraw=pd.concat(handoff);h=pair(hraw,keys,'total_ms');hs=pair(hraw,keys,'host_ms');rr=pair(pd.concat(raster),keys,'gpu_event_ms')
for name,table in [('HANDOFF_PAIRED_RESULTS',h),('HOST_PAIRED_RESULTS',hs),('RASTER_DIAGNOSTICS',rr)]:table.to_csv(HERE/'reports'/(name+'.csv'),index=False)
q=pd.read_csv(HERE/'reports/quality01/quality.csv');assert read('quality01/SUMMARY.json')['passed'] and q.same_ids.all()
qc=q.groupby(['scene','condition']).agg(psnr_loss=('psnr_loss','mean'),worst_psnr_loss=('psnr_loss','max'),baseline_psnr=('baseline_psnr','mean'),new_psnr=('new_psnr','mean'),baseline_ssim=('baseline_ssim','mean'),new_ssim=('new_ssim','mean'),baseline_lpips=('baseline_lpips','mean'),new_lpips=('new_lpips','mean'),minimum_image_psnr=('image_equivalence_psnr','min'),K=('K','median')).reset_index()
q.to_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv',index=False);qc.to_csv(HERE/'reports/QUALITY_PER_CONDITION.csv',index=False)
assert q.psnr_loss.mean()<=.05 and qc.psnr_loss.max()<=.2
v=read('FROZEN_CANDIDATE.json')['variant'];main=cpu[(cpu.variant==v)&cpu.condition.isin(PRIMARY)];mc=comp[(comp.variant==v)&comp.condition.isin(PRIMARY)&(comp.stage==4)];mh=h[h.condition.isin(PRIMARY)];qp=q[q.condition.isin(PRIMARY)]
slow=cpu[(cpu.variant==v)&(cpu.speedup<1)];slow.to_csv(HERE/'reports/ALL_CPU_SLOWER_CASES.csv',index=False)
result=dict(passed=True,variant=v,scenes=int(d.scene.nunique()),CPU_samples=len(d),component_records=len(st),checks=len(c),handoff_samples=len(hraw),quality_rows=len(q),
    primary_CPU_speedup=geo(main.speedup),primary_compensator_speedup=geo(mc.speedup),primary_handoff_speedup=geo(mh.speedup),primary_CPU_slower_rounds=int((main.speedup<1).sum()),primary_handoff_slower_rounds=int((mh.speedup<1).sum()),
    primary_mean_psnr_loss=float(qp.psnr_loss.mean()),worst_psnr_loss=float(q.psnr_loss.max()),worst_condition_mean_loss=float(qc.psnr_loss.max()),minimum_image_psnr=float(q.image_equivalence_psnr.min()),max_rgb_error=float(q.max_rgb_error.max()),radii_mismatches=int((~q.same_radii).sum()),score_ids_identical=True)
(HERE/'reports/RESULTS_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(pd.DataFrame(agg).query('variant==@v').round(6).to_string(index=False));print(json.dumps(result,indent=2))
