"""冻结后的双轮、全部慢例、单模块归因与质量；不重复计算旧优化收益。"""
from pathlib import Path
import json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;R=HERE/'reports'
def read(n):return json.loads((R/n).read_text())
def geo(x):return float(np.exp(np.log(x).mean()))
def pair(d,keys):
    a=d.groupby(keys+['variant']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),samples=('ms','size'),maximum=('ms','max')).reset_index()
    base=a[a.variant=='original'][keys+['ms','p95']].rename(columns={'ms':'baseline_ms','p95':'baseline_p95'})
    a=a[a.variant!='original'].merge(base,on=keys);a['speedup']=a.baseline_ms/a.ms;a['reduction_percent']=100*(1-a.ms/a.baseline_ms);return a
tables=[];checks=[];mem=[]
for name in read('MEASUREMENT_RUNS.json')['formal']:
    job=read('jobs/'+name+'.json');assert job['returncode']==0 and job['measurement_valid'] and not job['foreign_active_samples'] and not job['foreign_cpu_active_samples'],job
    assert read(name+'/SUMMARY.json')['passed']
    tables.append(pd.read_csv(R/name/'samples.csv'));checks.append(pd.read_csv(R/name/'checks.csv'));mem.append(pd.read_csv(R/name/'memory.csv'))
raw=pd.concat(tables);c=pd.concat(checks);m=pd.concat(mem)
assert raw.scene.nunique()==16 and raw['round'].nunique()==2
assert raw.groupby(['scene','round','condition','scope','variant']).size().eq(36).all(), '完整工作量不匹配'
assert c.groupby(['scene','round','condition','frame']).K.nunique().eq(1).all(), '严格K变化'
cpu=pair(raw,['scene','condition','scope','round']);cpu=cpu.merge(raw.groupby(['scene','condition','scope','round']).agg(N=('N','first'),M=('M','median'),K=('K','median')).reset_index(),on=['scene','condition','scope','round'])
cpu.to_csv(R/'PAIRED_RESULTS.csv',index=False);c.to_csv(R/'SELECTION_CHECKS.csv',index=False);m.to_csv(R/'MEMORY.csv',index=False);cpu[cpu.speedup<1].to_csv(R/'ALL_SLOWER_CASES.csv',index=False)
agg=[]
for (v,s,cond),g in cpu.groupby(['variant','scope','condition']):agg.append(dict(variant=v,scope=s,condition=cond,speedup=geo(g.speedup),baseline_ms=g.baseline_ms.median(),optimized_ms=g.ms.median(),slower_rounds=int((g.speedup<1).sum()),scene_rounds=len(g)))
pd.DataFrame(agg).to_csv(R/'AGGREGATE.csv',index=False)
primary=['local0.1','local0.2','local0.3','local0.4','local0.5'];g=cpu[(cpu.variant=='both')&cpu.condition.isin(primary)]
q=pd.read_csv(R/'quality03/quality.csv');qc=q.groupby(['scene','variant','condition']).agg(psnr_loss=('psnr_loss','mean'),worst_frame_loss=('psnr_loss','max'),baseline_psnr=('baseline_psnr','mean'),psnr=('psnr','mean'),ssim=('ssim','mean'),lpips=('lpips','mean'),minimum_jaccard=('jaccard','min'),K=('K','median')).reset_index()
q.to_csv(R/'QUALITY_ALL_FRAMES.csv',index=False);qc.to_csv(R/'QUALITY_PER_CONDITION.csv',index=False);assert read('quality03/SUMMARY.json')['passed']
v=q[q.variant=='both'];vp=v[v.condition.isin(primary)]
component=pd.read_csv(R/'components_final/samples.csv');job=read('jobs/components_final.json');assert job['measurement_valid'],job
co=pair(component,['scene','ratio','stage']);co.to_csv(R/'COMPONENT_RESULTS.csv',index=False)
f=read('FROZEN_CANDIDATE.json');cfg=lambda stage:'m%d_w%d'%((f['selector'],f['warps']) if stage=='selector' else (f['compensator'],f['comp_warps']))
comp_summary={stage:geo(co[(co.stage==stage)&(co.variant==cfg(stage))].speedup) for stage in ['selector','compensator']}
by_variant={}
for variant in ['selector','compensator','both']:
    x=cpu[(cpu.variant==variant)&cpu.condition.isin(primary)];by_variant[variant]={scope:dict(speedup=geo(z.speedup),slower_rounds=int((z.speedup<1).sum()),scene_rounds=len(z)) for scope,z in x.groupby('scope')}
result=dict(passed=True,scenes=int(raw.scene.nunique()),samples=len(raw),checks=len(c),quality_cases=len(q),components=comp_summary,primary=by_variant,
    primary_mean_psnr_loss=float(vp.psnr_loss.mean()),all_mean_psnr_loss=float(v.psnr_loss.mean()),worst_frame_loss=float(v.psnr_loss.max()),worst_condition_mean_loss=float(qc[qc.variant=='both'].psnr_loss.max()),
    minimum_jaccard=float(v.jaccard.min()),same_ids_fraction=float(v.same_ids.mean()),minimum_image_psnr=float(v.image_psnr.min()),max_rgb_error=float(v.max_rgb_error.max()),radii_mismatches=int((~v.same_radii).sum()),
    tensor_weight_bytes=int(m[m.variant=='both'].tensor_weight_bytes.max()),full_prepare=geo(cpu[(cpu.variant=='both')&(cpu.condition=='full')&(cpu.scope=='prepare')].speedup),full_pipeline=geo(cpu[(cpu.variant=='both')&(cpu.condition=='full')&(cpu.scope=='pipeline')].speedup))
save=R/'RESULTS_SUMMARY.json';save.write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2,ensure_ascii=False))
