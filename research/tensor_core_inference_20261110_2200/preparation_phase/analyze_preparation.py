from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;R=HERE/'reports'
read=lambda n:json.loads((R/n).read_text())
geo=lambda x:float(np.exp(np.log(x).mean()))
controller=read('FORMAL_CONTROLLER.json');assert controller['state']=='complete',controller
raw=[];checks=[];memory=[]
for name in controller['accepted']:
    j=read('jobs/'+name+'.json');assert j['returncode']==0 and j['measurement_valid'] and not j['foreign_active_samples'] and not j['foreign_cpu_active_samples'],j
    raw.append(pd.read_csv(R/name/'samples.csv'));checks.append(pd.read_csv(R/name/'checks.csv'));memory.append(pd.read_csv(R/name/'memory.csv'))
raw=pd.concat(raw,ignore_index=True);checks=pd.concat(checks,ignore_index=True);memory=pd.concat(memory,ignore_index=True)
keys=['scene','condition','scope','round'];groups=raw.groupby(keys+['variant']).ms
assert raw.scene.nunique()==16 and raw['round'].nunique()==2
assert raw.groupby(keys+['variant']).size().eq(48).all()
assert checks.groupby(['scene','condition','round','frame']).K.nunique().eq(1).all()
assert checks.groupby(['scene','condition','round','frame']).M.nunique().eq(1).all()
timings=groups.agg(p50='median',p95=lambda x:x.quantile(.95),maximum='max',samples='size').reset_index()
meta=raw.groupby(keys).agg(N=('N','first'),M=('M','median'),K=('K','median')).reset_index()
timings=timings.merge(meta,on=keys);timings.to_csv(R/'TIMINGS_ALL_VARIANTS.csv',index=False)
pairs=[]
for ref in ['A','B','C','D']:
    base=timings[timings.variant==ref][keys+['p50','p95']].rename(columns={'p50':'reference_p50','p95':'reference_p95'})
    t=timings.merge(base,on=keys);t=t[t.variant!=ref].copy();t['reference']=ref
    t['speedup']=t.reference_p50/t.p50;t['latency_reduction_percent']=100*(1-t.p50/t.reference_p50);t['saved_ms']=t.reference_p50-t.p50
    pairs.append(t)
pairs=pd.concat(pairs,ignore_index=True);pairs.to_csv(R/'PAIRED_ALL_VARIANTS.csv',index=False)
pairs[pairs.speedup<1].to_csv(R/'ALL_SLOWER_CASES.csv',index=False)
agg=[]
for (ref,variant,scope,condition),g in pairs.groupby(['reference','variant','scope','condition']):
    agg.append(dict(reference=ref,variant=variant,scope=scope,condition=condition,speedup=geo(g.speedup),reference_p50=g.reference_p50.median(),p50=g.p50.median(),
        reference_p95=g.reference_p95.median(),p95=g.p95.median(),saved_ms=g.saved_ms.median(),slower_rounds=int((g.speedup<1).sum()),scene_rounds=len(g)))
pd.DataFrame(agg).to_csv(R/'AGGREGATE_ALL_VARIANTS.csv',index=False)
primary=['local0.1','local0.2','local0.3','local0.4','local0.5'];summary=[]
for (ref,variant,scope),g in pairs[pairs.condition.isin(primary)].groupby(['reference','variant','scope']):
    summary.append(dict(reference=ref,variant=variant,scope=scope,speedup=geo(g.speedup),reference_p50=g.reference_p50.median(),p50=g.p50.median(),
        reference_p95=g.reference_p95.median(),p95=g.p95.median(),saved_ms=g.saved_ms.median(),slower_rounds=int((g.speedup<1).sum()),scene_rounds=len(g)))
summ=pd.DataFrame(summary);summ.to_csv(R/'PRIMARY_SUMMARY.csv',index=False)
memory.to_csv(R/'MEMORY_ALL_VARIANTS.csv',index=False);checks.to_csv(R/'SELECTION_ALL_VARIANTS.csv',index=False)
raw.groupby(['scene','condition','scope','variant','motion']).ms.agg(p50='median',p95=lambda x:x.quantile(.95),samples='size').reset_index().to_csv(R/'MOTION_TIMINGS.csv',index=False)
quality=pd.read_csv(R/'quality_final/quality.csv');quality.to_csv(R/'QUALITY_ALL_FRAMES.csv',index=False)
q=quality.groupby(['scene','condition','variant']).agg(psnr_loss=('psnr_loss','mean'),extra_C_psnr_loss=('extra_C_psnr_loss','mean'),
    worst_frame_loss=('psnr_loss','max'),baseline_psnr=('baseline_psnr','mean'),psnr=('psnr','mean'),ssim=('ssim','mean'),lpips=('lpips','mean'),K=('K','median'),M=('M','median')).reset_index()
q.to_csv(R/'QUALITY_PER_CONDITION.csv',index=False)
assert read('quality_final/SUMMARY.json')['summary']['D']['passed']
quality_summary={variant:dict(mean_psnr_loss=float(g.psnr_loss.mean()),worst_frame_loss=float(g.psnr_loss.max()),
    worst_condition_mean_loss=float(q[q.variant==variant].psnr_loss.max()),extra_C_mean_loss=float(g.extra_C_psnr_loss.mean()),
    same_ids_A_fraction=float(g.same_ids_A.mean()),same_ids_C_fraction=float(g.same_ids_C.mean())) for variant,g in quality.groupby('variant')}
component_runs=read('COMPONENT_MEASUREMENT.json')['accepted'];component_tables=[]
for name in component_runs:
    assert read('jobs/'+name+'.json')['measurement_valid']
    component_tables.append(pd.read_csv(R/name/'samples.csv'))
components=pd.concat(component_tables,ignore_index=True);assert components.scene.nunique()==16 and len(components)==9600
co=components.groupby(['scene','ratio','stage','variant']).ms.agg(p50='median',p95=lambda x:x.quantile(.95),samples='size').reset_index()
co.to_csv(R/'COMPONENT_RESULTS.csv',index=False)
result=dict(passed=True,scenes=16,rounds=2,samples=len(raw),selection_checks=len(checks),quality_rows=len(quality),quality=quality_summary,
    primary=summary,peak_projection_bytes=int(memory.projection_bytes.max()),measurements=controller['accepted'],rejected=controller['rejected'])
observations=[]
for name in controller['accepted']:
    for line in (HERE/'logs'/(name+'_MONITOR.jsonl')).read_text().splitlines():
        item=json.loads(line);parts=item['gpu'].split(',')
        observations.append(dict(clock_mhz=float(parts[2].strip().split()[0]),temperature_c=float(parts[3].strip().split()[0]),
            foreign_gpu=bool(item['foreign_active']),foreign_cpu=bool(item['foreign_cpu_active'])))
result['monitor']=dict(observations=len(observations),clock_mhz_min=min(x['clock_mhz'] for x in observations),
    clock_mhz_max=max(x['clock_mhz'] for x in observations),temperature_c_min=min(x['temperature_c'] for x in observations),
    temperature_c_max=max(x['temperature_c'] for x in observations),foreign_gpu_samples=sum(x['foreign_gpu'] for x in observations),
    foreign_cpu_samples=sum(x['foreign_cpu'] for x in observations))
(R/'RESULTS_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(summ[(summ.variant=='D')&summ.reference.isin(['A','B','C'])].to_string(index=False));print('PREPARATION_ANALYSIS_PASS',len(raw),quality_summary)
