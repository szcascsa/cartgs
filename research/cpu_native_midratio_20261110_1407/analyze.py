"""只汇总实测双轮数据；低Ratio退化和Full全部保留。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/'reports'/name).read_text())

frames=[];checks=[];stages=[];memory=[]
for run in ['formal01','formal02']:
    job=read('jobs/'+run+'.json');summary=read(run+'/SUMMARY.json')
    assert job['measurement_valid'] and not job['foreign_active_samples'] and not job['foreign_cpu_active_samples'],job
    assert summary['passed'] and summary['nonrotation_scores_bitwise'] and not summary['cuda_initialized']
    for sink,name in [(frames,'samples'),(checks,'checks'),(stages,'stages'),(memory,'memory')]:sink.append(pd.read_csv(HERE/'reports'/run/(name+'.csv')))
d=pd.concat(frames);c=pd.concat(checks);s=pd.concat(stages);mem=pd.concat(memory)
groups=c.groupby(['scene','round','condition','frame'])
assert groups.ids_sha256.nunique().eq(1).all() and groups.attributes_sha256.nunique().eq(1).all() and c.all_finite.all() and groups.score_sha256.nunique().eq(1).all() and c.rotation_max_error.max()<=1e-6
keys=['scene','slug','condition','round']
a=d.groupby(keys+['variant']).agg(ms=('ms','median'),p95=('ms',lambda v:v.quantile(.95)),maximum_ms=('ms','max'),N=('N','first'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
baseline=a[a.variant=='baseline'][keys+['ms','p95','maximum_ms']].rename(columns={'ms':'baseline_ms','p95':'baseline_p95','maximum_ms':'baseline_maximum_ms'})
a=a[a.variant!='baseline'].merge(baseline,on=keys);a['speedup']=a.baseline_ms/a.ms;a['reduction_percent']=100*(1-a.ms/a.baseline_ms)
for name,data in [('CPU_PAIRED_RESULTS',a),('CPU_CORRECTNESS',c),('CPU_STAGES',s),('CPU_MEMORY',mem)]:data.to_csv(HERE/'reports'/(name+'.csv'),index=False)
aggregate=[]
for (variant,condition),g in a.groupby(['variant','condition']):
    aggregate.append(dict(variant=variant,condition=condition,geometric_speedup=float(np.exp(np.log(g.speedup).mean())),
        median_baseline_ms=float(g.baseline_ms.median()),median_new_ms=float(g.ms.median()),min_speedup=float(g.speedup.min()),max_speedup=float(g.speedup.max()),
        slower_rounds=int((g.speedup<1).sum()),scene_rounds=len(g)))
pd.DataFrame(aggregate).to_csv(HERE/'reports/CPU_AGGREGATE.csv',index=False)
motion=d.groupby(keys+['motion','variant']).ms.median().unstack();motion['speedup']=motion.baseline/motion.combined
motion.to_csv(HERE/'reports/CPU_MOTION_RESULTS.csv')
slow=a[(a.variant=='combined')&(a.speedup<1)]
slow.to_csv(HERE/'reports/CPU_ALL_SLOWER_CASES.csv',index=False)
handoff=[];hchecks=[];hmem=[]
for run in ['handoff01','handoff02']:
    job=read('jobs/'+run+'.json');assert job['measurement_valid'] and not job['foreign_active_samples'] and not job['foreign_cpu_active_samples'],job
    assert read(run+'/SUMMARY.json')['passed']
    handoff.append(pd.read_csv(HERE/'reports'/run/'samples.csv'));hchecks.append(pd.read_csv(HERE/'reports'/run/'checks.csv'));hmem.append(pd.read_csv(HERE/'reports'/run/'memory.csv'))
h=pd.concat(handoff);hc=pd.concat(hchecks);hk=['scene','round','condition']
hs=h.groupby(hk+['variant']).agg(ms=('total_ms','median'),p95=('total_ms',lambda x:x.quantile(.95)),maximum_ms=('total_ms','max'),host_ms=('host_prepare_queue_ms','median'),K=('K','median'),samples=('total_ms','size')).reset_index()
hb=hs[hs.variant=='baseline'][hk+['ms','p95','host_ms']].rename(columns={'ms':'baseline_ms','p95':'baseline_p95','host_ms':'baseline_host_ms'})
hs=hs[hs.variant!='baseline'].merge(hb,on=hk);hs['speedup']=hs.baseline_ms/hs.ms;hs['reduction_percent']=100*(1-hs.ms/hs.baseline_ms)
hs.to_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv',index=False);hc.to_csv(HERE/'reports/HANDOFF_CORRECTNESS.csv',index=False);pd.concat(hmem).to_csv(HERE/'reports/HANDOFF_MEMORY.csv',index=False)
q=pd.read_csv(HERE/'reports/quality01/quality.csv');q.to_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv',index=False)
qc=q.groupby(['scene','condition']).agg(mean_psnr_loss=('psnr_loss','mean'),worst_psnr_loss=('psnr_loss','max'),same_ids=('same_ids','all'),same_rgb=('same_rgb','all'),same_radii=('same_radii','all'),
    baseline_psnr=('baseline_psnr','mean'),new_psnr=('new_psnr','mean'),baseline_ssim=('baseline_ssim','mean'),new_ssim=('new_ssim','mean'),baseline_lpips=('baseline_lpips','mean'),new_lpips=('new_lpips','mean')).reset_index()
qc.to_csv(HERE/'reports/QUALITY_PER_CONDITION.csv',index=False)
assert hc[['same_radii','same_ids']].all().all() and q[['same_ids','same_radii']].all().all()
assert q.psnr_loss.mean()<=.05 and qc.mean_psnr_loss.max()<=.2
result=dict(passed=True,scenes=int(d.scene.nunique()),CPU_samples=len(d),CPU_checks=len(c),CPU_nonrotation_scores_bitwise=True,CPU_rotation_bitwise=bool(c.rotation_bitwise.all()),CPU_rotation_max_abs_error=float(c.rotation_max_error.max()),
    handoff_samples=len(h),handoff_checks=len(hc),handoff_slower_rounds=int((hs.speedup<1).sum()),quality_rows=len(q),quality_RGB_bitwise=bool(q.same_rgb.all()),quality_max_rgb_error=float(q.max_rgb_error.max()),mean_psnr_loss=float(q.psnr_loss.mean()),worst_psnr_loss=float(q.psnr_loss.max()),
    CPU_slower_rounds=len(slow),aggregate=aggregate)
primary=['local0.1','local0.2','local0.3','local0.4','local0.5'];main=a[(a.variant=='combined')&a.condition.isin(primary)];he=hs[hs.condition.isin(primary)]
result.update(primary_CPU_speedup=float(np.exp(np.log(main.speedup).mean())),primary_handoff_speedup=float(np.exp(np.log(he.speedup).mean())),primary_CPU_slower_rounds=int((main.speedup<1).sum()),primary_handoff_slower_rounds=int((he.speedup<1).sum()))
(HERE/'reports/RESULTS_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(pd.DataFrame(aggregate).query("variant=='combined'").round(5).to_string(index=False));print(json.dumps({k:v for k,v in result.items() if k!='aggregate'},indent=2))
