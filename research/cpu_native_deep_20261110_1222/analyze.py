"""汇总全部场景/条件/慢例，独立区分CPU、Host排队和完整渲染延迟。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/'reports'/name).read_text())
frames=[];checks=[];stages=[];memory=[]
isolated_checks=[]
for run in ['isolated01','isolated02']:
    job=read('jobs/'+run+'.json');assert job['measurement_valid'] and not job['foreign_active_samples'] and not job['foreign_cpu_active_samples'],job
    summary=read(run+'/SUMMARY.json');assert summary['passed'] and not summary['cuda_initialized']
    for sink,name in [(frames,'samples'),(isolated_checks,'checks'),(stages,'stages'),(memory,'memory')]:sink.append(pd.read_csv(HERE/'reports'/run/(name+'.csv')))
for run in ['formal01','formal02']:checks.append(pd.read_csv(HERE/'reports'/run/'checks.csv'))
d=pd.concat(frames);c=pd.concat(checks);s=pd.concat(stages);mem=pd.concat(memory)
identities=pd.concat(isolated_checks);assert identities.groupby(['scene','round','condition','frame']).ids_sha256.nunique().eq(1).all()
identities.to_csv(HERE/'reports/CPU_ISOLATED_ID_CHECKS.csv',index=False)
keys=['scene','slug','condition','round']
a=d.groupby(keys+['variant']).agg(ms=('ms','median'),p95=('ms',lambda v:v.quantile(.95)),N=('N','first'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
baseline=a[a.variant=='baseline'][keys+['ms','p95']].rename(columns={'ms':'baseline_ms','p95':'baseline_p95'})
a=a[a.variant!='baseline'].merge(baseline,on=keys);a['speedup']=a.baseline_ms/a.ms;a['reduction_percent']=100*(1-a.ms/a.baseline_ms)
a.to_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv',index=False);c.to_csv(HERE/'reports/CPU_CORRECTNESS.csv',index=False)
s.to_csv(HERE/'reports/CPU_STAGES.csv',index=False);mem.to_csv(HERE/'reports/CPU_MEMORY.csv',index=False)
aggregate=[]
for (variant,condition),g in a.groupby(['variant','condition']):
    aggregate.append(dict(variant=variant,condition=condition,geometric_speedup=float(np.exp(np.log(g.speedup).mean())),
        median_baseline_ms=float(g.baseline_ms.median()),median_new_ms=float(g.ms.median()),min_speedup=float(g.speedup.min()),max_speedup=float(g.speedup.max()),
        slower_rounds=int((g.speedup<1).sum()),scene_rounds=len(g)))
pd.DataFrame(aggregate).to_csv(HERE/'reports/CPU_AGGREGATE.csv',index=False)
# 多线程池共驻留的所有原始慢例也独立保留，不用单池数据覆盖。
multi=pd.concat([pd.read_csv(HERE/'reports'/run/'samples.csv') for run in ['formal01','formal02']])
mp=multi.groupby(['scene','round','condition','variant']).ms.median().unstack();mp['speedup']=mp.baseline/mp.combined
mp.to_csv(HERE/'reports/CPU_MULTIPOOL_RESULTS.csv')
handoff=[]
for run in ['handoff01','handoff02']:
    assert read('jobs/'+run+'.json')['measurement_valid']
    assert read(run+'/SUMMARY.json')['passed']
    handoff.append(pd.read_csv(HERE/'reports'/run/'samples.csv'))
h=pd.concat(handoff);hk=['scene','round','condition'];hs=h.groupby(hk+['variant']).agg(ms=('total_ms','median'),p95=('total_ms',lambda x:x.quantile(.95)),host_ms=('host_prepare_queue_ms','median'),K=('K','median')).reset_index()
hb=hs[hs.variant=='baseline'][hk+['ms','host_ms']].rename(columns={'ms':'baseline_ms','host_ms':'baseline_host_ms'})
hs=hs[hs.variant!='baseline'].merge(hb,on=hk);hs['speedup']=hs.baseline_ms/hs.ms;hs['reduction_percent']=100*(1-hs.ms/hs.baseline_ms)
hs.to_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv',index=False)
q=pd.read_csv(HERE/'reports/quality02/quality.csv');q.to_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv',index=False)
qc=q.groupby(['scene','condition']).agg(mean_psnr_loss=('psnr_loss','mean'),worst_psnr_loss=('psnr_loss','max'),same_ids=('ids_equal','all'),
    ssim_reference=('reference_ssim','mean'),ssim_new=('cpu_ssim','mean'),lpips_reference=('reference_lpips','mean'),lpips_new=('cpu_lpips','mean')).reset_index()
qc.to_csv(HERE/'reports/QUALITY_PER_CONDITION.csv',index=False)
assert c.same_ids.all() and q.ids_equal.all() and q.psnr_loss.mean()<=.05 and qc.mean_psnr_loss.max()<=.2
result=dict(passed=True,scenes=d.scene.nunique(),CPU_samples=len(d),CPU_checks=len(c),CPU_all_bitwise=bool(c.bitwise.all()),CPU_max_abs_error=float(c.max_abs_error.max()),
    CPU_isolated_identity_checks=len(identities),CPU_multipool_samples=len(multi),
    handoff_samples=len(h),quality_rows=len(q),mean_psnr_loss=float(q.psnr_loss.mean()),worst_psnr_loss=float(q.psnr_loss.max()),minimum_image_equivalence_psnr=float(q.cpu_vs_reference_psnr.min()),
    aggregate=aggregate,handof_slower_rounds=int(((hs.variant=='native_all')&(hs.speedup<1)).sum()))
(HERE/'reports/RESULTS_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(pd.DataFrame(aggregate).query("variant=='combined'").round(5).to_string(index=False));print(json.dumps({k:v for k,v in result.items() if k!='aggregate'},indent=2))
