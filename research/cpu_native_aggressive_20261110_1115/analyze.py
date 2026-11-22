"""从同一冻结二进制生成全部性能/质量结果，不删反例。"""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--runs',nargs=2,default=['paired01','paired02']);args=p.parse_args()
frames=[];correct=[];stages=[];startup=[]
for name in args.runs:
    root=HERE/'reports'/name;summary=json.loads((root/'SUMMARY.json').read_text());assert summary['passed'] and not summary['cuda_initialized']
    job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text());assert job['measurement_valid'] and not job.get('foreign_cpu_active_samples',0) and not job['foreign_active_samples'],job
    frames.append(pd.read_csv(root/'samples.csv'));correct.append(pd.read_csv(root/'checks.csv'));stages.append(pd.read_csv(root/'stages.csv'));startup.append(pd.read_csv(root/'startup.csv'))
d=pd.concat(frames,ignore_index=True);checks=pd.concat(correct,ignore_index=True)
keys=['scene','slug','round','condition'];s=d.groupby(keys+['variant']).agg(median_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),samples=('ms','size'),N=('N','first'),M=('M','median'),K=('K','median')).reset_index()
old=s[s.variant=='pytorch'][keys+['median_ms','p95_ms']].rename(columns={'median_ms':'pytorch_ms','p95_ms':'pytorch_p95_ms'})
s=s[s.variant=='native'].merge(old,on=keys);s['speedup']=s.pytorch_ms/s.median_ms;s['saved_ms']=s.pytorch_ms-s.median_ms
s.to_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv',index=False);checks.to_csv(HERE/'reports/CPU_PAIRED_CHECKS.csv',index=False)
groups=[]
for condition,g in s.groupby('condition'):
    groups.append(dict(condition=condition,scenes=g.scene.nunique(),rounds=len(g),geometric_speedup=float(np.exp(np.log(g.speedup).mean())),
        median_native_ms=float(g.median_ms.median()),median_pytorch_ms=float(g.pytorch_ms.median()),
        min_speedup=float(g.speedup.min()),max_speedup=float(g.speedup.max()),slower=int((g.speedup<1).sum()),
        native_ms_min=float(g.median_ms.min()),native_ms_max=float(g.median_ms.max())))
a=pd.DataFrame(groups);a.to_csv(HERE/'reports/CPU_SPEED_AGGREGATE.csv',index=False)
pd.concat(stages,ignore_index=True).to_csv(HERE/'reports/CPU_STAGES.csv',index=False)
pd.concat(startup,ignore_index=True).to_csv(HERE/'reports/CPU_STARTUP_MEMORY.csv',index=False)
q=pd.read_csv(HERE/'reports/quality_all01/quality.csv');quality=q.groupby(['scene','condition']).agg(frames=('frame','size'),
    mean_psnr_loss=('psnr_loss','mean'),worst_psnr_loss=('psnr_loss','max'),min_image_equivalence=('cpu_vs_reference_psnr','min'),
    differing_ids=('ids_equal',lambda x:int((~x).sum())),mean_ssim_reference=('reference_ssim','mean'),mean_ssim_cpu=('cpu_ssim','mean'),
    mean_lpips_reference=('reference_lpips','mean'),mean_lpips_cpu=('cpu_lpips','mean')).reset_index()
quality['ssim_loss']=quality.mean_ssim_reference-quality.mean_ssim_cpu;quality['lpips_increase']=quality.mean_lpips_cpu-quality.mean_lpips_reference
quality.to_csv(HERE/'reports/CPU_QUALITY_PER_CONDITION.csv',index=False)
q.sort_values('psnr_loss',ascending=False).to_csv(HERE/'reports/CPU_QUALITY_ALL_FRAMES.csv',index=False)
assert q.psnr_loss.mean()<=.05 and quality.mean_psnr_loss.max()<=.2
summary=dict(passed=True,runs=args.runs,scenes=s.scene.nunique(),samples=len(d),checks=len(checks),
    all_cpu_only=True,IDs_equal_checks=int(checks.same_ids.sum()),nonfinite_checks=int((~checks.all_finite).sum()),
    quality_rows=len(q),quality_ID_changes=int((~q.ids_equal).sum()),mean_psnr_loss=float(q.psnr_loss.mean()),
    worst_psnr_loss=float(q.psnr_loss.max()),minimum_equivalence_psnr=float(q.cpu_vs_reference_psnr.min()),
    max_condition_ssim_loss=float(quality.ssim_loss.max()),max_condition_lpips_increase=float(quality.lpips_increase.max()),
    results=groups)
(HERE/'reports/RESULTS_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(a.round(5).to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
