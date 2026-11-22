from pathlib import Path
import json
import pandas as pd,numpy as np

HERE=Path(__file__).resolve().parent;R=HERE/'reports';runs=json.loads((R/'TRANSFER_MEASUREMENTS.json').read_text())['accepted']
data=[];checks=[]
for name in runs:
    job=json.loads((R/'jobs'/(name+'.json')).read_text());assert job['measurement_valid'],job
    data.append(pd.read_csv(R/name/'samples.csv'));checks.append(pd.read_csv(R/name/'checks.csv'))
d=pd.concat(data,ignore_index=True);q=pd.concat(checks,ignore_index=True)
assert d.scene.nunique()==4 and d['round'].nunique()==2 and len(q)==240
keys=['scene','ratio','scope','round'];a=d.groupby(keys+['variant']).ms.agg(p50='median',p95=lambda x:x.quantile(.95),samples='size').reset_index()
base=a[a.variant=='C_teacher'][keys+['p50','p95']].rename(columns={'p50':'reference_p50','p95':'reference_p95'})
a=a[a.variant=='D_preparation_teacher'].merge(base,on=keys);a['speedup']=a.reference_p50/a.p50
a=a.merge(d.groupby(keys).agg(N=('N','first'),M=('M','median'),K=('K','median')).reset_index(),on=keys)
a.to_csv(R/'TRANSFER_RESULTS.csv',index=False);q.to_csv(R/'TRANSFER_QUALITY.csv',index=False)
rows=[]
for (scene,scope),g in a.groupby(['scene','scope']):
    rows.append(dict(scene=scene,scope=scope,N=int(g.N.iloc[0]),reference_p50=g.reference_p50.median(),p50=g.p50.median(),
        reference_p95=g.reference_p95.median(),p95=g.p95.median(),speedup=float(np.exp(np.log(g.speedup).mean())),
        slower_rounds=int((g.speedup<1).sum()),round_conditions=len(g)))
pd.DataFrame(rows).to_csv(R/'TRANSFER_SUMMARY.csv',index=False)
summary=q.groupby(['scene','ratio']).psnr_loss.mean()
result=dict(passed=bool(q.psnr_loss.mean()<=.05 and summary.max()<=.2),samples=len(d),quality_checks=len(q),
    mean_psnr_loss=float(q.psnr_loss.mean()),worst_frame_loss=float(q.psnr_loss.max()),worst_condition_mean_loss=float(summary.max()),
    all_same_ids=bool(q.same_ids.all()),scope='准备迁移；Selector FP16和原Teacher CUDA补偿相同，无同场景学生预投影')
(R/'TRANSFER_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False));print(result);assert result['passed']
