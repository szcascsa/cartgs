"""只纳入控制器认可的独立测量；固定属性和真实Student LoD分别汇总。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PRIMARY=[f'local{x:g}' for x in [.1,.2,.3,.4,.5]]


def main():
    state=json.loads((HERE/'reports/RENDER_FINAL_CONTROLLER.json').read_text())
    assert state['state']=='complete'
    results={}
    for kind,prefix in [('student_lod','student_render_formal'),('nested','student_render_nested')]:
        paths=[HERE/'reports'/name for name in state['accepted'] if name.startswith(prefix)]
        assert len(paths)==(18 if kind=='student_lod' else 32)
        for data in ['samples','events','quality','checks','workloads','memory']:
            parts=[]
            for path in paths:
                f=path/(data+'.csv')
                if f.exists() and f.stat().st_size>1:
                    part=pd.read_csv(f);part['workload']=kind;part['job']=path.name;parts.append(part)
            if parts:results.setdefault(data,[]).extend(parts)
    frames={name:pd.concat(parts,ignore_index=True) for name,parts in results.items()}
    for name,data in frames.items():data.to_csv(HERE/'reports'/('RENDER_FINAL_'+name.upper()+'.csv.gz'),index=False,compression='gzip')
    data=frames['samples'];keys=['workload','scene','condition','round','variant']
    stats=data.groupby(keys).agg(p50_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),
        mean_ms=('ms','mean'),K=('K','median'),K_min=('K','min'),K_max=('K','max'),N=('N','first'),samples=('ms','size')).reset_index()
    for name,label in [('F2','F2'),('tile_safe_bitexp_stop','old38')]:
        ref=stats[stats.variant==name][keys[:-1]+['p50_ms','p95_ms']].rename(columns={'p50_ms':label+'_p50_ms','p95_ms':label+'_p95_ms'})
        stats=stats.merge(ref,on=keys[:-1]);stats[label+'_speedup']=stats[label+'_p50_ms']/stats.p50_ms
    stats.to_csv(HERE/'reports/RENDER_SCENE_RATIO.csv',index=False)
    stats[(stats.condition=='full')&(stats.variant=='encoded_signed_square')].to_csv(HERE/'reports/FULL_ACCEPTANCE.csv',index=False)
    slow=stats[(stats.F2_speedup<1)|(stats.old38_speedup<1)].copy()
    slow.to_csv(HERE/'reports/RENDER_ALL_SLOW_CASES.csv',index=False)
    fkeys=['workload','scene','condition','round','frame','motion','variant']
    per=data.groupby(fkeys).agg(p50_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),K=('K','first')).reset_index()
    ref=per[per.condition=='full'][['workload','scene','round','frame','variant','p50_ms','K']].rename(columns={'p50_ms':'T_ref','K':'K_ref'})
    per=per.merge(ref,on=['workload','scene','round','frame','variant'])
    per['normalized_K']=per.K/per.K_ref;per['normalized_T']=per.p50_ms/per.T_ref;per['ideal_gap']=per.normalized_T-per.normalized_K
    per.to_csv(HERE/'reports/RENDER_PER_FRAME.csv.gz',index=False,compression='gzip')
    fits=[];inversions=[]
    for values,part in per[(per.condition.str.startswith('local'))|(per.condition=='full')].groupby(['workload','scene','round','frame','variant']):
        part=part.sort_values('K');k=part.K.to_numpy();t=part.p50_ms.to_numpy()*1000
        if len(np.unique(k))<2:continue
        design=np.stack([np.ones_like(k),k],axis=1);b,a=np.linalg.lstsq(design,t,rcond=None)[0];prediction=b+a*k
        total=((t-t.mean())**2).sum();r2=1-((t-prediction)**2).sum()/total if total else 1.
        base=dict(zip(['workload','scene','round','frame','variant'],values))
        fits.append(dict(**base,intercept_us=b,slope_us_per_1k=a*1000,R2=r2,mean_ideal_gap=part.ideal_gap.mean(),
            raw_inversions=int((np.diff(t)<0).sum()),inversions_gt2pct=int((t[1:]<t[:-1]*.98).sum())))
        for i in range(len(k)-1):
            if k[i+1]>k[i] and t[i+1]<t[i]:inversions.append(dict(**base,lower_condition=part.iloc[i].condition,higher_condition=part.iloc[i+1].condition,
                lower_K=int(k[i]),higher_K=int(k[i+1]),lower_us=t[i],higher_us=t[i+1],drop_pct=(1-t[i+1]/t[i])*100))
    pd.DataFrame(fits).to_csv(HERE/'reports/RENDER_SCALING_FIT.csv',index=False)
    pd.DataFrame(inversions).to_csv(HERE/'reports/RENDER_INVERSIONS.csv',index=False)
    q=frames['quality'];quality=q.groupby(['workload','scene','condition','variant']).agg(psnr_loss=('psnr_loss','mean'),
        worst_frame_loss=('psnr_loss','max'),baseline_psnr=('baseline_psnr','mean'),psnr=('psnr','mean'),
        ssim=('ssim','mean'),baseline_ssim=('baseline_ssim','mean'),lpips=('lpips','mean'),baseline_lpips=('baseline_lpips','mean')).reset_index()
    quality.to_csv(HERE/'reports/RENDER_QUALITY_CONDITIONS.csv',index=False)
    overview=[]
    for (kind,variant),part in stats[stats.condition.isin(PRIMARY)].groupby(['workload','variant']):
        values=quality[(quality.workload==kind)&(quality.variant==variant)]
        overview.append(dict(workload=kind,variant=variant,F2_speedup=np.exp(np.log(part.F2_speedup).mean()),
            old38_speedup=np.exp(np.log(part.old38_speedup).mean()),F2_worst=part.F2_speedup.min(),old38_worst=part.old38_speedup.min(),
            mean_psnr_loss=values.psnr_loss.mean() if len(values) else 0.,worst_condition=values.psnr_loss.max() if len(values) else 0.,
            worst_frame=values.worst_frame_loss.max() if len(values) else 0.,p50_us=part.p50_ms.mean()*1000))
    table=pd.DataFrame(overview);table.to_csv(HERE/'reports/RENDER_FINAL_OVERVIEW.csv',index=False)
    print(table.to_string(index=False))
    print('FINAL_RENDER_ROWS', {key:len(value) for key,value in frames.items()})


if __name__=='__main__':main()
