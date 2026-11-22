"""逐场景/Ratio、真实K缩放与全部反序；候选资格仅按预注册门限。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/RENDER_AGGREGATE'


def geometric(values):return float(np.exp(np.log(values).mean()))


def main():
    OUT.mkdir(exist_ok=False)
    state=json.loads((HERE/'reports/RENDER_CONTROLLER.json').read_text());assert state['state']=='complete'
    extra=json.loads((HERE/'reports/QUALITY_SUPPLEMENT_CONTROLLER.json').read_text());assert extra['state']=='complete'
    quality=[];sets={key:[] for key in ['samples','events','checks','workloads','memory']}
    accepted=state['accepted']+extra['accepted']
    for name in accepted:
        folder=HERE/'reports'/name
        if name.startswith('quality'):
            quality.append(pd.read_csv(folder/'quality.csv'))
        if name.startswith('formal'):
            j=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text());assert j['measurement_valid']
            for key in sets:
                path=folder/(key+'.csv')
                if path.exists():
                    frame=pd.read_csv(path);frame['job']=name;sets[key].append(frame)
    quality=pd.concat(quality,ignore_index=True)
    assert len(quality)==35840,(len(quality),'32*2*14*10*4')
    assert not quality.duplicated(['scene','workload','condition','frame','variant']).any()
    quality.to_csv(OUT/'QUALITY_FRAMES.csv.gz',index=False,compression='gzip')
    q=quality.groupby(['workload','scene','condition','variant']).agg(psnr_loss=('psnr_loss','mean'),
        worst_frame_loss=('psnr_loss','max'),baseline_psnr=('baseline_psnr','mean'),psnr=('psnr','mean'),
        ssim=('ssim','mean'),baseline_ssim=('baseline_ssim','mean'),lpips=('lpips','mean'),baseline_lpips=('baseline_lpips','mean'),
        image_psnr=('image_psnr','mean'),max_rgb_error=('max_rgb_error','max'),frames=('frame','size')).reset_index()
    q.to_csv(OUT/'QUALITY_PER_CONDITION.csv',index=False)
    eligibility=q.groupby('variant').agg(mean_loss=('psnr_loss','mean'),worst_condition_loss=('psnr_loss','max'),
        worst_frame_loss=('worst_frame_loss','max')).reset_index()
    eligibility['passed']=(eligibility.mean_loss<=.2)&(eligibility.worst_condition_loss<=.5)
    eligibility.to_csv(OUT/'QUALITY_GATE.csv',index=False)
    data={key:pd.concat(value,ignore_index=True) for key,value in sets.items()}
    expected=dict(samples=1075200,events=215040,checks=107520,workloads=44800,memory=384)
    assert {k:len(v) for k,v in data.items()}==expected
    for key,frame in data.items():frame.to_csv(OUT/(key+'.csv.gz'),index=False,compression='gzip')
    checks=data['checks'];assert checks.radii_bitwise.all()
    assert checks[checks.variant=='rebuilt_F2'].rgb_bitwise.all()
    sample=data['samples']
    assert not sample.duplicated(['workload','scene','condition','variant','round','repeat','frame']).any()
    perf=sample.groupby(['workload','scene','split','condition','variant','round']).agg(
        p50_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),std_ms=('ms','std'),N=('N','first'),
        K_p50=('K','median'),K_min=('K','min'),K_max=('K','max'),samples=('ms','size')).reset_index()
    base=perf[perf.variant=='F2'][['workload','scene','condition','round','p50_ms']].rename(columns={'p50_ms':'F2_ms'})
    perf=perf.merge(base,on=['workload','scene','condition','round']);perf['speedup']=perf.F2_ms/perf.p50_ms
    perf['saved_us']=(perf.F2_ms-perf.p50_ms)*1000
    perf.to_csv(OUT/'PER_SCENE_RATIO_ROUND.csv',index=False)
    perf[perf.speedup<1].to_csv(OUT/'ALL_SLOWER_CASES.csv',index=False)
    cond=perf.groupby(['workload','condition','variant']).agg(speedup=('speedup',geometric),p50_ms=('p50_ms','median'),
        p95_ms=('p95_ms','median'),saved_us=('saved_us','mean'),slower_scene_rounds=('speedup',lambda x:int((x<1).sum()))).reset_index()
    cond.to_csv(OUT/'CONDITIONS.csv',index=False)
    pairs=sample.groupby(['workload','scene','condition','variant','round','frame']).agg(ms=('ms','median'),K=('K','first')).reset_index()
    reference=pairs[pairs.condition=='full'][['workload','scene','variant','round','frame','ms','K']].rename(columns={'ms':'ref_ms','K':'ref_K'})
    pairs=pairs.merge(reference,on=['workload','scene','variant','round','frame'])
    pairs['K_normalized']=pairs.K/pairs.ref_K;pairs['T_normalized']=pairs.ms/pairs.ref_ms
    pairs['ideal_gap']=pairs.T_normalized-pairs.K_normalized
    pairs.to_csv(OUT/'NORMALIZED_SCALING.csv',index=False)
    fits=[];inversions=[]
    for key,frame in pairs.groupby(['workload','scene','variant','round','frame']):
        values=frame[frame.condition.str.startswith('local')|frame.condition.isin(['fixed0','full'])].sort_values('K')
        design=np.column_stack([np.ones(len(values)),values.K]);coefficient=np.linalg.lstsq(design,values.ms,rcond=None)[0]
        predicted=design@coefficient;total=((values.ms-values.ms.mean())**2).sum()
        fits.append(dict(zip(['workload','scene','variant','round','frame'],key),b_ms=coefficient[0],a_ms_per_gaussian=coefficient[1],
            r_squared=float(1-((values.ms-predicted)**2).sum()/max(total,1e-30)),mean_ideal_gap=values.ideal_gap.mean()))
        for a,b in zip(values.iloc[:-1].to_dict('records'),values.iloc[1:].to_dict('records')):
            if b['K']>a['K'] and b['ms']<a['ms']:
                inversions.append(dict(zip(['workload','scene','variant','round','frame'],key),lower_condition=a['condition'],
                    higher_condition=b['condition'],lower_K=a['K'],higher_K=b['K'],lower_ms=a['ms'],higher_ms=b['ms'],
                    reversed_us=(a['ms']-b['ms'])*1000))
    pd.DataFrame(fits).to_csv(OUT/'LINEAR_FITS.csv',index=False)
    pd.DataFrame(inversions).to_csv(OUT/'ALL_COUNT_INVERSIONS.csv',index=False)
    main=perf[(perf.workload=='lod')&perf.condition.isin([f'local{x:g}' for x in [.1,.2,.3,.4,.5]])]
    gain=main.groupby('variant').agg(speedup=('speedup',geometric),slower_scene_rounds=('speedup',lambda x:int((x<1).sum()))).reset_index()
    gain.to_csv(OUT/'PRIMARY_RATIO_GAIN.csv',index=False)
    record=dict(quality_gate=eligibility.to_dict('records'),primary_gain=gain.to_dict('records'),expected_rows=expected,
        rejected=state['rejected'],no_images=True,selection_pending_joint_validation=True)
    (OUT/'SUMMARY.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    print('QUALITY_GATE',eligibility.to_string(index=False));print('PRIMARY_GAIN',gain.to_string(index=False))


if __name__=='__main__':main()
