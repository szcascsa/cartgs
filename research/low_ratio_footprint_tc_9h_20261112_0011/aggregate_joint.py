"""冻结联合A–F完整结果：质量、双轮、全部慢例与按场景bootstrap。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/JOINT_AGGREGATE'
PRIMARY=[f'local{x:g}' for x in [.1,.2,.3,.4,.5]]


def geometric(x):return float(np.exp(np.log(x).mean()))


def confidence(values):
    values=np.log(np.asarray(values));rng=np.random.default_rng(3407)
    samples=rng.choice(values,(4000,len(values)),replace=True).mean(axis=1)
    lo,hi=np.quantile(np.exp(samples),[.025,.975])
    return float(lo),float(hi)


def main():
    OUT.mkdir(exist_ok=False)
    state=json.loads((HERE/'reports/JOINT_FORMAL_CONTROLLER.json').read_text());assert state['state']=='complete'
    quality=[];sets={key:[] for key in ['samples','events','checks','memory']};quality_checks=[]
    for name in state['accepted']:
        folder=HERE/'reports'/name
        if name.startswith('joint_quality'):
            quality.append(pd.read_csv(folder/'quality.csv'));quality_checks.append(pd.read_csv(folder/'checks.csv'))
        elif name.startswith('joint_formal'):
            job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text());assert job['measurement_valid']
            for key in sets:
                d=pd.read_csv(folder/(key+'.csv'));d['job']=name;sets[key].append(d)
    q=pd.concat(quality,ignore_index=True);qc=pd.concat(quality_checks,ignore_index=True)
    model_groups=q[['scene','model']].drop_duplicates();groups=len(model_groups)
    assert len(q)==groups*14*10*7
    assert not q.duplicated(['scene','model','condition','frame','variant']).any()
    assert qc[['same_ids','rgb_bitwise','radii_bitwise']].all().all()
    q.to_csv(OUT/'QUALITY_FRAMES.csv.gz',index=False,compression='gzip')
    qc.to_csv(OUT/'QUALITY_EXACT_CHECKS.csv.gz',index=False,compression='gzip')
    per=q.groupby(['model','scene','split','condition','variant']).agg(psnr_loss=('psnr_loss','mean'),
        worst_frame_loss=('psnr_loss','max'),psnr=('psnr','mean'),baseline_psnr=('baseline_psnr','mean'),
        ssim=('ssim','mean'),baseline_ssim=('baseline_ssim','mean'),lpips=('lpips','mean'),baseline_lpips=('baseline_lpips','mean'),
        renderer_loss=('renderer_psnr_loss','mean'),same_ids=('same_ids','all'),id_overlap=('id_overlap','mean')).reset_index()
    per.to_csv(OUT/'QUALITY_PER_SCENE_RATIO.csv',index=False)
    gate=per.groupby(['model','variant']).agg(mean_loss=('psnr_loss','mean'),worst_condition_loss=('psnr_loss','max'),
        worst_frame_loss=('worst_frame_loss','max'),min_id_overlap=('id_overlap','min')).reset_index()
    gate['passed']=(gate.mean_loss<=.2)&(gate.worst_condition_loss<=.5)
    gate.to_csv(OUT/'QUALITY_GATE.csv',index=False)
    data={key:pd.concat(value,ignore_index=True) for key,value in sets.items()}
    expected=dict(samples=groups*2*14*10*8*22,events=groups*2*14*2*6*7,checks=groups*2*14*10*2,memory=groups*2*3)
    assert {key:len(value) for key,value in data.items()}==expected
    assert data['checks'][['same_ids','rgb_bitwise','radii_bitwise']].all().all()
    for key,value in data.items():value.to_csv(OUT/(key+'.csv.gz'),index=False,compression='gzip')
    samples=data['samples'];keys=['scene','model','condition','variant','scope','round']
    assert not samples.duplicated(keys+['frame','repeat']).any()
    perf=samples.groupby(keys).agg(p50_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),
        std_ms=('ms','std'),K_p50=('K','median'),K_min=('K','min'),K_max=('K','max'),N=('N','first'),samples=('ms','size')).reset_index()
    join=['scene','model','condition','scope','round']
    for base,label in [('A_fp32_F2','A'),('D_tensor_best_new','D')]:
        b=perf[perf.variant==base][join+['p50_ms']].rename(columns={'p50_ms':label+'_ms'})
        perf=perf.merge(b,on=join);perf['speedup_'+label]=perf[label+'_ms']/perf.p50_ms
        perf['saved_us_'+label]=(perf[label+'_ms']-perf.p50_ms)*1000
    perf.to_csv(OUT/'PER_SCENE_RATIO_ROUND.csv',index=False)
    perf[(perf.speedup_A<1)|(perf.speedup_D<1)].to_csv(OUT/'ALL_SLOWER_CASES.csv',index=False)
    perf[(perf.scope=='pipeline')&(perf.condition=='full')].to_csv(OUT/'ALL_FULL_COSTS.csv',index=False)
    summary=perf.groupby(['model','condition','variant','scope']).agg(p50_ms=('p50_ms','median'),p95_ms=('p95_ms','median'),
        speedup_A=('speedup_A',geometric),speedup_D=('speedup_D',geometric),saved_us_A=('saved_us_A','mean'),
        slower_A=('speedup_A',lambda x:int((x<1).sum())),slower_D=('speedup_D',lambda x:int((x<1).sum()))).reset_index()
    summary.to_csv(OUT/'CONDITIONS.csv',index=False)
    primary=perf[perf.condition.isin(PRIMARY)].groupby(['model','variant','scope']).agg(speedup_A=('speedup_A',geometric),
        speedup_D=('speedup_D',geometric),saved_us_A=('saved_us_A','mean'),p50_ms=('p50_ms','median'),p95_ms=('p95_ms','median'),
        slower_A=('speedup_A',lambda x:int((x<1).sum())),slower_D=('speedup_D',lambda x:int((x<1).sum()))).reset_index()
    primary.to_csv(OUT/'PRIMARY_RATIO_GAIN.csv',index=False)
    selection=[]
    pipe=perf[(perf.scope=='pipeline')&perf.condition.isin(PRIMARY)]
    for kind in sorted(pipe.model.unique()):
        for name in ['E_pending','F_native']:
            d=pipe[(pipe.model==kind)&(pipe.variant==name)]
            scenes=d.groupby('scene').speedup_D.apply(geometric)
            lo,hi=confidence(scenes);rounds=d.groupby('round').speedup_D.apply(geometric)
            selection.append(dict(model=kind,variant=name,speedup_D=geometric(scenes),ci95_low=lo,ci95_high=hi,
                round1=float(rounds.loc[1]),round2=float(rounds.loc[2]),slower_scene_rounds=int((d.speedup_D<1).sum()),
                scenes=len(scenes),passed=bool(lo>1 and (rounds>1).all())))
        d=pipe[(pipe.model==kind)&pipe.variant.isin(['E_pending','F_native'])].pivot(index=['scene','condition','round'],columns='variant',values='p50_ms')
        ratios=(d.E_pending/d.F_native).groupby('scene').apply(geometric);lo,hi=confidence(ratios)
        selection.append(dict(model=kind,variant='F_vs_E',speedup_D=geometric(ratios),ci95_low=lo,ci95_high=hi,
            scenes=len(ratios),passed=bool(lo>1)))
    selection=pd.DataFrame(selection);selection.to_csv(OUT/'SCHEDULER_SELECTION.csv',index=False)
    chosen={}
    for kind in sorted(pipe.model.unique()):
        s=selection[selection.model==kind].set_index('variant');winner='D_tensor_best_new'
        if s.loc['E_pending','passed']:winner='E_pending'
        if s.loc['F_native','passed'] and (winner=='D_tensor_best_new' or s.loc['F_vs_E','passed']):winner='F_native'
        assert bool(gate[(gate.model==kind)&(gate.variant==winner)].passed.iloc[0]);chosen[kind]=winner
    sample=samples[samples.scope=='pipeline'].groupby(['scene','model','condition','variant','round','frame']).agg(ms=('ms','median'),K=('K','first')).reset_index()
    reference=sample[sample.condition=='full'][['scene','model','variant','round','frame','ms','K']].rename(columns={'ms':'Full_ms','K':'Full_K'})
    sample=sample.merge(reference,on=['scene','model','variant','round','frame'])
    sample['K_normalized']=sample.K/sample.Full_K;sample['T_normalized']=sample.ms/sample.Full_ms
    sample.to_csv(OUT/'PIPELINE_NORMALIZED_SCALING.csv.gz',index=False,compression='gzip')
    sample[(sample.condition!='full')&(sample.T_normalized>1)].to_csv(OUT/'LOD_SLOWER_THAN_FULL.csv',index=False)
    record=dict(model_groups=groups,rows=expected,quality_gate=gate.to_dict('records'),primary_gain=primary.to_dict('records'),
        scheduler_selection=selection.to_dict('records'),chosen_by_model=chosen,rejected=state['rejected'],
        no_images=True,only_frozen_inference=True,quality_reference='同权重同模型kind的原FP32网络＋F2；Teacher/Student不混算',
        bootstrap='按场景配对、4000次重采样，95%区间；不把相关帧当独立样本')
    (OUT/'SUMMARY.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    print('JOINT_AGGREGATION_COMPLETE',groups,expected)
    print(primary[primary.scope=='pipeline'].round(5).to_string(index=False));print(selection.round(5).to_string(index=False));print('CHOSEN',chosen)


if __name__=='__main__':main()
