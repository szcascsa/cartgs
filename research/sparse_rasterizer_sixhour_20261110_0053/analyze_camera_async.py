"""固定内参候选独立配对验收；旧纯渲染和新准备计时不跨批次拼成配对。"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def stats(data,keys):
    return data.groupby(keys).agg(median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda s:s.quantile(.95)),
        mean_ms=('wall_ms','mean'),std_ms=('wall_ms','std'),samples=('wall_ms','size'),
        K_median=('K','median'),K_min=('K','min'),K_max=('K','max'),N=('N','first')).reset_index()


def refs(data,keys):
    for variant in ['A','B','G','F','Q']:
        other=data[data.variant==variant][keys+['median_ms','p95_ms']].rename(columns={name:variant+'_'+name for name in ['median_ms','p95_ms']})
        data=data.merge(other,on=keys,validate='many_to_one')
        data['speedup_'+variant]=data[variant+'_median_ms']/data.median_ms
        data['latency_reduction_'+variant+'_pct']=(1-data.median_ms/data[variant+'_median_ms'])*100
    return data


def native(value):
    if isinstance(value,dict):return {str(k):native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rounds',default='async_final03,async_final02');args=parser.parse_args()
    names=args.rounds.split(',');assert len(names)==2
    for name in names:
        summary=json.loads((HERE/'reports'/name/'SUMMARY.json').read_text())
        job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
        assert summary['passed'] and summary['scenes']==24 and job['measurement_valid'] and job['foreign_active_samples']==0
    data=pd.concat([pd.read_csv(HERE/'reports'/name/'samples.csv') for name in names],ignore_index=True)
    pipe=data[data.scope=='pipeline'];assert pipe.scene.nunique()==24 and set(pipe['round'])=={1,2}
    result=refs(stats(pipe,['scene','condition','variant']),['scene','condition'])
    repeats=pipe.groupby(['scene','condition','round','repeat','variant']).wall_ms.median().reset_index()
    repeat=repeats.groupby(['scene','condition','variant']).wall_ms.agg(repeat_min_median_ms='min',repeat_max_median_ms='max',repeat_std_ms='std',repeat_mean_ms='mean').reset_index()
    repeat['repeat_cv_pct']=100*repeat.repeat_std_ms/repeat.repeat_mean_ms
    result=result.merge(repeat,on=['scene','condition','variant'],validate='one_to_one')
    result['saved_Q_us']=(result.Q_median_ms-result.median_ms)*1000
    result.to_csv(HERE/'FINAL_ASYNC_PIPELINE_RESULTS.csv',index=False)
    by_round=refs(stats(pipe,['scene','condition','round','variant']),['scene','condition','round'])
    by_round.to_csv(HERE/'CAMERA_ASYNC_ROUND_RESULTS.csv',index=False)
    regression=by_round[(by_round.variant=='R')&(by_round.speedup_Q<1)].copy()
    regression['latency_increase_vs_Q_pct']=(regression.median_ms/regression.Q_median_ms-1)*100
    regression.to_csv(HERE/'ASYNC_VS_Q_REGRESSIONS.csv',index=False)
    gate=by_round[(by_round.variant=='R')&(by_round.condition=='full')].copy()
    for name in ['A','B','G','F','Q']:gate['pass_'+name]=gate.median_ms<=gate[name+'_median_ms']
    gate.to_csv(HERE/'FINAL_ASYNC_FULL_GATES.csv',index=False)
    stats(data[data.scope=='prepare'],['scene','condition','variant']).to_csv(HERE/'FINAL_ASYNC_PREPARATION_RESULTS.csv',index=False)
    parity=pd.concat([pd.read_csv(HERE/'reports'/name/'correctness.csv') for name in names],ignore_index=True)
    assert parity[['camera_bitwise','ids_bitwise','rgb_bitwise','radii_bitwise']].all().all()
    parity.to_csv(HERE/'CAMERA_ASYNC_CORRECTNESS.csv',index=False)
    view=pipe.groupby(['scene','condition','frame','variant']).agg(median_ms=('wall_ms','median'),K=('K','median'),N=('N','first')).reset_index()
    own=view[(view.condition=='full')&(view.variant=='R')][['scene','frame','median_ms']].rename(columns={'median_ms':'own_full_ms'})
    scaling=view[(view.variant=='R')&view.condition.str.startswith('local')].merge(own,on=['scene','frame'],validate='many_to_one')
    scaling['speedup_vs_own_full']=scaling.own_full_ms/scaling.median_ms
    scaling.to_csv(HERE/'FINAL_ASYNC_COUNT_VS_FULL.csv',index=False)
    low=result[(result.variant=='R')&result.condition.isin(['local0.01','local0.05','local0.1'])]
    summary=[]
    for condition,group in low.groupby('condition'):
        summary.append(dict(condition=condition,scenes=len(group),min_ms=group.median_ms.min(),max_ms=group.median_ms.max(),
            min_speedup_A=group.speedup_A.min(),median_speedup_A=group.speedup_A.median(),max_speedup_A=group.speedup_A.max(),
            min_speedup_B=group.speedup_B.min(),median_speedup_B=group.speedup_B.median(),max_speedup_B=group.speedup_B.max(),
            min_speedup_G=group.speedup_G.min(),median_speedup_G=group.speedup_G.median(),max_speedup_G=group.speedup_G.max(),
            min_speedup_F=group.speedup_F.min(),median_speedup_F=group.speedup_F.median(),max_speedup_F=group.speedup_F.max(),
            min_speedup_Q=group.speedup_Q.min(),median_speedup_Q=group.speedup_Q.median(),max_speedup_Q=group.speedup_Q.max()))
    negatives=scaling[scaling.condition.isin(['local0.01','local0.05','local0.1'])&(scaling.speedup_vs_own_full<1)]
    graph_memory=pd.concat([pd.read_csv(HERE/'reports'/name/'async_startup.csv') for name in names],ignore_index=True)
    graph_memory.to_csv(HERE/'CAMERA_ASYNC_MEMORY.csv',index=False)
    out=dict(passed=bool(gate[['pass_A','pass_B','pass_G','pass_F','pass_Q']].all().all()),formal_names=names,raw_rows=len(data),
        scenes=24,rounds=2,correctness_inputs=len(parity),full_failed_rows=gate[~gate[['pass_A','pass_B','pass_G','pass_F','pass_Q']].all(axis=1)].to_dict('records'),
        low_ratio_bounds=summary,projection_bytes=64,per_view_regressions_vs_own_full=len(negatives),
        per_round_regressions_vs_Q=len(regression),worst_regression_vs_Q_pct=regression.latency_increase_vs_Q_pct.max() if len(regression) else 0.,
        worst_speedup_vs_own_full=negatives.speedup_vs_own_full.min() if len(negatives) else 1.,
        note='R与A/B/G/F/Q来自新独立配对批次；渲染二进制未变，纯渲染复用formal02/formal04，不重复宣称收益。')
    (HERE/'CAMERA_ASYNC_FINAL_VERIFICATION.json').write_text(json.dumps(native(out),ensure_ascii=False,indent=2))
    print(json.dumps(native(out),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
