"""正式两轮结果汇总与预声明门限验收；失败和旧B差异保留。"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def read_rounds(names,filename):
    return pd.concat([pd.read_csv(HERE/'reports'/name/(filename+'.csv')) for name in names],ignore_index=True)


def statistics(frame,keys):
    return frame.groupby(keys,dropna=False).agg(median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda s:s.quantile(.95)),
        mean_ms=('wall_ms','mean'),std_ms=('wall_ms','std'),min_ms=('wall_ms','min'),max_ms=('wall_ms','max'),
        samples=('wall_ms','size'),K_median=('K','median'),K_min=('K','min'),K_max=('K','max'),N=('N','first')).reset_index()


def baselines(frame,keys):
    result=frame.copy()
    for name in ['A','B','C']:
        ref=frame[frame.variant==name][keys+['median_ms','p95_ms']].rename(columns={'median_ms':name+'_median_ms','p95_ms':name+'_p95_ms'})
        result=result.merge(ref,on=keys,validate='many_to_one')
        result['speedup_'+name]=result[name+'_median_ms']/result.median_ms
        result['latency_reduction_'+name+'_pct']=(1-result.median_ms/result[name+'_median_ms'])*100
    return result


def native_json(value):
    if isinstance(value,dict):return {str(k):native_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native_json(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rounds',default='formal02,formal04');args=parser.parse_args()
    names=args.rounds.split(',');assert len(names)==2
    summaries=[]
    for name in names:
        value=json.loads((HERE/'reports'/name/'SUMMARY.json').read_text());assert value['passed'] and value['scenes']==24; summaries.append(value)
        identity=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text());assert identity['state']=='complete' and identity['measurement_valid'] and identity['foreign_active_samples']==0
    data=read_rounds(names,'samples');assert set(data['round'])=={1,2} and data.scene.nunique()==24 and not data.plan_miss.any()
    summary=baselines(statistics(data[data.scope!='prepare'],['scene','condition','scope','variant']),['scene','condition','scope'])
    by_round=baselines(statistics(data[data.scope!='prepare'],['scene','condition','scope','round','variant']),['scene','condition','scope','round'])
    repeat=data.groupby(['scene','condition','scope','variant','round','repeat']).wall_ms.median().reset_index()
    variation=repeat.groupby(['scene','condition','scope','variant']).wall_ms.agg(repeat_min_median_ms='min',repeat_max_median_ms='max',repeat_std_median_ms='std',repeat_mean_median_ms='mean').reset_index()
    variation['repeat_cv_pct']=100*variation.repeat_std_median_ms/variation.repeat_mean_median_ms
    summary=summary.merge(variation,on=['scene','condition','scope','variant'],validate='one_to_one')
    summary.to_csv(HERE/'RASTER_AND_PIPELINE_RESULTS.csv',index=False)
    summary[summary.scope=='raster'].to_csv(HERE/'RASTERIZER_RESULTS.csv',index=False)
    summary[summary.scope=='pipeline'].to_csv(HERE/'PIPELINE_RESULTS.csv',index=False)
    by_round.to_csv(HERE/'ROUND_RESULTS.csv',index=False)
    statistics(data[data.scope=='prepare'],['scene','condition','scope','variant']).to_csv(HERE/'PREPARATION_RESULTS.csv',index=False)
    full=by_round[(by_round.condition=='full')&by_round.variant.isin(['C','G'])].copy()
    full['pass_original_A']=full.median_ms<=full.A_median_ms;full['pass_previous_B']=full.median_ms<=full.B_median_ms
    full.to_csv(HERE/'FULL_GATES.csv',index=False)
    assert len(full)==24*2*2*2
    parity=read_rounds(names,'correctness');new=parity[parity.variant.isin(['C','G'])]
    assert new.rgb_bitwise.all() and new.radii_bitwise.all()
    old=parity[parity.variant=='B'];mismatches=old[~old.rgb_bitwise|~old.radii_bitwise]
    mismatches.to_csv(HERE/'BASELINE_B_NUMERICAL_DIFFERENCES.csv',index=False)
    parity.groupby('variant').agg(inputs=('rgb_bitwise','size'),RGB_passed=('rgb_bitwise','sum'),radii_passed=('radii_bitwise','sum'),
        max_absolute_error=('max_absolute_error','max'),max_relative_error=('max_relative_error','max')).to_csv(HERE/'CORRECTNESS_SUMMARY.csv')
    events=read_rounds(names,'events');events.to_csv(HERE/'CUDA_EVENT_RESULTS.csv',index=False)
    workload=pd.read_csv(HERE/'reports'/names[0]/'workloads.csv');workload.to_csv(HERE/'TILE_WORKLOADS.csv',index=False)
    memory=read_rounds(names,'memory');memory.to_csv(HERE/'MEMORY_RESULTS.csv',index=False)
    startup=read_rounds(names,'startup');startup.to_csv(HERE/'PLAN_STARTUP_RESULTS.csv',index=False)
    # 同视角比较点数缩减后的实际时间，不把不同场景/视角直接混为同一输入。
    view=data[(data.variant=='G')&(data.scope.isin(['raster','pipeline']))].groupby(['scene','scope','condition','frame']).agg(
        median_ms=('wall_ms','median'),K=('K','median'),N=('N','first')).reset_index()
    base=view[view.condition=='full'][['scene','scope','frame','median_ms']].rename(columns={'median_ms':'full_ms'})
    scaling=view[view.condition.str.startswith('local')].merge(base,on=['scene','scope','frame'],validate='many_to_one')
    scaling['speedup_vs_own_full']=scaling.full_ms/scaling.median_ms;scaling['actual_retained_fraction']=scaling.K/scaling.N
    scaling.to_csv(HERE/'COUNT_VS_SPEED_PER_VIEW.csv',index=False)
    low=summary[(summary.variant=='G')&summary.condition.isin(['local0.01','local0.05','local0.1'])]
    low.to_csv(HERE/'LOW_RATIO_RESULTS.csv',index=False)
    bounds=[]
    for (scope,condition),group in low.groupby(['scope','condition']):
        bounds.append(dict(scope=scope,condition=condition,scenes=len(group),
            min_speedup_A=group.speedup_A.min(),median_speedup_A=group.speedup_A.median(),max_speedup_A=group.speedup_A.max(),
            min_speedup_B=group.speedup_B.min(),median_speedup_B=group.speedup_B.median(),max_speedup_B=group.speedup_B.max(),
            min_speedup_C=group.speedup_C.min(),median_speedup_C=group.speedup_C.median(),max_speedup_C=group.speedup_C.max(),
            worst_repeat_cv_pct=group.repeat_cv_pct.max(),min_K=group.K_min.min(),max_K=group.K_max.max()))
    full_g=full[(full.variant=='G')&(full.scope=='raster')]
    outcome=dict(passed_new_numerics=True,formal_names=names,rounds=2,scenes=24,raw_rows=len(data),new_parity_inputs=len(new),
        paired_fresh_preparation_checks=sum(x['checks'] for x in summaries),
        full_G_original_A_pass=bool(full_g.pass_original_A.all()),full_G_previous_B_pass=bool(full_g.pass_previous_B.all()),
        full_C_original_A_pass=bool(full[(full.variant=='C')&(full.scope=='raster')].pass_original_A.all()),
        full_C_previous_B_pass=bool(full[(full.variant=='C')&(full.scope=='raster')].pass_previous_B.all()),
        full_failed_rows=full[~full.pass_original_A|~full.pass_previous_B].to_dict('records'),
        full_speedup_A_range=[full_g.speedup_A.min(),full_g.speedup_A.max()],full_speedup_B_range=[full_g.speedup_B.min(),full_g.speedup_B.max()],
        previous_B_numeric_mismatch_rows=len(mismatches),previous_B_unique_failed_inputs=len(mismatches.drop_duplicates(['scene','condition','frame'])),
        previous_B_max_absolute_error=float(mismatches.max_absolute_error.max()) if len(mismatches) else 0.,
        low_ratio_bounds=bounds,max_workspace_MiB=(memory.groupby('variant').workspace_bytes.max()/2**20).to_dict(),
        graph_precompile_ms_range=[startup.graph_precompile_ms.min(),startup.graph_precompile_ms.max()],
        input_N_range=[data.N.min(),data.N.max()],per_view_low_ratio_regressions_vs_own_full=len(scaling[(scaling.condition.isin(['local0.01','local0.05','local0.1']))&(scaling.speedup_vs_own_full<1)]))
    (HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(native_json(outcome),indent=2,ensure_ascii=False))
    print(json.dumps(native_json(outcome),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
