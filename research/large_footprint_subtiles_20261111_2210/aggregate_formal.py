"""只汇总预注册的有效正式段；污染段保留但不参与性能结论。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/AGGREGATE'


def geometric(values):
    return float(np.exp(np.log(values).mean()))


def main():
    OUT.mkdir(exist_ok=False)
    controller=json.loads((HERE/'reports/FORMAL_CONTROLLER.json').read_text())
    assert controller['state']=='complete' and len(controller['accepted'])==32
    tables={k:[] for k in ('samples','events','checks','workloads','memory')}
    jobs=[];monitors=[]
    for name in controller['accepted']:
        job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
        assert job['returncode']==0 and job['measurement_valid']
        assert job['foreign_active_samples']==0 and job['foreign_cpu_active_samples']==0
        jobs.append(job)
        for key in tables:
            path=HERE/'reports'/name/(key+'.csv')
            if path.exists():
                frame=pd.read_csv(path);frame['job']=name;tables[key].append(frame)
        for line in (HERE/'logs'/(name+'_MONITOR.jsonl')).read_text().splitlines():
            item=json.loads(line);gpu=[x.strip() for x in item['gpu'].split(',')]
            monitors.append(dict(job=name,time=item['time'],utilization=gpu[0],memory=gpu[1],
                                 sm_clock=gpu[2],temperature=gpu[3],power=gpu[4]))
    data={key:pd.concat(value,ignore_index=True) for key,value in tables.items()}
    expected=dict(samples=230400,events=46080,checks=23040,workloads=3840,memory=192)
    assert {key:len(value) for key,value in data.items()}==expected
    checks=data['checks'];assert checks.rgb_bitwise.all() and checks.radii_bitwise.all()
    assert (checks.max_rgb_error==0).all()
    samples=data['samples'];assert samples.scene.nunique()==32
    assert len(samples[['scene','condition','round','repeat','frame','variant']].drop_duplicates())==len(samples)
    for key,value in data.items():value.to_csv(OUT/(key+'.csv.gz'),index=False,compression='gzip')
    pd.DataFrame(monitors).to_csv(OUT/'MONITOR.csv',index=False)
    perf=samples.groupby(['scene','split','condition','variant','round']).ms.agg(
        p50_ms='median',p95_ms=lambda x:x.quantile(.95),mean_ms='mean',std_ms='std',samples='size').reset_index()
    sizes=samples.groupby(['scene','condition']).agg(N=('N','first'),K_p50=('K','median'),K_min=('K','min'),K_max=('K','max')).reset_index()
    perf=perf.merge(sizes,on=['scene','condition'])
    ref=perf[perf.variant=='F2'][['scene','condition','round','p50_ms','p95_ms']].rename(
        columns={'p50_ms':'F2_p50_ms','p95_ms':'F2_p95_ms'})
    perf=perf.merge(ref,on=['scene','condition','round'])
    perf['speedup']=perf.F2_p50_ms/perf.p50_ms
    perf['latency_reduction_percent']=100*(1-perf.p50_ms/perf.F2_p50_ms)
    perf['saved_us']=(perf.F2_p50_ms-perf.p50_ms)*1000
    perf.to_csv(OUT/'PER_SCENE_RATIO_ROUND.csv',index=False)
    pairs=perf[perf.variant!='F2'].groupby(['scene','split','condition','variant']).agg(
        speedup=('speedup',geometric),min_round_speedup=('speedup','min'),max_round_speedup=('speedup','max'),
        saved_us=('saved_us','mean'),N=('N','first'),K_p50=('K_p50','first')).reset_index()
    pairs['faster_both_rounds']=pairs.min_round_speedup>=1
    pairs['slower_both_rounds']=pairs.max_round_speedup<1
    pairs.to_csv(OUT/'PAIRED_SPEEDUP.csv',index=False)
    conditions=perf.groupby(['condition','variant']).agg(
        speedup=('speedup',geometric),p50_ms=('p50_ms','median'),p95_ms=('p95_ms','median'),
        saved_us=('saved_us','mean'),scene_rounds=('scene','size')).reset_index()
    stable=pairs.groupby(['condition','variant']).agg(faster_both_rounds=('faster_both_rounds','sum'),
        slower_both_rounds=('slower_both_rounds','sum'),scenes=('scene','size')).reset_index()
    conditions=conditions.merge(stable,on=['condition','variant'],how='left')
    conditions.to_csv(OUT/'CONDITIONS.csv',index=False)
    splits=perf.groupby(['split','condition','variant']).agg(speedup=('speedup',geometric),
        saved_us=('saved_us','mean'),scene_rounds=('scene','size')).reset_index()
    splits.to_csv(OUT/'SPLITS.csv',index=False)
    e=data['events'].groupby(['scene','condition','variant','round']).cuda_ms.agg(
        p50_ms='median',p95_ms=lambda x:x.quantile(.95),samples='size').reset_index()
    er=e[e.variant=='F2'][['scene','condition','round','p50_ms']].rename(columns={'p50_ms':'F2_p50_ms'})
    e=e.merge(er,on=['scene','condition','round']);e['speedup']=e.F2_p50_ms/e.p50_ms
    e.to_csv(OUT/'CUDA_EVENTS.csv',index=False)
    # 当前帧各版本是轮换次序配对，先在每帧重复内取中位，避免单个离群样本决定结论。
    frame=samples.groupby(['scene','split','condition','round','frame','motion','variant']).ms.median().reset_index()
    pivot=frame.pivot(index=['scene','split','condition','round','frame','motion'],columns='variant',values='ms').reset_index()
    for name in ('common','exit8'):pivot[name+'_speedup']=pivot.F2/pivot[name]
    pivot.to_csv(OUT/'PER_FRAME_PAIRED.csv',index=False)
    workload=data['workloads'].groupby(['scene','condition']).agg(N=('N','first'),M=('M','median'),K=('K','median'),
        preprocess_visible=('preprocess_visible','median'),alpha_participating=('alpha_participating','median'),
        alpha_interactions=('alpha_interactions','median'),tile_instances=('tile_instances','median'),
        active_tiles=('active_tiles','median'),tiles=('tiles','first'),members_p50=('members_p50','median'),
        members_p95=('members_p95','median'),members_max=('members_max','max'),overflow_tiles=('overflow_tiles','max'),
        workspace_bytes=('workspace_bytes','max'),width=('width','first'),height=('height','first')).reset_index()
    workload.to_csv(OUT/'WORKLOADS.csv',index=False)
    slower=perf[(perf.variant!='F2') & (perf.speedup<1)].sort_values('speedup')
    slower.to_csv(OUT/'ALL_SLOWER_SCENE_ROUNDS.csv',index=False)
    meta=dict(expected_rows=expected,actual_rows={k:len(v) for k,v in data.items()},scenes=32,
              tune_scenes=4,heldout_scenes=28,rounds=2,views_per_scene=10,repeats_per_view=10,
              accepted_jobs=controller['accepted'],rejected_jobs=controller['rejected'],
              bitwise_pass=True,max_rgb_error=float(checks.max_rgb_error.max()),
              full_gate={n:bool(pairs[(pairs.variant==n)&(pairs.condition=='full')].faster_both_rounds.all()) for n in ('common','exit8')},
              speedup_by_condition=conditions.to_dict('records'))
    (OUT/'SUMMARY.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(conditions.to_string(index=False))
    print('FULL_GATE',meta['full_gate'])
    print('HELDOUT',splits[(splits['split']=='heldout')&(splits.variant!='F2')].to_string(index=False))


if __name__=='__main__':main()
