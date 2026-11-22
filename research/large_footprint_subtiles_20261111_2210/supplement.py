"""候选族、合成压力和真实负载证据，始终与正式独占墙钟分开。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/AGGREGATE'


def geometric(x):return float(np.exp(np.log(x).mean()))


def main():
    frames=[]
    for index in range(1,9):
        name=f'screen_clean{index:02d}'
        assert json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())['measurement_valid']
        table=pd.read_csv(HERE/'reports'/name/'summary.csv');table['job']=name;frames.append(table)
    screen=pd.concat(frames,ignore_index=True)
    screen.to_csv(OUT/'DEVELOPMENT_SCREEN_ALL.csv',index=False)
    screen.groupby(['job','variant','condition']).agg(speedup=('speedup',geometric),
        p50_ms=('p50_ms','median'),p95_ms=('p95_ms','median'),scenes=('scene','size')).reset_index().to_csv(
        OUT/'DEVELOPMENT_SCREEN_SUMMARY.csv',index=False)
    d=pd.read_csv(HERE/'reports/diagnose01/workloads.csv')
    d.to_csv(OUT/'SUBTILE_DIAGNOSTICS.csv',index=False)
    masks=d.groupby(['condition','side']).agg(regions=('tested_regions','sum'),kept=('kept_regions','sum'),
        rejected=('rejected_regions','sum'),false_rejections=('false_pixel_rejections','sum'),
        mean_keep=('keep_fraction','mean'),overflows=('overflow_tiles','max')).reset_index()
    masks['weighted_reject_fraction']=masks.rejected/masks.regions
    masks.to_csv(OUT/'SUBTILE_REJECTION.csv',index=False)
    state=json.loads((HERE/'reports/POST_CHECKS.json').read_text());assert state['state']=='complete'
    stress_name=state['stress'];assert json.loads((HERE/'reports/jobs'/(stress_name+'.json')).read_text())['measurement_valid']
    stress=pd.read_csv(HERE/'reports'/stress_name/'samples.csv');stress.to_csv(OUT/'STRESS_SAMPLES.csv.gz',index=False,compression='gzip')
    keys=['width','height','K','scale','opacity']
    stats=stress.groupby(keys+['variant']).ms.agg(p50_ms='median',p95_ms=lambda x:x.quantile(.95)).reset_index()
    base=stats[stats.variant=='previous_F2'][keys+['p50_ms']].rename(columns={'p50_ms':'F2_p50_ms'})
    stats=stats.merge(base,on=keys);stats['speedup']=stats.F2_p50_ms/stats.p50_ms
    stats.to_csv(OUT/'SYNTHETIC_STRESS.csv',index=False)
    work=pd.read_csv(OUT/'WORKLOADS.csv')
    perf=pd.read_csv(OUT/'PER_SCENE_RATIO_ROUND.csv')
    raster=perf[perf.variant=='F2'].groupby(['scene','condition']).p50_ms.mean().reset_index()
    work=work.merge(raster,on=['scene','condition'])
    full=work[work.condition=='full'][['scene','K','alpha_interactions','tile_instances','p50_ms']].rename(
        columns={k:'full_'+k for k in ('K','alpha_interactions','tile_instances','p50_ms')})
    work=work.merge(full,on='scene')
    for key in ('K','alpha_interactions','tile_instances','p50_ms'):work[key+'_vs_full']=work[key]/work['full_'+key]
    work['active_tile_fraction']=work.active_tiles/work.tiles
    work.to_csv(OUT/'COUNT_SCALING.csv',index=False)
    counts=work.groupby('condition').agg(K_vs_full=('K_vs_full','median'),
        interaction_vs_full=('alpha_interactions_vs_full','median'),instances_vs_full=('tile_instances_vs_full','median'),
        latency_vs_full=('p50_ms_vs_full','median'),active_tile_fraction=('active_tile_fraction','median'),
        K=('K','median'),alpha_interactions=('alpha_interactions','median')).reset_index()
    counts.to_csv(OUT/'COUNT_SCALING_SUMMARY.csv',index=False)
    memory=pd.read_csv(OUT/'memory.csv.gz');memory['MiB']=memory.workspace_bytes/(1<<20)
    full_perf=perf[(perf.condition=='full')&(perf.variant=='common')]
    result=dict(stress_conditions=len(stats[keys].drop_duplicates()),stress_bitwise_comparisons=len(stats),
        workspace_MiB=memory.groupby('variant').MiB.agg(['min','max']).to_dict('index'),
        full_regressions=full_perf[full_perf.speedup<1][['scene','round','speedup','saved_us']].to_dict('records'),
        low_ratio_rejection=masks[masks.condition=='local0.01'].to_dict('records'),
        synthetic_extremes=stats[stats.variant!='previous_F2'].groupby('variant').speedup.agg(['min','max',geometric]).to_dict('index'))
    (OUT/'SUPPLEMENT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('COUNT_SCALING',counts.to_string(index=False))
    print('SUPPLEMENT',json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
