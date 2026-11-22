"""逐视角反例、相同身份补偿对照、运动方式和保留集的真实边界。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    scene=pd.DataFrame(json.loads((HERE/'reports/SCENES.json').read_text()))[['id','split']].rename(columns={'id':'scene'})
    summary=pd.read_csv(HERE/'RASTER_AND_PIPELINE_RESULTS.csv').merge(scene,on='scene',validate='many_to_one')
    low=summary[(summary.variant=='G')&summary.condition.isin(['local0.01','local0.05','local0.1'])]
    low.groupby(['split','scope','condition']).agg(scenes=('scene','size'),
        min_speedup_A=('speedup_A','min'),median_speedup_A=('speedup_A','median'),
        min_speedup_B=('speedup_B','min'),median_speedup_B=('speedup_B','median'),
        max_speedup_B=('speedup_B','max')).to_csv(HERE/'HELDOUT_GENERALIZATION.csv')
    scaling=pd.read_csv(HERE/'COUNT_VS_SPEED_PER_VIEW.csv')
    scaling[scaling.speedup_vs_own_full<1].to_csv(HERE/'LOW_RATIO_VS_FULL_COUNTEREXAMPLES.csv',index=False)
    scaling.groupby(['scope','condition']).agg(views=('frame','size'),
        faster_than_full=('speedup_vs_own_full',lambda s:int((s>=1).sum())),
        median_speedup=('speedup_vs_own_full','median'),min_speedup=('speedup_vs_own_full','min'),
        max_speedup=('speedup_vs_own_full','max')).to_csv(HERE/'LOW_RATIO_VS_FULL_SUMMARY.csv')
    rounds=json.loads((HERE/'FINAL_VERIFICATION.json').read_text())['formal_names']
    samples=pd.concat([pd.read_csv(HERE/'reports'/name/'samples.csv') for name in rounds],ignore_index=True)
    motion=samples[samples.scope!='prepare'].groupby(['scene','scope','condition','motion','variant']).agg(
        median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda s:s.quantile(.95)),samples=('wall_ms','size')).reset_index()
    for name in ['A','B']:
        reference=motion[motion.variant==name].drop(columns=['variant','p95_ms','samples']).rename(columns={'median_ms':name+'_median_ms'})
        motion=motion.merge(reference,on=['scene','scope','condition','motion'],validate='many_to_one')
        motion['speedup_'+name]=motion[name+'_median_ms']/motion.median_ms
    motion.to_csv(HERE/'MOTION_RESULTS.csv',index=False)
    work=pd.read_csv(HERE/'TILE_WORKLOADS.csv')
    measured=samples[(samples.variant=='G')&(samples.scope=='raster')].groupby(['scene','condition','frame']).wall_ms.median().reset_index()
    work=work.merge(measured,on=['scene','condition','frame'],validate='one_to_one')
    work['tile_fraction_active']=work.active_tiles/work.tiles
    work['tile_instances_per_K']=work.tile_instances/work.K.replace(0,np.nan)
    work.to_csv(HERE/'WORKLOAD_AND_LATENCY.csv',index=False)
    counter=pd.read_csv(HERE/'reports/count_scaling01/samples.csv')
    values=counter.groupby(['scene','frame','condition','treatment','variant']).wall_ms.median().reset_index()
    counter_work=pd.read_csv(HERE/'reports/count_scaling01/workloads.csv')
    result=values.merge(counter_work,on=['scene','frame','condition','treatment'],validate='many_to_one')
    keys=['scene','frame','condition','variant']
    reference=result[result.treatment=='same_ids_master_attributes'][keys+['wall_ms','tile_instances','alpha_interactions']].rename(
        columns={'wall_ms':'master_attributes_ms','tile_instances':'master_tile_instances','alpha_interactions':'master_alpha_interactions'})
    result=result.merge(reference,on=keys,how='left',validate='many_to_one')
    result['compensation_latency_factor']=result.wall_ms/result.master_attributes_ms
    result['compensation_tile_factor']=result.tile_instances/result.master_tile_instances.replace(0,np.nan)
    result['compensation_alpha_factor']=result.alpha_interactions/result.master_alpha_interactions.replace(0,np.nan)
    result.to_csv(HERE/'SAME_ID_COMPENSATION_CONTROL.csv',index=False)
    result[(result.treatment=='with_compensation')&(result.variant=='G')].groupby('condition').agg(
        paired_views=('frame','size'),median_latency_factor=('compensation_latency_factor','median'),
        max_latency_factor=('compensation_latency_factor','max'),median_tile_factor=('compensation_tile_factor','median'),
        median_alpha_factor=('compensation_alpha_factor','median')).to_csv(HERE/'COMPENSATION_FOOTPRINT_SUMMARY.csv')
    output=dict(heldout_scenes=int(scene[scene.split=='heldout'].shape[0]),
        motion_min_speedup_B=motion[(motion.variant=='G')&motion.condition.isin(['local0.01','local0.05','local0.1'])].groupby(['scope','motion']).speedup_B.min().to_dict(),
        remaining_limit='Full没有Selector/Compensator；局部LoD存在准备成本，不能仅由纯Rasterizer加速推出总延迟必然低于Full。')
    print('BOUNDARY_ANALYSIS_COMPLETE',output)


if __name__=='__main__':main()
