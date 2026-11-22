"""追加控制实验：不使用归因结果重选候选或修改原Full门限。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/AGGREGATE'


def geometric(x):return float(np.exp(np.log(x).mean()))


def main():
    state=json.loads((HERE/'reports/ATTRIBUTION_CONTROLLER.json').read_text())
    assert state['state']=='complete' and len(state['accepted'])==32
    samples=[];checks=[]
    for name in state['accepted']:
        job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text());assert job['measurement_valid']
        for key,target in [('samples',samples),('checks',checks)]:
            table=pd.read_csv(HERE/'reports'/name/(key+'.csv'));table['job']=name;target.append(table)
    samples=pd.concat(samples,ignore_index=True);checks=pd.concat(checks,ignore_index=True)
    assert len(samples)==345600 and len(checks)==11520
    assert checks.rgb_bitwise.all() and checks.radii_bitwise.all() and (checks.max_rgb_error==0).all()
    samples.to_csv(OUT/'ATTRIBUTION_SAMPLES.csv.gz',index=False,compression='gzip')
    checks.to_csv(OUT/'ATTRIBUTION_CHECKS.csv.gz',index=False,compression='gzip')
    perf=samples.groupby(['scene','condition','round','variant']).ms.agg(
        p50_ms='median',p95_ms=lambda x:x.quantile(.95),samples='size').reset_index()
    pivot=perf.pivot(index=['scene','condition','round'],columns='variant',values='p50_ms').reset_index()
    for n in ('rebuilt_F2','common'):pivot[n+'_vs_F2']=pivot.F2/pivot[n]
    pivot['common_vs_rebuilt']=pivot.rebuilt_F2/pivot.common
    pivot['common_vs_rebuilt_saved_us']=(pivot.rebuilt_F2-pivot.common)*1000
    perf.to_csv(OUT/'ATTRIBUTION_PER_SCENE.csv',index=False)
    pivot.to_csv(OUT/'ATTRIBUTION_PAIRS.csv',index=False)
    summary=pivot.groupby('condition').agg(rebuilt_vs_F2=('rebuilt_F2_vs_F2',geometric),
        common_vs_F2=('common_vs_F2',geometric),common_vs_rebuilt=('common_vs_rebuilt',geometric),
        common_vs_rebuilt_saved_us=('common_vs_rebuilt_saved_us','mean')).reset_index()
    summary.to_csv(OUT/'ATTRIBUTION_SUMMARY.csv',index=False)
    (OUT/'ATTRIBUTION.json').write_text(json.dumps(dict(samples=len(samples),checks=len(checks),bitwise_pass=True,
        accepted=state['accepted'],rejected=state['rejected'],summary=summary.to_dict('records')),indent=2))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
