"""合成压力输入单独列出，不混入24组正式加速统计。"""
from pathlib import Path
import json,pandas as pd

HERE=Path(__file__).resolve().parent;name='footprint_stress02'
job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text());assert job['measurement_valid'] and job['foreign_active_samples']==0
root=HERE/'reports'/name;d=pd.read_csv(root/'samples.csv');checks=pd.read_csv(root/'checks.csv')
assert checks[['RGB_bitwise','radii_bitwise']].all().all()
key=['width','height','K','scale','opacity']
s=d.groupby(key+['variant']).agg(median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda x:x.quantile(.95))).reset_index()
ref=s[s.variant=='G'][key+['median_ms']].rename(columns={'median_ms':'G_ms'});s=s.merge(ref,on=key)
s['speedup_vs_G']=s.G_ms/s.median_ms;s['latency_change_pct_vs_G']=(s.median_ms/s.G_ms-1)*100
s=s.merge(checks[checks.variant=='C'][key+['tile_instances','members_max','overflow_tiles','preprocess_visible']],on=key)
s.to_csv(HERE/'reports/SYNTHETIC_STRESS_RESULTS.csv',index=False)
print(s[s.variant.isin(['C','D'])].groupby(['variant','scale']).speedup_vs_G.agg(['min','median','max']).round(5).to_string())
print('SLOWER_CASES',s[s.variant.isin(['C','D'])& (s.speedup_vs_G<1)].to_string(index=False))
