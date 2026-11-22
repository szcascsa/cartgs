"""同批次逐场景显示低Ratio收益、Full代价及候选反例。"""
from pathlib import Path
import argparse,json
import pandas as pd

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
root=HERE/'reports'/args.name
job=json.loads((HERE/'reports/jobs'/(args.name+'.json')).read_text());assert job['measurement_valid'] and job['foreign_active_samples']==0
summary=json.loads((root/'SUMMARY.json').read_text());assert summary['passed']
d=pd.read_csv(root/'samples.csv')
s=d.groupby(['scene','condition','variant']).agg(median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda x:x.quantile(.95)),K=('K','median')).reset_index()
b=s[s.variant=='previous_G'][['scene','condition','median_ms']].rename(columns={'median_ms':'previous_ms'})
s=s.merge(b,on=['scene','condition']);s['speedup']=s.previous_ms/s.median_ms;s['saved_us']=(s.previous_ms-s.median_ms)*1000
s.to_csv(root/'COMPARISON.csv',index=False)
print(s.groupby(['variant','condition']).agg(min_speedup=('speedup','min'),median_speedup=('speedup','median'),max_speedup=('speedup','max'),min_saved_us=('saved_us','min'),max_saved_us=('saved_us','max')).round(4).to_string())
print('\nALL_SCENES')
print(s[s.variant!='previous_G'].pivot(index=['scene','condition'],columns='variant',values='speedup').round(4).to_string())
