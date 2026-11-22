"""汇总三帧长尾的真实可见性，不进行额外渲染。"""
import csv
import statistics
from common import ROOT,GROUPS,save

OUT=ROOT/'evaluation/reports'
with (OUT/'TAIL_DIAGNOSTIC_POINTS.csv').open() as f:rows=list(csv.DictReader(f))
results=[]
for group in GROUPS:
    for case in ['view_0.01','view_0.05','view_0.1','points_2000']:
        all_rows=[r for r in rows if r['group']==group and r['case']==case]
        tail=[r for r in all_rows if float(r['xyz_normalized'])>10]
        result=dict(group=group,case=case,frames=3,selected_records=len(all_rows),tail_records=len(tail),
            tail_unique_gaussians=len({r['gaussian_row'] for r in tail}),tail_unique_anchors=len({r['anchor_row'] for r in tail}),
            tail_alpha_before=sum(float(r['alpha_before'])>0 for r in tail),tail_alpha_after=sum(float(r['alpha_after'])>0 for r in tail),
            tail_radii_zero=sum(float(r['radii_after'])==0 for r in tail),tail_behind_after=sum(float(r['depth_after'])<=.01 for r in tail),
            tail_opacity_median=statistics.median(float(r['opacity']) for r in tail) if tail else None,
            tail_opacity_below_001=sum(float(r['opacity'])<.01 for r in tail),
            tail_extent_code_above_9=sum(float(r['extent_code'])>9 for r in tail))
        results.append(result)
save(OUT/'TAIL_DIAGNOSTIC_SUMMARY.json',results)
print(results)
