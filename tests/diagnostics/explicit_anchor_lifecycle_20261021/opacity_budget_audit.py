"""只读复核原 checkpoint 中不可能超过 rasterizer alpha 截断的点及其 LoD 预算占用。"""
import csv
from pathlib import Path
import numpy as np
from plyfile import PlyData

root=Path('/home/prmi/zwj/octree-explicit-20261021')
out=Path('/home/prmi/zwj/octree-explicit-diagnostics-20261021/frozen_evidence')
rows=[]
for seed in (2026,2027):
    cloud=root/f'explicit_seed{seed}/10000_shutdown/ply/point_cloud/iteration_10000/point_cloud.ply'
    d=PlyData.read(cloud)['vertex'].data
    a=1/(1+np.exp(-d['opacity'].astype(float)))
    low=a<.02;cut=a<1/255
    row=dict(seed=seed,policy='Full',ratio=100,selected=len(a),
        selected_below_pruning_threshold=int(low.sum()),selected_below_renderer_alpha_cutoff=int(cut.sum()),
        effective_opacity_eligible=int((~cut).sum()))
    rows.append(row)
    for policy in ['cached_bernoulli','same_k']:
        for ratio in [1,5,10,15]:
            ids=np.load(root/'quality'/f'explicit_seed{seed}'/policy/f'LOD_{ratio}/selected_rows.npz')['rows']
            rows.append(dict(seed=seed,policy=policy,ratio=ratio,selected=len(ids),
                selected_below_pruning_threshold=int(low[ids].sum()),selected_below_renderer_alpha_cutoff=int(cut[ids].sum()),
                effective_opacity_eligible=int((~cut[ids]).sum())))
with (out/'original_lod_low_opacity_selection.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print([r for r in rows if r['policy']=='Full'])
