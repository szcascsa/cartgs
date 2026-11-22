"""仅CPU检查固定拓扑下各组Master终态差异，不渲染、不更新模型。"""
import json
import numpy as np
from plyfile import PlyData
from common import ROOT,BASE,PREVIOUS,GROUPS,save


def read(path):
    p=PlyData.read(str(path))['vertex'].data
    return {k:np.array(p[k],copy=True).astype(np.float64) for k in p.dtype.names}


def xyz(p):return np.stack([p[k] for k in ['x','y','z']],1)


def metrics(a):
    return {n:float(v) for n,v in zip(['p50','p90','p99','max'],list(np.quantile(a,[.5,.9,.99]))+[a.max()])}


def main():
    initial=read(ROOT/'runs/full_selector_view/step0_master.ply');models={}
    for g in ['baseline','mode','view']+GROUPS:
        folder=BASE/'runs/full_no_li' if g=='baseline' else (PREVIOUS if g in ['mode','view'] else ROOT)/'runs'/f'full_{g}'
        models[g]=read(folder/'23560_shutdown/ply/point_cloud/iteration_23560/point_cloud.ply')
        assert len(models[g]['x'])==len(initial['x'])==13298
    rows=[]
    for ref,base in [('initial',initial),('baseline',models['baseline'])]:
        for group,p in models.items():
            delta=np.linalg.norm(xyz(p)-xyz(base),axis=1)
            scale=np.exp(np.stack([p[f'scale_{i}']-base[f'scale_{i}'] for i in range(3)],1)).max(1)
            rows.append(dict(group=group,reference=ref,N=13298,xyz_meters=metrics(delta),maximum_axis_scale_ratio=metrics(scale)))
    save(ROOT/'evaluation/reports/MASTER_DRIFT.json',dict(passed=True,cpu_only=True,renderer_calls=0,
        correspondence='Fine topology fixed; original PLY row order preserved by verified initializer/entry',rows=rows,
        limitation='终态差异量化Master混杂，不推断其对PSNR的独立因果作用'))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
