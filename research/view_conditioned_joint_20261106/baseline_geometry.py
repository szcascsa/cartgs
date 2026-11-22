"""只补充基线前向属性统计，不训练、不调用Renderer、不重新计算图像指标。"""
import csv
import hashlib
import json
import torch
import numpy as np
from common import ROOT,BASE,save
import evaluate_full as evaluation

base=evaluation.base
OUT=ROOT/'evaluation/reports'


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (OUT/'PER_VIEW.csv').open() as f:
        expected={(r['case'],int(r['frame'])):r for r in csv.DictReader(f) if r['group']=='baseline'}
    scene=base.BudgetScene(BASE/'runs/full_no_li/23560_shutdown/ply','/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',1000,0,None,'pytorch')
    pc=scene.renderer.gaussians;records=[]
    for frame,v in enumerate(scene.views):
        c=scene.calibration
        camera=base.MiniCam(c['width'],c['height'],base.focal2fov(c['fx'],c['width']),base.focal2fov(c['fy'],c['height']),base.np.linalg.inv(base.np.asarray(v['pose'])))
        for scope,value in [('view',.01),('view',.05),('view',.1),('points',2000)]:
            kwargs={'ratio_budget':value} if scope=='view' else {'budget':value}
            data,ids,stats=scene.renderer.prepare(camera,**kwargs)
            case=f'{scope}_{value:g}';ref=expected[case,frame];digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            assert digest==ref['selection_sha256'] and len(ids)==int(ref['K'])
            reference=scene.model.anchors.offset_scales[scene.model.anchors.rows[ids]].reshape(-1)
            movement=(data['inputs']['means3D']-pc.get_xyz[ids]).norm(dim=1)/reference
            scale=(data['inputs']['scales']/pc.get_scaling[ids]).amax(1)
            q=torch.tensor([.5,.9,.99],device=movement.device)
            row=dict(group='baseline',case=case,frame=frame,K=len(ids),selection_sha256=digest)
            row.update(zip(['xyz_p50','xyz_p90','xyz_p99','xyz_max'],torch.quantile(movement,q).cpu().tolist()+[float(movement.max())]))
            row.update(zip(['scale_p50','scale_p90','scale_p99','scale_max'],torch.quantile(scale,q).cpu().tolist()+[float(scale.max())]))
            records.append(row)
    with (OUT/'BASELINE_GEOMETRY_PER_VIEW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for case in ['view_0.01','view_0.05','view_0.1','points_2000']:
        rows=[r for r in records if r['case']==case];assert len(rows)==792
        item=dict(group='baseline',case=case,frames=792)
        for key in ['xyz_p50','xyz_p90','xyz_p99','xyz_max','scale_p50','scale_p90','scale_p99','scale_max']:item[key+'_mean']=float(np.mean([r[key] for r in rows]))
        item.update(xyz_global_max=max(r['xyz_max'] for r in rows),scale_global_max=max(r['scale_max'] for r in rows));summary.append(item)
    save(OUT/'BASELINE_GEOMETRY_SUMMARY.json',dict(passed=True,rows=len(records),renderer_calls=0,quality_metrics_recomputed=0,
        selected_ids_match_original_evaluation=True,summary=summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
