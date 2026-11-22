"""最终模型实际输入钩子检查；只准备属性，不重复渲染图像。"""
import json
from pathlib import Path
import sys
import csv
import hashlib
from common import ROOT,GROUPS,save

OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
import torch
import numpy as np
from budget_viewer import BudgetScene
from anchor_frustum import anchor_frustum_mask
from utils import MiniCam,focal2fov


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=[]
    with (ROOT/'evaluation/reports/PER_VIEW.csv').open() as f:
        expected_rows={(r['group'],r['case'],int(r['frame'])):r for r in csv.DictReader(f)
            if r['group'] in GROUPS and int(r['frame']) in [0,396,791]}
    for group in GROUPS:
        checkpoint=ROOT/'runs'/f'full_{group}'/'23560_shutdown/ply'
        scene=BudgetScene(checkpoint,'/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',1000,0,None,'pytorch')
        model=scene.model;captured={}
        def selector_hook(module,args):captured['selector']=args[0]
        def compensation_hook(module,args):captured['compensation']=args[0]
        hooks=[model.local_emd[0].register_forward_pre_hook(selector_hook),model.spatial_compensation.view_projection.register_forward_pre_hook(compensation_hook)]
        for frame in [0,396,791]:
            c=scene.calibration;v=scene.views[frame]
            camera=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose'])))
            code=model.joint_view_code(camera)
            visible=anchor_frustum_mask(model.anchors.centers,model.anchors.offset_scales,camera.full_proj_transform,scene.renderer.occupied)
            source=torch.nonzero(visible[model.anchors.rows]).reshape(-1)
            for scope,value in [('global',r) for r in [.01,.05,.1,.7]]+[('view',r) for r in [.01,.05,.1,.7]]+[('points',b) for b in [1000,2000]]:
                captured.clear();scene.renderer.global_cache.clear()
                if scope=='global':data,ids,stats=scene.renderer.prepare(None,global_ratio=value)
                else:
                    kwargs={'ratio_budget':value} if scope=='view' else {'budget':value}
                    data,ids,stats=scene.renderer.prepare(camera,view_conditioned=True,**kwargs)
                if scope=='global':
                    assert captured['selector'][:,11:].count_nonzero()==0
                    assert 'compensation' not in captured,'Frozen global inference must not read camera projection'
                else:
                    expected=torch.cat((model.joint_branch_view_code(code)[model.anchors.rows[source]],code.new_ones((len(source),1))),1)
                    assert torch.equal(captured['selector'][:,11:],expected)
                    if len(ids):
                        expected=torch.cat((model.joint_branch_view_code(code,compensation=True)[model.anchors.rows[ids]],code.new_ones((len(ids),1))),1)
                        assert torch.equal(captured['compensation'],expected)
                assert all(torch.isfinite(x).all() for x in data['inputs'].values() if isinstance(x,torch.Tensor))
                case=f'{scope}_{value:g}' if scope!='points' else f'points_{value}'
                digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                reference=expected_rows[group,case,frame]
                assert digest==reference['selection_sha256'] and len(ids)==int(reference['K'])
                records.append(dict(group=group,frame=frame,scope=scope,value=value,K=len(ids),selection_sha256=digest,
                    separate_branch_conditions_match_training=True,same_as_evaluated_selection=True))
        for hook in hooks:hook.remove()
        del scene;torch.cuda.empty_cache()
    assert len(records)==60
    save(ROOT/'reports/FINAL_CONDITIONING_VALIDATION.json',dict(passed=True,cases=records,rendered_images=0))


if __name__=='__main__':main()
