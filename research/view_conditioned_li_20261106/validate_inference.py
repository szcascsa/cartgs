"""跨语言fixture与短程模型的全局/局部推理验收。"""
import json
from pathlib import Path
import shutil
import sys
from common import ROOT,GROUPS,SCHEMAS,sha,save

OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene
from renderer import render_prepared
from selector import load_selector
from utils import MiniCam,focal2fov


@torch.no_grad()
def main():
    validation=json.loads((ROOT/'reports/VALIDATION_STATUS.json').read_text());assert validation['state']=='native_smoke_complete'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checks=[]
    for schema in [2,3,4]:
      for device in ['cpu','cuda']:
        folder=Path(validation['test_root'])/('test_joint_unit' if schema==2 else 'test_joint_cross_unit/'+str(schema))/device
        model=load_selector(folder/'selector.pt',torch.device(device))
        f=torch.jit.load(str(folder/'fixture.pt'),map_location=device).state_dict()
        spatial=model.spatial_compensation
        for local in [False,True]:
            actual=spatial(f['xyz'],f['scale'],f['rotation'],f['dc'],f['opacity'],f['center'],f['reference'],.05,
                log_scale_input=True,anchor_feature=f['feature'],view_condition=f['condition'] if local else None)
            expected=f['comp_local' if local else 'comp_global']
            assert torch.equal(actual.view(torch.int32),expected.view(torch.int32)),f'Compensation fixture {device}/{local}'
            code=model.joint_branch_view_code(f['view_code'])
            condition=torch.cat((code,code.new_ones((17,1))),1)
            packed=torch.cat((f['input11'],condition if local else condition.new_zeros((17,8))),1)
            logits=model.soft_net(model.local_emd(packed))
            assert torch.equal(logits.view(torch.int32),f['selector_local' if local else 'selector_global'].view(torch.int32))
            checks.append(dict(schema=schema,device=device,local=local,bitwise=True))
    records=[]
    for group in GROUPS:
        folder=ROOT/'runs'/f'smoke_{group}';checkpoint=folder/'12016_shutdown/ply'
        manifest=json.loads((ROOT/'configs'/f'{group}.json').read_text())
        shutil.copyfile(Path(manifest['full_ply']).parent.parent/'CameraTrajectory_TUM.txt',folder/'CameraTrajectory_TUM.txt')
        scene=BudgetScene(checkpoint,'/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',1000,0,None,'pytorch')
        assert len(scene.views)==792 and scene.renderer.total==13298
        assert int(scene.model.joint_view_conditioning_schema)==SCHEMAS[group]
        assert scene.model.spatial_compensation.view_projection.weight.shape==(64,8)
        for frame in [0,396,791]:
            v=scene.views[frame];c=scene.calibration
            camera=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose'])))
            code=scene.model.joint_view_code(camera)
            assert not code.requires_grad and torch.isfinite(code).all()
            assert code.count_nonzero()>0
            assert bool(scene.model.joint_branch_view_code(code).count_nonzero())==(group in ['selector_li','both_li'])
            assert bool(scene.model.joint_branch_view_code(code,compensation=True).count_nonzero())==(group in ['compensator_li','both_li'])
            global_data,global_ids,_=scene.renderer.prepare(None,global_ratio=.1)
            global_image=render_prepared(camera,global_data,scene.background)['render'].clone()
            for kind,value in [('points',0),('points',1000),('points',2000)]+[('ratio',r) for r in [.01,.03,.05,.08,.1,.3,.7]]:
                kwargs={'budget':value} if kind=='points' else {'ratio_budget':value}
                data,ids,stats=scene.renderer.prepare(camera,view_conditioned=True,**kwargs)
                assert len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
                assert torch.isfinite(render_prepared(camera,data,scene.background)['render']).all()
                _,again,_=scene.renderer.prepare(camera,view_conditioned=True,**kwargs);assert ids.equal(again)
                records.append(dict(group=group,frame=frame,kind=kind,value=value,K=len(ids),finite=True))
            scene.renderer.global_cache.clear();data,after,_=scene.renderer.prepare(None,global_ratio=.1)
            assert after.equal(global_ids) and torch.equal(render_prepared(camera,data,scene.background)['render'],global_image)
        del scene;torch.cuda.empty_cache()
    save(ROOT/'reports/INFERENCE_VALIDATION.json',dict(passed=True,fixture_checks=checks,inference_cases=records,
        frozen_global_output_unchanged_after_local=True,rendered_frame_images=0))


if __name__=='__main__':main()
