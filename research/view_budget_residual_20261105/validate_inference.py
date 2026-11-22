"""短程checkpoint的Python视角推理验收；数值健康不代表最终质量。"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),
    str(OLD/'evaluation/python/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene
from renderer import render_prepared
from utils import MiniCam,focal2fov
from view_residual import anchor_view_code


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    assert json.loads((ROOT/'reports/VALIDATION_STATUS.json').read_text())['state']=='native_smoke_complete'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=[];identity={}
    for group in ['view_no_li','view_li01']:
        folder=ROOT/'runs'/f'smoke_{group}';checkpoint=folder/'12016_shutdown/ply'
        manifest=json.loads((ROOT/'configs'/f'{group}.json').read_text())
        trajectory=Path(manifest['full_ply']).parent.parent/'CameraTrajectory_TUM.txt'
        assert not (folder/'CameraTrajectory_TUM.txt').exists()
        shutil.copyfile(trajectory,folder/'CameraTrajectory_TUM.txt')
        for p in [checkpoint/'selector.pt',checkpoint/'selector_config.yaml',
                  checkpoint/'point_cloud/iteration_12016/point_cloud.ply']:identity[str(p)]=sha(p)
        scene=BudgetScene(checkpoint,'/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',1000,0,None,'pytorch')
        assert len(scene.views)==792 and scene.renderer.total==13298
        branch=scene.model.selector_view_residual
        assert sum(p.numel() for p in branch.parameters())==657 and int(branch.initialized)==1
        codes=[];view_logits=[]
        for frame in [0,396,791]:
            v=scene.views[frame];c=scene.calibration
            camera=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),
                np.linalg.inv(np.asarray(v['pose'])))
            data,global_ids,stats=scene.renderer.prepare(None,global_ratio=.1)
            global_image=render_prepared(camera,data,scene.background)['render'].clone()
            for budget in [0,1000,2000]:
                data,ids,stats=scene.renderer.prepare(camera,budget=budget,view_conditioned=True)
                assert len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
                output=render_prepared(camera,data,scene.background)['render']
                assert torch.isfinite(output).all()
                _,again,_=scene.renderer.prepare(camera,budget=budget,view_conditioned=True)
                assert ids.equal(again),'Fixed frozen input changed selected identities'
                records.append(dict(group=group,frame=frame,kind='points',value=budget,K=len(ids),finite=True))
            for ratio in [.01,.03,.05,.08,.3,.7]:
                data,ids,stats=scene.renderer.prepare(camera,ratio_budget=ratio,view_conditioned=True)
                assert len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
                assert torch.isfinite(render_prepared(camera,data,scene.background)['render']).all()
                records.append(dict(group=group,frame=frame,kind='ratio',value=ratio,K=len(ids),finite=True))
            scene.renderer.global_cache.clear()
            data,after,_=scene.renderer.prepare(None,global_ratio=.1)
            assert (after.equal(global_ids) and torch.equal(render_prepared(camera,data,scene.background)['render'],global_image)), \
                'Local view inference mutated global output'
            anchors=scene.model.anchors
            code=anchor_view_code(anchors.centers,anchors.offset_scales,camera.camera_center,camera.world_view_transform,
                camera.full_proj_transform,camera.image_height,camera.image_width,camera.znear,branch.scene_scale)
            assert code.shape==(2790,7) and not code.requires_grad and torch.isfinite(code).all()
            codes.append(code)
            pc=scene.gaussians
            _,_,logits=scene.model.forward_view_selection_features(pc.get_xyz,pc.get_scaling,
                pc.get_features_dc,pc.get_opacity,pc.get_xyz.new_tensor([[.1]]),camera,return_logits=True)
            view_logits.append(logits)
        assert not torch.equal(codes[0],codes[-1]),'Camera descriptor ignored viewpoint'
        assert not torch.equal(view_logits[0],view_logits[-1]),'Trained local logits ignored viewpoint'
        del scene;torch.cuda.empty_cache()
    assert all(sha(Path(p))==h for p,h in identity.items())
    report=dict(passed=True,groups=2,frame_checks=3,cases_per_frame=9,cases=len(records),images_saved=0,
        deterministic_selected_ids=True,local_did_not_mutate_global=True,camera_changes_local_logits=True,checkpoint_unchanged=True,
        checkpoint_identity=identity,records=records)
    (ROOT/'reports/INFERENCE_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','checkpoint_identity']},indent=2))


if __name__=='__main__':main()
