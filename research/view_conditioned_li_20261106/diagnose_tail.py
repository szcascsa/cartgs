"""仅诊断三组新LI模型，固定三帧核验XYZ行对应及Alpha参与。"""
import csv
import importlib.util
import json
from pathlib import Path
import torch
from common import ROOT,GROUPS,save
import evaluate_full as evaluation

base=evaluation.base
OUT=ROOT/'evaluation/reports'


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    spec=importlib.util.spec_from_file_location('fr1xyz_production_counts',p);binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    records=[];cases=[];checks=[]
    fields=['xyz_normalized','xyz_meters','scale_multiplier','opacity','depth_before','depth_after','view_depth_code','ndc_x','ndc_y','extent_code',
            'alpha_before','alpha_after','radii_after']
    for group in GROUPS:
        scene=base.BudgetScene(ROOT/'runs'/f'full_{group}'/'23560_shutdown/ply','/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',1000,0,None,'pytorch')
        model=scene.model;pc=scene.renderer.gaussians;spatial=model.spatial_compensation
        for frame in [0,396,791]:
            c=scene.calibration;v=scene.views[frame]
            camera=base.MiniCam(c['width'],c['height'],base.focal2fov(c['fx'],c['width']),base.focal2fov(c['fy'],c['height']),base.np.linalg.inv(base.np.asarray(v['pose'])))
            full=base.counted_render(binding,camera,scene.full_data,scene.background)
            code=model.joint_view_code(camera)
            for scope,value in [('view',.01),('view',.05),('view',.1),('points',2000)]:
                kwargs={'ratio_budget':value} if scope=='view' else {'budget':value}
                data,ids,stats=scene.renderer.prepare(camera,view_conditioned=True,**kwargs)
                owners=model.anchors.rows[ids];reference=model.anchors.offset_scales[owners]
                comp_code=model.joint_branch_view_code(code,compensation=True)
                condition=torch.cat((comp_code[owners],code.new_ones((len(ids),1))),1)
                raw=spatial(pc.get_xyz[ids],pc._scaling[ids],pc.get_rotation[ids],pc.get_features_dc[ids],pc._opacity[ids],
                    model.anchors.centers[owners],reference,stats['ratio'],log_scale_input=True,
                    anchor_feature=model.anchors.compensation_lookup(ids,stats['ratio']),view_condition=condition)
                expected=pc.get_xyz[ids]+raw[:,:3]
                assert torch.equal(expected.view(torch.int32),data['inputs']['means3D'].view(torch.int32)),'XYZ/selected row mismatch'
                out=base.counted_render(binding,camera,data,scene.background);assert len(out[5])==len(ids)
                delta=(data['inputs']['means3D']-pc.get_xyz[ids]).norm(dim=1)
                movement=delta/reference.reshape(-1);factor=(data['inputs']['scales']/pc.get_scaling[ids]).amax(1)
                before=torch.cat((pc.get_xyz[ids],torch.ones_like(pc.get_xyz[ids][:,:1])),1)@camera.world_view_transform
                after=torch.cat((data['inputs']['means3D'],torch.ones_like(data['inputs']['means3D'][:,:1])),1)@camera.world_view_transform
                alpha_before=full[5][ids]>0;alpha_after=out[5]>0;tail=movement>10
                item=dict(group=group,frame=frame,case=f'{scope}_{value:g}',K=len(ids),tail_threshold=10.,tail_points=int(tail.sum()),
                    xyz_p99=float(torch.quantile(movement,.99)),xyz_max=float(movement.max()),xyz_meters_max=float(delta.max()),
                    tail_alpha_before_fraction=float(alpha_before[tail].float().mean()) if tail.any() else None,
                    tail_alpha_after_fraction=float(alpha_after[tail].float().mean()) if tail.any() else None,
                    tail_opacity_mean=float(data['inputs']['opacities'].reshape(-1)[tail].mean()) if tail.any() else None,
                    tail_near_plane_extent_fraction=float((code[owners,6][tail]>9).float().mean()) if tail.any() else None)
                cases.append(item);checks.append(dict(group=group,frame=frame,case=item['case'],xyz_matches_direct_network_bitwise=True))
                packed=torch.stack((movement,delta,factor,data['inputs']['opacities'].reshape(-1),before[:,2],after[:,2],code[owners,3],code[owners,4],
                    code[owners,5],code[owners,6],alpha_before.float(),alpha_after.float(),out[3].float()),1).cpu().tolist()
                for row,anchor,values in zip(ids.cpu().tolist(),owners.cpu().tolist(),packed):
                    records.append(dict(group=group,frame=frame,case=item['case'],gaussian_row=row,anchor_row=anchor,**dict(zip(fields,values))))
            del full
        del scene;torch.cuda.empty_cache()
    with (OUT/'TAIL_DIAGNOSTIC_POINTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    save(OUT/'TAIL_DIAGNOSTIC_CASES.json',dict(cases=cases,checks=checks,point_records=len(records),extra_new_model_renders=45,
        baseline_rerenders=0,images_saved=0,scope='固定首/中/末三帧；>10仅用于分组诊断，不修改训练或成功门槛'))
    print(json.dumps(dict(cases=len(cases),points=len(records),checks=len(checks))),flush=True)


if __name__=='__main__':main()
