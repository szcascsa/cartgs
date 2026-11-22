"""只读复用原无视角指标，只渲染mode/view两组并记录补偿幅度。"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
from budget_viewer import BudgetScene
import view_residual,spatial_compensation,selector,anchor_budget,renderer,anchor_frustum
MODULES={m.__name__:str(Path(m.__file__).resolve()) for m in [view_residual,spatial_compensation,selector,anchor_budget,renderer,anchor_frustum]}
assert all(Path(p).parent==ROOT/'source/python' for p in MODULES.values())
sys.path.insert(0,str(OLD/'source/research/global_view_joint_lod_20261105'))
import evaluate_full_experiment as base
sys.path.insert(0,str(ROOT/'source/python'))

assert base.BudgetScene is BudgetScene
OUT=ROOT/'evaluation/reports'
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
TRAIN=[.01,.05,.1,.3,.5,.7]
GEOMETRY_FIELDS=['xyz_p50','xyz_p90','xyz_p99','xyz_max','scale_p50','scale_p90','scale_p99','scale_max','rotation_p99_deg','opacity_mean']
CASES=[('full','full',None)]+[(f'{scope}_{r:g}',scope,r) for scope in ['global','view'] for r in RATIOS]
CASES += [(f'matched_{r:g}','matched',r) for r in [.1,.2,.3,.4,.5]]
CASES += [(f'points_{b}','points',b) for b in [1000,2000,3000,4000,5000,10000]]



@base.torch.no_grad()
def main():
    assert json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())['state']=='training_complete'
    OUT.mkdir(exist_ok=True);assert not (OUT/'PER_VIEW.csv').exists()
    base.save(OUT/'SOURCE_MODULES.json',MODULES)
    base.torch.set_num_threads(2);base.torch.backends.cuda.matmul.allow_tf32=False
    base.torch.backends.cudnn.allow_tf32=False;base.torch.backends.cudnn.benchmark=False
    started=time.monotonic()
    binding_path=Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    spec=importlib.util.spec_from_file_location('fr1xyz_production_counts',binding_path)
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    lpips=base.LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    protected=json.loads((ROOT/'reports/HISTORICAL_INPUTS.json').read_text())
    for name,value in protected.items():assert base.sha(Path(name))==value,name
    previous_csv=PREVIOUS/'evaluation/reports/PER_VIEW.csv'
    protected[str(previous_csv)]=base.sha(previous_csv)
    with previous_csv.open() as prior:old_rows=[dict(r,group='baseline',view_conditioned=False,diagnostic=False) for r in csv.DictReader(prior) if r['group']=='no_li']
    assert len(old_rows)==26928
    reference=[r for r in old_rows if r['case']=='full' and r['group']=='baseline']
    assert len(reference)==792 and [int(r['frame']) for r in reference]==list(range(792))
    parity=[];all_rows=list(old_rows)
    frame_keys=[(str(r['timestamp']),r['image']) for r in reference]
    fields=base.FIELDS+['network_ratio','source_global_contributing_K','view_conditioned','diagnostic']+GEOMETRY_FIELDS
    with (OUT/'PER_VIEW.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(old_rows)
        for group in ['mode','view']:
            folder=ROOT/'runs'/f'full_{group}';checkpoint=folder/'23560_shutdown/ply'
            for p in [checkpoint/'selector.pt',checkpoint/'selector_config.yaml',checkpoint/'cameras.json',
                      checkpoint/'point_cloud/iteration_23560/point_cloud.ply',folder/'CameraTrajectory_TUM.txt']:
                protected[str(p)]=base.sha(p)
            scene=base.BudgetScene(checkpoint,'/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',5000,0,None,'pytorch')
            assert len(scene.views)==792 and scene.renderer.total==13298
            keys=[(str(v['timestamp']),v['image']) for v in scene.views]
            assert keys==frame_keys
            bank={str((Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')/v['img_name']).resolve())
                  for v in json.loads((checkpoint/'cameras.json').read_text())}
            # Frozen global exports are view independent; retain the finite 11-ratio set.
            global_exports={r:scene.renderer.prepare(None,global_ratio=r) for r in RATIOS}
            predictions,truths,pending=[],[],[]

            def flush():
                if not pending:return
                image,target=base.torch.stack(predictions),base.torch.stack(truths)
                d=image-target
                metric=base.torch.stack([-10*base.torch.log10(d.square().mean((1,2,3))),
                    base.structural_similarity_index_measure(image,target,data_range=1.,reduction='none'),
                    lpips(image,target).reshape(-1),d.abs().mean((1,2,3))],1);lpips.reset()
                assert base.torch.isfinite(metric).all()
                for row,values in zip(pending,metric.cpu().tolist()):
                    row.update(zip(['psnr','ssim','lpips','photo_l1'],values));writer.writerow(row);all_rows.append(row)
                predictions.clear();truths.clear();pending.clear()

            for frame,v in enumerate(scene.views):
                c=scene.calibration
                camera=base.MiniCam(c['width'],c['height'],base.focal2fov(c['fx'],c['width']),
                    base.focal2fov(c['fy'],c['height']),base.np.linalg.inv(base.np.asarray(v['pose'])))
                truth=scene.ground_truth(frame);assert truth is not None
                global_counts={}
                for name,scope,value in CASES:
                    conditioned=scope in ['view','matched','points']
                    if scope=='full':
                        data,ids=scene.full_data,scene.full_ids
                        stats=dict(candidate_gaussians=scene.renderer.total,eligible_candidates=scene.renderer.total,
                                   requested_budget=scene.renderer.total,budget_shortfall=0,ratio=None)
                    elif scope=='global':data,ids,stats=global_exports[value]
                    elif scope in ['view','view_off']:data,ids,stats=scene.renderer.prepare(camera,ratio_budget=value,view_conditioned=conditioned)
                    else:
                        budget=global_counts[value] if scope=='matched' else value
                        data,ids,stats=scene.renderer.prepare(camera,budget=budget,view_conditioned=conditioned)
                    if scope in ['view','matched','points','view_off','points_off']:
                        assert len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
                    output=base.counted_render(binding,camera,data,scene.background)
                    geometry={}
                    if scope!='full' and len(ids):
                        pc=scene.gaussians if hasattr(scene,'gaussians') else scene.renderer.gaussians
                        inputs=data['inputs'];owners=scene.model.anchors.rows[ids]
                        displacement=(inputs['means3D']-pc.get_xyz[ids]).norm(dim=1)/scene.model.anchors.offset_scales[owners].reshape(-1)
                        scale=(inputs['scales']/pc.get_scaling[ids]).amax(1)
                        quantiles=base.torch.tensor([.5,.9,.99],device=displacement.device)
                        geometry.update(zip(GEOMETRY_FIELDS[:4],base.torch.quantile(displacement,quantiles).cpu().tolist()+[float(displacement.max())]))
                        geometry.update(zip(GEOMETRY_FIELDS[4:8],base.torch.quantile(scale,quantiles).cpu().tolist()+[float(scale.max())]))
                        dot=(inputs['rotations']*pc.get_rotation[ids]).sum(1).abs().clamp(0,1)
                        geometry['rotation_p99_deg']=float(base.torch.quantile(2*dot.acos()*180/math.pi,.99))
                        geometry['opacity_mean']=float(inputs['opacities'].mean())
                    projected,contributed=int((output[3]>0).sum()),int((output[5]>0).sum())
                    assert 0<=contributed<=projected<=len(ids)
                    if scope=='global':global_counts[value]=contributed
                    if frame in [0,396,791]:
                        normal=base.render_prepared(camera,data,scene.background,return_radii=True)
                        assert base.torch.equal(normal['render'].view(base.torch.int32),output[2].view(base.torch.int32))
                        assert base.torch.equal(normal['radii'],output[3])
                        parity.append(dict(group=group,case=name,frame=frame,bitwise=True))
                    row=dict(group=group,case=name,scope=scope,ratio=None if scope in ['full','points','points_off'] else value,
                        ratio_split='full' if scope=='full' else 'fixed_points' if scope in ['points','points_off'] else 'train' if value in TRAIN else 'unseen',
                        frame=frame,timestamp=v['timestamp'],image=v['image'],bank=str(Path(v['image']).resolve()) in bank,
                        N=scene.renderer.total,candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
                        requested_K=stats['requested_budget'],K=len(ids),projected_K=projected,contributing_K=contributed,
                        budget_shortfall=stats['budget_shortfall'],selection_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                        network_ratio=stats['ratio'],source_global_contributing_K=global_counts[value] if scope=='matched' else None,
                        view_conditioned=conditioned,diagnostic=scope in ['view_off','points_off'])
                    row.update(geometry)
                    predictions.append(output[2].clamp(0,1));truths.append(truth);pending.append(row)
                    if len(pending)==8:flush()
                    del output,data
                if frame%20==0 or frame==791:
                    flush();stream.flush()
                    state=dict(state='running',group=group,frame=frame+1,frames=792,rows=len(all_rows),seconds=time.monotonic()-started)
                    base.save(ROOT/'evaluation/STATUS.json',state);print(json.dumps(state),flush=True)
            flush();del scene,global_exports;base.torch.cuda.empty_cache()
    after={p:base.sha(Path(p)) for p in protected};assert after==protected
    base.save(OUT/'INPUT_INTEGRITY.json',dict(before=protected,after=after,passed=True))
    base.save(OUT/'RASTERIZER_PARITY.json',parity)
    assert len(all_rows)==80784 and len(parity)==204
    base.save(ROOT/'evaluation/STATUS.json',dict(state='complete',groups=3,rendered_groups=2,reused_groups=1,frames=792,main_cases_per_group=len(CASES),diagnostic_cases_per_new_group=0,
        rows=len(all_rows),new_rows=53856,reused_rows=len(old_rows),images_saved=0,parity_checks=len(parity),seconds=time.monotonic()-started))


if __name__=='__main__':
    try:main()
    except Exception as error:
        base.save(ROOT/'evaluation/STATUS.json',dict(state='failed',error=repr(error)));raise
