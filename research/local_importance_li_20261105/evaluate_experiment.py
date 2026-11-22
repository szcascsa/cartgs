"""对两组冻结模型进行全轨迹评价，包括ratio、固定点数和贡献匹配预算。"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
sys.path.insert(0,str(OLD/'source/research/global_view_joint_lod_20261105'))
import evaluate_full_experiment as base

OUT=ROOT/'evaluation/reports'
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
TRAIN=[.01,.05,.1,.3,.5,.7]
CASES=[('full','full',None)]+[(f'{scope}_{r:g}',scope,r) for scope in ['global','view'] for r in RATIOS]
CASES += [(f'matched_{r:g}','matched',r) for r in [.1,.2,.3,.4,.5]]
CASES += [(f'points_{b}','points',b) for b in [1000,2000,3000,4000,5000,10000]]


@base.torch.no_grad()
def main():
    assert json.loads((ROOT/'reports/EXPERIMENT_STATUS.json').read_text())['state']=='training_complete'
    OUT.mkdir(exist_ok=True);assert not (OUT/'PER_VIEW.csv').exists()
    base.torch.set_num_threads(2);base.torch.backends.cuda.matmul.allow_tf32=False
    base.torch.backends.cudnn.allow_tf32=False;base.torch.backends.cudnn.benchmark=False
    started=time.monotonic()
    binding_path=Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    spec=importlib.util.spec_from_file_location('fr1xyz_production_counts',binding_path)
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    lpips=base.LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    protected={};parity=[];all_rows=[];frame_keys=None
    fields=base.FIELDS+['network_ratio','source_global_contributing_K']
    with (OUT/'PER_VIEW.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group in ['no_li','li01']:
            folder=ROOT/'runs'/f'full_{group}';checkpoint=folder/'23560_shutdown/ply'
            for p in [checkpoint/'selector.pt',checkpoint/'selector_config.yaml',checkpoint/'cameras.json',
                      checkpoint/'point_cloud/iteration_23560/point_cloud.ply',folder/'CameraTrajectory_TUM.txt']:
                protected[str(p)]=base.sha(p)
            scene=base.BudgetScene(checkpoint,'/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',5000,0,None,'pytorch')
            assert len(scene.views)==792 and scene.renderer.total==13298
            keys=[(v['timestamp'],v['image']) for v in scene.views]
            if frame_keys is None:frame_keys=keys
            assert keys==frame_keys
            bank={str((Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')/v['img_name']).resolve())
                  for v in json.loads((checkpoint/'cameras.json').read_text())}
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
                    if scope=='full':
                        data,ids=scene.full_data,scene.full_ids
                        stats=dict(candidate_gaussians=scene.renderer.total,eligible_candidates=scene.renderer.total,
                                   requested_budget=scene.renderer.total,budget_shortfall=0,ratio=None)
                    elif scope=='global':data,ids,stats=scene.renderer.prepare(None,global_ratio=value)
                    elif scope=='view':data,ids,stats=scene.renderer.prepare(camera,ratio_budget=value)
                    else:
                        budget=global_counts[value] if scope=='matched' else value
                        data,ids,stats=scene.renderer.prepare(camera,budget=budget)
                    if scope in ['view','matched','points']:
                        assert len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
                    output=base.counted_render(binding,camera,data,scene.background)
                    projected,contributed=int((output[3]>0).sum()),int((output[5]>0).sum())
                    assert 0<=contributed<=projected<=len(ids)
                    if scope=='global':global_counts[value]=contributed
                    if frame in [0,396,791]:
                        normal=base.render_prepared(camera,data,scene.background,return_radii=True)
                        assert base.torch.equal(normal['render'].view(base.torch.int32),output[2].view(base.torch.int32))
                        assert base.torch.equal(normal['radii'],output[3])
                        parity.append(dict(group=group,case=name,frame=frame,bitwise=True))
                    row=dict(group=group,case=name,scope=scope,ratio=None if scope in ['full','points'] else value,
                        ratio_split='full' if scope=='full' else 'fixed_points' if scope=='points' else 'train' if value in TRAIN else 'unseen',
                        frame=frame,timestamp=v['timestamp'],image=v['image'],bank=str(Path(v['image']).resolve()) in bank,
                        N=scene.renderer.total,candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
                        requested_K=stats['requested_budget'],K=len(ids),projected_K=projected,contributing_K=contributed,
                        budget_shortfall=stats['budget_shortfall'],selection_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                        network_ratio=stats['ratio'],source_global_contributing_K=global_counts[value] if scope=='matched' else None)
                    predictions.append(output[2].clamp(0,1));truths.append(truth);pending.append(row)
                    if len(pending)==8:flush()
                    del output,data
                if frame%20==0 or frame==791:
                    flush();stream.flush()
                    state=dict(state='running',group=group,frame=frame+1,frames=792,rows=len(all_rows),seconds=time.monotonic()-started)
                    base.save(ROOT/'evaluation/STATUS.json',state);print(json.dumps(state),flush=True)
            flush();del scene;base.torch.cuda.empty_cache()
    after={p:base.sha(Path(p)) for p in protected};assert after==protected
    base.save(OUT/'INPUT_INTEGRITY.json',dict(before=protected,after=after,passed=True))
    base.save(OUT/'RASTERIZER_PARITY.json',parity)
    assert len(all_rows)==2*792*len(CASES)
    base.save(ROOT/'evaluation/STATUS.json',dict(state='complete',groups=2,frames=792,cases_per_group=len(CASES),
        rows=len(all_rows),images_saved=0,parity_checks=len(parity),seconds=time.monotonic()-started))


if __name__=='__main__':
    try:main()
    except Exception as error:
        base.save(ROOT/'evaluation/STATUS.json',dict(state='failed',error=repr(error)));raise
