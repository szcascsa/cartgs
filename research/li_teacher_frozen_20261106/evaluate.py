"""固定18D+LI模型，三类集合×补偿开关×两ratio，完整轨迹评价。"""
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
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene
import view_residual
from renderer import prepare_render_data,render_prepared
from selector import slot_eligibility,compensated_attributes
from anchor_frustum import anchor_frustum_mask,compact_gaussians
from utils import MiniCam,focal2fov
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

assert Path(view_residual.__file__).resolve()==ROOT/'source/python/view_residual.py'
OUT=ROOT/'evaluation/reports'
RATIOS=[.01,.05]
SMOKE=[0,68,88,169,178,396,791]
FIELDS=['frame','timestamp','image','bank','ratio','selection','compensation','N','candidate_N','eligible_N',
    'requested_K','K','projected_K','contributing_K','selection_sha256','teacher_active','teacher_K',
    'training_float_ratio_teacher_K','teacher_overlap_fraction','full_li_selected_fraction','full_li_global_fraction',
    'full_li_pool_global_fraction','render_alpha_mass','xyz_p99_reference','xyz_max_reference','scale_ratio_p99',
    'photo_train','psnr','ssim','lpips','photo_l1','previous_psnr_abs_error']


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def load_binding():
    path=ROOT/'build/li_teacher_frozen.so'
    spec=importlib.util.spec_from_file_location('li_teacher_frozen',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def native_render(binding,camera,data,background,statistic=2,training=False):
    v=data['inputs'];empty=torch.empty(0)
    get=lambda name:v[name] if v[name] is not None else empty
    args=[background,get('means3D'),get('colors_precomp'),get('opacities'),get('scales'),get('rotations'),
        get('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,get('dc'),get('shs'),camera.camera_center]
    return binding.render(args,camera.image_height,camera.image_width,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
        data['sh_degree'],statistic,0 if training else 1)


def python_teacher(q,gi,rows,eligible,ratio):
    mask=torch.zeros_like(eligible)
    order=torch.nonzero(eligible).flatten()
    order=order[rows[order].argsort(stable=True)]
    order=order[gi[order].argsort(descending=True,stable=True)]
    order=order[q[order].argsort(descending=True,stable=True)]
    mask[order[:min(math.floor(ratio*len(q)),len(order))]]=True
    return mask


def unit(binding):
    o=dict(device='cuda',dtype=torch.float32)
    tests=0
    for q,gi,uids,elig,r in [([1,1,0,0],[2,3,9,9],[9,8,7,6],[1,1,1,1],.5),
        ([1,1,1,1],[2,2,2,2],[9,8,7,6],[1,1,1,1],.5),
        ([2,3,4,5],[4,3,2,1],[0,1,2,3],[0,1,0,1],1.),
        ([1,2,3,4],[0,0,0,0],[0,1,2,3],[1,1,1,1],0.),
        ([0,0],[2,3],[0,1],[1,1],.5),([],[],[],[],.5),
        ([1,2],[2,3],[0,1],[0,0],.5)]:
        q=torch.tensor(q,**o);gi=torch.tensor(gi,**o)
        rows=torch.tensor(uids,device='cuda',dtype=torch.long);e=torch.tensor(elig,device='cuda',dtype=torch.bool)
        result=binding.teacher(q,gi,rows,e,r)
        active=len(q)>0 and bool(((q>0)&e).any())
        if active:assert torch.equal(result.bool(),python_teacher(q,gi,rows,e,r))
        else:assert result is None
        tests+=1
    return tests


def selected_data(scene,compact,rows,mask,ratio,compensation):
    attributes=compensated_attributes(scene.model,compact,mask,ratio,source_rows=rows) if compensation else None
    if attributes is not None:
        attributes['scaling']=attributes['scaling'].log().exp()
        attributes['rotation']=scene.gaussians.rotation_activation(attributes['rotation'])
    return prepare_render_data(compact,mask,attributes),rows[mask]


@torch.no_grad()
def main(smoke=False):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    config=json.loads((ROOT/'configs/experiment.json').read_text());started=time.monotonic()
    assert json.loads((ROOT/'reports/BUILD.json').read_text())['passed']
    if not smoke:assert json.loads((ROOT/'reports/SMOKE.json').read_text())['passed']
    binding=load_binding();unit_count=unit(binding)
    before=json.loads((ROOT/'reports/INPUTS_BEFORE.json').read_text())
    assert all(sha(Path(n))==h for n,h in before.items()),'输入已变化'
    scene=BudgetScene(config['checkpoint'],config['dataset'],5000,0,None,'pytorch')
    for p in scene.model.parameters():p.requires_grad_(False)
    scene.model.eval()
    assert scene.model.selector_view_residual.schema==2 and len(scene.views)==792 and scene.renderer.total==13298
    pc=scene.gaussians;anchors=scene.model.anchors
    global_gi=torch.jit.load(str(Path(config['checkpoint'])/'flexgs_importance.pt'),map_location='cuda').state_dict()['0'].detach()
    assert global_gi.shape==(13298,) and torch.isfinite(global_gi).all()
    bank={str((Path(config['dataset'])/v['img_name']).resolve()) for v in json.loads((Path(config['checkpoint'])/'cameras.json').read_text())}
    assert len(bank)==72
    with Path(config['old_metrics']).open() as f:
        prior={(int(r['frame']),float(r['ratio']),r['case'].startswith('view_off')):r for r in csv.DictReader(f)
            if r['group']=='raw_li01' and r['case'] in ['view_0.01','view_0.05','view_off_0.01','view_off_0.05']}
    assert len(prior)==3168
    lpips=LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    frames=SMOKE if smoke else range(792)
    parity=[];stability=[];total_rows=0;max_prior_error=0.;native_sort_checks=0
    path=OUT/('SMOKE_PER_VIEW.csv' if smoke else 'PER_VIEW.csv');assert not path.exists(),'禁止覆盖评价'
    predictions=[];truths=[];photos=[];pending=[]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()

        def flush():
            nonlocal total_rows,max_prior_error
            if not pending:return
            image,target=torch.stack(predictions),torch.stack(truths)
            diff=image-target
            metrics=torch.stack([-10*torch.log10(diff.square().mean((1,2,3))),
                structural_similarity_index_measure(image,target,data_range=1.,reduction='none'),
                lpips(image,target).reshape(-1),diff.abs().mean((1,2,3)),torch.stack(photos)],1)
            lpips.reset();assert torch.isfinite(metrics).all()
            for row,values in zip(pending,metrics.cpu().tolist()):
                row.update(zip(['psnr','ssim','lpips','photo_l1','photo_train'],values))
                if row['compensation'] and row['selection'] in ['base','view']:
                    old=prior[(row['frame'],row['ratio'],row['selection']=='base')]
                    assert row['selection_sha256']==old['selection_sha256'] and row['K']==int(old['K'])
                    error=abs(row['psnr']-float(old['psnr']));assert error<=2e-5,('既有评价不匹配',row,error)
                    max_prior_error=max(max_prior_error,error);row['previous_psnr_abs_error']=error
                writer.writerow(row);total_rows+=1
            predictions.clear();truths.clear();photos.clear();pending.clear()

        for frame in frames:
            v=scene.views[frame];c=scene.calibration
            camera=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose'])))
            truth=scene.ground_truth(frame);isbank=str(Path(v['image']).resolve()) in bank
            full=native_render(binding,camera,scene.full_data,scene.background,training=True)
            q=full[2].detach()/float(camera.image_height*camera.image_width)
            assert q.shape==(13298,) and torch.isfinite(q).all() and (q>=0).all()
            if frame in SMOKE:
                normal=render_prepared(camera,scene.full_data,scene.background,return_radii=True)
                assert torch.equal(normal['render'].view(torch.int32),full[0].view(torch.int32)) and torch.equal(normal['radii'],full[1])
                repeated=native_render(binding,camera,scene.full_data,scene.background,training=True)
                q_again=repeated[2]/float(camera.image_height*camera.image_width)
                stability.append(dict(frame=frame,raw_max_abs_error=float((q-q_again).abs().max()),bitwise=torch.equal(q.view(torch.int32),q_again.view(torch.int32)),ratios=[]))
                parity.append(dict(frame=frame,selection='full',bitwise=True))
                del normal,repeated
            visible=anchor_frustum_mask(anchors.centers,anchors.offset_scales,camera.full_proj_transform,scene.renderer.occupied)
            rows=torch.nonzero(visible[anchors.rows]).flatten();compact=compact_gaussians(pc,rows)
            eligible=slot_eligibility(scene.model,compact.get_opacity,rows);local=q[rows];n=len(rows);available=int(eligible.sum())
            denominator=max(float(local.sum()),1e-20);global_mass=max(float(q.sum()),1e-20)
            for ratio in RATIOS:
                target=math.floor(ratio*n);k=min(target,available)
                labels=binding.teacher(local,global_gi[rows],rows,eligible,ratio)
                active=labels is not None
                teacher=labels.bool() if active else python_teacher(local,global_gi[rows],rows,eligible,ratio)
                assert torch.equal(teacher,python_teacher(local,global_gi[rows],rows,eligible,ratio));native_sort_checks+=1
                assert int(teacher.sum())==k
                if frame in SMOKE:
                    again=binding.teacher(q_again[rows],global_gi[rows],rows,eligible,ratio)
                    same=torch.equal(teacher,again.bool()) if again is not None else not active
                    stability[-1]['ratios'].append(dict(ratio=ratio,teacher_ids_equal=same,changed_points=0 if same else int((teacher!=again.bool()).sum())))
                chosen={}
                for selection in ['base','view','teacher']:
                    for compensation in [False,True]:
                        if selection=='teacher':data,ids=selected_data(scene,compact,rows,teacher,ratio,compensation)
                        else:
                            data,ids,stats=scene.renderer.prepare(camera,ratio_budget=ratio,compensation=compensation,view_conditioned=selection=='view')
                            assert stats['candidate_gaussians']==n and stats['eligible_candidates']==available and stats['requested_budget']==target and stats['ratio']==ratio
                        assert len(ids)==k
                        if selection in chosen:assert ids.equal(chosen[selection]),'补偿开关改变选点'
                        else:chosen[selection]=ids.clone()
                        result=native_render(binding,camera,data,scene.background)
                        assert torch.isfinite(result[0]).all() and torch.isfinite(result[2]).all()
                        projected=int((result[1]>0).sum());contributed=int((result[2]>0).sum());assert 0<=contributed<=projected<=k
                        if frame in SMOKE:
                            normal=render_prepared(camera,data,scene.background,return_radii=True)
                            assert torch.equal(normal['render'].view(torch.int32),result[0].view(torch.int32)) and torch.equal(normal['radii'],result[1])
                            parity.append(dict(frame=frame,ratio=ratio,selection=selection,compensation=compensation,bitwise=True));del normal
                        if k:
                            reference=anchors.offset_scales[anchors.rows[ids]].flatten()
                            displacement=(data['inputs']['means3D']-pc.get_xyz[ids]).norm(dim=1)/reference
                            scale=(data['inputs']['scales']/pc.get_scaling[ids]).max(1).values
                            geometry=torch.stack([displacement.quantile(.99),displacement.max(),scale.quantile(.99)]).cpu().tolist()
                        else:geometry=[0.,0.,1.]
                        mass=float(q[ids].sum());overlap=int(torch.isin(ids,rows[teacher]).sum())/max(1,k)
                        row=dict(frame=frame,timestamp=v['timestamp'],image=v['image'],bank=isbank,ratio=ratio,selection=selection,compensation=compensation,
                            N=13298,candidate_N=n,eligible_N=available,requested_K=target,K=k,projected_K=projected,contributing_K=contributed,
                            selection_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),teacher_active=active,teacher_K=k,
                            training_float_ratio_teacher_K=min(math.floor(float(np.float32(ratio))*n),available),
                            teacher_overlap_fraction=overlap,full_li_selected_fraction=mass/denominator,full_li_global_fraction=mass/global_mass,
                            full_li_pool_global_fraction=denominator/global_mass,render_alpha_mass=float(result[2].sum())/float(camera.image_height*camera.image_width),
                            xyz_p99_reference=geometry[0],xyz_max_reference=geometry[1],scale_ratio_p99=geometry[2],previous_psnr_abs_error='')
                        predictions.append(result[0].clamp(0,1));truths.append(truth);photos.append(binding.photo(result[0],truth));pending.append(row)
                        if len(pending)==8:flush()
                        del data,result
            if frame%20==0 or frame==791 or smoke:
                flush();stream.flush()
                state=dict(state='running',smoke=smoke,frame=frame+1,frames=792,rows=total_rows,seconds=time.monotonic()-started)
                save(ROOT/'evaluation'/('SMOKE_STATUS.json' if smoke else 'STATUS.json'),state);print(state,flush=True)
            del full
        flush()
    scene.renderer.check_state()
    assert all(sha(Path(n))==h for n,h in before.items()),'输入被修改'
    assert all(not p.requires_grad and p.grad is None for p in scene.model.parameters())
    assert total_rows==(84 if smoke else 9504)
    save(OUT/('SMOKE_PARITY.json' if smoke else 'RASTERIZER_PARITY.json'),parity)
    save(OUT/('SMOKE_TEACHER_STABILITY.json' if smoke else 'TEACHER_STABILITY.json'),stability)
    maps=Path('/proc/self/maps').read_text();assert str(ROOT/'build/li_teacher_frozen.so') in maps
    (ROOT/'reports'/('smoke_maps.txt' if smoke else 'evaluation_maps.txt')).write_text(maps)
    value=dict(state='complete',passed=True,smoke=smoke,rows=total_rows,frames=len(frames),unit_tests=unit_count,
        rasterizer_bitwise_checks=len(parity),native_teacher_sort_checks=native_sort_checks,prior_psnr_max_abs_error=max_prior_error,
        optimizer_updates=0,images_saved=0,seconds=time.monotonic()-started)
    save(ROOT/'reports/SMOKE.json' if smoke else ROOT/'evaluation/STATUS.json',value)
    print(value,flush=True)


if __name__=='__main__':
    smoke='--smoke' in sys.argv
    try:main(smoke)
    except Exception as error:
        save(ROOT/'evaluation'/('SMOKE_STATUS.json' if smoke else 'STATUS.json'),dict(state='failed',error=repr(error)));raise
