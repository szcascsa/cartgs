"""同一选点与几何下，逐帧对照Other Head保留／归零，不训练或保存渲染图片。"""
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from common import ROOT,PREVIOUS,OLD,CHECKPOINT,DATASET,TRAIN,RATIOS,CASES,sha,save

sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene
import spatial_compensation,selector,anchor_budget,renderer
from renderer import render_prepared
from utils import MiniCam,focal2fov
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

OUT=ROOT/'evaluation/reports'
METRICS=['psnr','ssim','lpips','photo_l1']
FIELDS=['variant','case','scope','ratio','ratio_split','frame','timestamp','image','bank','N','candidate_N','eligible_N',
    'requested_K','K','projected_K','contributing_K','budget_shortfall','network_ratio','selection_sha256',
    'opacity_delta_abs_mean','opacity_delta_abs_max','dc_delta_norm_mean','dc_delta_norm_p99']+METRICS


def counted(binding,camera,data,background):
    inputs=data['inputs'];empty=torch.Tensor([])
    get=lambda name:inputs[name] if inputs[name] is not None else empty
    return binding.render_with_counts(background,get('means3D'),get('colors_precomp'),get('opacities'),get('scales'),get('rotations'),
        1.,get('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
        camera.image_height,camera.image_width,get('dc'),get('shs'),data['sh_degree'],camera.camera_center,False,False,True,binding.Mode.Inference)


def equal_bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def remove_other(packet,ids,pc):
    inputs=dict(packet['inputs'])
    inputs['opacities']=pc.get_opacity[ids].contiguous()
    inputs['dc']=pc.get_features_dc[ids].contiguous()
    result=dict(packet,inputs=inputs)
    for key in ['means3D','means2D','scales','rotations','shs']:
        assert inputs[key] is packet['inputs'][key] and equal_bits(inputs[key],packet['inputs'][key])
    assert all(torch.isfinite(v).all() for v in inputs.values() if isinstance(v,torch.Tensor))
    return result


@torch.no_grad()
def main():
    assert not (OUT/'PER_VIEW.csv').exists()
    assert json.loads((ROOT/'reports/PREFLIGHT.json').read_text())['passed']
    modules={m.__name__:str(Path(m.__file__).resolve()) for m in [spatial_compensation,selector,anchor_budget,renderer]}
    assert all(Path(p).parent==ROOT/'source/python' for p in modules.values())
    save(ROOT/'reports/SOURCE_MODULES.json',modules)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    binding_path=Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    spec=importlib.util.spec_from_file_location('fr1xyz_production_counts',binding_path)
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    lpips=LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    with (PREVIOUS/'evaluation/reports/PER_VIEW.csv').open() as f:
        historical={(v['case'],int(v['frame'])):v for v in csv.DictReader(f) if v['group']=='both_li'}
    assert len(historical)==34*792
    scene=BudgetScene(CHECKPOINT,DATASET,5000,0,None,'pytorch');pc=scene.renderer.gaussians;model=scene.model
    assert len(scene.views)==792 and scene.renderer.total==13298 and int(model.joint_view_conditioning_schema)==2
    assert 'other' in model.spatial_compensation.heads and len(model.spatial_compensation.heads)==4
    assert not hasattr(model,'gaussian_compensation')
    state_before={n:t.detach().clone() for n,t in model.state_dict().items()}
    global_packets={r:scene.renderer.prepare(None,global_ratio=r) for r in RATIOS}
    bank={str((DATASET/v['img_name']).resolve()) for v in json.loads((CHECKPOINT/'cameras.json').read_text())}
    parity=[];packet_checks=[];prior_difference={k:0. for k in METRICS};rows_written=0;started=time.monotonic()
    process=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(ROOT/'reports/EVALUATOR_PROCESS.json',process)
    images=[];truths=[];pending=[]

    def prepare(scope,value,camera):
        if scope=='global':return scene.renderer.prepare(None,global_ratio=value)
        if scope=='view':return scene.renderer.prepare(camera,ratio_budget=value,view_conditioned=True)
        return scene.renderer.prepare(camera,budget=value,view_conditioned=True)

    def verify_zero_output(scope,value,camera,expected,ids):
        def zero(module,args,out):
            assert out.ndim==2 and out.shape[1]==14
            value=out.clone();value[:,10:14]=0
            assert equal_bits(value[:,:10],out[:,:10]);return value
        scene.renderer.global_cache.clear()
        hook=model.spatial_compensation.register_forward_hook(zero)
        try:actual,other_ids,_=prepare(scope,value,camera)
        finally:hook.remove();scene.renderer.global_cache.clear()
        assert equal_bits(other_ids,ids)
        assert actual['sh_degree']==expected['sh_degree']
        for key,a in actual['inputs'].items():
            b=expected['inputs'][key]
            assert (a is None and b is None) or (isinstance(a,torch.Tensor) and equal_bits(a,b)),key
        return True

    with (OUT/'PER_VIEW.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()

        def flush():
            nonlocal rows_written
            if not pending:return
            prediction,target=torch.stack(images),torch.stack(truths);difference=prediction-target
            values=torch.stack([-10*torch.log10(difference.square().mean((1,2,3))),
                structural_similarity_index_measure(prediction,target,data_range=1.,reduction='none'),
                lpips(prediction,target).reshape(-1),difference.abs().mean((1,2,3))],1);lpips.reset()
            assert torch.isfinite(values).all()
            for row,metric in zip(pending,values.cpu().tolist()):
                row.update(zip(METRICS,metric))
                if row['variant']=='other_on':
                    old=historical[row['case'],row['frame']]
                    for key in METRICS:prior_difference[key]=max(prior_difference[key],abs(row[key]-float(old[key])))
                writer.writerow(row);rows_written+=1
            images.clear();truths.clear();pending.clear()

        for frame,v in enumerate(scene.views):
            old_full=historical['full',frame]
            assert str(v['timestamp'])==str(old_full['timestamp']) and v['image']==old_full['image']
            c=scene.calibration;camera=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose'])))
            truth=scene.ground_truth(frame);assert truth is not None
            for name,scope,value in CASES:
                packet,ids,stats=global_packets[value] if scope=='global' else prepare(scope,value,camera)
                digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest();old=historical[name,frame]
                assert digest==old['selection_sha256'] and len(ids)==int(old['K'])
                assert stats['candidate_gaussians']==int(old['candidate_N']) and stats['eligible_candidates']==int(old['eligible_N'])
                off=remove_other(packet,ids,pc)
                if frame in [0,396,791]:
                    packet_checks.append(dict(case=name,frame=frame,K=len(ids),geometry_bitwise=True,
                        zero_network_output_equals_packet=verify_zero_output(scope,value,camera,off,ids),selection_sha256=digest))
                opacity_delta=(packet['inputs']['opacities']-off['inputs']['opacities']).abs().reshape(-1)
                dc_delta=(packet['inputs']['dc']-off['inputs']['dc']).reshape(len(ids),-1).norm(dim=1)
                appearance=dict(opacity_delta_abs_mean=float(opacity_delta.mean()),opacity_delta_abs_max=float(opacity_delta.max()),
                    dc_delta_norm_mean=float(dc_delta.mean()),dc_delta_norm_p99=float(torch.quantile(dc_delta,.99)))
                for variant,data in [('other_on',packet),('other_off',off)]:
                    output=counted(binding,camera,data,scene.background)
                    assert torch.isfinite(output[2]).all()
                    projected,contributing=int((output[3]>0).sum()),int((output[5]>0).sum())
                    assert 0<=contributing<=projected<=len(ids)
                    if frame in [0,396,791]:
                        normal=render_prepared(camera,data,scene.background,return_radii=True)
                        assert equal_bits(normal['render'],output[2]) and equal_bits(normal['radii'],output[3])
                        parity.append(dict(case=name,frame=frame,variant=variant,bitwise=True))
                        if variant=='other_on':assert projected==int(old['projected_K']) and contributing==int(old['contributing_K'])
                    row=dict(variant=variant,case=name,scope=scope,ratio=value if scope!='points' else None,
                        ratio_split='fixed_points' if scope=='points' else ('train' if value in TRAIN else 'unseen'),frame=frame,
                        timestamp=v['timestamp'],image=v['image'],bank=str(Path(v['image']).resolve()) in bank,N=scene.renderer.total,
                        candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],requested_K=stats['requested_budget'],
                        K=len(ids),projected_K=projected,contributing_K=contributing,budget_shortfall=stats['budget_shortfall'],
                        network_ratio=stats['ratio'],selection_sha256=digest,**appearance)
                    pending.append(row);images.append(output[2].clamp(0,1));truths.append(truth)
                    if len(pending)==8:flush()
                del off,packet,output
            if frame%20==0 or frame==791:
                flush();stream.flush()
                status=dict(state='running',frame=frame+1,frames=792,rows=rows_written,seconds=time.monotonic()-started)
                save(ROOT/'evaluation/STATUS.json',status);print(json.dumps(status),flush=True)
        flush()
    assert rows_written==44352 and len(packet_checks)==84 and len(parity)==168
    assert state_before.keys()==model.state_dict().keys() and all(equal_bits(v,model.state_dict()[n]) for n,v in state_before.items())
    assert prior_difference['psnr']<1e-4 and prior_difference['ssim']<1e-5 and prior_difference['lpips']<1e-5
    save(OUT/'CORRECTNESS.json',dict(passed=True,packet_checks=packet_checks,rasterizer_checks=parity,
        all_frame_selection_checks=22176,model_state_bitwise_unchanged=True,geometry_preserved_every_case=True,
        historical_on_max_absolute_difference=prior_difference,metric_note='原历史评价batch顺序不同，指标记录最大误差；原始渲染颜色与radii单独逐bit核验'))
    save(ROOT/'evaluation/STATUS.json',dict(state='complete',frames=792,cases=28,variants=2,rows=rows_written,
        images_saved=0,seconds=time.monotonic()-started))


if __name__=='__main__':
    try:main()
    except Exception as error:
        save(ROOT/'evaluation/STATUS.json',dict(state='failed',error=repr(error)));raise
