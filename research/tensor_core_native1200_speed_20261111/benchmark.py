"""1200×680原生分辨率A/C/D配对：网络/准备不计入纯渲染，完整路径独立计量。"""
from helpers import *


def psnr(image,gt):return float(-10*torch.log10((image.clamp(0,1)-gt).square().mean().clamp_min(1e-30)))


@torch.no_grad()
def main():
    out=HERE/'reports';out.mkdir(exist_ok=False)
    c,d,cm,dm=configure();torch.set_num_threads(2)
    base=json.loads((TC/'reports/PROTECTED_BEFORE.json').read_text())
    base.update(json.loads((Path('/home/prmi/zwj/experiments/tensor_core_square2000_speed_20261111/reports/PROTECTED_AFTER.json')).read_text()))
    for root,config in [(TC,c),(PREP,d)]:
        base[config['binary']]=config['binary_sha256']
        for name,value in config['sources'].items():base[str(root/name)]=value
    before={name:sha(name) for name in base};save('PROTECTED_BEFORE.json',before)
    save('PROTOCOL.json',dict(size=SIZE,slugs=SLUGS,ratios=RATIOS,rounds=2,repeats=6,views=18,
        variants=dict(A='FP32 CUDA Student + R camera',C='FP16 Tensor network only + R camera',D='C + CUDA preparation/projection/device counts + native C++ camera'),
        precision='C/D network FP16 multiply, FP32 accumulate; physical decode FP32, D static projection FP32',
        scopes=['camera_wall','engine_prepare_wall','prepare_wall','raster_event','pipeline_wall'],
        rendering='only G raster CUDA Event after camera and attributes are ready; excludes networks, upload and CPU dispatch',
        full='private Master packet prepared once per variant, no per-frame map copy',
        quality='native 1200x680 GT on 3 matched trajectory views per condition; gate mean loss<=.05dB, scene/condition mean<=.2dB; native image error and original GT quality',
        camera='original horizontal/vertical FoV; actual native 1200/680 width/height passed into view features',
        student='same frozen 3534-parameter student, Teacher binding and map across A/C/D',
        no_timing_sum=True,training=False,production_changed=False,saved_images=False))
    rows=[];checks=[];quality=[];identities=[];memory=[]
    for slug in SLUGS:
        spec,source=p.load_scene(slug);student,path,payload=p.student_for(source,slug)
        runs={name:engine(source,student,name) for name in ['A','C','D']}
        cameras={name:Camera(source,name,dm) for name in runs}
        native={name:Camera(source,name,dm,(1200,680)) for name in runs}
        views=frames_for(source)
        full={name:run.full() for name,run in runs.items()}
        def request(name,cam,condition):
            if condition=='full':return *full[name],dict(candidate_gaussians=runs[name].total)
            return runs[name].prepare(cam,ratio_budget=float(condition[5:]))
        render=p.create_renderer(cameras['A'].create(views[0][1]),source.scene.background,runs['A'].total,use_graph=True)
        for name,run in runs.items():
            memory.append(dict(scene=spec['id'],variant=name,N=run.total,
                projected_bytes=run._pipeline._preparation_bytes() if name=='D' else 0,
                packed_bytes=run._pipeline._tensor_bytes() if name!='A' else 0))
        identities.append(dict(scene=spec['id'],checkpoint=spec['checkpoint'],student=str(path),student_sha256=sha(path),
            fingerprint=source.fingerprint,N=runs['A'].total,student_step=payload['step'],parameters=3534))
        # A/G parity is checked separately for each version's own predictions.
        for _,frame in views:
            for condition in CONDITIONS:
                reference=None;reference_ids=None;tc_ids=None;tc_m=None
                for name in ['A','C','D']:
                    cam=cameras[name].create(frame);packet,ids,stats=request(name,cam,condition)
                    image,radii=render.render(cam,packet,source.scene.background)
                    original=p.render_prepared(cam,packet,source.scene.background,return_radii=True)
                    assert p.bits_equal(image,original['render']) and p.bits_equal(radii,original['radii'])
                    assert torch.isfinite(image).all()
                    if reference is None:reference=image.clone();reference_ids=ids.clone();reference_m=stats.get('candidate_gaussians',len(ids))
                    assert len(ids)==len(reference_ids) and stats.get('candidate_gaussians',len(ids))==reference_m
                    if name=='C':tc_ids=ids.clone();tc_m=stats.get('candidate_gaussians',len(ids))
                    if name=='D':assert torch.equal(ids,tc_ids) and stats.get('candidate_gaussians',len(ids))==tc_m
                    checks.append(dict(scene=spec['id'],frame=frame,condition=condition,variant=name,K=len(ids),
                        M=stats.get('candidate_gaussians',len(ids)),same_ids_A=torch.equal(ids,reference_ids),
                        ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),rgb_radii_bitwise=True,
                        image_psnr_vs_A=float(-10*torch.log10((image-reference).square().mean().clamp_min(1e-30))),
                        max_rgb_vs_A=float((image-reference).abs().max())))
        # Existing approximation gate is evaluated on original native GT; native GT with no resizing.
        quality_frames=[int(f*(len(source.scene.views)-1)) for f in [.17,.57,.93]]
        for frame in quality_frames:
            gt=source.scene.ground_truth(frame).clamp(0,1)
            for condition in CONDITIONS:
                ref_psnr=None
                for name in ['A','C','D']:
                    cam=native[name].create(frame);packet,ids,stats=request(name,cam,condition)
                    image=p.render_prepared(cam,packet,source.scene.background)['render'];score=psnr(image,gt)
                    if ref_psnr is None:ref_psnr=score
                    quality.append(dict(scene=spec['id'],frame=frame,condition=condition,variant=name,K=len(ids),
                        psnr=score,psnr_loss_vs_A=ref_psnr-score))
        for _ in range(2):
            for _,frame in views:
                for condition in CONDITIONS:
                    for name in runs:
                        cam=cameras[name].create(frame);packet,ids,stats=request(name,cam,condition)
                        render.render(cam,packet,source.scene.background)
        torch.cuda.synchronize()
        for round_index in range(2):
            ordered=views if round_index==0 else list(reversed(views))
            for repeat in range(6):
                for vi,(_,frame) in enumerate(ordered):
                    for ci,condition in enumerate(p.rotate(CONDITIONS,repeat+vi+round_index)):
                        for name in p.rotate(list(runs),ci+repeat+vi+round_index):
                            cam=cameras[name].create(frame)
                            packet,ids,stats=request(name,cam,condition)
                            for scope in ['camera_wall','engine_prepare_wall','prepare_wall','pipeline_wall']:
                                torch.cuda.synchronize();tick=time.perf_counter_ns()
                                current=cam if scope=='engine_prepare_wall' else cameras[name].create(frame)
                                if scope!='camera_wall':packet,ids,stats=request(name,current,condition)
                                if scope=='pipeline_wall':render.render(current,packet,source.scene.background)
                                torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                                rows.append(dict(scene=spec['id'],variant=name,condition=condition,scope=scope,
                                    round=round_index,repeat=repeat,frame=frame,N=runs[name].total,
                                    M=stats.get('candidate_gaussians',len(ids)),K=len(ids),ms=ms))
                            if repeat<2:
                                ms,_=event(lambda:render.render(current,packet,source.scene.background))
                                rows.append(dict(scene=spec['id'],variant=name,condition=condition,scope='raster_event',
                                    round=round_index,repeat=repeat,frame=frame,N=runs[name].total,
                                    M=stats.get('candidate_gaussians',len(ids)),K=len(ids),ms=ms))
                print('PROGRESS',spec['id'],round_index,repeat,len(rows),flush=True)
        assert render.native.stats()[1:3]==[0,0]
        for name,data in [('samples.csv',rows),('checks.csv',checks),('quality_native.csv',quality),('memory.csv',memory)]:write(name,data)
        save('MODEL_IDENTITIES.json',identities)
        del source,student,runs,cameras,native,full,render,packet,ids,image,radii,original,reference,ref_psnr,gt,run
        gc.collect();torch.cuda.empty_cache()
    after={name:sha(name) for name in before};changed=[name for name in before if before[name]!=after[name]]
    save('PROTECTED_AFTER.json',after)
    q={}
    for name in ['C','D']:
        group=[r for r in quality if r['variant']==name];losses={}
        for r in group:losses.setdefault((r['scene'],r['condition']),[]).append(r['psnr_loss_vs_A'])
        mean=float(np.mean([r['psnr_loss_vs_A'] for r in group]));worst=max(map(np.mean,losses.values()))
        q[name]=dict(mean_loss=mean,worst_condition_mean=float(worst),worst_frame=max(r['psnr_loss_vs_A'] for r in group),passed=bool(mean<=.05 and worst<=.2))
    result=dict(passed=not changed and all(r['passed'] for r in q.values()),changed=changed,
        protected_files=len(before),samples=len(rows),checks=len(checks),quality_rows=len(quality),quality=q,
        renderer_unchanged=True,training=False,production_changed=False,saved_images=0)
    save('VALIDATION.json',result);assert result['passed'],result;print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
