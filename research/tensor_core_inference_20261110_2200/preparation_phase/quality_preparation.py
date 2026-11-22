"""D/B相对A以及D相对已验收C的质量；不保存图像。"""
from prep_study import *
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true')
    p.add_argument('--flags',default='1,1,1');p.add_argument('--initial',action='store_true');p.add_argument('--native',action='store_true');args=p.parse_args()
    flags=tuple(map(int,args.flags.split(',')));assert len(flags)==3
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[];identities=[]
    specs=[s for s in SCENES if s['dataset_kind']=='Replica' and (not args.quick or s['slug'] in TUNE)]
    conditions=['local0.1','local0.3','local0.5'] if args.quick else CONDITIONS
    for spec in specs:
        _,source=load_scene(spec['slug']);student,path,payload=student_for(source,spec['slug'])
        engines={x:make_engine(source,student,x) for x in ['A','C']}
        engines['D']=make_engine(source,student,frustum=flags[0],projection=flags[1],device=flags[2])
        if args.native:
            from native_camera import NativeCameraPreparer
            native=NativeCameraPreparer(MiniCam)
        if args.initial:engines['B']=make_engine(source,student,'B')
        identities.append(dict(scene=spec['id'],checkpoint=spec['checkpoint'],student=str(path),student_sha256=sha(path),fingerprint=source.fingerprint,N=engines['A'].total))
        for frame in map(int,np.array([.17,.57,.93])*(len(source.scene.views)-1)):
            cam=source.camera(frame);gt=source.scene.ground_truth(frame).clamp(0,1)
            dcam=native.from_source(source,frame) if args.native else cam
            for condition in conditions:
                packet,base_ids,stats=request(engines['A'],cam,condition)
                a=render_prepared(cam,packet,source.scene.background,return_radii=True)
                ref=a['render'];rx=ref.clamp(0,1)
                baseline_psnr=float(-10*torch.log10((rx-gt).square().mean().clamp_min(1e-30)))
                cpack,cids,cstats=request(engines['C'],cam,condition)
                cimage=render_prepared(cam,cpack,source.scene.background,return_radii=True)['render']
                cpsnr=float(-10*torch.log10((cimage.clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
                for name in ['D']+(['B'] if args.initial else []):
                    current=dcam if name=='D' else cam
                    packet,ids,info=request(engines[name],current,condition)
                    assert len(ids)==len(base_ids)
                    if name=='D':
                        assert info.get('candidate_gaussians',len(ids))==cstats.get('candidate_gaussians',len(cids))
                        if not args.native:assert torch.equal(ids,cids)
                    rendered=render_prepared(current,packet,source.scene.background,return_radii=True);image=rendered['render'];assert torch.isfinite(image).all();x=image.clamp(0,1)
                    psnr=float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30)));overlap=int(torch.isin(ids,base_ids).sum());union=len(ids)+len(base_ids)-overlap
                    rows.append(dict(scene=spec['id'],variant=name,condition=condition,frame=frame,N=engines[name].total,M=info.get('candidate_gaussians',len(ids)),K=len(ids),
                        baseline_psnr=baseline_psnr,C_psnr=cpsnr,psnr=psnr,psnr_loss=baseline_psnr-psnr,extra_C_psnr_loss=cpsnr-psnr,
                        ssim=float(ssim(x[None],gt[None],data_range=1.)),lpips=float(perceptual(x[None]*2-1,gt[None]*2-1)),
                        same_ids_A=torch.equal(ids,base_ids),same_ids_C=torch.equal(ids,cids),jaccard=overlap/union if union else 1.,
                        image_psnr=float(-10*torch.log10((image-ref).square().mean().clamp_min(1e-30))),max_rgb_error=float((image-ref).abs().max()),same_radii_A=torch.equal(a['radii'],rendered['radii'])))
        write_csv(out/'quality.csv',rows);save(out/'MODEL_IDENTITIES.json',identities)
        print('PREPARATION_QUALITY',spec['id'],len(rows),flush=True)
        del engines,source,student,packet,image,cimage,ref,rx,gt,x;gc.collect();torch.cuda.empty_cache()
    summary={}
    for variant in {r['variant'] for r in rows}:
        data=[r for r in rows if r['variant']==variant];groups={}
        for r in data:groups.setdefault((r['scene'],r['condition']),[]).append(r['psnr_loss'])
        mean=float(np.mean([r['psnr_loss'] for r in data]));worst=max(float(np.mean(x)) for x in groups.values())
        summary[variant]=dict(mean_psnr_loss=mean,worst_condition_mean_loss=worst,worst_frame_loss=max(r['psnr_loss'] for r in data),
            extra_C_mean_loss=float(np.mean([r['extra_C_psnr_loss'] for r in data])),passed=mean<=.05 and worst<=.2)
    save(out/'SUMMARY.json',dict(passed=all(x['passed'] for x in summary.values()),rows=len(rows),flags=flags,native_camera=args.native,summary=summary,images_saved=0))
    assert summary['D']['passed'],summary


if __name__=='__main__':main()
