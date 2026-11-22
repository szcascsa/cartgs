"""原始GT、相同输入/K的渲染画质；近似候选独立门限，不保存RGB。"""
from nine_study import *
from final_benchmark import native_modules


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--modes',required=True)
    parser.add_argument('--tune',action='store_true');parser.add_argument('--group',type=int);parser.add_argument('--nested',action='store_true')
    parser.add_argument('--conditions')
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    native_modules(json.loads((previous.PRIOR/'reports/FROZEN_CANDIDATE.json').read_text()))
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    scenes=[s for s in SCENES if not args.tune or s['slug'] in TUNE]
    if args.group is not None:scenes=scenes[2*args.group:2*args.group+2]
    rows=[];identities=[]
    for s in scenes:
        _,source=load_source(s['slug']);bg=source.scene.background
        engines={MODES[m]:Renderer(m) for m in map(int,args.modes.split(','))};base=BaselineRenderer()
        views=motions(source,quick=args.tune)
        cam=source.camera(views[0][1]);base.precompile(source.runtime.total,cam,bg)
        for e in engines.values():e.precompile(source.runtime.total,cam,bg)
        conditions=['local0.01','local0.05','local0.1','local0.3','local0.5','full'] if args.tune else [f'local{x:g}' for x in [.01,.03,.05,.1,.2,.3,.4,.5,.7]]+['fixed1000','fixed5000','full']
        if args.conditions:conditions=args.conditions.split(',')
        for motion,frame in views:
            cam=source.camera(frame);gt=source.scene.ground_truth(frame).clamp(0,1)
            full,_,_=request(source,cam,'full')
            for condition in conditions:
                if args.nested:
                    if condition.startswith('fixed'):ratio=(min(len(full['inputs']['means3D']),int(condition[5:]))+.01)/max(1,len(full['inputs']['means3D']))
                    else:ratio=1. if condition=='full' else float(condition[5:])
                    packet,ids=nested_packet(full,ratio)
                else:packet,ids,_=request(source,cam,condition)
                ref,radii=base.render(cam,packet,bg);refx=ref.clamp(0,1)
                baseline_psnr=float(-10*torch.log10((refx-gt).square().mean().clamp_min(1e-30)))
                baseline_ssim=float(ssim(refx[None],gt[None],data_range=1.))
                baseline_lpips=float(perceptual(refx[None]*2-1,gt[None]*2-1))
                for name,engine in engines.items():
                    image,r=engine.render(cam,packet,bg);finite=bool(torch.isfinite(image).all());assert finite,(s['id'],condition,name,frame)
                    assert bits_equal(r,radii),(s['id'],condition,name,'radii')
                    x=image.clamp(0,1);psnr=float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30)))
                    row=dict(scene=s['id'],split='tune' if s['slug'] in TUNE else 'heldout',workload='nested' if args.nested else 'lod',
                        condition=condition,frame=frame,motion=motion,variant=name,N=source.runtime.total,K=len(ids),
                        baseline_psnr=baseline_psnr,psnr=psnr,psnr_loss=baseline_psnr-psnr,
                        baseline_ssim=baseline_ssim,ssim=float(ssim(x[None],gt[None],data_range=1.)),
                        baseline_lpips=baseline_lpips,lpips=float(perceptual(x[None]*2-1,gt[None]*2-1)),
                        image_psnr=float(-10*torch.log10((image-ref).square().mean().clamp_min(1e-30))),
                        max_rgb_error=float((image-ref).abs().max()),rgb_bitwise=bits_equal(image,ref),radii_bitwise=True,finite=finite)
                    if engine.mode==0:assert row['rgb_bitwise'],row
                    rows.append(row)
            del gt,full,ref,refx,image,x,packet
        append(out/'quality.csv',rows);rows=[]
        identities.append(dict(scene=s['id'],checkpoint=s['checkpoint'],fingerprint=source.fingerprint,N=source.runtime.total))
        print('QUALITY_SCENE_COMPLETE',s['id'],args.modes,flush=True)
        del engines,source,base;gc.collect();torch.cuda.empty_cache()
    import pandas as pd
    data=pd.read_csv(out/'quality.csv');group=data.groupby(['variant','scene','workload','condition']).psnr_loss.mean()
    results={}
    for name in data.variant.unique():
        values=group.loc[name];sub=data[data.variant==name]
        results[name]=dict(equal_condition_mean_psnr_loss=float(values.mean()),worst_condition_psnr_loss=float(values.max()),
            worst_frame_psnr_loss=float(sub.psnr_loss.max()),worst_frame=sub.loc[sub.psnr_loss.idxmax()].to_dict(),
            passed=bool(values.mean()<=.2 and values.max()<=.5))
    save(out/'SUMMARY.json',dict(rows=len(data),results=results,no_images=True))
    save(out/'INPUT_IDENTITIES.json',identities)
    print('QUALITY_COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':main()
