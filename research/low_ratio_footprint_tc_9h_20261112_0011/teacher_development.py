"""Teacher同权重TC开发：先画质/身份与warp等价，再独立轮转整帧测速。"""
from joint_study import *
from pending_runtime import PendingRenderer
from teacher_runtime import TeacherTensorRenderer
from balanced_order import order

CONDITIONS=['local0.01','local0.05','local0.1','local0.2','local0.3','local0.4','local0.5','local0.7','fixed0','fixed1','full']


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--group',required=True,type=int);parser.add_argument('--round',required=True,type=int)
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();quality=[];checks=[];samples=[];numerics=[]
    def metrics(image,gt):
        x=image.clamp(0,1);assert bool(torch.isfinite(x).all())
        return dict(psnr=float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30))),
            ssim=float(ssim(x[None],gt[None],data_range=1.)),lpips=float(perceptual(x[None]*2-1,gt[None]*2-1)))
    for slug in TUNE[args.group*2:args.group*2+2]:
        spec,source=load_source(slug);bg=source.scene.background;networks=network_variants(source,None)
        factory=networks['tensor_best'][1];views=motions(source,quick=True)
        if args.round==2:views.reverse()
        ref=PendingRenderer(source,None,mode,native_dispatch=True)
        engines={f'{kind}_w{w}':TeacherTensorRenderer(source,mode,m,w) for kind,m in [('fp16',3),('tf32',4)] for w in [1,2,4]}
        engines={'reference_FP32_comp':ref,**engines}
        cam=factory.from_source(source,views[0][1]);base=BaselineRenderer();base.precompile(source.runtime.total,cam,bg)
        for engine in engines.values():engine.precompile(cam,bg)
        for ci,condition in enumerate(CONDITIONS if args.round==1 else CONDITIONS[::-1]):
            expected={}
            for motion,frame in views:
                cam=factory.from_source(source,frame);gt=source.scene.ground_truth(frame).clamp(0,1)
                packet,ids,info=p.request(networks['fp32'][0],cam,condition);aimage,aradii=base.render(cam,packet,bg)
                baseline=metrics(aimage,gt);results={name:engine.render(cam,bg,condition) for name,engine in engines.items()}
                for name,(image,radii,chosen,stats) in results.items():
                    reference=results['reference_FP32_comp'];same=torch.equal(chosen,reference[2])
                    assert same,(slug,condition,name,'Teacher TC改变了Selector身份')
                    expected[name,frame]=len(chosen)
                    if name!='reference_FP32_comp':
                        pair=results[name.split('_')[0]+'_w2']
                        row=dict(scene=spec['id'],condition=condition,frame=frame,variant=name,round=args.round,K=len(chosen),
                            same_ids=same,warp_rgb_bitwise=bits_equal(image,pair[0]),warp_radii_bitwise=bits_equal(radii,pair[1]))
                        checks.append(row);assert row['warp_rgb_bitwise'] and row['warp_radii_bitwise'],row
                    if name=='reference_FP32_comp' or name.endswith('w2'):
                        result=metrics(image,gt)
                        quality.append(dict(scene=spec['id'],condition=condition,frame=frame,variant=name,round=args.round,
                            K=len(chosen),baseline_psnr=baseline['psnr'],psnr_loss=baseline['psnr']-result['psnr'],
                            baseline_ssim=baseline['ssim'],baseline_lpips=baseline['lpips'],**result))
                    if condition!='full' and name.endswith('w2'):
                        current=engines[name].network.prepare(cam,**({'ratio_budget':float(condition[5:])} if condition.startswith('local') else {'budget':int(condition[5:])}))[0]
                        original=ref.network.prepare(cam,**({'ratio_budget':float(condition[5:])} if condition.startswith('local') else {'budget':int(condition[5:])}))[0]
                        for key in ['means3D','scales','rotations','opacities','dc','shs']:
                            x,y=current['inputs'][key],original['inputs'][key];error=(x-y).abs()
                            numerics.append(dict(scene=spec['id'],condition=condition,frame=frame,variant=name,field=key,
                                max_abs=float(error.max()) if error.numel() else 0.,mean_abs=float(error.mean()) if error.numel() else 0.))
                del results,gt
            torch.cuda.synchronize()
            for repeat in range(8):
                for j,(motion,frame) in enumerate(views):
                    for name in order(list(engines),repeat+j+ci,reverse=args.round==2):
                        torch.cuda.synchronize();tick=time.perf_counter_ns();cam=factory.from_source(source,frame)
                        image,radii,ids,stats=engines[name].render(cam,bg,condition);torch.cuda.synchronize()
                        elapsed=(time.perf_counter_ns()-tick)*1e-6;assert len(ids)==expected[name,frame]
                        samples.append(dict(scene=spec['id'],condition=condition,frame=frame,motion=motion,variant=name,
                            round=args.round,repeat=repeat,N=source.runtime.total,K=len(ids),ms=elapsed))
            print('TEACHER_TENSOR_DEVELOPMENT',spec['id'],condition,args.round,flush=True)
        del engines,ref,networks,source,base;gc.collect();torch.cuda.empty_cache()
    for name,data in [('quality',quality),('checks',checks),('samples',samples),('numerics',numerics)]:append(out/(name+'.csv'),data)
    save(out/'DONE.json',dict(passed=True,checks=len(checks),warp_exact=True,all_selector_ids_same=True,no_images=True))


if __name__=='__main__':main()
