"""投影/Head并行/持久block/当前位姿四项分解；仅使用预先开发集。"""
from joint_study import *
from pending_runtime import PendingRenderer
from pose_runtime import PoseRenderer
from teacher_plus_runtime import TeacherPlusRenderer
from balanced_order import order

CONDITIONS=['local0.01','local0.05','local0.1','local0.2','local0.3','local0.4','local0.5','local0.7','fixed0','fixed1','full']
CONFIGS={'seq_raw':(3,False),'heads_raw':(5,False),'seq_projected':(3,True),
    'heads_projected':(5,True),'persistent_projected':(7,True),'persistent_raw':(7,False),'tf32_projected':(8,True)}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--group',required=True,type=int);parser.add_argument('--round',required=True,type=int)
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();quality=[];checks=[];samples=[];memory=[]
    def metrics(image,gt):
        x=image.clamp(0,1);assert bool(torch.isfinite(x).all())
        return dict(psnr=float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30))),
            ssim=float(ssim(x[None],gt[None],data_range=1.)),lpips=float(perceptual(x[None]*2-1,gt[None]*2-1)))
    for slug in TUNE[args.group*2:args.group*2+2]:
        spec,source=load_source(slug);bg=source.scene.background;networks=network_variants(source,None)
        factory=networks['tensor_best'][1];views=motions(source,quick=True)
        if args.round==2:views.reverse()
        cam=factory.from_source(source,views[0][1]);base=BaselineRenderer();base.precompile(source.runtime.total,cam,bg)
        engines={'F_native':PendingRenderer(source,None,mode,native_dispatch=True),'G_current_pose':PoseRenderer(source,None,mode)}
        for name,(math_mode,projected) in CONFIGS.items():
            tick=time.perf_counter_ns();engine=TeacherPlusRenderer(source,mode,math_mode,projected,1)
            engines[name]=engine
            memory.append(dict(scene=spec['id'],variant=name,math_mode=math_mode,projected=projected,
                N=source.runtime.total,projection_bytes=engine.network._pipeline._preparation_bytes(),
                tensor_bytes=engine.network._pipeline._tensor_bytes(),startup_ms=(time.perf_counter_ns()-tick)*1e-6))
        engines['persistent_pose']=engines['persistent_projected']
        for engine in engines.values():engine.precompile(cam,bg)
        def render(name,frame,condition):
            if name in ['G_current_pose','persistent_pose']:
                return engines[name].render_from_source(source,frame,bg,condition,factory)
            return engines[name].render(factory.from_source(source,frame),bg,condition)
        for ci,condition in enumerate(CONDITIONS if args.round==1 else CONDITIONS[::-1]):
            expected={}
            for motion,frame in views:
                cam=factory.from_source(source,frame);gt=source.scene.ground_truth(frame).clamp(0,1)
                packet,ids,info=p.request(networks['fp32'][0],cam,condition);aimage,aradii=base.render(cam,packet,bg)
                baseline=metrics(aimage,gt);results={name:render(name,frame,condition) for name in engines}
                for name,(image,radii,ids,stats) in results.items():
                    assert torch.equal(ids,results['F_native'][2]),(slug,condition,name,'Selector IDs changed')
                    expected[name,frame]=len(ids)
                    result=metrics(image,gt)
                    quality.append(dict(scene=spec['id'],condition=condition,frame=frame,variant=name,round=args.round,K=len(ids),
                        baseline_psnr=baseline['psnr'],psnr_loss=baseline['psnr']-result['psnr'],
                        baseline_ssim=baseline['ssim'],baseline_lpips=baseline['lpips'],**result))
                for a,b in [('heads_raw','seq_raw'),('heads_projected','seq_projected'),
                    ('persistent_projected','heads_projected'),('persistent_raw','heads_raw'),
                    ('persistent_pose','persistent_projected'),('G_current_pose','F_native')]:
                    x,y=results[a],results[b]
                    row=dict(scene=spec['id'],condition=condition,frame=frame,variant=a,reference=b,K=len(x[2]),round=args.round,
                        same_ids=torch.equal(x[2],y[2]),rgb_bitwise=bits_equal(x[0],y[0]),radii_bitwise=bits_equal(x[1],y[1]))
                    checks.append(row);assert all(row[k] for k in ['same_ids','rgb_bitwise','radii_bitwise']),row
                del results,gt
            torch.cuda.synchronize()
            for repeat in range(8):
                for j,(motion,frame) in enumerate(views):
                    for name in order(list(engines),repeat+j+ci,reverse=args.round==2):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        image,radii,ids,stats=render(name,frame,condition);torch.cuda.synchronize()
                        elapsed=(time.perf_counter_ns()-tick)*1e-6;assert len(ids)==expected[name,frame]
                        samples.append(dict(scene=spec['id'],condition=condition,frame=frame,motion=motion,variant=name,
                            round=args.round,repeat=repeat,N=source.runtime.total,K=len(ids),ms=elapsed))
            print('TEACHER_PLUS_DEVELOPMENT',spec['id'],condition,args.round,flush=True)
        del engines,networks,source,base;gc.collect();torch.cuda.empty_cache()
    for name,data in [('quality',quality),('checks',checks),('samples',samples),('memory',memory)]:append(out/(name+'.csv'),data)
    save(out/'DONE.json',dict(passed=True,checks=len(checks),independent_toggles=True,no_images=True))


if __name__=='__main__':main()
