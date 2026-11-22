"""仅3组开发场景：单模块及双模块Tensor路径，连续和跳转视角配对。"""
from study import *
import gc
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--warps',default='1,2,4');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];checks=[]
    variants=[('original',0,0,2,True)]+[(f'{kind}_w{w}',mode,mode,w,False) for mode,kind in [(1,'tf32'),(2,'fp16')] for w in map(int,args.warps.split(','))]
    variants.extend([('tf32_selector',1,0,2,False),('tf32_compensator',0,1,2,False)])
    variants.extend([('fp16_s4_c1',2,2,4,False),('fp16_s4_c2',2,2,4,False),('fp16_selector',2,0,4,False),('fp16_compensator',0,2,2,False)])
    overrides={'fp16_s4_c1':1,'fp16_s4_c2':2}
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug);n=len(source.scene.views)
        frames=[int(.35*n),int(.35*n)+1,int(.35*n)+2,int(.11*n),int(.63*n),int(.89*n)]
        factory=create_camera_preparer(MiniCam,variant='R');cams=[factory.from_source(source,f) for f in frames]
        engines={name:engine(source,student,sm,cm,w,orig,comp_warps=overrides.get(name)) for name,sm,cm,w,orig in variants}
        for e in engines.values():
            for cam in cams:
                for r in PRIMARY:request(e,cam,f'local{r:g}')
        torch.cuda.synchronize()
        for ci,r in enumerate(PRIMARY):
            condition=f'local{r:g}'
            for repeat in range(4):
                for vi,cam in enumerate(cams):
                    for name in rotate(list(engines),repeat+vi+ci):
                        torch.cuda.synchronize();tick=time.perf_counter_ns();packet,ids,stats=request(engines[name],cam,condition);torch.cuda.synchronize()
                        rows.append(dict(scene=slug,variant=name,ratio=r,repeat=repeat,frame=frames[vi],M=stats['candidate_gaussians'],K=len(ids),ms=(time.perf_counter_ns()-tick)/1e6))
            for vi in [0,3,5]:
                cam=cams[vi];gt=source.scene.ground_truth(frames[vi]).clamp(0,1);a,aid,_=request(engines['original'],cam,condition)
                ref=render_prepared(cam,a,source.scene.background,return_radii=True);ref_psnr=float(-10*torch.log10((ref['render'].clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
                for name,e in engines.items():
                    packet,ids,stats=request(e,cam,condition);assert len(ids)==len(aid)
                    image=render_prepared(cam,packet,source.scene.background,return_radii=True);assert torch.isfinite(image['render']).all()
                    psnr=float(-10*torch.log10((image['render'].clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
                    checks.append(dict(scene=slug,variant=name,ratio=r,frame=frames[vi],K=len(ids),same_ids=torch.equal(ids,aid),overlap=int(torch.isin(ids,aid).sum()),psnr_loss=ref_psnr-psnr,
                        image_psnr=float(-10*torch.log10((image['render']-ref['render']).square().mean().clamp_min(1e-30))),max_rgb_error=float((image['render']-ref['render']).abs().max())))
        write_csv(out/'samples.csv',rows);write_csv(out/'quality.csv',checks)
        print('TC_TUNE',slug,len(rows),len(checks),flush=True)
        del engines,source,student,packet,a,ref,image,gt;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),quality_checks=len(checks),scenes=len(TUNE),binary_sha256=sha(tc.deployment_extension().__file__)))
if __name__=='__main__':main()
