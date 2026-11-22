"""16组保留验证；分离Selector近似和Compensator近似的图像影响。"""
from study import *
import gc
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--tune-only',action='store_true');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());sm,cm,w=[frozen[k] for k in ['selector','compensator','warps']];cw=frozen.get('comp_warps',w)
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[];attributes=[];identities=[]
    for spec in [s for s in SCENES if s['dataset_kind']=='Replica' and (not args.tune_only or s['slug'] in TUNE)]:
        slug=spec['slug'];_,source=load_scene(slug);student,path,payload=student_for(source,slug)
        engines=dict(original=engine(source,student,original=True),selector=engine(source,student,sm,0,w,comp_warps=cw),compensator=engine(source,student,0,cm,w,comp_warps=cw),both=engine(source,student,sm,cm,w,comp_warps=cw))
        identities.append(dict(scene=spec['id'],checkpoint=spec['checkpoint'],student=str(path),student_sha256=sha(path),fingerprint=source.fingerprint,N=engines['original'].total,step=payload['step']))
        for frame in map(int,np.array([.17,.57,.93])*(len(source.scene.views)-1)):
            cam=source.camera(frame);gt=source.scene.ground_truth(frame).clamp(0,1)
            for condition in CONDITIONS:
                base,base_ids,stats=request(engines['original'],cam,condition)
                ref=render_prepared(cam,base,source.scene.background,return_radii=True);refim=ref['render'];refx=refim.clamp(0,1)
                refpsnr=float(-10*torch.log10((refx-gt).square().mean().clamp_min(1e-30)))
                refssim=float(ssim(refx[None],gt[None],data_range=1.));reflpips=float(perceptual(refx[None]*2-1,gt[None]*2-1))
                for name in ['selector','compensator','both']:
                    packet,ids,info=request(engines[name],cam,condition);assert len(ids)==len(base_ids)
                    if name=='compensator':assert torch.equal(ids,base_ids)
                    result=render_prepared(cam,packet,source.scene.background,return_radii=True);im=result['render'];assert torch.isfinite(im).all();x=im.clamp(0,1)
                    psnr=float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30)));overlap=int(torch.isin(ids,base_ids).sum());union=len(ids)+len(base_ids)-overlap
                    rows.append(dict(scene=spec['id'],variant=name,condition=condition,frame=frame,N=engines[name].total,M=info.get('candidate_gaussians',len(ids)),K=len(ids),same_ids=torch.equal(ids,base_ids),overlap=overlap,jaccard=overlap/union if union else 1.,
                        baseline_psnr=refpsnr,psnr=psnr,psnr_loss=refpsnr-psnr,baseline_ssim=refssim,ssim=float(ssim(x[None],gt[None],data_range=1.)),baseline_lpips=reflpips,lpips=float(perceptual(x[None]*2-1,gt[None]*2-1)),
                        image_psnr=float(-10*torch.log10((im-refim).square().mean().clamp_min(1e-30))),max_rgb_error=float((im-refim).abs().max()),same_radii=torch.equal(result['radii'],ref['radii'])))
                    if torch.equal(ids,base_ids):
                        for key,value in difference(base,packet).items():attributes.append(dict(scene=spec['id'],variant=name,condition=condition,frame=frame,node=key,max_abs=value))
        write_csv(out/'quality.csv',rows);write_csv(out/'attributes.csv',attributes);save(out/'MODEL_IDENTITIES.json',identities)
        print('TC_QUALITY',spec['id'],len(rows),flush=True)
        del engines,source,student,base,base_ids,ref,refim,refx,gt,packet,result,im,x;gc.collect();torch.cuda.empty_cache()
    groups={}
    for r in rows:groups.setdefault((r['variant'],r['scene'],r['condition']),[]).append(r['psnr_loss'])
    summary={}
    for v in ['selector','compensator','both']:
        g=[r for r in rows if r['variant']==v];mean=float(np.mean([r['psnr_loss'] for r in g]));worst_condition=max(float(np.mean(vals)) for key,vals in groups.items() if key[0]==v)
        summary[v]=dict(mean_psnr_loss=mean,worst_frame_loss=max(r['psnr_loss'] for r in g),worst_condition_mean_loss=worst_condition,minimum_jaccard=min(r['jaccard'] for r in g),passed=mean<=.05 and worst_condition<=.2)
    passed=all(x['passed'] for x in summary.values());save(out/'SUMMARY.json',dict(passed=passed,rows=len(rows),summary=summary,images_saved=0));assert passed,summary
if __name__=='__main__':main()
