"""仅比较优化前后同一个学生；原Teacher蒸馏损失不重复归因。"""
from common import *
from PIL import Image
import gc

def ground_truth(source,record):
    dataset=Path(source['spec']['dataset']);name=Path(record.get('img_name',record.get('image','')));path=next(p for p in [dataset/name.name,dataset/'results'/name.name,dataset/'rgb'/name.name,name] if p.is_file())
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).cuda().permute(2,0,1).float()/255

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--variant',default='vector');p.add_argument('--tune-only',action='store_true');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    from renderer import render_prepared
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[];background=torch.zeros(3,device='cuda')
    specs=[s for s in SCENES if s['slug'] in ['Replica__office2__rgbd','Replica__room0__rgbd','Replica__office1__mono']] if args.tune_only else SCENES
    for spec in specs:
        source=load_scene(spec['slug']);noise=torch.load(ORIGINAL/'noise_snapshots'/(spec['slug']+'_seed0.pt'),map_location='cpu',weights_only=True);noise=noise['noise'] if isinstance(noise,dict) else noise;source['reference'].noise.copy_(noise)
        old=native(source,'baseline');new=native(source,args.variant)
        for frame in map(int,np.linspace(.11,.89,3)*(len(source['records'])-1)):
            cam=camera(source['records'][frame]);gpu_cam=cuda_camera(cam);gt=ground_truth(source,source['records'][frame])
            for condition in CONDITIONS:
                a,ids,stats=request(old,cam,condition);b,selected,info=request(new,cam,condition);assert torch.equal(ids,selected)
                packets=dict(baseline=a,optimized=b);images={};metrics={};radii={}
                for name,packet in packets.items():
                    result=render_prepared(gpu_cam,cuda_packet(packet),background,return_radii=True);image=result['render'];assert torch.isfinite(image).all();images[name]=image;radii[name]=result['radii'];x=image.clamp(0,1)
                    metrics[name]=dict(psnr=float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30))),ssim=float(ssim(x[None],gt[None],data_range=1.)),lpips=float(perceptual(x[None]*2-1,gt[None]*2-1)))
                diff=images['optimized']-images['baseline'];eq=float(-10*torch.log10(diff.square().mean().clamp_min(1e-30)))
                rows.append(dict(scene=spec['id'],slug=spec['slug'],frame=frame,condition=condition,N=new.total,M=info['candidate_gaussians'],K=len(ids),same_ids=True,same_radii=torch.equal(radii['baseline'],radii['optimized']),
                    baseline_psnr=metrics['baseline']['psnr'],new_psnr=metrics['optimized']['psnr'],psnr_loss=metrics['baseline']['psnr']-metrics['optimized']['psnr'],
                    baseline_ssim=metrics['baseline']['ssim'],new_ssim=metrics['optimized']['ssim'],baseline_lpips=metrics['baseline']['lpips'],new_lpips=metrics['optimized']['lpips'],
                    same_rgb=torch.equal(images['optimized'],images['baseline']),image_equivalence_psnr=eq,max_rgb_error=float(diff.abs().max())))
        write_csv(out/'quality.csv',rows);print('STUDENT_FAST_QUALITY',spec['id'],len(rows),flush=True)
        del source,old,new,a,b,packets,images,metrics,gt,result,image,diff;gc.collect();torch.cuda.empty_cache()
    grouped={}
    for row in rows:grouped.setdefault((row['scene'],row['condition']),[]).append(row['psnr_loss'])
    passed=statistics.mean(r['psnr_loss'] for r in rows)<=.05 and max(statistics.mean(v) for v in grouped.values())<=.2
    save(out/'SUMMARY.json',dict(passed=passed,variant=args.variant,scenes=len(specs),rows=len(rows),mean_psnr_loss=statistics.mean(r['psnr_loss'] for r in rows),worst_psnr_loss=max(r['psnr_loss'] for r in rows),
        worst_condition_mean_loss=max(statistics.mean(v) for v in grouped.values()),minimum_image_equivalence_psnr=min(r['image_equivalence_psnr'] for r in rows),radii_mismatches=sum(not r['same_radii'] for r in rows),max_rgb_error=max(r['max_rgb_error'] for r in rows),images_saved=False))
    assert passed,'Additional backend quality gate failed'

if __name__=='__main__':main()
