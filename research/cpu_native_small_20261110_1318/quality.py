"""与本轮开始的Deep CPU输出配对渲染；同权重/固定键/原Rasterizer，不保存图。"""
from common import *
from PIL import Image
import gc

def ground_truth(source,record):
    dataset=Path(source['spec']['dataset']);name=Path(record.get('img_name',record.get('image','')))
    candidates=[dataset/name.name,dataset/'results'/name.name,dataset/'rgb'/name.name,name]
    path=next(p for p in candidates if p.is_file());array=np.asarray(Image.open(path).convert('RGB')).copy()
    return torch.from_numpy(array).cuda().permute(2,0,1).float()/255

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    from renderer import render_prepared
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[];background=torch.zeros(3,device='cuda')
    conditions=['local0.01','local0.05','local0.1','local0.3','local0.5','local0.7','local0.03','local0.08','local0.2','local0.4','local0.6','fixed0','fixed1000','fixed5000','global0.01','global0.1','full']
    for spec in SCENES:
        source=load_scene(spec['slug']);noise=torch.load(BASE/'noise_snapshots'/(spec['slug']+'_seed0.pt'),map_location='cpu',weights_only=True)
        if isinstance(noise,dict):noise=noise['noise']
        source['reference'].noise.copy_(noise);old=native(source,'baseline');new=native(source,'combined');records=source['records']
        for frame in map(int,np.linspace(.11,.89,3)*(len(records)-1)):
            cam=camera(records[frame]);gpu_cam=cuda_camera(cam);gt=ground_truth(source,records[frame])
            for condition in conditions:
                a,ids,stats=request(old,cam,condition);b,selected,info=request(new,cam,condition);assert torch.equal(ids,selected)
                original=render_prepared(gpu_cam,cuda_packet(a),background,return_radii=True);result=render_prepared(gpu_cam,cuda_packet(b),background,return_radii=True)
                rgb=result['render'];reference=original['render'];assert torch.equal(rgb,reference) and torch.equal(result['radii'],original['radii'])
                psnr=lambda x:float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30)))
                psnr_old=psnr(reference);psnr_new=psnr(rgb)
                rows.append(dict(scene=spec['id'],slug=spec['slug'],frame=frame,condition=condition,N=new.total,M=info['candidate_gaussians'],K=len(ids),same_ids=True,
                    same_rgb=True,same_radii=True,max_rgb_error=float((rgb-reference).abs().max()),baseline_psnr=psnr_old,new_psnr=psnr_new,psnr_loss=psnr_old-psnr_new,
                    baseline_ssim=float(ssim(reference[None].clamp(0,1),gt[None],data_range=1.)),new_ssim=float(ssim(rgb[None].clamp(0,1),gt[None],data_range=1.)),
                    baseline_lpips=float(perceptual(reference[None].clamp(0,1)*2-1,gt[None]*2-1)),new_lpips=float(perceptual(rgb[None].clamp(0,1)*2-1,gt[None]*2-1))))
        write_csv(out/'quality.csv',rows);print('SMALL_QUALITY',spec['id'],len(rows),flush=True)
        del source,old,new,a,b,original,result,rgb,reference,gt;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,rows=len(rows),same_rgb=True,same_radii=True,max_rgb_error=0.,mean_psnr_loss=statistics.mean(r['psnr_loss'] for r in rows),
        worst_psnr_loss=max(r['psnr_loss'] for r in rows),no_render_images=True,binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
