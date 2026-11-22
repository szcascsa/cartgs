"""Teacher、普通学生和原生CPU学生三方图像对照，不保存图。"""
from common import *
from PIL import Image
import gc

def ground_truth(source,record):
    dataset=Path(source['spec']['dataset']);name=Path(record.get('img_name',record.get('image','')))
    path=next(p for p in [dataset/name.name,dataset/'results'/name.name,dataset/'rgb'/name.name,name] if p.is_file())
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).cuda().permute(2,0,1).float()/255

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    from renderer import render_prepared
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[];background=torch.zeros(3,device='cuda')
    for spec in SCENES:
        source=load_scene(spec['slug']);noise=torch.load(ORIGINAL/'noise_snapshots'/(spec['slug']+'_seed0.pt'),map_location='cpu',weights_only=True)
        if isinstance(noise,dict):noise=noise['noise']
        source['reference'].noise.copy_(noise);teacher=native(source,'teacher');engine=native(source,'student');student,_=load_student(student_path(source),'cpu',teacher_fingerprint(source['model'],source['pc']))
        for frame in map(int,np.linspace(.11,.89,3)*(len(source['records'])-1)):
            cam=camera(source['records'][frame]);gpu_cam=cuda_camera(cam);gt=ground_truth(source,source['records'][frame])
            for condition in CONDITIONS:
                a,ids,info=request(teacher,cam,condition);b,selected,stats=request(engine,cam,condition);assert torch.equal(ids,selected)
                full=condition=='full';ratio=1. if full else float(stats['ratio'])
                ordinary,_=ordinary_student(source,student,cam,ids,ratio,not condition.startswith('global')) if len(ids) and not full else (b,None)
                packets=dict(teacher=a,student=b,ordinary=ordinary);images={};metrics={};radii={}
                for name,packet in packets.items():
                    result=render_prepared(gpu_cam,cuda_packet(packet),background,return_radii=True);image=result['render'];assert torch.isfinite(image).all();images[name]=image;radii[name]=result['radii']
                    clamped=image.clamp(0,1);metrics[name]=dict(psnr=float(-10*torch.log10((clamped-gt).square().mean().clamp_min(1e-30))),ssim=float(ssim(clamped[None],gt[None],data_range=1.)),lpips=float(perceptual(clamped[None]*2-1,gt[None]*2-1)))
                diff=images['student']-images['ordinary'];backend_psnr=float(-10*torch.log10(diff.square().mean().clamp_min(1e-30)))
                rows.append(dict(scene=spec['id'],slug=spec['slug'],frame=frame,condition=condition,ratio=ratio,N=engine.total,M=stats['candidate_gaussians'],K=len(ids),same_ids=True,
                    teacher_psnr=metrics['teacher']['psnr'],student_psnr=metrics['student']['psnr'],ordinary_student_psnr=metrics['ordinary']['psnr'],
                    distillation_psnr_loss=metrics['teacher']['psnr']-metrics['ordinary']['psnr'],total_psnr_loss=metrics['teacher']['psnr']-metrics['student']['psnr'],backend_psnr_loss=metrics['ordinary']['psnr']-metrics['student']['psnr'],
                    teacher_ssim=metrics['teacher']['ssim'],student_ssim=metrics['student']['ssim'],ordinary_ssim=metrics['ordinary']['ssim'],teacher_lpips=metrics['teacher']['lpips'],student_lpips=metrics['student']['lpips'],ordinary_lpips=metrics['ordinary']['lpips'],
                    backend_image_equivalence_psnr=backend_psnr,backend_max_rgb_error=float(diff.abs().max()),backend_radii_equal=torch.equal(radii['student'],radii['ordinary'])))
        write_csv(out/'quality.csv',rows);print('STUDENT_QUALITY',spec['id'],len(rows),flush=True)
        del source,teacher,engine,student,images,metrics,packets,a,b,ordinary,gt,result,image,diff;gc.collect();torch.cuda.empty_cache()
    per_condition={}
    for row in rows:per_condition.setdefault((row['scene'],row['condition']),[]).append(row['backend_psnr_loss'])
    passed=statistics.mean(r['backend_psnr_loss'] for r in rows)<=.05 and max(statistics.mean(v) for v in per_condition.values())<=.2
    save(out/'SUMMARY.json',dict(passed=passed,rows=len(rows),scenes=len(SCENES),ids_identical=True,images_saved=False,
        mean_backend_psnr_loss=statistics.mean(r['backend_psnr_loss'] for r in rows),worst_backend_psnr_loss=max(r['backend_psnr_loss'] for r in rows),
        mean_teacher_student_psnr_loss=statistics.mean(r['total_psnr_loss'] for r in rows),worst_teacher_student_psnr_loss=max(r['total_psnr_loss'] for r in rows),
        minimum_backend_image_equivalence_psnr=min(r['backend_image_equivalence_psnr'] for r in rows),backend_radii_mismatch_conditions=sum(not r['backend_radii_equal'] for r in rows)))
    assert passed,'CPU backend quality failed; retain model and report independently'

if __name__=='__main__':main()
