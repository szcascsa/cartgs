"""CPU准备真实渲染验证；仅GPU执行原Rasterizer及独立质量评价。"""
from cpu_study import *
from PIL import Image

def ground_truth(source,record):
    dataset=Path(source['spec']['dataset']);name=Path(record.get('img_name',record.get('image','')))
    candidates=[dataset/name.name,dataset/'results'/name.name,dataset/'rgb'/name.name,name]
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:raise FileNotFoundError(candidates)
    array=np.asarray(Image.open(path).convert('RGB')).copy()
    return torch.from_numpy(array).cuda().permute(2,0,1).float()/255


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--slugs',default='Replica__office2__rgbd,TUM_RGBD__freiburg1_xyz__rgbd,Replica__room0__rgbd')
    p.add_argument('--all-scenes',action='store_true');p.add_argument('--student',action='store_true');p.add_argument('--frames',type=int,default=3)
    p.add_argument('--gpu-oracle',action='store_true')
    p.add_argument('--perceptual',action='store_true');args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    from renderer import render_prepared
    if args.perceptual:
        import lpips
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim
        perceptual=lpips.LPIPS(net='alex').cuda().eval()
    rows=[];slugs=[s['slug'] for s in SCENES] if args.all_scenes else args.slugs.split(',')
    conditions=['local0.01','local0.05','local0.1','local0.3','local0.7','local0.03','local0.08','local0.2','local0.4','local0.6','fixed0','fixed1000','fixed5000','global0.1','full']
    for slug in slugs:
        source=load_scene(slug);ref=source['reference'];model=source['model'];pc=source['pc']
        student=load_student(source['student_path'],'cpu')[0] if args.student else None
        oracle=ref
        if args.gpu_oracle:
            gpu_model=copy.deepcopy(model).cuda().eval().requires_grad_(False);gpu_pc=copy.copy(pc)
            for name in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']:setattr(gpu_pc,name,getattr(pc,name).cuda())
            oracle=FixedBudgetRenderer(gpu_model,gpu_pc,read_config(source['checkpoint']/'selector_config.yaml'),0,'pytorch')
            ref.noise.copy_(oracle.noise.cpu())
            (HERE/'noise_snapshots').mkdir(exist_ok=True)
            torch.save(dict(noise=ref.noise.detach().clone(),seed=0,checkpoint=str(source['checkpoint'])),HERE/'noise_snapshots'/(slug+'_seed0.pt'))
        engine=CpuRenderer(model,pc,ref.noise,6,student,True,True,True)
        indices=np.linspace(.11,.89,args.frames)*(len(source['records'])-1)
        for frame in map(int,indices):
            record=source['records'][frame];cam=camera(record);gpu_cam=cuda_camera(cam);gt=ground_truth(source,record)
            assert gt.shape==(3,cam.image_height,cam.image_width)
            for condition in conditions:
                truth,ids,stats=call(oracle,gpu_cam if args.gpu_oracle else cam,condition);ids=ids.cpu();packet,selected,info=call(engine,cam,condition)
                if not condition.startswith('global'):assert len(selected)==len(ids),(slug,condition,'K')
                if condition!='full':assert info['candidate_gaussians']==stats['candidate_gaussians'],(slug,condition,'M')
                base=render_prepared(gpu_cam,cuda_packet(truth),torch.zeros(3,device='cuda'),return_radii=True)
                result=render_prepared(gpu_cam,cuda_packet(packet),torch.zeros(3,device='cuda'),return_radii=True)
                image,original=result['render'],base['render'];diff=(image-original).abs()
                psnr=lambda x,y:float(-10*torch.log10((x-y).square().mean().clamp_min(1e-30)))
                ref_psnr=psnr(original,gt);new_psnr=psnr(image,gt);equivalence=psnr(image,original)
                overlap=len(set(selected.tolist())&set(ids.tolist()))/max(1,len(ids));same=torch.equal(selected,ids)
                entry=dict(scene=source['spec']['id'],slug=slug,frame=frame,condition=condition,N=len(pc.get_xyz),M=stats['candidate_gaussians'],
                    K_reference=len(ids),K_cpu=len(selected),ids_equal=same,id_overlap=overlap,
                    reference_psnr=ref_psnr,cpu_psnr=new_psnr,psnr_loss=ref_psnr-new_psnr,
                    cpu_vs_reference_psnr=equivalence,max_rgb_error=float(diff.max()),mean_rgb_error=float(diff.mean()),
                    reference_ssim=None,cpu_ssim=None,reference_lpips=None,cpu_lpips=None,
                    max_xyz_error=None,max_scale_error=None,max_rotation_error=None,max_dc_error=None,student=args.student)
                if same and len(ids):
                    for k,label in [('means3D','xyz'),('scales','scale'),('rotations','rotation'),('dc','dc')]:entry['max_'+label+'_error']=float((packet['inputs'][k]-truth['inputs'][k].cpu()).abs().max())
                if args.perceptual:
                    entry.update(reference_ssim=float(ssim(original[None].clamp(0,1),gt[None],data_range=1.)),
                        cpu_ssim=float(ssim(image[None].clamp(0,1),gt[None],data_range=1.)),
                        reference_lpips=float(perceptual(original[None].clamp(0,1)*2-1,gt[None]*2-1)),
                        cpu_lpips=float(perceptual(image[None].clamp(0,1)*2-1,gt[None]*2-1)))
                rows.append(entry);assert torch.isfinite(image).all()
                print('CPU_QUALITY',slug,frame,condition,'loss',entry['psnr_loss'],'equiv',equivalence,'overlap',overlap,flush=True)
            write_csv(out/'quality.csv',rows)
        del source,ref,model,pc,engine,packet,truth,image,original,base,result,gt,oracle
        if args.gpu_oracle:del gpu_model,gpu_pc
        torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,rows=len(rows),scenes=len(slugs),student=args.student,
        binary_sha256=sha(extension().__file__),gpu_oracle=args.gpu_oracle,mean_psnr_loss=statistics.mean(r['psnr_loss'] for r in rows),
        worst_psnr_loss=max(r['psnr_loss'] for r in rows),minimum_equivalence_psnr=min(r['cpu_vs_reference_psnr'] for r in rows),
        differing_ID_conditions=sum(not r['ids_equal'] for r in rows),no_render_images=True,
        complete_preparation_device='CPU',GPU='原FP32 Rasterizer＋评价；网络未在GPU运行'))
    print('CPU_QUALITY_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
