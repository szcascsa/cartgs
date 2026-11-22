"""同权重Teacher/Student分别评价四组和pending；网络误差与渲染误差分列。"""
from joint_study import *
from joint_benchmark import CONDITIONS
from pending_runtime import PendingRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--group',required=True,type=int);parser.add_argument('--pending',action='store_true')
    parser.add_argument('--native',action='store_true')
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[];checks=[];identities=[]
    def metrics(image,gt):
        assert bool(torch.isfinite(image).all()),'联合推理输出非有限'
        x=image.clamp(0,1)
        return dict(psnr=float(-10*torch.log10((x-gt).square().mean().clamp_min(1e-30))),
            ssim=float(ssim(x[None],gt[None],data_range=1.)),
            lpips=float(perceptual(x[None]*2-1,gt[None]*2-1)))
    for spec in SCENES[args.group*2:args.group*2+2]:
        _,source=load_source(spec['slug']);student,path,payload=student_for_spec(source,spec)
        for kind,weights in [('teacher',None)]+([('student',student)] if student is not None else []):
            networks=network_variants(source,weights);factory=networks['tensor_best'][1]
            bg=source.scene.background;views=motions(source);cam=factory.from_source(source,views[0][1])
            renders={'F2':BaselineRenderer(),'new':Renderer(mode)}
            for r in renders.values():r.precompile(source.runtime.total,cam,bg)
            variants=benchmark_variants(networks,renders)
            pending=PendingRenderer(source,weights,mode) if args.pending else None
            if pending:pending.precompile(cam,bg)
            native=PendingRenderer(source,weights,mode,native_dispatch=True) if args.native else None
            if native:native.precompile(cam,bg)
            dispatch={name:engine for name,engine in [('E_pending',pending),('F_native',native)] if engine is not None}
            identities.append(dict(scene=spec['id'],model=kind,fingerprint=source.fingerprint,
                student=str(path) if weights is not None else None,
                student_sha256=sha(path) if weights is not None else None,
                tensor_selector=True,tensor_compensator=weights is not None))
            for motion,frame in views:
                cam=factory.from_source(source,frame);gt=source.scene.ground_truth(frame).clamp(0,1)
                for condition in CONDITIONS:
                    packets={};images={};ids_by_name={};radii_by_name={};results={}
                    for name,((engine,_),renderer) in variants.items():
                        if id(engine) not in packets:packets[id(engine)]=p.request(engine,cam,condition)
                        packet,ids,info=packets[id(engine)];image,radii=renderer.render(cam,packet,bg)
                        images[name]=image;ids_by_name[name]=ids;radii_by_name[name]=radii
                        results[name]=metrics(image,gt)
                    base='A_fp32_F2';reference=results[base];base_ids=ids_by_name[base]
                    for name in variants:
                        ids=ids_by_name[name];assert len(ids)==len(base_ids),(spec['id'],kind,condition,'strict K',name)
                        same=torch.equal(ids,base_ids)
                        overlap=1. if not len(ids) else float(torch.isin(ids,base_ids).float().mean())
                        pair='A_fp32_F2' if name=='B_fp32_new' else 'C_tensor_best_F2' if name=='D_tensor_best_new' else base
                        if name in ['B_fp32_new','D_tensor_best_new']:
                            assert torch.equal(ids,ids_by_name[pair]),'纯渲染对照改变了Gaussian身份'
                            assert bits_equal(radii_by_name[name],radii_by_name[pair]),'radii变化'
                        row=dict(scene=spec['id'],split='tune' if spec['slug'] in TUNE else 'heldout',model=kind,
                            condition=condition,frame=frame,motion=motion,variant=name,N=source.runtime.total,K=len(ids),
                            baseline_psnr=reference['psnr'],baseline_ssim=reference['ssim'],baseline_lpips=reference['lpips'],
                            psnr_loss=reference['psnr']-results[name]['psnr'],same_ids=same,id_overlap=overlap,
                            renderer_pair=pair,renderer_psnr_loss=results[pair]['psnr']-results[name]['psnr'],
                            max_rgb_error=float((images[name]-images[base]).abs().max()),**results[name])
                        rows.append(row)
                    for name,engine in dispatch.items():
                        image,radii,ids,info=engine.render(cam,bg,condition);ref='D_tensor_best_new'
                        check=dict(scene=spec['id'],model=kind,condition=condition,frame=frame,variant=name,K=len(ids),
                            same_ids=torch.equal(ids,ids_by_name[ref]),rgb_bitwise=bits_equal(image,images[ref]),
                            radii_bitwise=bits_equal(radii,radii_by_name[ref]))
                        checks.append(check);assert all(check[n] for n in ['same_ids','rgb_bitwise','radii_bitwise']),check
                        row=next(r for r in reversed(rows) if r['variant']==ref).copy()
                        row.update(variant=name,renderer_pair=ref,renderer_psnr_loss=0.)
                        rows.append(row)
                print('JOINT_QUALITY_FRAME',spec['id'],kind,frame,flush=True)
            append(out/'quality.csv',rows);rows=[]
            del networks,renders,variants,pending,native,dispatch,packets,images,radii_by_name,ids_by_name
            gc.collect();torch.cuda.empty_cache()
        del source,student;gc.collect();torch.cuda.empty_cache()
    append(out/'checks.csv',checks);save(out/'IDENTITIES.json',identities)
    save(out/'DONE.json',dict(passed=True,checks=len(checks),no_images=True,quality_gate_pending_aggregation=True))


if __name__=='__main__':main()
