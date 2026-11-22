from prep_study import *
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);checks=[];frustum=[]
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug)
        old=make_engine(source,student,'C');b=old._base
        mod=dp.deployment_extension()
        for fraction in [.12,.43,.81]:
            cam=source.camera(int(fraction*len(source.scene.views)))
            x=mod._preparation_frustum(b.anchors.centers,cam.full_proj_transform,b.anchors.offset_scales,
                old._reference.occupied,b.eligible_counts,0)
            for mode in [1,2]:
                y=mod._preparation_frustum(b.anchors.centers,cam.full_proj_transform,b.anchors.offset_scales,
                    old._reference.occupied,b.eligible_counts,mode)
                row=dict(scene=slug,frame_fraction=fraction,mode=mode,mask_mismatches=int((x[0]!=y[0]).sum()),
                    bitwise_distance=bits_equal(x[1],y[1]),max_distance_error=float((x[1]-y[1]).abs().max()))
                frustum.append(row)
                if mode==1:assert row['mask_mismatches']==0,row
        cam=source.camera(int(.43*len(source.scene.views)))
        for flags in [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]:
            run=make_engine(source,student,frustum=flags[0],projection=flags[1],device=flags[2])
            for cond in ['fixed0','fixed1','fixed17','fixed1000','local0.01','local0.1','local0.3','local0.5','local0.7']:
                a,ids,stats=request(run,cam,cond);r,ref,rs=request(old,cam,cond)
                assert stats['candidate_gaussians']==rs['candidate_gaussians'] and len(ids)==len(ref)
                assert torch.equal(ids,ref),(slug,flags,cond)
                assert all(torch.isfinite(x).all() for x in a['inputs'].values() if isinstance(x,torch.Tensor))
                equal=all(bits_equal(x,r['inputs'][key]) for key,x in a['inputs'].items() if isinstance(x,torch.Tensor))
                if flags==(0,0,0):assert equal,(slug,flags,cond)
                checks.append(dict(scene=slug,flags=str(flags),condition=cond,K=len(ids),same_ids=True,
                    same_attributes=equal,cache_bytes=run._pipeline._preparation_bytes(),
                    max_attribute_error=max(difference(a,r).values(),default=0.)))
            del run;gc.collect();torch.cuda.empty_cache()
        if slug==TUNE[0]:
            first=make_engine(source,student,'B');_,ids,stats=first.prepare(cam,ratio_budget=.3)
            checks.append(dict(scene=slug,flags='B_frozen_initial',condition='local0.3',K=len(ids),
                same_ids=None,same_attributes=None,cache_bytes=0,max_attribute_error=None))
            del first
        del old,source,student,b;gc.collect();torch.cuda.empty_cache()
        write_csv(out/'checks.csv',checks);write_csv(out/'frustum.csv',frustum)
        print('PREPARATION_SMOKE',slug,len(checks),flush=True)
    save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),binary_sha256=sha(mod.__file__)))


if __name__=='__main__':main()
