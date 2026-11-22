"""先核对关闭开关与v31逐位相同，再检查两种Tensor路径和严格K。"""
from study import *
import gc
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--slug',default=TUNE[0]);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    slug=args.slug;_,source=load_scene(slug);student,path,payload=student_for(source,slug)
    cam=source.camera(len(source.scene.views)//2)
    original=engine(source,student,original=True);off=engine(source,student)
    conditions=['fixed0','fixed1','fixed17','local0.1','local0.3','local0.5','local0.15','local0.7','full']
    base={};checks=[]
    for c in conditions:
        a,ids,info=request(original,cam,c);b,bids,_=request(off,cam,c)
        assert torch.equal(ids,bids)
        assert all(bits_equal(v,b['inputs'][k]) for k,v in a['inputs'].items() if isinstance(v,torch.Tensor)),c
        base[c]=(a,ids,info)
    for mode in [1,2]:
        for sm,cm in [(mode,0),(0,mode),(mode,mode)]:
            run=engine(source,student,sm,cm)
            for c in conditions:
                p,ids,info=request(run,cam,c);ref,rids,old=base[c];assert len(ids)==len(rids)
                assert all(torch.isfinite(v).all() for v in p['inputs'].values() if isinstance(v,torch.Tensor))
                if not sm:assert torch.equal(ids,rids)
                loss={};image_psnr=300.
                if len(ids):
                    a=render_prepared(cam,ref,source.scene.background,return_radii=True)['render']
                    b=render_prepared(cam,p,source.scene.background,return_radii=True)['render']
                    image_psnr=float(-10*torch.log10((a-b).square().mean().clamp_min(1e-30)))
                if torch.equal(ids,rids):loss=difference(ref,p)
                checks.append(dict(condition=c,selector_mode=sm,compensator_mode=cm,K=len(ids),same_ids=torch.equal(ids,rids),image_psnr=image_psnr,errors=json.dumps(loss)))
                print('TC_SMOKE',sm,cm,c,len(ids),torch.equal(ids,rids),image_psnr,flush=True)
            del run;gc.collect();torch.cuda.empty_cache()
    write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),off_bitwise=True,student_sha256=sha(path),binary_sha256=sha(tc.deployment_extension().__file__)))
if __name__=='__main__':main()
