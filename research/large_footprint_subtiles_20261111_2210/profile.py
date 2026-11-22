"""F2/精确公共乘积/提前退出的Graph节点，移动与跳转视角独立计时。"""
from footprint_study import *
from formal import verify_frozen


@torch.no_grad()
def main():
    f=verify_frozen();cases=[];owners=[]
    for slug in ('Replica__office2__rgbd','VECtor__corner-slow__stereo'):
        _,source=load_source(slug);bg=source.scene.background;views=motions(source,quick=True)[::3]
        cam=source.camera(views[0][1]);runs={'F2':BaselineRenderer(),'common':Renderer(f['mode']),'exit8':Renderer(18)}
        for e in runs.values():e.precompile(source.runtime.total,cam,bg)
        for cond in ('local0.01','local0.05','local0.1','local0.3','full'):
            for _,frame in views:
                cam=source.camera(frame);packet,_,_=request(source,cam,cond)
                ref=p.render_prepared(cam,packet,bg,return_radii=True)
                for e in runs.values():
                    c=compare(e,cam,packet,bg,ref);assert c['rgb_bitwise'] and c['radii_bitwise']
                    for _ in range(4):e.render(cam,packet,bg)
                cases.append((slug,cond,cam,packet,bg,runs))
        owners.append(source)
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for rep in range(6):
        for j,(slug,cond,cam,packet,bg,runs) in enumerate(cases):
            for n in order(list(runs),rep+j,reverse=rep%2==1):
                torch.cuda.nvtx.range_push('footprint/'+slug+'/'+cond+'/'+n)
                runs[n].render(cam,packet,bg);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('FOOTPRINT_PROFILE_PASS',len(cases)*6*3,flush=True)


if __name__=='__main__':main()
