from common import *
from baseline import previous_runtime,previous_extension
from deployment_runtime import deployment_extension,DeploymentRenderer
from components import sample
from micro import gpu_time
from large_scene import LARGE


@torch.no_grad()
def main():
    ext=deployment_extension();check=Comparisons('SMALL_TILE_BITS.json');rows=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        ref=previous_runtime(s);run=DeploymentRenderer(s.model,s.gaussians,s.config)
        frame=len(s.views)//2;cam=camera(s,frame)
        for ratio in [.01,.1,.7]:
            _,args=sample(s,ref,frame,ratio);pool=ref.prepare(cam,ratio_budget=.7)[1]
            for k in [0,1,16,64,128,256,384,511,512,768,1000,1024,1025,2000]:
                arguments=(args[0],pool[:k],*args[2:]);baseline=previous_extension()._compensate_tiled(*arguments,0)
                for tile in [6416,8208,4104]:
                    f=lambda tile=tile:ext._compensate_tiled(*arguments,tile)
                    value=f()
                    for j in range(3):check.same(str((group,ratio,k,tile,j)),baseline[j],value[j])
                    if k and ratio==.1:rows.append(dict(group=group,k=k,tile=tile,**gpu_time(f)))
        for kind in [16,8208,4104]:
            run._pipeline._small_tile_kind(kind)
            for k in [128,511,512,768,1000,1024,1025]:
                a,ia,_=ref.prepare(cam,budget=k);b,ib,_=run.prepare(cam,budget=k)
                label=str((group,'graph',kind,k));check.same(label,ia,ib);check.packet(label,a,b)
                samples=[]
                for i in range(140):
                    c=camera(s,(i*73+19)%len(s.views));start=time.perf_counter();run.prepare(c,budget=k)
                    if i>=20:samples.append((time.perf_counter()-start)*1000)
                rows.append(dict(group=group,k=k,tile=kind,prepare_ms=float(np.median(samples))))
        check.finish(group=group);save(ROOT/'reports/SMALL_TILES.json',dict(rows=rows));print('SMALL_TILES',group,flush=True)
        del s,ref,run;torch.cuda.empty_cache()
    check.finish();save(ROOT/'reports/SMALL_TILES.json',dict(passed=True,rows=rows))


if __name__=='__main__':main()
