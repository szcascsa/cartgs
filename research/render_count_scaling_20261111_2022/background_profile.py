"""第一候选与背景优化的真实Kernel对照，覆盖反例视角。"""
from render_study import *
from background_runtime import Renderer
from scaling_runtime import Renderer as FirstRenderer
import formal
import importlib.util
from balanced_order import order


@torch.no_grad()
def main():
    formal.frozen()
    spec=importlib.util.spec_from_file_location('background_formal_profile',HERE/'background_stage/formal.py')
    current=importlib.util.module_from_spec(spec);spec.loader.exec_module(current)
    current.frozen()
    cases=[];owners=[];metadata=[]
    for slug in ('Replica__office2__rgbd','VECtor__corridors-dolly__stereo'):
        _,source=load_source(slug)
        views=motions(source,quick=True)[::3] if slug.startswith('Replica') else [('continuous',653),('continuous',654)]
        cam=source.camera(views[0][1]);bg=source.scene.background
        runs={'G':p.create_renderer(cam,bg,source.runtime.total,use_graph=True),'C':old.Renderer(21),
              'F1':FirstRenderer(18),'F':Renderer()}
        for n,e in runs.items():
            if n!='G':e.precompile(source.runtime.total,cam,bg)
        conditions=('local0.01','local0.1','local0.3','local0.5','full') if slug.startswith('Replica') else ('local0.01','local0.05','full')
        for cond in conditions:
            for _,frame in views:
                cam=source.camera(frame);packet,ids,_=request(source,cam,cond)
                ref=p.render_prepared(cam,packet,bg,return_radii=True)
                for e in runs.values():
                    for _ in range(4):image,radii=e.render(cam,packet,bg)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                metadata.append(dict(scene=slug,condition=cond,frame=frame,K=len(ids),
                                     tile_instances=int(runs['C'].native.snapshot()[3].sum())))
                cases.append((slug,cond,frame,cam,packet,bg,runs))
        owners.append(source)
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for rep in range(6):
        for j,(slug,cond,frame,cam,packet,bg,runs) in enumerate(cases):
            for n in order(list(runs),rep+j,reverse=rep%2==1):
                torch.cuda.nvtx.range_push('scaling/'+slug+'/'+cond+'/'+n)
                runs[n].render(cam,packet,bg);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    save(HERE/'background_stage/reports/PROFILE_INPUTS.json',metadata)
    print('BACKGROUND_PROFILE_PASS',len(cases)*6*4,flush=True)


if __name__=='__main__':main()
