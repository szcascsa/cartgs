"""只对Rasterizer的真实Graph节点做Nsight归因；不混入正式墙钟测量。"""
from render_study import *
from formal import frozen


@torch.no_grad()
def main():
    f=frozen()
    _,source=load_source('Replica__office2__rgbd')
    views=motions(source,quick=True)[::3]
    cam=source.camera(views[0][1]);bg=source.scene.background
    runs={'G':p.create_renderer(cam,bg,source.runtime.total,use_graph=True),'C':old.Renderer(21),'F':Renderer(f['mode'])}
    for n,e in runs.items():
        if n!='G':e.precompile(source.runtime.total,cam,bg)
    cases=[]
    for condition in ('local0.01','local0.1','local0.3','local0.5','full'):
        for _,frame in views:
            cam=source.camera(frame);packet,ids,_=request(source,cam,condition)
            ref=p.render_prepared(cam,packet,bg,return_radii=True)
            for e in runs.values():
                for _ in range(4):image,radii=e.render(cam,packet,bg)
                assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
            cases.append((condition,frame,cam,packet))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for rep in range(6):
        for j,(condition,frame,cam,packet) in enumerate(cases):
            for n in p.rotate(list(runs),rep+j):
                torch.cuda.nvtx.range_push('scaling/'+condition+'/'+n)
                runs[n].render(cam,packet,bg)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('PROFILE_PASS',len(cases)*6*len(runs),flush=True)


if __name__=='__main__':main()
