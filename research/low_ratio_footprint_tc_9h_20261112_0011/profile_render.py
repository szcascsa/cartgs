"""冻结版本的原始Graph节点Profile；只用于Kernel归因，不当独占墙钟。"""
from nine_study import *
from balanced_order import order


@torch.no_grad()
def main():
    accepted=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text());cases=[];owners=[]
    for slug in ['Replica__office2__rgbd','VECtor__corner-slow__stereo']:
        s,source=load_source(slug);bg=source.scene.background;views=motions(source,quick=True)[::3]
        cam=source.camera(views[0][1]);runs={'F2':BaselineRenderer(),MODES[accepted['mode']]:Renderer(accepted['mode']),
            'tile_safe_fast':Renderer(40)}
        for e in runs.values():e.precompile(source.runtime.total,cam,bg)
        for workload in ['lod','nested']:
            for condition in ['local0.01','local0.05','local0.1','local0.3','local0.5','full']:
                for _,frame in views:
                    cam=source.camera(frame)
                    if workload=='nested':
                        full,_,_=request(source,cam,'full');packet,_=nested_packet(full,1. if condition=='full' else float(condition[5:]))
                    else:packet,_,_=request(source,cam,condition)
                    for e in runs.values():
                        for _ in range(3):e.render(cam,packet,bg)
                    cases.append((slug,workload,condition,cam,packet,bg,runs))
        owners.append(source)
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(6):
        for j,(scene,workload,condition,cam,packet,bg,runs) in enumerate(cases):
            for name in order(list(runs),repeat+j,reverse=repeat%2==1):
                torch.cuda.nvtx.range_push('/'.join(['nine_render',scene,workload,condition,name]))
                runs[name].render(cam,packet,bg);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('NINE_RENDER_PROFILE_COMPLETE',len(cases)*6*3,flush=True)


if __name__=='__main__':main()
