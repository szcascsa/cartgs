"""同地图/视角/属性的嵌套删点归因；不作为部署选点或质量结果。"""
from render_study import *
import argparse
from formal import frozen
from diagnostics_check import diagnostics
from final_benchmark import native_modules
from runtime import SparseRenderer


@torch.no_grad()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True)
    a.add_argument('--repeats',type=int,default=16)
    args=a.parse_args()
    f=frozen();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    config=json.loads((PRIOR/'reports/FROZEN_CANDIDATE.json').read_text())
    native_modules(config)
    samples,workloads,checks=[],[],[]
    for s in SCENES:
        _,source=load_source(s['slug'])
        full,_=source.runtime.full()
        total=source.runtime.total
        generator=torch.Generator(device='cpu').manual_seed(104729)
        permutation=torch.randperm(total,generator=generator).cuda()
        views=motions(source,quick=True)[::3]
        camera=source.camera(views[0][1])
        runs={'G':p.create_renderer(camera,source.scene.background,total,use_graph=True),'C':old.Renderer(21),'F':Renderer(f['mode'])}
        for n,e in runs.items():
            if n!='G':e.precompile(total,camera,source.scene.background)
        diagnostic=SparseRenderer(**config['regular'])
        for motion,frame in views:
            cam=source.camera(frame);packets=[]
            previous=torch.empty(0,dtype=torch.int64,device='cuda')
            for ratio in (.01,.05,.1,.2,.3,.4,.5,.7,1.):
                ids=permutation[:max(1,math.floor(total*ratio))].sort().values
                assert bool(torch.isin(previous,ids).all())
                previous=ids
                inputs={k:v.index_select(0,ids) if isinstance(v,torch.Tensor) else v for k,v in full['inputs'].items()}
                packet=dict(sh_degree=full['sh_degree'],inputs=inputs)
                reference=p.render_prepared(cam,packet,source.scene.background,return_radii=True)
                for n,e in runs.items():
                    for _ in range(3):image,radii=e.render(cam,packet,source.scene.background)
                    assert bits_equal(image,reference['render']) and bits_equal(radii,reference['radii'])
                    checks.append(dict(scene=s['id'],frame=frame,ratio=ratio,variant=n,K=len(ids),bitwise=True))
                value=diagnostics(diagnostic,cam,packet,source.scene.background)
                workloads.append(dict(scene=s['id'],frame=frame,ratio=ratio,N=total,**value))
                packets.append((ratio,len(ids),packet))
            for rep in range(args.repeats):
                # 次序轮换；不把K从小到大固定执行造成的频率变化算成单调关系。
                for j in p.rotate(list(range(len(packets))),rep):
                    ratio,k,packet=packets[j]
                    for name in p.rotate(list(runs),rep+j):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        runs[name].render(cam,packet,source.scene.background)
                        torch.cuda.synchronize()
                        samples.append(dict(scene=s['id'],frame=frame,motion=motion,ratio=ratio,N=total,K=k,
                                            variant=name,repeat=rep,ms=(time.perf_counter_ns()-tick)*1e-6))
        for name,rows in [('samples',samples),('workloads',workloads),('checks',checks)]:append(out/(name+'.csv'),rows)
        samples,workloads,checks=[],[],[]
        print('CAUSAL_SCENE_PASS',s['id'],flush=True)
        del source,full,runs,diagnostic,packets,packet,reference,image,radii
        gc.collect();torch.cuda.empty_cache()
    save(out/'DONE.json',dict(passed=True,scenes=len(SCENES),scope='嵌套Master属性删点，只做渲染耗时归因',no_images=True))


if __name__=='__main__':main()
