from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""冻结后32组运动/跳转视角配对；所有Pipeline请求重新准备输入。"""
from render_study import *
from balanced_order import order
import formal as first_formal
from background_runtime import Renderer,extension
from scaling_runtime import Renderer as FirstRenderer
HERE=HERE/"background_stage"
TUNE=TUNE+["VECtor__corridors-dolly__stereo"]
import argparse
import shutil
from native_camera import NativeCameraPreparer
from diagnostics_check import diagnostics
from runtime import SparseRenderer
from final_benchmark import native_modules

CONDITIONS=[f'local{x:g}' for x in (.01,.05,.1,.2,.3,.4,.5,.7)]+['fixed1000','fixed5000','full']


def identities():
    paths={}
    for s in SCENES:
        c=Path(s['checkpoint'])
        files=list(c.glob('point_cloud/iteration_*/point_cloud.ply'))
        files += [c/n for n in ('selector.pt','selector_config.yaml','cameras.json')]
        files += [Path(s['trajectory'])]
        if s.get('frame_csv'):files.append(Path(s['frame_csv']))
        for f in files:paths[str(f)]=sha(f)
    for mode,(path,expected) in catalog.frozen_paths().items():
        assert sha(path)==expected
        paths[path]=expected
    original=json.loads((HERE/'reports/BASELINE_SHA256.json').read_text())
    for f,h in original.items():assert sha(f)==h, f
    manifest=HERE/'reports/INPUT_SHA256.json'
    if manifest.is_file():assert json.loads(manifest.read_text())==paths
    else:save(manifest,paths)
    return len(paths)


def frozen():
    f=json.loads((HERE/'reports/FROZEN.json').read_text())
    assert f['binary_sha256']==sha(extension().__file__)
    for n,h in f['sources'].items():assert sha(HERE/n)==h,n
    return f


@torch.no_grad()
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',required=True)
    a.add_argument('--round',type=int,required=True,choices=(1,2))
    a.add_argument('--group',type=int,required=True)
    a.add_argument('--repeats',type=int,default=6)
    a.add_argument('--diagnostics',action='store_true')
    args=a.parse_args()
    f=frozen()
    out=HERE/'reports'/args.output
    out.mkdir(exist_ok=False)
    first_formal.frozen()
    config=json.loads((PRIOR/'reports/FROZEN_CANDIDATE.json').read_text())
    native_modules(config)
    scenes=SCENES[args.group*2:args.group*2+2]
    assert scenes
    if args.round==2:scenes=list(reversed(scenes))
    rows,checks,workloads,events,memory=[],[],[],[],[]
    for s in scenes:
        assert shutil.disk_usage(HERE).free>=5*(1<<30)
        _,source=load_source(s['slug'])
        factory=NativeCameraPreparer(p.MiniCam)
        # 所有渲染版本消费同一个冻结网络，避免把网络数值差异混入光栅比较。
        network=p.make_engine(source,None,'D',frustum=1,projection=0,device=1,compensator=0)
        views=motions(source)
        if args.round==2:views.reverse()
        initial=factory.from_source(source,views[0][1])
        runs={'G':p.create_renderer(initial,source.scene.background,source.runtime.total,use_graph=True),
              'C':old.Renderer(21),'F1':FirstRenderer(18),'F':Renderer(f['mode'])}
        for name,run in runs.items():
            if name!='G':run.precompile(source.runtime.total,initial,source.scene.background)
            memory.append(dict(scene=s['id'],variant=name,round=args.round,N=source.runtime.total,
                               workspace_bytes=run.native.workspace_bytes(),plans=run.native.stats()[3]))
        diagnostic=SparseRenderer(**config['regular']) if args.diagnostics else None
        for ci,condition in enumerate(CONDITIONS if args.round==1 else CONDITIONS[::-1]):
            packets=[]
            for motion,frame in views:
                cam=factory.from_source(source,frame)
                packet,ids,info=p.request(network,cam,condition)
                ref=p.render_prepared(cam,packet,source.scene.background,return_radii=True)
                for name,run in runs.items():
                    image,radii=run.render(cam,packet,source.scene.background)
                    row=dict(scene=s['id'],variant=name,condition=condition,frame=frame,round=args.round,K=len(ids),
                             rgb_bitwise=bits_equal(image,ref['render']),radii_bitwise=bits_equal(radii,ref['radii']))
                    checks.append(row)
                    assert row['rgb_bitwise'] and row['radii_bitwise'],row
                    for _ in range(2):run.render(cam,packet,source.scene.background)
                packets.append((motion,frame,cam,packet,ids.clone(),ref['render']))
                if diagnostic is not None:
                    value=diagnostics(diagnostic,cam,packet,source.scene.background)
                    workloads.append(dict(scene=s['id'],condition=condition,frame=frame,motion=motion,
                                          N=source.runtime.total,M=info['candidate_gaussians'],**value))
            torch.cuda.synchronize()
            for scope in ('raster','pipeline'):
                for rep in range(args.repeats):
                    for j,(motion,frame,cam,packet,ids,ref) in enumerate(packets):
                        for name in order(list(runs),rep+j+ci+args.round,reverse=args.round==2):
                            before=runs[name].native.stats()
                            torch.cuda.synchronize()
                            tick=time.perf_counter_ns()
                            if scope=='pipeline':
                                current=factory.from_source(source,frame)
                                fresh,fresh_ids,_=p.request(network,current,condition)
                                image,radii=runs[name].render(current,fresh,source.scene.background)
                            else:
                                image,radii=runs[name].render(cam,packet,source.scene.background)
                            torch.cuda.synchronize()
                            ms=(time.perf_counter_ns()-tick)*1e-6
                            after=runs[name].native.stats()
                            assert after[1:3]==before[1:3],'unexpected graph miss'
                            if scope=='pipeline':
                                assert torch.equal(ids,fresh_ids) and bits_equal(image,ref)
                            rows.append(dict(scene=s['id'],split='tune' if s['slug'] in TUNE else 'heldout',
                                             condition=condition,round=args.round,scope=scope,variant=name,repeat=rep,
                                             frame=frame,motion=motion,N=network.total,K=len(ids),ms=ms))
            # CUDA Event 独立测量，与同步墙钟和Profile Kernel累计分开。
            for _,frame,cam,packet,ids,_ in (packets[0],packets[-1]):
                for name,run in runs.items():
                    for rep in range(10):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        run.render(cam,packet,source.scene.background)
                        end.record();end.synchronize()
                        events.append(dict(scene=s['id'],condition=condition,round=args.round,variant=name,frame=frame,
                                           repeat=rep,K=len(ids),cuda_ms=begin.elapsed_time(end)))
            print('FORMAL_CONDITION',args.round,s['id'],condition,flush=True)
        print('FORMAL_SCENE_PASS',s['id'],network.total,flush=True)
        for name,run in runs.items():assert run.native.stats()[1:3]==[0,0]
        del source,network,runs,diagnostic,factory,packets,packet,ref,image,radii
        gc.collect()
        torch.cuda.empty_cache()
    for name,data in [('samples',rows),('checks',checks),('workloads',workloads),('events',events),('memory',memory)]:
        append(out/(name+'.csv'),data)
    frozen()
    save(out/'DONE.json',dict(passed=True,round=args.round,group=args.group,scenes=[s['id'] for s in scenes],
                              samples=len(rows),checks=len(checks),no_render_images=True))


if __name__=='__main__':main()
