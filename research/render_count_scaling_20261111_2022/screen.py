"""固定开发场景；同一当前输入轮转测速，不根据保留验证集调参。"""
from render_study import *
import argparse


@torch.no_grad()
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',required=True)
    a.add_argument('--modes',default='0,1,2,3,4')
    a.add_argument('--repeats',type=int,default=14)
    args=a.parse_args()
    out=HERE/'reports'/args.output
    out.mkdir(exist_ok=False)
    config=json.loads((FOOTPRINT/'reports/FROZEN_CANDIDATE.json').read_text())
    assert sha(config['binary'])==config['binary_sha256']
    rows=[]
    for slug in TUNE:
        s,source=load_source(slug)
        camera=source.camera(int(.43*len(source.scene.views)))
        runs={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))}
        runs['G']=p.create_renderer(camera,source.scene.background,source.runtime.total,use_graph=True)
        for run in runs.values():
            run.precompile(source.runtime.total,camera,source.scene.background)
        for condition in ('local0.01','local0.05','local0.1','local0.3','local0.5','local0.7','full'):
            prepared=[]
            for motion,frame in motions(source,quick=True):
                camera=source.camera(frame)
                packet,ids,info=request(source,camera,condition)
                ref=p.render_prepared(camera,packet,source.scene.background,return_radii=True)
                for run in runs.values():
                    for _ in range(3):
                        image,radii=run.render(camera,packet,source.scene.background)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                prepared.append((motion,frame,camera,packet,len(ids)))
            torch.cuda.synchronize()
            for rep in range(args.repeats):
                for j,(motion,frame,camera,packet,k) in enumerate(prepared):
                    for name in p.rotate(list(runs),rep+j):
                        torch.cuda.synchronize()
                        tick=time.perf_counter_ns()
                        image,radii=runs[name].render(camera,packet,source.scene.background)
                        torch.cuda.synchronize()
                        rows.append(dict(scene=s['id'],condition=condition,variant=name,repeat=rep,frame=frame,
                                         motion=motion,K=k,ms=(time.perf_counter_ns()-tick)*1e-6))
            print('SCREEN',s['id'],condition,flush=True)
        del runs,source,prepared,packet,ref,image,radii
        gc.collect()
        torch.cuda.empty_cache()
    append(out/'samples.csv',rows)
    save(out/'DONE.json',dict(samples=len(rows),scenes=TUNE,modes=args.modes))


if __name__=='__main__':main()
