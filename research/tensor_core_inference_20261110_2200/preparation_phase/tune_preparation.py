"""开发场景上完整移动视角配对，不按保留集选择开关。"""
from prep_study import *
import gc

CONFIGS={'C':(0,0,0),'D_off':(0,0,0),'F':(1,0,0),'P':(0,1,0),'S':(0,0,1),
         'FP':(1,1,0),'FS':(1,0,1),'PS':(0,1,1),'FPS':(1,1,1)}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];checks=[];memory=[]
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug)
        factory=create_camera_preparer(MiniCam,variant='R');n=len(source.scene.views)
        frames=[int(.43*n),int(.43*n)+1,int(.09*n),int(.81*n)]
        engines={name:make_engine(source,student,'C' if name=='C' else 'D',
            frustum=f,projection=p,device=s) for name,(f,p,s) in CONFIGS.items()}
        for name,e in engines.items():
            for ratio in [.1,.3,.5]:
                for frame in frames:e.prepare(factory.from_source(source,frame),ratio_budget=ratio)
            memory.append(dict(scene=slug,variant=name,static_workspace_bytes=e._pipeline._preparation_bytes() if name!='C' else 0))
        cam=factory.from_source(source,frames[0]);renderer=create_renderer(cam,source.scene.background,engines['C'].total,use_graph=True)
        for ratio in [.1,.3,.5]:
            for frame in frames:
                cam=factory.from_source(source,frame);ref,rids,rs=engines['C'].prepare(cam,ratio_budget=ratio)
                for name,e in engines.items():
                    packet,ids,stats=e.prepare(cam,ratio_budget=ratio)
                    assert torch.equal(ids,rids) and stats['candidate_gaussians']==rs['candidate_gaussians'],(slug,name,ratio,frame)
                    checks.append(dict(scene=slug,variant=name,ratio=ratio,frame=frame,K=len(ids),M=stats['candidate_gaussians'],same_ids=True,
                        same_attributes=all(bits_equal(v,ref['inputs'][k]) for k,v in packet['inputs'].items() if isinstance(v,torch.Tensor))))
            for repeat in range(10):
                for vi,frame in enumerate(frames):
                    for name in rotate(list(engines),repeat+vi):
                        e=engines[name]
                        for scope in ['engine_prepare','prepare','pipeline']:
                            cam=factory.from_source(source,frame);torch.cuda.synchronize()
                            tick=time.perf_counter_ns()
                            if scope!='engine_prepare':cam=factory.from_source(source,frame)
                            packet,ids,stats=e.prepare(cam,ratio_budget=ratio)
                            if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(scene=slug,variant=name,ratio=ratio,scope=scope,repeat=repeat,frame=frame,ms=ms,N=e.total,K=len(ids)))
            print('PREPARATION_TUNE',slug,ratio,len(rows),flush=True)
        write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'memory.csv',memory)
        del engines,source,student,renderer,factory,packet,ref;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),binary_sha256=sha(dp.deployment_extension().__file__)))


if __name__=='__main__':main()
