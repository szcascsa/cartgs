"""冻结后四组主对照及逐项关闭；同工作量、移动/跳转视角、双轮反序。"""
from prep_study import *
from native_camera import NativeCameraPreparer
import gc


def frozen_config():
    f=json.loads((HERE/'reports/FROZEN_JOINT_CANDIDATE.json').read_text())
    assert sha(dp.deployment_extension().__file__)==f['binary_sha256']
    return f


def variants_for(source,student,config):
    p=config['projection'];r=create_camera_preparer(MiniCam,variant='R');h=NativeCameraPreparer(MiniCam)
    engines={v:(make_engine(source,student,v),r) for v in ['A','B','C']}
    for name,flags,camera in [('D',(1,p,1),h),('D_noF',(0,p,1),h),('D_noP',(1,0,1),h),
        ('D_noS',(1,p,0),h),('D_noH',(1,p,1),r),('CUDA_preparation',(1,0,1),h)]:
        run=make_engine(source,student,frustum=flags[0],projection=flags[1],device=flags[2],
            selector=0 if name=='CUDA_preparation' else 2,compensator=0 if name=='CUDA_preparation' else 2)
        engines[name]=(run,camera)
    return engines


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--round',required=True,type=int)
    parser.add_argument('--slice',required=True,type=int);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);config=frozen_config();rows=[];checks=[];memory=[]
    specs=[s for s in SCENES if s['dataset_kind']=='Replica'][args.slice*2:args.slice*2+2]
    if args.round%2==0:specs.reverse()
    for spec in specs:
        _,source=load_scene(spec['slug']);student,_,_=student_for(source,spec['slug'])
        engines=variants_for(source,student,config);n=len(source.scene.views)
        frames=[int(.43*n),int(.43*n)+1,int(.43*n)+2,int(.07*n),int(.71*n),int(.97*n)]
        if args.round%2==0:frames.reverse()
        basecam=engines['A'][1].from_source(source,frames[0]);renderer=create_renderer(basecam,source.scene.background,engines['A'][0].total,use_graph=True)
        torch.cuda.reset_peak_memory_stats();start_memory=torch.cuda.memory_allocated()
        for name,(engine,factory) in engines.items():
            for condition in CONDITIONS:
                for frame in frames:
                    cam=factory.from_source(source,frame);packet,ids,stats=request(engine,cam,condition);renderer.render(cam,packet,source.scene.background)
            memory.append(dict(scene=spec['id'],variant=name,round=args.round,N=engine.total,
                projection_bytes=engine._pipeline._preparation_bytes() if name not in ['A','B','C'] else 0,
                tensor_bytes=engine._pipeline._tensor_bytes() if name!='A' else 0,
                start_allocated=0,peak_allocated=0,peak_reserved=0))
        memory.append(dict(scene=spec['id'],variant='all_engines_process',round=args.round,N=engines['A'][0].total,
            projection_bytes=0,tensor_bytes=0,
            start_allocated=start_memory,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
        for ci,condition in enumerate(CONDITIONS):
            for frame in frames:
                acam=engines['A'][1].from_source(source,frame);ap,ai,ast=request(engines['A'][0],acam,condition)
                cp,cc,cst=request(engines['C'][0],acam,condition)
                for name,(engine,factory) in engines.items():
                    cam=factory.from_source(source,frame);packet,ids,stats=request(engine,cam,condition)
                    assert len(ids)==len(ai) and stats.get('candidate_gaussians',len(ids))==ast.get('candidate_gaussians',len(ai)),(spec['id'],name,condition,frame)
                    if name.startswith('D') and name!='D_noS':assert torch.equal(ids,cc),(spec['id'],name,condition,frame)
                    checks.append(dict(scene=spec['id'],variant=name,round=args.round,condition=condition,frame=frame,K=len(ids),
                        M=stats.get('candidate_gaussians',len(ids)),same_ids_A=torch.equal(ids,ai),same_ids_C=torch.equal(ids,cc)))
            for repeat in range(8):
                for vi,frame in enumerate(frames):
                    for name in rotate(list(engines),repeat+vi+ci+args.round):
                        engine,factory=engines[name]
                        for scope in ['engine_prepare','prepare','pipeline','camera']:
                            cam=factory.from_source(source,frame)
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope!='engine_prepare':cam=factory.from_source(source,frame)
                            if scope!='camera':packet,ids,stats=request(engine,cam,condition)
                            if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(scene=spec['id'],variant=name,condition=condition,scope=scope,round=args.round,repeat=repeat,frame=frame,
                                motion='continuous' if frame in [int(.43*n),int(.43*n)+1,int(.43*n)+2] else 'jump',N=engine.total,
                                M=stats.get('candidate_gaussians',engine.total),K=len(ids),device_counts=stats.get('device_counts',False),ms=ms))
            print('PREPARATION_FORMAL',args.round,spec['id'],condition,len(rows),flush=True)
        write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'memory.csv',memory)
        del engines,source,student,renderer,packet,ap,cp;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),round=args.round,scenes=len(specs),binary_sha256=config['binary_sha256']))


if __name__=='__main__':main()
