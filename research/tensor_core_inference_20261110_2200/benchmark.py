"""双轮移动视角；原v31、两个单模块和两模块Tensor路径配对。"""
from study import *
import gc
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--all-scenes',action='store_true');p.add_argument('--slice',type=int);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);f=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());sm,cm,w=[f[k] for k in ['selector','compensator','warps']];cw=f.get('comp_warps',w)
    assert sha(tc.deployment_extension().__file__)==f['binary_sha256'];rows=[];checks=[];memory=[]
    specs=[s for s in SCENES if s['dataset_kind']=='Replica' and (args.all_scenes or s['slug'] in [*TUNE,'Replica__room2__mono'])]
    if args.slice is not None:
        assert 0<=args.slice<8;specs=specs[args.slice*2:args.slice*2+2]
    if args.round%2==0:specs.reverse()
    for si,spec in enumerate(specs):
        slug=spec['slug'];_,source=load_scene(slug);student,_,_=student_for(source,slug);n=len(source.scene.views)
        frames=[int(.43*n),int(.43*n)+1,int(.43*n)+2,int(.07*n),int(.71*n),int(.97*n)]
        factory=create_camera_preparer(MiniCam,variant='R');cams=[factory.from_source(source,x) for x in frames]
        engines=dict(original=engine(source,student,original=True),selector=engine(source,student,sm,0,w,comp_warps=cw),compensator=engine(source,student,0,cm,w,comp_warps=cw),both=engine(source,student,sm,cm,w,comp_warps=cw))
        master,master_ids=engines['original'].full();renderer=create_renderer(cams[0],source.scene.background,engines['original'].total,use_graph=True)
        for name,e in engines.items():
            for cam in cams:
                for c in CONDITIONS:
                    packet,ids,stats=request(e,cam,c);renderer.render(cam,packet,source.scene.background)
            memory.append(dict(scene=spec['id'],variant=name,round=args.round,tensor_weight_bytes=e._pipeline._tensor_bytes() if name!='original' else 0))
        torch.cuda.synchronize()
        for ci,c in enumerate(CONDITIONS):
            for vi,cam in enumerate(cams):
                packets={name:request(e,cam,c) for name,e in engines.items()};ref=packets['original'];K=len(ref[1])
                for name,(packet,ids,stats) in packets.items():
                    assert len(ids)==K
                    if name=='compensator':assert torch.equal(ids,ref[1])
                    if c!='full' and K and name=='both':assert engines[name].tensor_config==dict(selector=sm,compensator=cm,warps=w,comp_warps=cw)
                    checks.append(dict(scene=spec['id'],variant=name,condition=c,round=args.round,frame=frames[vi],K=K,same_ids=torch.equal(ids,ref[1]),overlap=int(torch.isin(ids,ref[1]).sum())))
            for repeat in range(6):
                order=list(range(6)) if args.round%2 else list(reversed(range(6)))
                for vi in order:
                    for name in rotate(list(engines),repeat+vi+ci+args.round):
                        e=engines[name]
                        for scope in ['prepare','pipeline']:
                            torch.cuda.synchronize();tick=time.perf_counter_ns();cam=factory.from_source(source,frames[vi]);packet,ids,stats=request(e,cam,c)
                            if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                            torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(scene=spec['id'],variant=name,condition=c,scope=scope,round=args.round,repeat=repeat,frame=frames[vi],motion='continuous' if vi<3 else 'jump',N=e.total,M=stats.get('candidate_gaussians',e.total),K=len(ids),device_counts=stats.get('device_counts',False),ms=elapsed))
            print('TC_BENCH',args.round,spec['id'],c,len(rows),flush=True)
        for name,data in [('samples',rows),('checks',checks),('memory',memory)]:write_csv(out/(name+'.csv'),data)
        del engines,source,student,renderer,factory,cams,packets,packet,ids,master,master_ids,ref;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,scenes=len(specs),samples=len(rows),checks=len(checks),round=args.round,binary_sha256=f['binary_sha256'],includes_current_camera=True,images_saved=0))
if __name__=='__main__':main()
