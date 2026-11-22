"""当前位姿C++边界：独立候选与原A/D/F同帧轮转，禁止旧视角命中。"""
from joint_study import *
from joint_benchmark import CONDITIONS
from pending_runtime import PendingRenderer
from pose_runtime import PoseRenderer
from balanced_order import order


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--group',required=True,type=int);parser.add_argument('--round',required=True,type=int)
    parser.add_argument('--tune',action='store_true');parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    scenes=[s for s in SCENES if not args.tune or s['slug'] in TUNE][2*args.group:2*args.group+2]
    if args.round==2:scenes.reverse()
    mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode'];rows=[];checks=[];events=[];memory=[]
    for spec in scenes:
        _,source=load_source(spec['slug']);student,path,payload=student_for_spec(source,spec)
        for kind,weights in [('teacher',None)]+([('student',student)] if student is not None else []):
            networks=network_variants(source,weights);factory=networks['tensor_best'][1];bg=source.scene.background
            views=motions(source,quick=args.quick)
            if args.round==2:views.reverse()
            cam=factory.from_source(source,views[0][1]);raster={'A_fp32_F2':BaselineRenderer(),'D_tensor_best_new':Renderer(mode)}
            for r in raster.values():r.precompile(source.runtime.total,cam,bg)
            native=PendingRenderer(source,weights,mode,native_dispatch=True);native.precompile(cam,bg)
            pose=PoseRenderer(source,weights,mode);pose.precompile(cam,bg)
            variants=['A_fp32_F2','D_tensor_best_new','F_native','G_current_pose']
            def render(name,frame,condition):
                if name=='G_current_pose':return pose.render_from_source(source,frame,bg,condition,factory)
                cam=factory.from_source(source,frame)
                if name=='F_native':return native.render(cam,bg,condition)
                engine=networks['fp32' if name=='A_fp32_F2' else 'tensor_best'][0]
                packet,ids,info=p.request(engine,cam,condition);image,radii=raster[name].render(cam,packet,bg)
                return image,radii,ids,info
            conditions=['local0.1','local0.3','local0.5','full'] if args.quick else CONDITIONS
            for ci,condition in enumerate(conditions if args.round==1 else conditions[::-1]):
                expected={}
                for motion,frame in views:
                    refs={name:render(name,frame,condition) for name in variants}
                    for name in variants:
                        a,b=refs[name],refs['D_tensor_best_new']
                        if name in ['F_native','G_current_pose']:
                            row=dict(scene=spec['id'],model=kind,condition=condition,frame=frame,variant=name,round=args.round,K=len(a[2]),
                                same_ids=torch.equal(a[2],b[2]),rgb_bitwise=bits_equal(a[0],b[0]),radii_bitwise=bits_equal(a[1],b[1]))
                            checks.append(row);assert all(row[n] for n in ['same_ids','rgb_bitwise','radii_bitwise']),row
                        expected[name,frame]=len(a[2])
                        for _ in range(2):render(name,frame,condition)
                    del refs
                for repeat in range(4 if args.quick else 10):
                    for j,(motion,frame) in enumerate(views):
                        for name in order(variants,repeat+j+ci,reverse=args.round==2):
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            image,radii,ids,stats=render(name,frame,condition);torch.cuda.synchronize()
                            elapsed=(time.perf_counter_ns()-tick)*1e-6
                            assert len(ids)==expected[name,frame]
                            rows.append(dict(scene=spec['id'],model=kind,condition=condition,variant=name,frame=frame,motion=motion,
                                round=args.round,repeat=repeat,N=source.runtime.total,K=len(ids),ms=elapsed))
                for _,frame in [views[0],views[-1]]:
                    for repeat in range(6):
                        for name in order(variants,repeat+ci,reverse=args.round==2):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record();render(name,frame,condition);b.record();b.synchronize()
                            events.append(dict(scene=spec['id'],model=kind,condition=condition,frame=frame,variant=name,
                                round=args.round,repeat=repeat,cuda_ms=a.elapsed_time(b)))
                append(out/'samples.csv',rows);rows=[]
                print('CURRENT_POSE_BENCHMARK',spec['id'],kind,condition,args.round,flush=True)
            memory.append(dict(scene=spec['id'],model=kind,round=args.round,workspace_F=native.native.workspace_bytes(),
                workspace_G=pose.native.workspace_bytes()))
            del networks,raster,native,pose;gc.collect();torch.cuda.empty_cache()
        del source,student;gc.collect();torch.cuda.empty_cache()
    append(out/'checks.csv',checks);append(out/'events.csv',events);append(out/'memory.csv',memory)
    save(out/'DONE.json',dict(passed=True,comparisons=len(checks),same_renderer_bitwise=True,no_images=True,
        scope='当前CPU位姿、当前选择/补偿及渲染完整延迟；无同帧预测缓存'))


if __name__=='__main__':main()
