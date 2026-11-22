"""补充最早已存在的C++位姿调度＋逐位F2，避免只与Python拆分入口比较。"""
from joint_final_study import *
from balanced_order import order


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--group',type=int,required=True);parser.add_argument('--round',type=int,required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];events=[];checks=[]
    specs=student_specs()[args.group*2:args.group*2+2]
    if args.round==2:specs.reverse()
    for spec in specs:
        source,student,_=load_case(spec);network,factory=network_variants(source,student)['tensor_best']
        config=json.loads((HERE/'reports/FROZEN_STUDENT.json').read_text())['config'];bg=source.scene.background
        engines={'C_native_F2':PoseRenderer(source,student,0),'G_cpp_pose':PoseRenderer(source,student,38),'I_final':FinalStudentRenderer(source,student,182,**config)}
        views=motions(source)
        if args.round==2:views.reverse()
        cam=factory.from_source(source,views[0][1]);base=BaselineRenderer();final=EncodedSignRenderer(182)
        base.precompile(source.runtime.total,cam,bg);final.precompile(source.runtime.total,cam,bg)
        for engine in engines.values():engine.precompile(cam,bg)
        for ci,condition in enumerate(CONDITIONS if args.round==1 else CONDITIONS[::-1]):
            for motion,frame in views:
                cam=factory.from_source(source,frame);packet,ids,_=p.request(network,cam,condition)
                ref=base.render(cam,packet,bg);new=final.render(cam,packet,bg)
                for name in ['C_native_F2','I_final']:
                    image,radii,actual,_=engines[name].render_from_source(source,frame,bg,condition,factory)
                    expected=ref if name=='C_native_F2' else new
                    row=dict(scene=spec['id'],condition=condition,frame=frame,variant=name,round=args.round,K=len(ids),
                        same_ids=torch.equal(ids,actual),rgb_bitwise=bits_equal(image,expected[0]),radii_bitwise=bits_equal(radii,expected[1]))
                    checks.append(row);assert row['same_ids'] and row['rgb_bitwise'] and row['radii_bitwise'],row
            for repeat in range(10):
                for j,(motion,frame) in enumerate(views):
                    for name in order(list(engines),repeat+j+ci,reverse=args.round==2):
                        torch.cuda.synchronize();start=time.perf_counter_ns()
                        image,radii,ids,_=engines[name].render_from_source(source,frame,bg,condition,factory)
                        torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-start)*1e-6
                        rows.append(dict(scene=spec['id'],condition=condition,frame=frame,motion=motion,variant=name,round=args.round,
                            repeat=repeat,N=source.runtime.total,K=len(ids),ms=elapsed))
            for _,frame in [views[0],views[-1]]:
                for repeat in range(6):
                    for name in order(list(engines),repeat+ci,reverse=args.round==2):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();engines[name].render_from_source(source,frame,bg,condition,factory);end.record();end.synchronize()
                        events.append(dict(scene=spec['id'],condition=condition,frame=frame,variant=name,round=args.round,repeat=repeat,cuda_ms=start.elapsed_time(end)))
            print('NATIVE_CONTROL',spec['id'],condition,args.round,flush=True)
        del engines,network,source,student,base,final;gc.collect();torch.cuda.empty_cache()
    for name,data in [('samples',rows),('events',events),('checks',checks)]:append(out/(name+'.csv'),data)
    save(out/'DONE.json',dict(passed=True,samples=len(rows),checks=len(checks),native_F2_bitwise=True))


if __name__=='__main__':main()
