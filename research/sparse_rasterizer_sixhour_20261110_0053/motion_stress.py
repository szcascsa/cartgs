"""不预热具体视角或容量，连续运动/随机跳转/预算轮转的全请求压力对照。"""
from harness import *
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--frames',type=int,default=400)
    parser.add_argument('--max-plans',type=int,default=512);parser.add_argument('--ordered',action='store_true');parser.add_argument('--device-clear',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[];startup=[];checks=0
    generator=np.random.default_rng(7013)
    for scene in SCENES:
        if scene['split']!='tune':continue
        item,source=load_scene(scene['slug']);n=len(source.scene.views)
        variants={'B':baseline_B(),'C':SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,ordered_coverage=args.ordered),
            'G':GraphRenderer(plan_granularity=512,max_plans=args.max_plans,ordered_coverage=args.ordered,device_clear=args.device_clear)}
        functions={'A':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
        functions.update({name:(lambda cam,p,e=engine:e.render(cam,p,source.scene.background)[0]) for name,engine in variants.items()})
        torch.cuda.synchronize();camera=source.camera(int(.15*n));tick=time.perf_counter_ns()
        variants['G'].precompile(source.runtime.total,camera,source.scene.background);torch.cuda.synchronize()
        startup.append(dict(scene=scene['id'],reserve_ms=(time.perf_counter_ns()-tick)/1e6))
        forbidden={frame for _,frame in views_for(source,'final')}
        eligible=[j for j in range(n) if j not in forbidden]
        conditions=['local0.01','local0.05','local0.1','local0.3','local0.7','fixed1000','fixed5000','full']
        for i in range(args.frames):
            frame=(int(.15*n)+i)%n if i<args.frames//2 else int(generator.choice(eligible))
            while frame in forbidden:frame=(frame+1)%n
            motion='continuous' if i<args.frames//2 else 'jump';condition=conditions[i%len(conditions)]
            camera=source.camera(frame);names=list(functions);shift=i%len(names);names=names[shift:]+names[:shift]
            if i%2:names.reverse()
            pair=[]
            for name in names:
                before=variants['G'].native.stats() if name=='G' else None
                torch.cuda.synchronize();tick=time.perf_counter_ns()
                packet,ids,stats=prepare(source,camera,frame,condition);image=functions[name](camera,packet);torch.cuda.synchronize()
                ms=(time.perf_counter_ns()-tick)/1e6;after=variants['G'].native.stats() if name=='G' else None
                rows.append(dict(scene=scene['id'],iteration=i,motion=motion,condition=condition,frame=frame,variant=name,
                    K=len(ids),candidate_N=stats['candidate_gaussians'],wall_ms=ms,
                    plan_miss=bool(after and after[1]>before[1]),workspace_bytes=after[4] if after else 0))
                pair.append((ids.clone(),image))
            # 旧B的Tile FMA历史差异另行披露；新C/G以原A为唯一数值判据。
            reference=pair[names.index('A')]
            assert all(torch.equal(value[0],reference[0]) and (name=='B' or bits_equal(value[1],reference[1])) for name,value in zip(names,pair));checks+=1
            if (i+1)%100==0:
                write_csv(output/'samples.csv',rows);print('MOTION_STRESS',scene['id'],i+1,variants['G'].native.stats(),flush=True)
        startup[-1]['final_graph_stats']=variants['G'].native.stats()
        assert variants['G'].native.stats()[1]==0 and variants['G'].native.stats()[2]==0
        del source,variants,pair,packet;torch.cuda.empty_cache()
    write_csv(output/'samples.csv',rows);write_json(output/'SUMMARY.json',dict(passed=True,frames=args.frames,
        max_plans=args.max_plans,checks=checks,rows=len(rows),startup=startup,
        workload='模型加载按容量预编译计划；每帧重新准备，混合Ratio/固定K/Full与跳转视角；无具体视角预热，建图成本单列'))
    print('MOTION_STRESS_COMPLETE',checks,flush=True)


if __name__=='__main__':main()
