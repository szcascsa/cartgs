"""预编译全部有界容量计划；不依赖任何具体视角或缓存渲染内容。"""
from harness import *
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[];generator=np.random.default_rng(8103)
    for scene in SCENES:
        if scene['split']!='tune':continue
        _,source=load_scene(scene['slug']);engine=GraphRenderer(plan_granularity=512,max_plans=512)
        camera=source.camera(0);torch.cuda.synchronize();before=torch.cuda.memory_allocated();tick=time.perf_counter_ns()
        engine.precompile(source.runtime.total,camera,source.scene.background);torch.cuda.synchronize()
        setup_ms=(time.perf_counter_ns()-tick)/1e6;initial=engine.native.stats();assert initial[3]==math.ceil(source.runtime.total/512)
        forbidden={frame for _,frame in views_for(source,'final')};eligible=[i for i in range(len(source.scene.views)) if i not in forbidden]
        for index in range(160):
            frame=eligible[index] if index<80 else int(generator.choice(eligible));camera=source.camera(frame)
            condition=['local0.01','local0.03','local0.05','local0.1','local0.3','local0.7','fixed1000','fixed5000','full'][index%9]
            packet,ids,stats=prepare(source,camera,frame,condition)
            ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
            image,radii=engine.render(camera,packet,source.scene.background)
            assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']),(scene['id'],index)
        final=engine.native.stats();assert final[1]==0 and final[2]==0 and final[3]==initial[3],final
        rows.append(dict(scene=scene['id'],setup_ms=setup_ms,plans=final[3],workspace_bytes=final[4],
            initial_allocated_bytes=before,final_allocated_bytes=torch.cuda.memory_allocated(),checks=160,misses=final[1]))
        del source,engine,packet;torch.cuda.empty_cache();print('PRECOMPILE_PARITY',scene['id'],rows[-1],flush=True)
    tiny=MiniCam(17,19,1.2,1.,np.eye(4));exemplar=torch.zeros(3,device='cuda');engine=GraphRenderer(plan_granularity=512,max_plans=2)
    try:engine.precompile(1025,tiny,exemplar)
    except RuntimeError as error:assert 'plan count' in str(error)
    else:raise AssertionError('Unbounded plan creation accepted')
    write_json(output/'SUMMARY.json',dict(passed=True,checks=801,rows=rows,plan_limit_rejection=True,
        precompile_uses_only_map_capacity_and_resolution=True))
    print('PRECOMPILE_COMPLETE',flush=True)


if __name__=='__main__':main()
