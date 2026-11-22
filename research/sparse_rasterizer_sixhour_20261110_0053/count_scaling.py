"""用同一选点身份做补偿开关归因；不改最终推理路径或重新训练。"""
from harness import *
from final_benchmark import native_modules,rotate
from graph_runtime import GraphRenderer
from runtime import SparseRenderer
from diagnostics_check import diagnostics


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    samples=[];work=[];checks=0
    for scene in SCENES:
        _,source=load_scene(scene['slug']);g=GraphRenderer(**config['graph']);c=SparseRenderer(**config['regular'])
        views=views_for(source,'final');views=views[:3]+views[-3:]
        g.precompile(source.runtime.total,source.camera(views[0][1]),source.scene.background)
        full,full_ids=source.runtime.full()
        for motion,frame in views:
            camera=source.camera(frame)
            for condition in ['local0.01','local0.05','local0.1','local0.3','local0.7','full']:
                packet,ids,stats=prepare(source,camera,frame,condition)
                uncompensated=dict(sh_degree=full['sh_degree'],inputs={name:(value[ids].contiguous() if isinstance(value,torch.Tensor) else value)
                    for name,value in full['inputs'].items()})
                inputs={'with_compensation':packet}
                if condition!='full':inputs['same_ids_master_attributes']=uncompensated
                refs={}
                for treatment,value in inputs.items():
                    ref=render_prepared(camera,value,source.scene.background,return_radii=True)
                    image,radii=g.render(camera,value,source.scene.background)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']);checks+=1
                    refs[treatment]=ref['render']
                    d=diagnostics(c,camera,value,source.scene.background)
                    work.append(dict(scene=scene['id'],N=source.runtime.total,frame=frame,motion=motion,condition=condition,treatment=treatment,
                        selected_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),**d))
                jobs=[(treatment,name) for treatment in inputs for name in ['A','G']]
                for repeat in range(8):
                    for treatment,name in rotate(jobs,repeat):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        image=render_prepared(camera,inputs[treatment],source.scene.background)['render'] if name=='A' else g.render(camera,inputs[treatment],source.scene.background)[0]
                        torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-tick)/1e6
                        samples.append(dict(scene=scene['id'],split=scene['split'],N=source.runtime.total,frame=frame,motion=motion,condition=condition,
                            treatment=treatment,variant=name,repeat=repeat,K=len(ids),wall_ms=elapsed))
        write_csv(output/'samples.csv',samples);write_csv(output/'workloads.csv',work)
        del source,full,g,c,packet,inputs,uncompensated;torch.cuda.empty_cache()
        print('COUNT_SCALING',scene['id'],len(samples),flush=True)
    write_json(output/'SUMMARY.json',dict(passed=True,checks=checks,scenes=len(SCENES),samples=len(samples),
        semantics='保持真实Selector的同一Gaussian身份与K；只在独立归因对照中替换为对应Master属性；最终推理仍使用原补偿',
        compensation_control_is_not_new_algorithm=True))
    print('COUNT_SCALING_COMPLETE',len(samples),flush=True)


if __name__=='__main__':main()
