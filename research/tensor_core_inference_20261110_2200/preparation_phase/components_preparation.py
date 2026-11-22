"""全16组的Selector/Compensator固定身份CUDA Event，静态预计算和动态投影成本区分。"""
from prep_study import *
import gc


def graph_for(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):out=fn()
    return graph


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--slice',type=int);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);f=json.loads((HERE/'reports/FROZEN_JOINT_CANDIDATE.json').read_text());rows=[]
    specs=[s for s in SCENES if s['dataset_kind']=='Replica']
    if args.slice is not None:specs=specs[args.slice*2:args.slice*2+2]
    for spec in specs:
        _,source=load_scene(spec['slug']);student,_,_=student_for(source,spec['slug'])
        engines={name:make_engine(source,student,name) for name in ['A','C']}
        engines['D']=make_engine(source,student,frustum=1,projection=f['projection'],device=1)
        run=engines['A'];b=run._base;a=b.anchors
        cam=source.camera(int(.43*len(source.scene.views)))
        visible=anchor_frustum_mask(a.centers,a.offset_scales,cam.full_proj_transform,run._reference.occupied)
        candidates=torch.nonzero(visible[b.owners],as_tuple=False).flatten()
        view=run._ext.view(a.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),cam.world_view_transform,
            cam.full_proj_transform,b.model.selector_view_scene_scale,float(cam.image_width)/cam.image_height,cam.znear)
        prefix=run._ext._selector_prefix(b.selector_static,run._selector[0])
        packed={'C':prior.tc.deployment_extension()._tensor_pack(run._selector,run._compensator,run._view_weight,2,2),
                'D':engines['D']._pipeline._tensor_state()}
        for ratio in PRIMARY:
            _,ids,_=run.prepare(cam,ratio_budget=ratio);weights=[1.]+[-1.]*(len(run.knots)-1)
            for j in range(1,len(run.knots)):
                if ratio<=run.knots[j-1]+1e-8:break
                weights[j]=min(1.,(ratio-run.knots[j-1])/(run.knots[j]-run.knots[j-1]))
            lo,hi=b.spatial.ratio_encoding_values;t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
            code=[t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)]
            sa=(b.selector_static,candidates,b.owners,view,run._reference.noise,run._selector,ratio,True,b.joint_schema)
            ca=(b.offset,ids,b.owners,b.reference,b.tables,run._compensator,b.compensated_master,view,run._view_weight,weights,code,
                True,b.joint_schema,hasattr(b.spatial,'softplus_scale_schema'))
            for stage in ['selector','compensator']:
                funcs={}
                if stage=='selector':
                    funcs['A']=lambda:run._ext._selector_prefixed(*sa,True,prefix)
                    for name,mod,mode,warps in [('C',prior.tc.deployment_extension(),2,4),('D',dp.deployment_extension(),2,4)]:
                        funcs[name]=lambda name=name,mod=mod,mode=mode,warps=warps:mod._tensor_selector(*sa,prefix,packed[name],mode,warps)
                else:
                    funcs['A']=lambda:run._ext._compensate_tiled(*ca,16)
                    funcs['C']=lambda:prior.tc.deployment_extension()._tensor_compensate(*ca,packed['C'],2,2,b.empty)
                    funcs['D']=lambda:dp.deployment_extension()._tensor_compensate(*ca,packed['D'],2,2,b.empty)
                graphs={name:graph_for(fn) for name,fn in funcs.items()}
                for repeat in range(20):
                    for name in rotate(list(graphs),repeat):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record();graphs[name].replay();end.record();end.synchronize()
                        rows.append(dict(scene=spec['id'],ratio=ratio,stage=stage,variant=name,repeat=repeat,ms=start.elapsed_time(end)/16,N=run.total,M=len(candidates),K=len(ids)))
                del graphs,funcs
        write_csv(out/'samples.csv',rows);print('PREPARATION_COMPONENTS',spec['id'],len(rows),flush=True)
        del engines,source,student,run,b,a,packed,sa,ca;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),scope='固定身份组件Event；D补偿包含所有Anchor动态投影，是保守算子边界；不作为真实移动整帧速度'))


if __name__=='__main__':main()
