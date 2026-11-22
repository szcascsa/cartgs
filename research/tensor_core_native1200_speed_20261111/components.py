"""两网络单独CUDA Event；同一候选/选择身份；绝不并入光栅耗时。"""
from helpers import *


def capture(function):
    for _ in range(3):out=function()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):out=function()
    return graph,out


@torch.no_grad()
def main():
    root=HERE/'reports/components';root.mkdir(exist_ok=False)
    config,d,cm,dm=configure();torch.set_num_threads(2);rows=[];checks=[]
    for slug in SLUGS:
        scene,source=p.load_scene(slug);student,_,_=p.student_for(source,slug)
        runs={name:engine(source,student,name) for name in ['A','C','D']}
        a=runs['A'];b=a._base;anchors=b.anchors
        camera=Camera(source,'A',dm)
        frames=frames_for(source);frames=[frames[i][1] for i in [0,1,2,12,14,17]]
        prefix=a._ext._selector_prefix(b.selector_static,a._selector[0])
        packed={'C':cm._tensor_pack(a._selector,a._compensator,a._view_weight,2,2),
                'D':runs['D']._pipeline._tensor_state()}
        for frame in frames:
            cam=camera.create(frame)
            visible=p.anchor_frustum_mask(anchors.centers,anchors.offset_scales,cam.full_proj_transform,a._reference.occupied)
            candidates=torch.nonzero(visible[b.owners],as_tuple=False).flatten()
            view=a._ext.view(anchors.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),cam.world_view_transform,
                cam.full_proj_transform,b.model.selector_view_scene_scale,float(cam.image_width)/cam.image_height,cam.znear)
            for ratio in RATIOS:
                _,ids,stats=a.prepare(cam,ratio_budget=ratio)
                weights=[1.]+[-1.]*(len(a.knots)-1)
                for k in range(1,len(a.knots)):
                    if ratio<=a.knots[k-1]+1e-8:break
                    weights[k]=min(1.,(ratio-a.knots[k-1])/(a.knots[k]-a.knots[k-1]))
                lo,hi=b.spatial.ratio_encoding_values;x=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
                code=[x,math.sin(math.pi*x/2),math.cos(math.pi*x/2)]
                sa=(b.selector_static,candidates,b.owners,view,a._reference.noise,a._selector,ratio,True,b.joint_schema)
                ca=(b.offset,ids,b.owners,b.reference,b.tables,a._compensator,b.compensated_master,view,a._view_weight,
                    weights,code,True,b.joint_schema,hasattr(b.spatial,'softplus_scale_schema'))
                for stage in ['selector','compensator']:
                    if stage=='selector':
                        functions={'A':lambda:a._ext._selector_prefixed(*sa,True,prefix),
                            'C':lambda:cm._tensor_selector(*sa,prefix,packed['C'],2,4),
                            'D':lambda:dm._tensor_selector(*sa,prefix,packed['D'],2,4)}
                    else:
                        functions={'A':lambda:a._ext._compensate_tiled(*ca,16),
                            'C':lambda:cm._tensor_compensate(*ca,packed['C'],2,2,b.empty),
                            'D':lambda:dm._tensor_compensate(*ca,packed['D'],2,2,b.empty)}
                    graphs={};reference=functions['A']()
                    for name,function in functions.items():
                        result=function();assert int(result[2])==0 and all(torch.isfinite(t).all() for t in result[:2])
                        checks.append(dict(scene=scene['id'],frame=frame,ratio=ratio,stage=stage,variant=name,
                            M=len(candidates),K=len(ids),max_output_error=float((result[0]-reference[0]).abs().max()) if result[0].numel() else 0.,
                            max_secondary_error=float((result[1]-reference[1]).abs().max()) if result[1].numel() else 0.))
                        graphs[name]=capture(function)
                    for round_index in range(2):
                        for repeat in range(8):
                            for name in p.rotate(list(graphs),repeat+round_index):
                                ms,_=event(graphs[name][0].replay)
                                rows.append(dict(scene=scene['id'],frame=frame,ratio=ratio,stage=stage,variant=name,
                                    round=round_index,repeat=repeat,N=a.total,M=len(candidates),K=len(ids),ms=ms/16))
                    del graphs,functions,reference,result
            print('COMPONENT',scene['id'],frame,len(rows),flush=True)
        write('samples.csv',rows,root);write('checks.csv',checks,root)
        del runs,source,student,a,b,anchors,packed,sa,ca;gc.collect();torch.cuda.empty_cache()
    save('VALIDATION.json',dict(passed=True,samples=len(rows),checks=len(checks),same_input_ids=True,
        scope='16 repeated fixed-input graph calls / CUDA Event; Selector includes fixed prefix read and status; Comp includes decode/pack; D also all-Anchor projection',
        not_pipeline=True,not_pure_mma=True,not_rendering=True),root)
    print('COMPONENTS_COMPLETE',len(rows),len(checks),flush=True)


if __name__=='__main__':main()
