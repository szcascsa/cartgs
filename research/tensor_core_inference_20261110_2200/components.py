"""相同候选/选中身份的网络组件CUDA Event微测；与完整准备墙钟分开。"""
from study import *
import gc
def graph_for(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):out=fn()
    return graph,out
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];checks=[];mod=tc.deployment_extension()
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug);run=engine(source,student,original=True);b=run._base;a=b.anchors;ext=run._ext
        cam=source.camera(int(.43*len(source.scene.views)))
        visible=anchor_frustum_mask(a.centers,a.offset_scales,cam.full_proj_transform,run._reference.occupied)
        candidates=torch.nonzero(visible[b.owners],as_tuple=False).flatten()
        view=ext.view(a.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),cam.world_view_transform,cam.full_proj_transform,b.model.selector_view_scene_scale,float(cam.image_width)/cam.image_height,cam.znear)
        prefix=ext._selector_prefix(b.selector_static,run._selector[0])
        packed={mode:mod._tensor_pack(run._selector,run._compensator,run._view_weight,mode,mode) for mode in [1,2]}
        for ratio in PRIMARY:
            packet,ids,_=run.prepare(cam,ratio_budget=ratio);weights=[1.]+[-1.]*(len(run.knots)-1)
            for j in range(1,len(run.knots)):
                if ratio<=run.knots[j-1]+1e-8:break
                weights[j]=min(1.,(ratio-run.knots[j-1])/(run.knots[j]-run.knots[j-1]))
            lo,hi=b.spatial.ratio_encoding_values;t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
            code=[t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)]
            selector_args=(b.selector_static,candidates,b.owners,view,run._reference.noise,run._selector,ratio,True,b.joint_schema)
            comp_args=(b.offset,ids,b.owners,b.reference,b.tables,run._compensator,b.compensated_master,view,run._view_weight,weights,code,True,b.joint_schema,hasattr(b.spatial,'softplus_scale_schema'))
            original_selector=ext._selector_prefixed(*selector_args,True,prefix);original_comp=ext._compensate_tiled(*comp_args,16)
            for stage in ['selector','compensator']:
                funcs={'original':(lambda:ext._selector_prefixed(*selector_args,True,prefix)) if stage=='selector' else (lambda:ext._compensate_tiled(*comp_args,16))}
                for mode in [1,2]:
                    for warps in [1,2,4]:
                        if stage=='selector':funcs[f'm{mode}_w{warps}']=lambda mode=mode,warps=warps:mod._tensor_selector(*selector_args,prefix,packed[mode],mode,warps)
                        else:funcs[f'm{mode}_w{warps}']=lambda mode=mode,warps=warps:mod._tensor_compensate(*comp_args,packed[mode],mode,warps,b.empty)
                graphs={}
                for name,fn in funcs.items():
                    value=fn();ref=original_selector if stage=='selector' else original_comp;assert int(value[2])==0
                    checks.append(dict(scene=slug,ratio=ratio,stage=stage,variant=name,M=len(candidates),K=len(ids),max_primary_error=float((value[0]-ref[0]).abs().max()),max_secondary_error=float((value[1]-ref[1]).abs().max())))
                    graphs[name]=graph_for(fn)
                for repeat in range(30):
                    for name in rotate(list(graphs),repeat):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();graphs[name][0].replay();end.record();end.synchronize()
                        rows.append(dict(scene=slug,ratio=ratio,stage=stage,variant=name,repeat=repeat,M=len(candidates),K=len(ids),ms=begin.elapsed_time(end)/16))
                del graphs,funcs,value;gc.collect();torch.cuda.empty_cache()
        write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);print('TC_COMPONENT',slug,len(rows),flush=True)
        del source,student,run,b,a,packet,packed,original_selector,original_comp,selector_args,comp_args;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),binary_sha256=sha(mod.__file__),scope='CUDA Event：16次固定输入网络入口Graph，包含原有状态清零；不代表每帧完整准备'))
if __name__=='__main__':main()
