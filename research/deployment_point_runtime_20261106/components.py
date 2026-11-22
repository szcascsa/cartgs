from common import *
from deployment_runtime import DeploymentRenderer
import statistics
import math


@torch.no_grad()
def main():
    s=scene();ref=reference_renderer(s);fast=DeploymentRenderer(s.model,s.gaussians,s.config)
    b=ref.inference;f=fast._base;cam=camera(s,396);code=b.view_code(cam);fast_code=fast.view_code(cam)
    rows,_,available=b.candidates(b.frustum(cam.full_proj_transform,ref.occupied));reports=[]
    for ratio in [.01,.05,.1,.3,.5,.7]:
        logits=b.logits(rows,ratio,code,True);scores=b.scores(logits,ref.noise,rows)
        ids=rows[b.select(scores,b.eligible[rows],min(math.floor(len(rows)*ratio),available))]
        controls=b.controls(ratio);weights,ratio_code=fast._controls(ratio)
        def base_selector():return b.scores(b.logits(rows,ratio,code,True),ref.noise,rows)
        def fast_selector():return fast._ext.selector(f.selector_static,rows,f.owners,fast_code,fast._reference.noise,
            fast._selector,ratio,True,f.joint_schema)
        def base_comp():return b._compute(ids,controls,True,code,True)
        def fast_comp():return fast._ext.compensate(f.offset,ids,f.owners,f.reference,f.tables,fast._compensator,
            f.compensated_master,fast_code,fast._view_weight,weights,ratio_code,True,f.joint_schema,True)
        functions={'selector':{'v3':base_selector,'deployment':fast_selector},
                   'compensation_decode':{'v3':base_comp,'deployment':fast_comp},
                   'two_networks_decode':{'v3':lambda:(base_selector(),base_comp()),
                                          'deployment':lambda:(fast_selector(),fast_comp())}}
        for stage,backends in functions.items():
            values={tag:dict(host=[],event=[]) for tag in backends}
            for fn in backends.values():
                for _ in range(16):fn()
            for i in range(80):
                order=list(backends)
                if i%2:order.reverse()
                for tag in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();t=time.perf_counter();start.record()
                    backends[tag]();end.record();end.synchronize()
                    values[tag]['host'].append((time.perf_counter()-t)*1000)
                    values[tag]['event'].append(start.elapsed_time(end))
            result=dict(ratio=ratio,N=len(rows),K=len(ids),stage=stage,summary={tag:{key:dict(median_ms=statistics.median(v),
                p90_ms=float(np.percentile(v,90)),p99_ms=float(np.percentile(v,99)),under2ms=float(np.mean(np.array(v)<2)))
                for key,v in samples.items()} for tag,samples in values.items()},raw_ms=values)
            reports.append(result);print('COMPONENT',ratio,stage,{tag:values['event']['median_ms'] for tag,values in result['summary'].items()},flush=True)
    save(ROOT/'reports/COMPONENTS.json',dict(results=reports,excludes='candidate construction, selection sort, finite readback, rasterizer',
        event_includes='CPU submission gaps and concurrent GPU interference',fixed_input=True))


if __name__=='__main__':main()
