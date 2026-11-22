from common import *
from deployment_runtime import DeploymentRenderer,deployment_extension
from baseline import previous_runtime
from large_scene import LARGE
import gc


@torch.no_grad()
def main():
    ext=deployment_extension();results=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        model=previous_runtime(s);b=model._base
        # Measure the actual persistent prefix and a warm initialization kernel.
        ext._selector_prefix(b.selector_static,model._selector[0]);torch.cuda.synchronize()
        retained=torch.cuda.memory_allocated()
        begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start=time.perf_counter();begin.record()
        prefix=ext._selector_prefix(b.selector_static,model._selector[0])
        end.record();end.synchronize()
        row=dict(group=group,N=model.total,A=b.anchors.centers.shape[0],prefix_bytes=prefix.nbytes,
            allocated_bytes=torch.cuda.memory_allocated()-retained,
            initialization_gpu_ms=begin.elapsed_time(end),initialization_wall_ms=(time.perf_counter()-start)*1000)
        assert prefix.nbytes==model.total*128
        del prefix,model,b;gc.collect();torch.cuda.empty_cache()
        costs=[]
        for enabled in [False,True]:
            before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            run=DeploymentRenderer(s.model,s.gaussians,s.config,
                selector_static_prefix=enabled,visible_view=enabled,small_compensation_tile=enabled)
            fresh=torch.cuda.memory_allocated()-before
            captures=[]
            for k in [1,128,767,768,1000,1024,1025,5000]:
                cam=camera(s,0);torch.cuda.synchronize();start=time.perf_counter()
                _,_,stats=run.prepare(cam,budget=k)
                captures.append(dict(k=k,first_request_ms=(time.perf_counter()-start)*1000,graph_count=stats["graph_count"]))
            torch.cuda.synchronize();warm=torch.cuda.memory_allocated()-before
            costs.append(dict(enabled=enabled,fresh_bytes=fresh,warm_bytes=warm,captures=captures,
                peak_extra_bytes=torch.cuda.max_memory_allocated()-before))
            del run;gc.collect();torch.cuda.empty_cache()
        row['instances']=costs;results.append(row)
        save(ROOT/'reports/SPACE_COST.json',dict(passed=True,results=results))
        print(row,flush=True)
        del s;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
