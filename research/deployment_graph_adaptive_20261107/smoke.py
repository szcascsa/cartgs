from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
import ctypes
from large_scene import LARGE


class Frame(ctypes.Structure):
    _fields_=[(n,ctypes.c_int64) for n in ['n','visible','available','budget','k']]+[
        ('ratio',ctypes.c_double),('log_ratio',ctypes.c_float),('weights',ctypes.c_float*6),
        ('ratio_code',ctypes.c_float*3),('status',ctypes.c_int),('has_ratio',ctypes.c_int)]


def control(runtime):
    return Frame.from_buffer_copy(runtime._pipeline._control().cpu().numpy().tobytes())


@torch.no_grad()
def main():
    group=sys.argv[1] if len(sys.argv)>1 else 'tum'
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config)
    rows=[]
    for frame in np.linspace(0,len(s.views)-1,5,dtype=int):
        frame=int(frame)
        for request in [dict(ratio_budget=r) for r in RATIOS]+[dict(budget=k) for k in [0,1,2000,5000,100000]]:
            cam=camera(s,frame);x,ix,sx=a.prepare(cam,**request);y,iy,sy=b.prepare(cam,**request)
            dc=control(b)
            assert sx['ratio']==sy['ratio'] and sx['candidate_gaussians']==dc.n and len(iy)==dc.k
            exact_ids=torch.equal(ix,iy)
            error={k:float((v-y['inputs'][k]).abs().max()) if v.numel() else 0 for k,v in x['inputs'].items() if isinstance(v,torch.Tensor)}
            row=dict(frame=frame,request=request,ids_equal=exact_ids,max_attributes=error,
                log_ratio_error=abs(dc.log_ratio-float(np.float32(np.log(np.float32(dc.ratio))))) if dc.has_ratio else 0)
            rows.append(row);save(ROOT/'reports'/f'SMOKE_{group}.json',dict(passed=False,rows=rows))
            assert exact_ids,row
        print('SMOKE',frame,len(rows),flush=True)
    save(ROOT/'reports'/f'SMOKE_{group}.json',dict(passed=True,cases=len(rows),rows=rows))


if __name__=='__main__':main()
