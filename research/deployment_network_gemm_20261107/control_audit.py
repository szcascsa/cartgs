from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from smoke import control
import ctypes
import math
import random
from large_scene import LARGE


@torch.no_grad()
def main():
    group=sys.argv[1] if len(sys.argv)>1 else 'tum'
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config)
    libm=ctypes.CDLL('libm.so.6');libm.logf.argtypes=[ctypes.c_float];libm.logf.restype=ctypes.c_float
    rng=random.Random(923);rows=[]
    requests=[dict(ratio_budget=r) for r in RATIOS]+[dict(ratio_budget=10**rng.uniform(-5,0)) for _ in range(150)]
    requests += [dict(budget=rng.randint(1,b.total*2)) for _ in range(150)]
    for i,request in enumerate(requests):
        frame=(i*73+19)%len(s.views)
        cam=camera(s,frame);x,ix,sx=a.prepare(cam,**request);y,iy,sy=b.prepare(cam,**request);f=control(b)
        assert sx['ratio']==sy['ratio'] and len(ix)==len(iy)
        ratio=f.ratio;weights=[1.]+[-1.]*(len(b.knots)-1)
        for j in range(1,len(weights)):
            if ratio<=b.knots[j-1]+1e-8:break
            weights[j]=min(1.,(ratio-b.knots[j-1])/(b.knots[j]-b.knots[j-1]))
        lo,hi=b._base.spatial.ratio_encoding_values
        t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
        expected=np.asarray([libm.logf(ratio),*weights,*([0.]*(6-len(weights))),t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)],dtype=np.float32)
        actual=np.asarray([f.log_ratio,*f.weights,*f.ratio_code],dtype=np.float32)
        exact_ids=torch.equal(ix,iy)
        attr_exact=exact_ids and all(torch.equal(v.contiguous().view(torch.uint8),y['inputs'][k].contiguous().view(torch.uint8))
            for k,v in x['inputs'].items() if isinstance(v,torch.Tensor))
        row=dict(request=request,N=f.n,K=f.k,ids_same=exact_ids,attributes_bitwise=attr_exact,
            controls_bitwise=bool(np.array_equal(expected.view('u4'),actual.view('u4'))),
            control_max_abs=float(np.max(np.abs(expected-actual))))
        if not attr_exact:
            px=render_prepared(cam,x,s.background)['render'];py=render_prepared(cam,y,s.background)['render']
            gt=s.ground_truth(frame)
            psnr=lambda z:-10*float(torch.log10((z.clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
            row.update(pixel_max_abs=float((px-py).abs().max()),psnr_delta=psnr(py)-psnr(px))
            assert row['psnr_delta']>=-.1,row
        rows.append(row)
    save(ROOT/'reports'/f'CONTROL_AUDIT_{group}.json',dict(passed=True,cases=len(rows),
        control_mismatches=sum(not v['controls_bitwise'] for v in rows),
        active_control_mismatches=sum(not v['controls_bitwise'] and v['K']>0 for v in rows),
        ids_mismatches=sum(not v['ids_same'] for v in rows),attribute_mismatches=sum(not v['attributes_bitwise'] for v in rows),
        max_control_abs=max(v['control_max_abs'] for v in rows),rows=rows))
    print('CONTROL_AUDIT_PASS',len(rows),flush=True)


if __name__=='__main__':main()
