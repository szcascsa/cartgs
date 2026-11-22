"""同一Core的Inductor控制；FP32、TF32关闭，输入Fourier不交给近似sin编译。"""
import json
import time
import torch
from baseline_profile import ROOT,fixture,measure
from implementation import Core,Execution
from parity import compare


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    sc,f,net,payload=fixture();ex=Execution(net);ids=f.sets[.01,'same_k'];x=ex.pack(ids).detach().requires_grad_()
    scale=f.features.anchor_scale[ids].detach().requires_grad_();condition=ex.condition(.01);core=Core(net)
    import torch._inductor.config as config
    config.compile_threads=1
    start=time.perf_counter();compiled=torch.compile(core,fullgraph=True,options={'triton.cudagraphs':False})
    rows=[]
    try:
        out=compiled(x,scale,condition);torch.cuda.synchronize();cold_ms=(time.perf_counter()-start)*1000
        compare(core(x,scale,condition),out,'delta',rows,'inductor',len(ids))
        vjp=torch.randn_like(out);parameters=tuple(net.parameters())+(x,scale)
        a=torch.autograd.grad(core(x,scale,condition),parameters,vjp);b=torch.autograd.grad(out,parameters,vjp)
        for i,(aa,bb) in enumerate(zip(a,b)):compare(aa,bb,'gradient_'+str(i),rows,'inductor',len(ids),True)
        result=dict(cold_compile_ms=cold_ms,parity=all(v['pass_'] for v in rows),rows=rows,
            reference_core=measure(lambda:core(x,scale,condition)),compiled_core=measure(lambda:compiled(x,scale,condition)),
            scope='Core only at K68; not end-to-end; input encoding outside; no performance recommendation without all-shape gates')
    except Exception as error:result=dict(error=str(error),parity=False,scope='experimental compiler probe; not recommended')
    (ROOT/'compiler_probe.json').write_text(json.dumps(result,indent=2));print('COMPILER_PROBE',result.get('parity'),result.get('error',''),flush=True)


if __name__=='__main__':main()
