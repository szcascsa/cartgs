"""解码Graph的逐张量/全参数梯度gate，随后才测真实Renderer步骤。"""
import copy
import json
import torch
from baseline_profile import ROOT,r,fixture,measure
from implementation import Execution
from decoded_graph import DecodedExecution,decode
from parity import compare


def main():
    torch.backends.cuda.matmul.allow_tf32=False;sc,f,net,payload=fixture();ref=Execution(net,'reference')
    candidate_net=r.make(f,net.cfg,5100);candidate_net.load_state_dict(payload['state_dict']);optimized=DecodedExecution(candidate_net,sc.render_bases)
    rows=[];perf=[]
    for ratio in r.rt.RATIOS:
        net.load_state_dict(payload['state_dict']);candidate_net.load_state_dict(payload['state_dict'])
        ids=f.sets[ratio,'same_k'];K=len(ids);delta=ref(ratio,ids)
        original=(delta,*decode(delta,ids,(*sc.render_bases[:4],sc.render_bases[5])));candidate=optimized(ratio,ids)
        for name,a,b in zip(['delta','xyz','scale','rotation','dc','opacity'],original,candidate):compare(a,b,name,rows,'decoded_graph',K)
        torch.manual_seed(91);vjp=tuple(torch.randn_like(value)*.001 for value in original)
        ga=torch.autograd.grad(original,tuple(net.parameters()),vjp);gb=torch.autograd.grad(candidate,tuple(candidate_net.parameters()),vjp)
        for (name,p),a,b in zip(net.named_parameters(),ga,gb):compare(a,b,'gradient/'+name,rows,'decoded_graph',K,True)
        with torch.no_grad():
            a=sc.render(ids,ref(ratio,ids),sc.train[0])[0];b=optimized.render(sc,ratio,ids,sc.train[0])
            compare(a,b,'renderer',rows,'decoded_graph',K);compare(r.rt.photo(a,sc.train[0]['gt']),r.rt.photo(b,sc.train[0]['gt']),'photo',rows,'decoded_graph',K)
        r.rt.write_csv(ROOT/'decoded_parity.csv',rows)
        assert all(row['pass_'] for row in rows),[row for row in rows if not row['pass_']]
        opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
        candidate_opt=torch.optim.Adam(candidate_net.parameters(),lr=.001,eps=1e-15,fused=True);candidate_opt.load_state_dict(copy.deepcopy(payload['optimizer']))
        def step_reference():
            opt.zero_grad(set_to_none=True);loss=r.rt.photo(sc.render(ids,ref(ratio,ids),sc.train[0])[0],sc.train[0]['gt']);loss.backward();opt.step()
        def step_optimized():
            candidate_opt.zero_grad(set_to_none=True);loss=r.rt.photo(optimized.render(sc,ratio,ids,sc.train[0]),sc.train[0]['gt']);loss.backward();candidate_opt.step()
        for method,fn in [('reference',step_reference),('decoded_graph',step_optimized)]:
            net.load_state_dict(payload['state_dict']);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
            candidate_net.load_state_dict(payload['state_dict']);candidate_opt.load_state_dict(copy.deepcopy(payload['optimizer']))
            perf.append(dict(method=method,ratio=ratio,K=K,**measure(fn,35)))
        r.rt.write_csv(ROOT/'decoded_probe_performance.csv',perf);print('DECODED_PROBE',K,flush=True)
    (ROOT/'decoded_probe_complete.json').write_text(json.dumps(dict(parity_rows=len(rows),all_pass=all(row['pass_'] for row in rows)),indent=2))


if __name__=='__main__':main()
