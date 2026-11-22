"""Tensor尾部、空候选、独立输出、Stream和切换开关的安全验收。"""
from study import *
import copy,gc
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);_,source=load_scene(TUNE[0]);student,_,_=student_for(source,TUNE[0]);cam=source.camera(len(source.scene.views)//2)
    checks=[];baseline=engine(source,student,original=True,graph=False)
    modes=[(1,1,1),(1,1,2),(1,1,4),(2,2,1),(2,2,2),(2,2,4)] if not args.quick else [(1,1,2),(2,2,2)]
    budgets=[0,1,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,1000,5000] if not args.quick else [0,1,15,16,17,63,64,65]
    modes=[(sm,cm,w,w) for sm,cm,w in modes]
    modes.append((2,2,4,2))
    for sm,cm,w,cw in modes:
        run=engine(source,student,sm,cm,w,graph=False,comp_warps=cw)
        for budget in budgets:
            packet,ids,stats=run.prepare(cam,budget=budget);_,old,oldstats=baseline.prepare(cam,budget=budget)
            assert len(ids)==len(old) and len(ids)==min(budget,stats['eligible_candidates'])
            assert all(torch.isfinite(x).all() for x in packet['inputs'].values() if isinstance(x,torch.Tensor))
            assert len(ids)<2 or bool((ids[1:]>ids[:-1]).all())
            checks.append(dict(case='strict_K_tail',selector=sm,compensator=cm,warps=w,budget=budget,K=len(ids),passed=True))
        for ratio in [.01,.03,.08,.1,.15,.25,.45,.5,.7,1.]:
            a,ids,stats=run.prepare(cam,ratio_budget=ratio);assert len(ids)==min(int(np.floor(ratio*stats['candidate_gaussians'])),stats['eligible_candidates'])
            checks.append(dict(case='ratio_unseen',selector=sm,compensator=cm,warps=w,budget=ratio,K=len(ids),passed=True))
        saved,ids,_=run.prepare(cam,ratio_budget=.3);snapshot={key:x.clone() for key,x in saved['inputs'].items() if isinstance(x,torch.Tensor)}
        other=source.camera(len(source.scene.views)//5)
        for ratio in [.1,.5,.2]:run.prepare(other,ratio_budget=ratio)
        assert all(torch.equal(saved['inputs'][key],x) for key,x in snapshot.items())
        checks.append(dict(case='output_independence',selector=sm,compensator=cm,warps=w,budget=0,K=len(ids),passed=True))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):packet,selected,_=run.prepare(cam,ratio_budget=.3)
        stream.synchronize();assert torch.equal(selected,ids)
        assert all(torch.equal(packet['inputs'][key],x) for key,x in snapshot.items())
        checks.append(dict(case='nondefault_stream',selector=sm,compensator=cm,warps=w,budget=0,K=len(ids),passed=True))
        if not args.quick:
            for kw in [dict(budget=-1),dict(ratio_budget=float('nan')),dict(ratio_budget=-.1),dict(ratio_budget=1.1)]:
                try:run.prepare(cam,**kw)
                except (ValueError,RuntimeError):pass
                else:raise AssertionError(kw)
            checks.append(dict(case='invalid_requests_rejected',selector=sm,compensator=cm,warps=w,budget=0,K=0,passed=True))
            run._pipeline._tensor_options(0,0,w,cw);a,ids,_=run.prepare(cam,ratio_budget=.3);b,old,_=baseline.prepare(cam,ratio_budget=.3)
            assert torch.equal(ids,old) and all(bits_equal(v,b['inputs'][k]) for k,v in a['inputs'].items() if isinstance(v,torch.Tensor))
            run._pipeline._tensor_options(sm,cm,w,cw);run.prepare(cam,ratio_budget=.3)
            checks.append(dict(case='toggle_repack_restore',selector=sm,compensator=cm,warps=w,budget=0,K=len(ids),passed=True))
        del run;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),binary_sha256=sha(tc.deployment_extension().__file__)));write_csv(out/'checks.csv',checks);print('TENSOR_BOUNDARIES_PASS',len(checks),flush=True)
if __name__=='__main__':main()
