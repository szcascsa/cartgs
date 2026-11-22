from prep_study import *
from native_camera import NativeCameraPreparer
import copy,gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);checks=[]
    config=json.loads((HERE/'reports/FROZEN_JOINT_CANDIDATE.json').read_text())
    mode=config['projection'];mod=dp.deployment_extension()
    specs=[TUNE[0],TUNE[1]]
    for slug in specs:
        _,source=load_scene(slug);student,_,_=student_for(source,slug)
        base=make_engine(source,student,'C');run=make_engine(source,student,frustum=1,projection=mode,device=1)
        native=NativeCameraPreparer(MiniCam);frame=int(.43*len(source.scene.views));cam=native.from_source(source,frame)
        budgets=[0,1,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,32768,32769,100000]
        if args.quick:budgets=[0,1,15,16,17,31,32,33,65,257,32769]
        for budget in budgets:
            packet,ids,stats=run.prepare(cam,budget=budget);_,ref,rs=base.prepare(cam,budget=budget)
            assert torch.equal(ids,ref) and len(ids)==min(budget,stats['eligible_candidates'])
            assert all(torch.isfinite(x).all() for x in packet['inputs'].values() if isinstance(x,torch.Tensor))
            checks.append(dict(scene=slug,case='tail_capacity_K',request=budget,K=len(ids),passed=True))
        for ratio in [.01,.03,.075,.1,.2,.3,.4,.5,.6,.7,1.]:
            packet,ids,stats=run.prepare(cam,ratio_budget=ratio);_,ref,rs=base.prepare(cam,ratio_budget=ratio)
            assert torch.equal(ids,ref) and len(ids)==min(int(np.floor(ratio*stats['candidate_gaussians'])),stats['eligible_candidates'])
            checks.append(dict(scene=slug,case='ratio_unseen',request=ratio,K=len(ids),passed=True))
        saved,ids,_=run.prepare(cam,ratio_budget=.3);snapshot={k:v.clone() for k,v in saved['inputs'].items() if isinstance(v,torch.Tensor)}
        for ratio in [.1,.5,.2]:run.prepare(native.from_source(source,1),ratio_budget=ratio)
        assert all(bits_equal(x,saved['inputs'][k]) for k,x in snapshot.items())
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):other=native.from_source(source,frame);packet,selected,_=run.prepare(other,ratio_budget=.3)
        stream.synchronize();assert torch.equal(selected,ids) and all(bits_equal(x,packet['inputs'][k]) for k,x in snapshot.items())
        checks.append(dict(scene=slug,case='nondefault_stream_and_independent_output',request=0,K=len(ids),passed=True))
        for local in [False,True]:
            for compensation in [False,True]:
                packet,ids,stats=run.prepare(cam,budget=1000,view_conditioned=local,compensation=compensation)
                _,ref,_=base.prepare(cam,budget=1000,view_conditioned=local,compensation=compensation)
                assert torch.equal(ids,ref)
                checks.append(dict(scene=slug,case=f'mode_{local}_{compensation}',request=1000,K=len(ids),passed=True))
        if not args.quick:
            run._pipeline._preparation_options(0,0,0)
            packet,ids,_=run.prepare(cam,ratio_budget=.3);ref,rids,_=base.prepare(cam,ratio_budget=.3)
            assert torch.equal(ids,rids) and all(bits_equal(x,ref['inputs'][k]) for k,x in packet['inputs'].items() if isinstance(x,torch.Tensor))
            run._pipeline._preparation_options(1,mode,1)
            packet,ids,_=run.prepare(cam,ratio_budget=.3)
            assert all(bits_equal(x,packet['inputs'][k]) for k,x in snapshot.items())
            packed=run._pipeline._tensor_state();fixed=packed[5].clone();packed[5].fill_(float('nan'))
            try:run.prepare(cam,budget=1000)
            except RuntimeError as e:assert 'Nonfinite' in str(e)
            else:raise AssertionError('Projected input nonfinite was not rejected')
            packed[5].copy_(fixed);run.prepare(cam,budget=1000)
            checks.append(dict(scene=slug,case='toggle_rebuild_and_nonfinite',request=1000,K=1000,passed=True))
        c=source.scene.calibration;pose=np.eye(4);pose[:3,3]=1e6
        empty=native.create_from_pose(c['width'],c['height'],2*math.atan(c['width']/(2*c['fx'])),2*math.atan(c['height']/(2*c['fy'])),pose)
        a,ids,stats=run.prepare(empty,budget=1000);b,rids,rs=base.prepare(empty,budget=1000)
        assert len(ids)==len(rids)==0
        checks.append(dict(scene=slug,case='empty_frustum',request=1000,K=0,passed=True))
        del run,base,student,source;gc.collect();torch.cuda.empty_cache()
    # 同一FP32运算顺序的通用视锥检查，覆盖CTA尾部和超大坐标/尺度。
    counts=[0,1,15,16,17,255,256,257,1025] if not args.quick else [0,1,257]
    generator=torch.Generator(device='cuda').manual_seed(3407)
    for n in counts:
        centers=torch.randn((n,3),device='cuda',generator=generator)*20
        scale=torch.rand((n,),device='cuda',generator=generator)*30;occupied=torch.ones(n,dtype=torch.bool,device='cuda')
        eligible=torch.arange(n,device='cuda',dtype=torch.int64)%7
        pm=torch.randn((4,4),device='cuda',generator=generator).t()
        a=mod._preparation_frustum(centers,pm,scale,occupied,eligible,0);b=mod._preparation_frustum(centers,pm,scale,occupied,eligible,1)
        assert torch.equal(a[0],b[0]);checks.append(dict(scene='synthetic',case='frustum_capacity',request=n,K=int(b[0].sum()),passed=True))
    write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),binary_sha256=sha(mod.__file__)))
    print('PREPARATION_BOUNDARIES_PASS',len(checks),flush=True)


if __name__=='__main__':main()
