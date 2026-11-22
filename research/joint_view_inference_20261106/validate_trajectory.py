"""792帧34条件的实际选点和逐像素回归，不保存渲染图片。"""
from common import *
import csv


@torch.no_grad()
def main():
    s=scene();reference=reference_renderer(s);candidate=FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda_graph')
    prior=Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/evaluation/reports/PER_VIEW.csv')
    prior_hash=sha(prior)
    with prior.open() as f:expected={(int(v['frame']),v['case']):v for v in csv.DictReader(f) if v['group']=='both_li'}
    assert len(expected)==792*34
    output=ROOT/'evaluation/TRAJECTORY_PARITY.csv';started=time.monotonic();count=0;checks=0
    with output.open('w',newline='') as f:
        fields=['frame','case','K','candidate_N','selection_sha256','image_bitwise','radii_bitwise','attributes_bitwise']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for frame in range(792):
            cam=camera(s,frame)
            for case in [c for n,c in expected if n==frame]:
                prior_row=expected[frame,case]
                if case=='full':
                    a=b=s.full_data;ia=ib=s.full_ids;sa=sb=dict(candidate_gaussians=len(ia))
                else:
                    scope,value=case.split('_');value=float(value)
                    kwargs=(dict(global_ratio=value) if scope=='global' else dict(ratio_budget=value) if scope=='view' else
                            dict(budget=int(prior_row['requested_K'])))
                    kwargs['view_conditioned']=scope!='global'
                    a,ia,sa=reference.prepare(cam,**kwargs)
                    b,ib,sb=candidate.prepare(cam,measure_time=False,**kwargs)
                assert torch.equal(ia,ib) and len(ia)==int(prior_row['K'])
                digest=hashlib.sha256(ib.cpu().numpy().tobytes()).hexdigest();assert digest==prior_row['selection_sha256']
                for key,tensor in a['inputs'].items():
                    if tensor is None:assert b['inputs'][key] is None
                    else:
                        assert torch.equal(tensor.contiguous().reshape(-1).view(torch.uint8),b['inputs'][key].contiguous().reshape(-1).view(torch.uint8)),(frame,case,key)
                        checks+=1
                ra=render_prepared(cam,a,s.background,return_radii=True);rb=render_prepared(cam,b,s.background,return_radii=True)
                assert torch.equal(ra['render'].view(torch.int32),rb['render'].view(torch.int32)),(frame,case,'image')
                assert torch.equal(ra['radii'],rb['radii']),(frame,case,'radii')
                checks+=3;count+=1
                writer.writerow(dict(frame=frame,case=case,K=len(ib),candidate_N=sb['candidate_gaussians'],selection_sha256=digest,
                    image_bitwise=True,radii_bitwise=True,attributes_bitwise=True))
            if frame%20==0 or frame==791:
                f.flush();state=dict(state='running',frame=frame+1,frames=792,cases=count,checks=checks,seconds=time.monotonic()-started)
                save(ROOT/'reports/TRAJECTORY_STATUS.json',state);print(json.dumps(state),flush=True)
    assert count==26928 and sha(prior)==prior_hash
    save(ROOT/'reports/TRAJECTORY_VALIDATION.json',dict(passed=True,frames=792,cases=count,comparisons=checks,images_saved=0,
        prior_metrics=str(prior),prior_sha256=sha(prior),metrics_scope='逐像素与原路径一致；不重复运行SSIM/LPIPS，不用PSNR近似代替等价',
        seconds=time.monotonic()-started))
    save(ROOT/'reports/TRAJECTORY_STATUS.json',dict(state='complete',frames=792,cases=count,checks=checks))


if __name__=='__main__':main()
