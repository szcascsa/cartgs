from common import *
from deployment_runtime import DeploymentRenderer
import csv
import statistics


@torch.no_grad()
def main():
    s=scene();ref=reference_renderer(s);fast=DeploymentRenderer(s.model,s.gaussians,s.config)
    requests=[('r'+str(r),dict(ratio_budget=r)) for r in RATIOS]
    requests += [('k'+str(k),dict(budget=k)) for k in [1000,2000,3000,5000,10000]]
    rows=[];started=time.monotonic()
    fields=['frame','condition','N','K','ids_same','id_retention','base_psnr','fast_psnr','delta_psnr',
            'image_max_abs','image_mean_abs','image_pair_psnr','pixel_over_1e4_fraction',
            'xyz_max_abs','scale_max_abs','rotation_max_abs','radii_same','finite']
    with (ROOT/'evaluation/PER_FRAME.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for frame in range(792):
            cam=camera(s,frame);gt=s.ground_truth(frame)
            for name,request in requests:
                a,ia,sa=ref.prepare(cam,**request,view_conditioned=True,measure_time=False)
                b,ib,sb=fast.prepare(cam,**request)
                assert len(ia)==len(ib) and sa['candidate_gaussians']==sb['candidate_gaussians']
                ra=render_prepared(cam,a,s.background,return_radii=True)
                rb=render_prepared(cam,b,s.background,return_radii=True)
                x,y=ra['render'],rb['render'];diff=(x-y).abs()
                psnr=lambda image:-10*float(torch.log10((image.clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
                pa,pb=psnr(x),psnr(y)
                same=torch.equal(ia,ib)
                if same:retention=1.
                else:
                    pos=torch.searchsorted(ia,ib).clamp_max(max(0,len(ia)-1))
                    retention=float((ia[pos]==ib).float().mean()) if len(ia) else 1.
                row=dict(frame=frame,condition=name,N=sa['candidate_gaussians'],K=len(ib),ids_same=same,
                    id_retention=retention,base_psnr=pa,fast_psnr=pb,delta_psnr=pb-pa,
                    image_max_abs=float(diff.max()),image_mean_abs=float(diff.mean()),
                    image_pair_psnr=-10*float(torch.log10(diff.square().mean().clamp_min(1e-30))),
                    pixel_over_1e4_fraction=float((diff>1e-4).float().mean()),radii_same=torch.equal(ra['radii'],rb['radii']),
                    finite=bool(torch.isfinite(y).all()),
                    **{label:float((a['inputs'][key]-b['inputs'][key]).abs().max()) if same and len(ia) else None
                       for label,key in [('xyz_max_abs','means3D'),('scale_max_abs','scales'),('rotation_max_abs','rotations')]})
                assert row['finite'];writer.writerow(row);rows.append(row)
            if frame%20==0 or frame==791:
                f.flush();save(ROOT/'reports/QUALITY_STATUS.json',dict(state='running',frames=frame+1,cases=len(rows),elapsed_s=time.monotonic()-started))
                print('QUALITY',frame+1,len(rows),flush=True)
    summaries=[]
    for name,_ in requests:
        values=[v for v in rows if v['condition']==name];delta=[v['delta_psnr'] for v in values]
        summary=dict(condition=name,frames=len(values),base_psnr=statistics.mean(v['base_psnr'] for v in values),
            fast_psnr=statistics.mean(v['fast_psnr'] for v in values),mean_delta=statistics.mean(delta),
            p01_delta=float(np.percentile(delta,1)),worst_delta=min(delta),ids_same_fraction=statistics.mean(v['ids_same'] for v in values),
            minimum_id_retention=min(v['id_retention'] for v in values),max_image_abs=max(v['image_max_abs'] for v in values),
            median_image_pair_psnr=statistics.median(v['image_pair_psnr'] for v in values))
        summary['passed']=summary['mean_delta']>=-.01 and summary['p01_delta']>=-.05 and summary['worst_delta']>=-.1
        summaries.append(summary)
    save(ROOT/'reports/QUALITY.json',dict(passed=all(v['passed'] for v in summaries),frames=792,cases=len(rows),summaries=summaries,
        acceptance=dict(mean_loss_db=.01,p99_view_loss_db=.05,worst_view_loss_db=.1),images_saved=0))
    save(ROOT/'reports/QUALITY_STATUS.json',dict(state='complete',frames=792,cases=len(rows)))


if __name__=='__main__':main()
