from common import *
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    s=scene();reference=reference_renderer(s);runtime=DeploymentRenderer(s.model,s.gaussians,s.config)
    results=[]
    for frame in [0,198,396,594,791]:
        cam=camera(s,frame)
        for ratio in [.01,.03,.1,.3,.7]:
            a,ia,sa=reference.prepare(cam,ratio_budget=ratio,view_conditioned=True,measure_time=False)
            b,ib,sb=runtime.prepare(cam,ratio_budget=ratio)
            assert len(ia)==len(ib)
            ra=render_prepared(cam,a,s.background);rb=render_prepared(cam,b,s.background)
            gt=s.ground_truth(frame)
            psnr=lambda x:-10*float(torch.log10((x.clamp(0,1)-gt).square().mean()))
            diff=(ra['render']-rb['render']).abs()
            row=dict(frame=frame,ratio=ratio,K=len(ib),ids_same=torch.equal(ia,ib),
                image_max_abs=float(diff.max()),image_mean_abs=float(diff.mean()),
                psnr_base=psnr(ra['render']),psnr_fast=psnr(rb['render']),
                base_ms=sa['prepare_ms'],fast_ms=sb['prepare_ms'])
            results.append(row);print(json.dumps(row),flush=True)
    save(ROOT/'reports/SMOKE.json',dict(results=results))


if __name__=='__main__':main()
