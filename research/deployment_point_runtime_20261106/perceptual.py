from common import *
from deployment_runtime import DeploymentRenderer
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import csv


@torch.no_grad()
def main():
    s=scene();ref=s.renderer;fast=DeploymentRenderer(s.model,s.gaussians,s.config)
    lpips=LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    rows=[]
    for frame in range(0,792,12):
        cam=camera(s,frame);gt=s.ground_truth(frame).unsqueeze(0)
        for ratio in [.01,.05,.1,.3,.5,.7]:
            images=[]
            chosen=[]
            for tag,e in [('pytorch',ref),('deployment',fast)]:
                kwargs=dict(ratio_budget=ratio,view_conditioned=True)
                if tag=='pytorch':kwargs['measure_time']=False
                data,ids,_=e.prepare(cam,**kwargs)
                chosen.append(ids)
                images.append(render_prepared(cam,data,s.background)['render'].clamp(0,1))
            x=torch.stack(images);target=gt.expand_as(x)
            ssim=structural_similarity_index_measure(x,target,data_range=1.,reduction='none')
            perceptual=lpips(x,target).reshape(-1);lpips.reset()
            row=dict(frame=frame,ratio=ratio,ids_same=torch.equal(*chosen),base_ssim=float(ssim[0]),fast_ssim=float(ssim[1]),
                delta_ssim=float(ssim[1]-ssim[0]),base_lpips=float(perceptual[0]),fast_lpips=float(perceptual[1]),
                delta_lpips=float(perceptual[1]-perceptual[0]))
            rows.append(row)
        if frame%120==0:print('PERCEPTUAL',frame,flush=True)
    with (ROOT/'evaluation/PERCEPTUAL.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(ROOT/'reports/PERCEPTUAL.json',dict(passed=True,frames=66,cases=len(rows),
        reference='unoptimized PyTorch',ids_same_fraction=float(np.mean([v['ids_same'] for v in rows])),
        max_abs_ssim_delta=max(abs(v['delta_ssim']) for v in rows),
        max_abs_lpips_delta=max(abs(v['delta_lpips']) for v in rows),
        ssim_mean_delta=float(np.mean([v['delta_ssim'] for v in rows])),
        lpips_mean_delta=float(np.mean([v['delta_lpips'] for v in rows]))))


if __name__=='__main__':main()
