"""实验完成后核验共同冻结地图；不参与候选选择。"""
import json
from pathlib import Path
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from runtime import ROOT,Scene,Network,RATIOS,photo,write_csv,save_json
from transform_field import _normalize_quaternion

@torch.no_grad()
def main():
    (ROOT/'analysis').mkdir(exist_ok=True)
    metrics=[];metadata=[];outputs=[]
    for name in ['fr1_xyz','fr2_xyz']:
        scene=Scene(name)
        occupancy=torch.bincount(scene.owners,minlength=scene.M)
        occupied=occupancy[occupancy>0].float()
        metadata.append(dict(scene=name,N=scene.N,feature_rows=scene.M,occupied_anchor_rows=int((occupancy>0).sum()),
            unused_feature_rows=int((occupancy==0).sum()),occupancy_min=float(occupied.min()),
            occupancy_median=float(occupied.median()),occupancy_max=float(occupied.max()),
            eligible=int(scene.eligible.sum()),K=scene.K,source_files=scene.hashes))
        ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda()
        lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
        ids=torch.arange(scene.N,device='cuda')
        for split,views in [('Bank',scene.bank_full),('non-Bank',scene.load_test())]:
            for v in views:
                image,radii,alpha=scene.render(ids,None,v);pred=image.clamp(0,1);gt=v['gt']
                metrics.append(dict(scene=name,split=split,path=v['path'],N=scene.N,visible=int((radii>0).sum()),
                    psnr=float(-10*torch.log10((pred-gt).square().mean())),ssim=float(ssim(pred[None],gt[None])),
                    lpips=float(lpips(pred[None],gt[None])),photo=float(photo(image,gt))))
                ssim.reset();lpips.reset()
        phase='confirm' if name=='fr1_xyz' else 'external'
        for path in (ROOT/'results'/name/phase).glob('*/*/complete.json'):
            info=json.loads(path.read_text());net=Network(scene,info['config'],info['seed']).eval()
            payload=torch.load(info['checkpoint'],map_location='cuda');net.load_state_dict(payload['state_dict']);del payload
            for ratio in RATIOS:
                p,h,logits=net(ratio);margin=logits.double()-torch.logit(scene.uniform.double());margin[~scene.eligible]=-torch.inf
                order=torch.argsort(margin,descending=True,stable=True)
                for policy,ids in [('same_k',order[:scene.K[ratio]]),('cached',torch.nonzero(p>scene.uniform).flatten())]:
                    if not len(ids):continue
                    residual=net.compensate(h,ids,ratio)
                    xyz=scene.g.get_xyz[ids]+residual[:,:3]
                    scales=scene.g.get_scaling[ids]*residual[:,3:6].exp()
                    ident=torch.zeros_like(residual[:,6:10]);ident[:,0]=1
                    rotation=_normalize_quaternion(ident+residual[:,6:10])
                    dc=scene.g.get_features_dc[ids].squeeze(1)+residual[:,11:14]
                    alpha=(scene.g._opacity[ids]+residual[:,10:11]).sigmoid()
                    finite=torch.cat([residual,xyz,scales,rotation,dc,alpha],1).isfinite().all(1)
                    displacement=residual[:,:3].norm(dim=1)
                    outputs.append(dict(scene=name,config=info['config']['id'],seed=info['seed'],ratio=ratio,policy=policy,
                        selected=len(ids),nonfinite_activated_rows=int((~finite).sum()),probabilities_finite=bool(p.isfinite().all()),
                        xyz_delta_p50=float(displacement.median()),xyz_delta_p90=float(torch.quantile(displacement,.9)),
                        xyz_delta_max=float(displacement.max()),logscale_delta_max=float(residual[:,3:6].abs().max()),
                        scale_max=float(scales.max()),rotation_angle_p90=float(torch.quantile(2*torch.acos(rotation[:,0].abs().clamp(0,1)),.9)),
                        opacity_mean=float(alpha.mean()),shdc_delta_abs_max=float(residual[:,11:14].abs().max())))
            del net
        assert scene.unchanged()
        del scene,ssim,lpips;torch.cuda.empty_cache()
    write_csv(ROOT/'analysis/source_full_per_view.csv',metrics)
    save_json(ROOT/'analysis/source_topology.json',metadata)
    write_csv(ROOT/'analysis/network_output_diagnostics.csv',outputs)
    print('SOURCE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
