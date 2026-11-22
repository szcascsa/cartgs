"""固定终点评价；Cached 与相同 K 分开，Bank 与 non-Bank 分开。"""
from pathlib import Path
import csv,json,sys
import cv2,numpy as np,torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from evo.tools import file_interface

root=Path('/home/prmi/zwj/anchor-si-20261024');source=root/'source';data=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
base=Path('/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply')
sys.path.insert(0,str(source/'python'))
from selector import load_selector,load_selector_metadata,compensated_attributes,mask_inactive_slots
from gaussian_model import GaussianModel
from renderer import prepare_render_data,_make_rasterizer
from utils import MiniCam,focal2fov
from diff_gaussian_rasterization import _C

torch.set_num_threads(2);torch.set_grad_enabled(False)
protocol=json.loads((root/'protocol.json').read_text());rgb=np.loadtxt(data/'rgb.txt',dtype=str,comments='#')
indices=json.loads(Path('/home/prmi/zwj/private-compensation-online-20261022/validation/preregistration.json').read_text())['evaluation_frames']
ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda();bg=torch.zeros(3,device='cuda')
ratios=[.01,.025,.05,.075,.1,.125,.15]
def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def load(checkpoint,selector=None):
    cloud=next((checkpoint/'point_cloud').glob('iteration_*'));g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
    m=load_selector(str(selector or checkpoint/'selector.pt'),'cuda').requires_grad_(False)
    metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda');cameras=json.loads((checkpoint/'cameras.json').read_text())
    trajectory=file_interface.read_tum_trajectory_file(str(checkpoint.parents[1]/'CameraTrajectory_TUM.txt'))
    return dict(g=g,m=m,meta=metadata,cameras=cameras,trajectory=trajectory,N=len(g.get_xyz),checkpoint=str(checkpoint))
def views(model,include_bank,excluded):
    intr=model['cameras'][0];views=[]
    if include_bank:
        for i,c in enumerate(model['cameras']):
            pose=np.eye(4);pose[:3,:3]=c['rotation'];pose[:3,3]=c['position']
            cam=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(pose))
            image=cv2.cvtColor(cv2.imread(str(data/c['img_name'])),cv2.COLOR_BGR2RGB)
            views.append((f'bank_{i}',c['img_name'],'Bank',cam,torch.as_tensor(image,device='cuda').permute(2,0,1).float().contiguous()/255))
    for i in indices:
        name=rgb[i,1]
        if name in excluded:continue
        j=int(np.argmin(abs(model['trajectory'].timestamps-float(rgb[i,0]))))
        assert abs(model['trajectory'].timestamps[j]-float(rgb[i,0]))<.08
        cam=MiniCam(intr['width'],intr['height'],focal2fov(intr['fx'],intr['width']),focal2fov(intr['fy'],intr['height']),np.linalg.inv(model['trajectory'].poses_se3[j]))
        image=cv2.cvtColor(cv2.imread(str(data/name)),cv2.COLOR_BGR2RGB)
        views.append((f'nonbank_{i}',name,'nonBank',cam,torch.as_tensor(image,device='cuda').permute(2,0,1).float().contiguous()/255))
    return views
def selection(model,ratio):
    m,g=model['m'],model['g'];p,h=m.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features_dc,g.get_opacity,torch.tensor([[ratio]],device='cuda'))
    logits=m.soft_net(m.local_emd[3](m.local_emd[2](h)));eligible=~(m.anchors.slot_inactive.flatten()&(g.get_opacity.flatten()<1/255))
    cached=model['meta']['cached_uniform']<mask_inactive_slots(m,p,g.get_opacity)
    margin=((logits[:,1]-logits[:,0]).double()-torch.logit(model['meta']['cached_uniform'].double())).cpu().numpy();margin[~eligible.cpu().numpy()]=-np.inf
    order=np.lexsort((np.arange(model['N']),-margin))
    return cached,h,order,int(eligible.sum())
def evaluate(models,group,include_bank):
    excluded={c['img_name'] for model in models.values() for c in model['cameras']};rows=[];selection_rows=[]
    current_ratios=ratios if group=='frozen' else [.01,.05,.1,.15]
    if group=='frozen':
        reference=load(base);common={r:int(selection(reference,r)[0].sum()) for r in current_ratios}
    else:
        minimum=min(model['N'] for model in models.values());capacity=min(selection(model,.01)[3] for model in models.values())
        common={r:min(int(np.floor(r*minimum)),capacity) for r in current_ratios}
    (root/(group+'_evaluation_protocol.json')).write_text(json.dumps(dict(ratios=current_ratios,common_K=common,evaluation_RGB_indices=indices,bank_union=sorted(excluded),tuned_on_eval=False),indent=2))
    for name,model in models.items():
        g,m=model['g'],model['m'];evaluation_views=views(model,include_bank,excluded);cases=[('Full',1.,None,None)]
        for ratio in current_ratios:
            cached,h,order,capacity=selection(model,ratio);k=common[ratio];assert k<=capacity
            same=torch.zeros_like(cached);same[torch.as_tensor(order[:k].copy(),device='cuda')]=True
            cases.extend([('cached',ratio,cached,h),('same_K',ratio,same,h)])
        for policy,ratio,mask,h in cases:
            attrs=compensated_attributes(m,g,mask.float(),ratio,h) if mask is not None else None
            inputs=prepare_render_data(g,mask,attrs)['inputs'];k=len(inputs['means3D'])
            chosen=torch.arange(model['N'],device='cuda') if mask is None else torch.nonzero(mask).flatten()
            owner=m.anchors.rows.index_select(0,chosen);anchors=torch.unique(owner)
            selected_per_anchor=torch.bincount(owner,minlength=len(m.anchors.features))
            selection_rows.append(dict(model=name,policy=policy,ratio=ratio,N=model['N'],K=k,selected_anchors=len(anchors),
                members_per_selected_anchor=float(selected_per_anchor[anchors].float().mean()) if len(anchors) else 0,
                mean_base_opacity=float(g.get_opacity.index_select(0,chosen).mean()) if k else 0,
                mean_base_scale=float(g.get_scaling.index_select(0,chosen).mean()) if k else 0))
            for view_id,image_id,split,cam,truth in evaluation_views:
                image,radii=_make_rasterizer(cam,g.active_sh_degree,bg,1.,inference_only=True)(**inputs)
                pred=image.clamp(0,1).unsqueeze(0);target=truth.unsqueeze(0)
                rows.append(dict(model=name,policy=policy,ratio=ratio,view=view_id,image=image_id,split=split,N=model['N'],K=k,visible=int((radii>0).sum()),effective_proxy=int(((radii>0)&(inputs['opacities'].flatten()>=1/255)).sum()),
                    psnr=float(-10*torch.log10((pred-target).square().mean())),ssim=float(ssim(pred,target)),lpips=float(lpips(pred,target)),photo=float(.8*(image-truth).abs().mean()+.2*(1-_C.fusedssim(.0001,.0009,image.contiguous(),truth).mean()))))
                ssim.reset();lpips.reset()
            write(root/(group+'_quality_per_view.csv'),rows);print('QUALITY',name,policy,ratio,flush=True)
    summary=[]
    for name in models:
        for policy,ratio in [('Full',1.)]+[(p,r) for p in ['cached','same_K'] for r in current_ratios]:
            for split in (['Bank','nonBank'] if include_bank else ['nonBank']):
                subset=[row for row in rows if row['model']==name and row['policy']==policy and row['ratio']==ratio and row['split']==split]
                if not subset:continue
                row=dict(model=name,policy=policy,ratio=ratio,split=split,views=len(subset),N=subset[0]['N'],K=subset[0]['K'])
                for metric in ['psnr','ssim','lpips','photo','visible','effective_proxy']:
                    values=[v[metric] for v in subset];row[metric]=float(np.mean(values))
                    if metric=='psnr':
                        for tag,q in [('median',50),('p10',10),('p90',90)]:row['psnr_'+tag]=float(np.percentile(values,q))
                summary.append(row)
    write(root/(group+'_quality_summary.csv'),summary)
    write(root/(group+'_selection_summary.csv'),selection_rows)

group=sys.argv[1] if len(sys.argv)>1 else 'frozen'
if group=='frozen':models={f'{arm}_{seed}':load(base,root/f'frozen_{arm}_{seed}'/'selector.pt') for seed in [2026,2027] for arm in ['off','gaussian','anchor']}
else:models={f'{arm}_{seed}':load(next((root/f'online_{arm}_{seed}').glob('*_shutdown/ply'))) for seed in [2026,2027] for arm in ['off','anchor']}
evaluate(models,group,group=='frozen')
