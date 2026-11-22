"""独立冻结地图协议；生产源码只读快照、Bank筛选、锁定后non-Bank评价。"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;REFERENCE=Path('/home/prmi/zwj/neural-v1-20261027');REPO=Path('/home/prmi/zwj/mycartgs')
sys.path.insert(0,str(REFERENCE));import run as old
# Historical loaders use their module ROOT for outputs; point it at this new study.
old.ROOT=ROOT;old.rt.ROOT=ROOT
from af_model import AnchorField
spec=importlib.util.spec_from_file_location('production_spatial_snapshot',ROOT/'production_spatial.py')
prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)
TRAIN=[.01,.05,.1,.15];RATIOS=sorted(TRAIN+[.025,.075,.125])
CONFIG=dict(id='SpatialLocalXYZ',kind='self',width=64,bands=6,film=True,output='local')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):old.rt.save_json(path,value)
def csv(path,rows):old.rt.write_csv(path,rows)


class Study:
    def __init__(self,scene):
        self.scene=old.rt.Scene(scene);self.fixture=old.Fixture(self.scene,3100 if scene=='fr1_xyz' else 4100)
        f=self.fixture.features;sc=self.scene
        with torch.no_grad():
            ids=torch.arange(sc.N,device='cuda');self.x=f.inputs(ids,.01,CONFIG)
            self.owners=sc.owners.clone();self.uids=sc.source.anchors.block_uids.clone();self.counts=torch.bincount(self.owners,minlength=sc.M)
            assert len(torch.unique(self.uids))==sc.M and (self.counts>=1).all() and self.counts.max()<=10
            sums=torch.zeros(sc.M,19,device='cuda').index_add_(0,self.owners,f.own)
            mean=sums/self.counts[:,None]
            second=torch.zeros_like(mean).index_add_(0,self.owners,f.own.square())/self.counts[:,None]
            maximum=torch.full_like(mean,-float('inf'));maximum.scatter_reduce_(0,self.owners[:,None].expand(-1,19),f.own,reduce='amax',include_self=True)
            self.context=torch.cat([mean,maximum,(second-mean.square()).clamp_min(0).sqrt(),self.counts[:,None].float().log1p()],1)
            gen=torch.Generator(device='cuda').manual_seed(9741)
            self.fake_owners=self.owners[torch.randperm(sc.N,generator=gen,device='cuda')]
            self.shuffle_anchors=torch.randperm(sc.M,generator=gen,device='cuda')
        self.anchor_scale=f.anchor_scale;self.M=sc.M
        self.checkpoint=REFERENCE/('results/controls/fr1_xyz/SpatialLocalXYZ/5100/step_12000.pt' if scene=='fr1_xyz' else 'results/fr2_local_followup/fr2_xyz/SpatialLocalXYZ/6100/step_12000.pt')
        self.payload=torch.load(self.checkpoint,map_location='cuda')
        self.world_center=sc.g.get_xyz.median(0).values
        self.world_spread=(sc.g.get_xyz-self.world_center).square().mean(0).sqrt().clamp_min(.1)
        self.mean=f.mean;self.stddev=f.std
        info=dict(scene=scene,N=sc.N,M=sc.M,occupancy={str(n):int((self.counts==n).sum()) for n in range(1,11)},K={str(r):len(self.fixture.sets[r,'same_k']) for r in RATIOS},
            checkpoint=str(self.checkpoint),checkpoint_sha256=sha(self.checkpoint),source=self.fixture.manifest,
            train_views=len(sc.train),bank_holdout_views=len(sc.validation),test_views=len(sc.test_paths),normalization='same full frozen map as original 12k; production physical55 function checked',
            production_difference='production fixes statistics at first online compensation; this fixed-map study fixes them on the historical frozen map',
            shared_anchor_feature=True,gaussian_private_parameters=False)
        save(ROOT/f'fixture_{scene}.json',info)

    def make(self,cfg,seed,route='continue'):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);base=prod.SpatialCompensation().cuda()
        if route=='continue':
            missing,unexpected=base.load_state_dict(self.payload['state_dict'],strict=False)
            assert set(missing)=={'world_center','world_spread','mean','stddev','normalization_ready'} and not unexpected
        with torch.no_grad():
            for name in ['world_center','world_spread','mean','stddev']:getattr(base,name).copy_(getattr(self,name))
            base.normalization_ready.fill_(1)
        return AnchorField(base,self,cfg)

    def optimizer(self,net,cfg,route):
        base=list(p for p in net.base.parameters() if p.requires_grad)
        optimizer=torch.optim.Adam(base if base else net.extra_parameters(),lr=.001,eps=1e-15,fused=True)
        if base and route=='continue':optimizer.load_state_dict(copy.deepcopy(self.payload['optimizer']))
        extra=net.extra_parameters()
        if extra and base:optimizer.add_param_group(dict(params=extra,lr=.001))
        features=net.feature_parameters()
        if features:optimizer.add_param_group(dict(params=features,lr=cfg.get('feature_lr',.001)))
        return optimizer


@torch.no_grad()
def evaluate(study,net,split='validation',full=False,ablation='normal',ratios=RATIOS):
    sc=study.scene
    if split=='test':views=sc.load_test()
    elif split=='bank_train':
        paths={v['path'] for v in sc.train};views=[v for v in sc.bank_full if v['path'] in paths]
    else:views=sc.validation
    if full:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
    rows=[]
    if net is not None:net.ablation=ablation
    for ratio in ratios:
        ids=study.fixture.sets[ratio,'same_k'];delta=study.fixture.teacher[ratio,'same_k'] if net is None else net(ratio,ids)
        assert delta.isfinite().all()
        for vi,v in enumerate(views):
            image,radii,alpha=sc.render(ids,delta,v);pred=image.clamp(0,1);gt=v['gt']
            record=dict(scene=sc.name,split=split,ratio=ratio,view=vi,path=v['path'],selected=len(ids),visible=int((radii>0).sum()),
                effective_proxy=int(((radii>0)&(alpha>=1/255)).sum()),psnr=float(-10*torch.log10((pred-gt).square().mean())),photo=float(old.rt.photo(image,gt)),ablation=ablation)
            if full:
                record.update(ssim=float(ssim(pred[None],gt[None])),lpips=float(lpips(pred[None],gt[None])));ssim.reset();lpips.reset()
            rows.append(record)
    if net is not None:net.ablation='normal'
    return rows


def parameter_audit(net):
    rows=[]
    for name,p in net.named_parameters():
        rows.append(dict(name=name,shape=list(p.shape),numel=p.numel(),trainable=p.requires_grad,scope='Anchor abstraction' if name=='embedding' else 'shared neural function'))
    assert not any('gaussian' in v['name'] for v in rows)
    return rows


def train(study,job,tier):
    cfg=job['config'];seed=job['seed'];route=job.get('route','continue');steps=job.get('steps',4000)
    directory=ROOT/'results'/tier/study.scene.name/cfg['id']/str(seed);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return json.loads((directory/'complete.json').read_text())
    net=study.make(cfg,seed,route);optimizer=study.optimizer(net,cfg,route)
    schedule=old.rt.schedule(study.scene,seed,steps//4)
    curve=[];validation=[];first_grads={};sample_visits=torch.zeros(study.M,device='cuda');grad_visits=torch.zeros_like(sample_visits);actual_updates=torch.zeros_like(sample_visits)
    counts_by_ratio=torch.zeros(4,study.M,device='cuda');start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    checks=set([min(steps,1000),steps//2//4*4,steps]);baseline=study.make(dict(feature='none'),seed,route)
    with torch.no_grad():
        diffs=[]
        for ratio in RATIOS:
            ids=study.fixture.sets[ratio,'same_k'];diffs.append(float((net(ratio,ids)-baseline(ratio,ids)).abs().max()))
        assert max(diffs)==0,('step0 mismatch',cfg,diffs)
    del baseline
    for step in range(1,steps+1):
        ri=(step-1)%4;ratio=TRAIN[ri];ids=study.fixture.sets[ratio,'same_k'];view=study.scene.train[schedule[(step-1)//4]]
        optimizer.zero_grad(set_to_none=True);delta=net(ratio,ids);image,_,_=study.scene.render(ids,delta,view)
        loss=old.rt.photo(image,view['gt']);assert loss.isfinite();loss.backward()
        owner=study.fake_owners[ids] if net.kind=='fake' else study.owners[ids]
        active=torch.bincount(owner,minlength=study.M)>0;sample_visits.add_(active);counts_by_ratio[ri].add_(active)
        has_embedding=hasattr(net,'embedding') and net.embedding.requires_grad
        if has_embedding:
            e=net.embedding;before=e.detach().clone();gradient=(e.grad.abs().sum(1)>0)
            if e.shape[0]==study.M:grad_visits.add_(gradient)
        if step<=4:first_grads[str(ratio)]={name:float(p.grad.norm()) for name,p in net.named_parameters() if p.grad is not None}
        optimizer.step()
        if has_embedding and e.shape[0]==study.M:actual_updates.add_((e.detach()-before).abs().sum(1)>0)
        if step%100 in [1,2,3,4] or step==steps:
            curve.append(dict(step=step,ratio=ratio,path=view['path'],photo=float(loss),xyz_p90=float(delta[:,:3].norm(dim=1).quantile(.9)),
                feature_norm=float(e.norm()) if has_embedding else 0.,feature_grad=float(e.grad.norm()) if has_embedding else 0.))
        if step in checks:
            vals=evaluate(study,net)
            for v in vals:v['step']=step
            validation+=vals;csv(directory/'validation_curve.csv',validation);csv(directory/'training_curve.csv',curve)
            torch.save(dict(config=cfg,seed=seed,steps=step,route=route,state_dict=net.state_dict(),optimizer=optimizer.state_dict(),source_checkpoint=str(study.checkpoint),
                uids=study.uids,owners=study.owners),directory/f'step_{step}.pt')
        if step%1000==0:
            save(ROOT/'current.json',dict(tier=tier,scene=study.scene.name,model=cfg['id'],seed=seed,step=step,total=steps));print('STEP',cfg['id'],step,flush=True)
    assert study.scene.unchanged()
    torch.save(dict(uids=study.uids,occupancy=study.counts,sample_visits=sample_visits,gradient_visits=grad_visits,actual_updates=actual_updates,ratio_visits=counts_by_ratio,
        embedding=net.embedding.detach() if hasattr(net,'embedding') else None),directory/'feature_diagnostics.pt')
    final=[v for v in validation if v['step']==steps];audit=parameter_audit(net)
    info=dict(config=cfg,seed=seed,route=route,steps=steps,prior_steps=12000 if route=='continue' else 0,tier=tier,scene=study.scene.name,
        checkpoint=str(directory/f'step_{steps}.pt'),source_checkpoint=str(study.checkpoint),source_checkpoint_sha256=sha(study.checkpoint),
        parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),parameter_inventory=audit,step0_max_abs=0.,seconds=time.monotonic()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),first_gradients=first_grads,ratio_exposures={str(r):steps//4 for r in TRAIN},
        optimizer_steps={name:int(optimizer.state[p]['step']) for name,p in net.named_parameters() if p in optimizer.state},
        bank_holdout_psnr=float(np.mean([v['psnr'] for v in final])),bank_low=float(np.mean([v['psnr'] for v in final if v['ratio'] in [.01,.05]])),
        bank_unseen=float(np.mean([v['psnr'] for v in final if v['ratio'] not in TRAIN])),frame_order_sha256=hashlib.sha256(np.asarray(schedule,dtype=np.int64).tobytes()).hexdigest(),
        code_sha256={p.name:sha(p) for p in ROOT.glob('af_*.py')},no_test_selection=True)
    save(directory/'complete.json',info);print('COMPLETE',cfg['id'],seed,steps,info['bank_holdout_psnr'],info['bank_low'],info['bank_unseen'],info['seconds'],flush=True)
    return info
