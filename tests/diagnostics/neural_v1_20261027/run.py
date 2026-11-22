"""固定真实R3集合的神经补偿研究；Bank-only筛选，默认不读non-Bank。"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
import matrix_runtime as rt
from neural import Features,SharedNeural,RatioExperts,SinusoidalField,audit_parameters

ROOT=Path(__file__).resolve().parent;OLD=Path('/home/prmi/zwj/autonomous-compensation-20261027')
rt.ROOT=ROOT
LOADED_CODE_SHA={n:rt.sha(ROOT/n) for n in ['run.py','neural.py','matrix_runtime.py']}
for name,digest in LOADED_CODE_SHA.items():
    snap=ROOT/'code_versions'/(Path(name).stem+'_'+digest[:12]+'.py');snap.parent.mkdir(exist_ok=True)
    if not snap.exists():snap.write_bytes((ROOT/name).read_bytes())


class Fixture:
    def __init__(self,scene,source_seed=3100,source_family='R3'):
        self.scene=scene;self.source_seed=source_seed;self.source_family=source_family
        tier='B_confirm' if scene.name=='fr1_xyz' else 'B_fr2'
        path=OLD/'results'/tier/scene.name/source_family/str(source_seed)/'formal_raw'
        sets=np.load(path/'selected_sets.npz');values=np.load(path/'compensation.npz')
        self.sets={(float(k.split('_')[0]),k.split('_',1)[1]):torch.tensor(v,device='cuda') for k,v in sets.items()}
        self.teacher={(float(k.split('_')[0]),k.split('_',1)[1]):torch.tensor(v,device='cuda') for k,v in values.items()}
        for ratio in rt.RATIOS:assert len(self.sets[ratio,'same_k'])==scene.K[ratio]
        self.features=Features(scene,self.sets)
        self.manifest=dict(source=str(path),source_seed=source_seed,source_family=source_family,
            set_sha256=rt.sha(path/'selected_sets.npz'),compensation_sha256=rt.sha(path/'compensation.npz'),
            frozen='Master/Anchor/Encoder/Selector/selection identities/poses/eligibility/uniform',
            teacher_usage='reference only unless explicit teacher config; never a candidate parameter')
        rt.save_json(ROOT/f'fixture_{scene.name}_{source_family}_{source_seed}.json',self.manifest)


class Reference59(nn.Module):
    def __init__(self,fixture,cfg,seed):
        super().__init__();self.cfg=cfg
        path=Path('/home/prmi/zwj/s00-rotation-encoder-20261026/run.py')
        spec=importlib.util.spec_from_file_location('historical_baseline',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rt.ROOT=ROOT
        self.model=module.Network(fixture.scene,module.config('baseline'),seed)
        self.model.encoder.requires_grad_(cfg.get('train_encoder',False));self.model.selector.requires_grad_(False)
        self.fixture=fixture
    def forward(self,ratio,ids,policy='same_k'):
        raw=self.model.raw(ratio)
        h=torch.relu(self.model.encoder(raw))
        return self.model.compensate(h,ids,ratio)


def make(fixture,cfg,seed):
    if cfg['kind']=='ratio_experts':return RatioExperts(fixture.features,cfg,seed)
    if cfg['kind']=='siren':return SinusoidalField(fixture.features,cfg,seed)
    return Reference59(fixture,cfg,seed) if cfg['kind']=='reference59' else SharedNeural(fixture.features,cfg,seed)


def render(fixture,net,ids,delta,view):
    sc=fixture.scene
    if net is None or net.cfg.get('output')!='affine':return sc.render(ids,delta,view)
    a=delta[:,3:9]
    transform=torch.stack([a[:,0],a[:,3],a[:,4],a[:,3],a[:,1],a[:,5],a[:,4],a[:,5],a[:,2]],1).reshape(-1,3,3)
    transform=transform+torch.eye(3,device='cuda')[None]
    cov=transform@fixture.features.covariance[ids]@transform.transpose(1,2)
    packed=cov[:,[0,0,0,1,1,2],[0,1,2,1,2,2]].contiguous()
    g=sc.g;opacity=(g._opacity[ids]+delta[:,9:10]).sigmoid()
    image,radii=rt._make_rasterizer(view['cam'],g.active_sh_degree,sc.bg,1.)(
        means3D=(g.get_xyz[ids]+delta[:,:3]).contiguous(),means2D=torch.zeros_like(g.get_xyz[ids]),
        dc=(g.get_features_dc[ids]+delta[:,10:13,None].transpose(1,2)).contiguous(),shs=g.get_features_rest[ids].contiguous(),
        colors_precomp=None,opacities=opacity.contiguous(),scales=None,rotations=None,cov3D_precomp=packed)
    return image,radii,opacity.flatten()


@torch.no_grad()
def evaluate(fixture,net,split='validation',full=False,policies=('same_k',)):
    sc=fixture.scene;views=sc.validation if split=='validation' else sc.load_test() if split=='test' else sc.bank_full
    if split=='bank_train':
        paths={v['path'] for v in sc.train};views=[v for v in sc.bank_full if v['path'] in paths]
    if full:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
    rows=[]
    for ratio in rt.RATIOS:
        for policy in policies:
            ids=fixture.sets[ratio,policy];delta=fixture.teacher[ratio,policy] if net is None else net(ratio,ids,policy)
            assert delta.isfinite().all()
            for vi,v in enumerate(views):
                image,radii,alpha=render(fixture,net,ids,delta,v);pred=image.clamp(0,1);gt=v['gt']
                row=dict(ratio=ratio,policy=policy,split=split,view=vi,path=v['path'],selected=len(ids),
                    visible=int((radii>0).sum()),effective_proxy=int(((radii>0)&(alpha>=1/255)).sum()),
                    psnr=float(-10*torch.log10((pred-gt).square().mean())),photo=float(rt.photo(image,gt)))
                if full:
                    row.update(ssim=float(ssim(pred[None],gt[None])),lpips=float(lpips(pred[None],gt[None])));ssim.reset();lpips.reset()
                rows.append(row)
    return rows


def train(fixture,cfg,seed,steps,tier):
    directory=ROOT/'results'/tier/fixture.scene.name/cfg['id']/str(seed);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return json.loads((directory/'complete.json').read_text())
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=make(fixture,cfg,seed)
    if cfg['kind']!='reference59':rt.save_json(directory/'parameter_audit.json',dict(pure_shared=True,parameters=audit_parameters(net),identity_trainable=False))
    if cfg.get('initial_checkpoint'):
        payload=torch.load(cfg['initial_checkpoint'],map_location='cuda');net.load_state_dict(payload['state_dict'])
    else:
        with torch.no_grad():assert all(torch.count_nonzero(net(r,fixture.sets[r,'same_k']))==0 for r in rt.TRAIN)
    optimizer=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=cfg.get('lr',.001),eps=1e-15,fused=True)
    if cfg.get('resume_optimizer',False):optimizer.load_state_dict(payload['optimizer'])
    offset=cfg.get('schedule_cycle_offset',0)
    sequence=rt.schedule(fixture.scene,seed,steps//4+offset)[offset:];curve=[];checks=sorted(set([steps//2//4*4,steps]));validation=[]
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();updates={}
    for step in range(1,steps+1):
        ri=(step-1)%4;ratio=rt.TRAIN[ri];view=fixture.scene.train[sequence[(step-1)//4]];ids=fixture.sets[ratio,'same_k']
        cycle=cfg.get('cycle',False)
        if not cycle or ri==0:optimizer.zero_grad(set_to_none=True)
        delta=net(ratio,ids);image,_,_=render(fixture,net,ids,delta,view)
        photo=rt.photo(image,view['gt']);loss=photo
        assert loss.isfinite(),(cfg,step)
        (loss/4 if cycle else loss).backward()
        if step%100 in [1,2,3,4] or step==steps:
            norms={name:float(p.grad.norm()) for name,p in net.named_parameters() if p.grad is not None}
            assert all(np.isfinite(list(norms.values())))
            curve.append(dict(step=step,ratio=ratio,path=view['path'],photo=float(photo),loss=float(loss),
                grad_norm=float(sum(x*x for x in norms.values())**.5),xyz_p90=float(delta[:,:3].norm(dim=1).quantile(.9))))
            if step<=104:updates[str(step)]=norms
        if not cycle or ri==3:optimizer.step()
        if step in checks:
            vals=evaluate(fixture,net)
            for row in vals:row.update(step=step)
            validation.extend(vals);rt.write_csv(directory/'validation_curve.csv',validation);rt.write_csv(directory/'training_curve.csv',curve)
            torch.save(dict(config=cfg,seed=seed,step=step,state_dict=net.state_dict(),optimizer=optimizer.state_dict()),directory/f'step_{step}.pt')
        if step%1000==0:rt.save_json(ROOT/('status_'+tier+'.json'),dict(config=cfg['id'],seed=seed,step=step,total=steps))
    assert fixture.scene.unchanged()
    final=[v for v in validation if v['step']==steps]
    info=dict(config=cfg,seed=seed,steps=steps,tier=tier,source=fixture.manifest,
        initial_checkpoint=cfg.get('initial_checkpoint'),initial_checkpoint_sha256=rt.sha(cfg['initial_checkpoint']) if cfg.get('initial_checkpoint') else None,
        prior_steps=int(payload['step']) if cfg.get('initial_checkpoint') else 0,
        parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        pure_shared=cfg['kind']!='reference59',seconds=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
        optimizer_clocks={name:int(optimizer.state[p]['step']) for name,p in net.named_parameters() if p in optimizer.state},
        ratio_exposures={r:steps//4 for r in rt.TRAIN},checkpoint=str(directory/f'step_{steps}.pt'),
        validation_psnr=float(np.mean([v['psnr'] for v in final])),validation_train=float(np.mean([v['psnr'] for v in final if v['ratio'] in rt.TRAIN])),
        validation_unseen=float(np.mean([v['psnr'] for v in final if v['ratio'] not in rt.TRAIN])),
        code_sha256=LOADED_CODE_SHA,code_sha_scope='loaded at process import; snapshot saved',master_frozen=True,fixed_selection=True)
    rt.save_json(directory/'early_gradients.json',updates);rt.save_json(directory/'complete.json',info)
    print('COMPLETE',cfg['id'],seed,steps,info['validation_psnr'],info['validation_train'],info['validation_unseen'],info['seconds'],flush=True)
    del net,optimizer;gc.collect();torch.cuda.empty_cache();return info


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue',required=True);args=parser.parse_args();queue=json.loads(Path(args.queue).read_text())
    scene=rt.Scene(queue.get('scene','fr1_xyz'));fixture=Fixture(scene,queue.get('source_seed',3100),queue.get('source_family','R3'));runs=[]
    for job in queue['jobs']:
        runs.append(train(fixture,job['config'],job['seed'],job.get('steps',4000),queue['tier']))
        rt.save_json(ROOT/(Path(args.queue).stem+'_results.json'),runs)
    rt.save_json(ROOT/(Path(args.queue).stem+'_complete.json'),dict(runs=len(runs)))


if __name__=='__main__':main()
