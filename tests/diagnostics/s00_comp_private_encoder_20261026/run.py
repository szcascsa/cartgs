"""冻结Selector与Cached集合，对照固定编码和同容量补偿专用编码。"""
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-comp-private-encoder-20261026')
SOURCE=Path('/home/prmi/zwj/s00-split-no-raw-20261026')
OLD=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(SOURCE/'run.py')==PROTOCOL['source_run_sha256']
spec=importlib.util.spec_from_file_location('source_no_raw',SOURCE/'run.py')
source_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_module)
rt.ROOT=ROOT
SourceNetwork=source_module.Network


class FrozenSelection:
    def __init__(self,scene,seed):
        self.scene=scene
        self.seed=seed
        path=SOURCE/'results/fr1_xyz/SharedE16Split4NoRaw'/str(seed)/'step_12000.pt'
        payload=torch.load(path,map_location='cuda')
        self.source=SourceNetwork(scene,payload['config'],seed)
        self.source.load_state_dict(payload['state_dict'])
        self.source.eval().requires_grad_(False)
        self.snapshots={k:v.detach().clone() for k,v in self.source.state_dict().items()}
        self.source_path=path
        self.source_sha=rt.sha(path)
        self.p={}
        self.h={}
        self.ids={}
        self.ids_snapshot={}
        self.budget={}
        data={}
        metadata=[]
        with torch.no_grad():
            for ratio in rt.RATIOS:
                p,h,logits=self.source(ratio)
                margin=logits.double()-torch.logit(scene.uniform.double())
                margin[~scene.eligible]=-torch.inf
                sets=dict(cached=torch.nonzero(p>scene.uniform).flatten(),
                    same_k=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values)
                self.p[ratio]=p.detach()
                self.h[ratio]=h.detach()
                self.budget[ratio]=(p.sum()-ratio*scene.N).abs()/max(1,ratio*scene.N)
                self.ids[ratio]=sets
                self.ids_snapshot[ratio]={policy:ids.clone() for policy,ids in sets.items()}
                for policy,ids in sets.items():
                    assert len(ids)>0 and len(ids.unique())==len(ids)
                    array=ids.cpu().numpy()
                    key=f'{ratio:g}_{policy}'
                    data[key]=array
                    metadata.append(dict(ratio=ratio,policy=policy,K=len(ids),rows_sha256=hashlib.sha256(array.tobytes()).hexdigest()))
        self.raw={ratio:torch.cat([scene.attrs,torch.full((scene.N,1),math.log(ratio),device='cuda')],1).detach() for ratio in rt.RATIOS}
        np.savez(ROOT/f'selected_sets_{seed}.npz',**data)
        rt.save_json(ROOT/f'frozen_selection_{seed}.json',dict(seed=seed,source_checkpoint=str(path),source_sha256=self.source_sha,
            source_checkpoint_step=12000,training_policy='fixed original cached sets, no resampling',sets=metadata))

    def unchanged(self):
        return (self.scene.unchanged() and rt.sha(self.source_path)==self.source_sha
            and all(torch.equal(v,self.source.state_dict()[k]) for k,v in self.snapshots.items())
            and all(p.grad is None for p in self.source.parameters())
            and all(torch.equal(ids,self.ids_snapshot[ratio][policy]) for ratio,sets in self.ids.items() for policy,ids in sets.items()))


class Compensation(nn.Module):
    def __init__(self,frozen,private):
        super().__init__()
        self.frozen=frozen
        self.private=private
        self.encoder=copy.deepcopy(frozen.source.encoder)
        self.encoder.requires_grad_(private)
        self.head=copy.deepcopy(frozen.source.compensation).requires_grad_(True)
        self.tables=nn.ParameterList([nn.Parameter(t.detach().clone()) for t in frozen.source.extra_tables])

    def forward(self,ratio,ids):
        h=torch.relu(self.encoder(self.frozen.raw[ratio][ids]))
        z=self.frozen.source.query(self.tables,ratio,self.frozen.scene.owners[ids])
        return self.head(torch.cat([z,h],1))


def group_norm(net,prefix):
    return math.sqrt(sum(float(p.grad.square().sum()) for name,p in net.named_parameters()
        if name.startswith(prefix) and p.grad is not None))


@torch.no_grad()
def evaluate(frozen,net,split='validation',full_metrics=False,policies=('cached','same_k')):
    scene=frozen.scene
    views=scene.validation if split=='validation' else scene.load_test()
    if full_metrics:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda()
        lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
    rows=[]
    for ratio in rt.RATIOS:
        for policy in policies:
            ids=frozen.ids[ratio][policy]
            delta=net(ratio,ids)
            assert delta.isfinite().all()
            for i,view in enumerate(views):
                image,radii,alpha=scene.render(ids,delta,view)
                prediction=image.clamp(0,1)
                truth=view['gt']
                row=dict(ratio=ratio,policy=policy,split=split,view=i,path=view['path'],selected=len(ids),
                    visible=int((radii>0).sum()),effective_proxy=int(((radii>0)&(alpha>=1/255)).sum()),
                    psnr=float(-10*torch.log10((prediction-truth).square().mean())),photo=float(rt.photo(image,truth)))
                if full_metrics:
                    row.update(ssim=float(ssim(prediction[None],truth[None])),lpips=float(lpips(prediction[None],truth[None])))
                    ssim.reset()
                    lpips.reset()
                rows.append(row)
    return rows


def gate(frozen):
    a,b=Compensation(frozen,False),Compensation(frozen,True)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert all(p.data_ptr()!=dict(b.named_parameters())[name].data_ptr() for name,p in a.named_parameters())
    error=0.
    with torch.no_grad():
        for ratio in rt.RATIOS:
            for policy,ids in frozen.ids[ratio].items():
                da,db=a(ratio,ids),b(ratio,ids)
                assert torch.equal(da,db)
                original=frozen.source.compensate(frozen.h[ratio],ids,ratio)
                assert torch.allclose(da,original,atol=2e-5,rtol=2e-5)
                error=max(error,float((da-original).abs().max()))
    ratio=.05
    ids=frozen.ids[ratio]['cached']
    view=frozen.scene.train[0]
    norms={}
    for name,net in [('frozen',a),('private',b)]:
        net.zero_grad(set_to_none=True)
        image,_,_=frozen.scene.render(ids,net(ratio,ids),view)
        rt.photo(image,view['gt']).backward()
        assert all(p.grad is None for p in frozen.source.parameters())
        if name=='frozen':
            assert all(p.grad is None for p in net.encoder.parameters())
        else:
            assert all(p.grad is not None and p.grad.isfinite().all() for p in net.encoder.parameters())
            assert group_norm(net,'encoder.')>0
        assert all(net.tables[j].grad is None for j in [0,2,3])
        assert net.tables[1].grad is not None and net.tables[1].grad.norm()>0
        norms[name]=dict(encoder=group_norm(net,'encoder.'),head=group_norm(net,'head.'),latent=group_norm(net,'tables.'))
    assert frozen.unchanged()
    rt.save_json(ROOT/f'correctness_{frozen.seed}.json',dict(initial_compensation_A_B_exact=True,
        maximum_source_batch_shape_difference=error,initial_function_matches_source_within_tolerance=True,
        selected_sets_exact_same=True,original_selector_and_encoder_frozen=True,
        only_private_encoder_receives_compensation_gradient=True,active_latent_gradient=True,
        inactive_ratio_gradient_none=True,real_photo_gradient_probe=norms,
        private_encoder_shape='11->32 ReLU',extra_trainable_encoder_parameters=384,
        total_compensation_stored_parameters=sum(p.numel() for p in b.parameters())))
    del a,b
    gc.collect()


def train(frozen,private):
    kind='private_encoder' if private else 'frozen_encoder'
    directory=ROOT/'results/fr1_xyz'/kind/str(frozen.seed)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        return json.loads((directory/'complete.json').read_text())
    torch.manual_seed(frozen.seed)
    torch.cuda.manual_seed_all(frozen.seed)
    net=Compensation(frozen,private)
    initial_encoder={k:v.clone() for k,v in net.encoder.state_dict().items()}
    optimizer=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.001,eps=1e-15,fused=True)
    sequence=rt.schedule(frozen.scene,frozen.seed,3000)
    curve=[]
    validation=[]
    exposures={r:0 for r in rt.TRAIN}
    start=time.monotonic()
    for step in range(1,12001):
        ratio=rt.TRAIN[(step-1)%4]
        view=frozen.scene.train[sequence[(step-1)//4]]
        ids=frozen.ids[ratio]['cached']
        optimizer.zero_grad(set_to_none=True)
        delta=net(ratio,ids)
        p=frozen.p[ratio][ids]
        gate=1+p-p.detach()
        image,_,_=frozen.scene.render(ids,delta,view,gate)
        photo=rt.photo(image,view['gt'])
        loss=photo+.01*frozen.budget[ratio]
        assert loss.isfinite(),(kind,frozen.seed,step)
        loss.backward()
        if (step-1)%100<4 or step==12000:
            norms={name:float(parameter.grad.norm()) for name,parameter in net.named_parameters() if parameter.grad is not None}
            assert all(math.isfinite(value) for value in norms.values())
            curve.append(dict(step=step,ratio=ratio,path=view['path'],selected=len(ids),photo=float(photo),
                frozen_budget=float(frozen.budget[ratio]),encoder_grad_norm=group_norm(net,'encoder.'),
                head_grad_norm=group_norm(net,'head.'),latent_grad_norm=group_norm(net,'tables.'),
                delta_xyz_std=float(delta[:,:3].std()),delta_rotation_std=float(delta[:,6:10].std())))
        optimizer.step()
        exposures[ratio]+=1
        if step in [4000,8000,12000]:
            for row in evaluate(frozen,net):
                row.update(step=step)
                validation.append(row)
            assert frozen.unchanged()
            if not private:
                assert all(torch.equal(value,net.encoder.state_dict()[key]) for key,value in initial_encoder.items())
            rt.write_csv(directory/'training_curve.csv',curve)
            rt.write_csv(directory/'validation_per_view.csv',validation)
            torch.save(dict(kind=kind,seed=frozen.seed,step=step,state_dict=net.state_dict(),optimizer=optimizer.state_dict()),directory/f'step_{step}.pt')
        if step%1000==0:
            rt.save_json(ROOT/'status.json',dict(stage='TRAINING',kind=kind,seed=frozen.seed,step=step,total=12000))
    clocks={name:int(optimizer.state[p]['step']) if p in optimizer.state else 0 for name,p in net.named_parameters()}
    assert all(value==3000 for value in exposures.values())
    assert all(value==3000 for name,value in clocks.items() if name.startswith('tables.'))
    assert all(value==12000 for name,value in clocks.items() if name.startswith('head.'))
    assert all(value==(12000 if private else 0) for name,value in clocks.items() if name.startswith('encoder.'))
    encoder_change=math.sqrt(sum(float((net.encoder.state_dict()[k]-v).square().sum()) for k,v in initial_encoder.items()))
    assert (encoder_change>0) if private else (encoder_change==0)
    info=dict(kind=kind,seed=frozen.seed,steps=12000,ratio_exposures=exposures,adam_clocks=clocks,
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        stored_parameters=sum(p.numel() for p in net.parameters()),encoder_change_l2=encoder_change,
        source_checkpoint=str(frozen.source_path),source_checkpoint_sha256=frozen.source_sha,
        frozen_source_unchanged=frozen.unchanged(),checkpoint=str(directory/'step_12000.pt'),seconds=time.monotonic()-start)
    rt.save_json(directory/'complete.json',info)
    for split in ['validation','test']:
        rows=evaluate(frozen,net,split,full_metrics=True)
        for row in rows:
            row.update(kind=kind,seed=frozen.seed)
        rt.write_csv(directory/(split+'_full.csv'),rows)
    print('TRAINED',kind,frozen.seed,encoder_change,flush=True)
    del net,optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return info


def main():
    old=json.loads((OLD/'delivery_verification.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==old['prototype_sha256']['runtime.py']
    assert rt.sha(Path(old['renderer_binary']))==old['renderer_sha256']
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((SOURCE/'results/fr1_xyz/scene_manifest.json').read_text())
    runs=[]
    for seed in PROTOCOL['seeds']:
        frozen=FrozenSelection(scene,seed)
        gate(frozen)
        initial=Compensation(frozen,False)
        for split in ['validation','test']:
            rt.write_csv(ROOT/f'initial_{seed}_{split}.csv',evaluate(frozen,initial,split,full_metrics=True))
        del initial
        for private in [False,True]:
            runs.append(train(frozen,private))
            rt.save_json(ROOT/'runs.json',runs)
            rt.save_json(ROOT/'status.json',dict(stage='COMPLETE_MODEL',completed=len(runs),total=4,seed=seed,private=private))
        assert frozen.unchanged()
        del frozen
        gc.collect()
        torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='TRAINING_COMPLETE',completed=4,total=4))


if __name__=='__main__':
    try:
        main()
    except Exception:
        (ROOT/'run.exit').write_text('1')
        raise
    (ROOT/'run.exit').write_text('0')
