"""冻结真实地图、原Cached ST与photo；研究网络容量，不接入生产。"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn

ROOT=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
REPO=Path('/home/prmi/zwj/mycartgs')
sys.path.insert(0,str(REPO/'python'))
from selector import load_selector,load_selector_metadata
from gaussian_model import GaussianModel
from renderer import _make_rasterizer
from transform_field import _normalize_quaternion,_multiply_quaternion
from utils import MiniCam,focal2fov
from diff_gaussian_rasterization import _C
from evo.tools import file_interface
from evo.core import sync

TRAIN=[.01,.05,.1,.15]
RATIOS=sorted(TRAIN+[.025,.075,.125])
ATTR=['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation']
torch.set_num_threads(2)

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class SSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        ctx.save_for_backward(x,y);return _C.fusedssim(.0001,.0009,x,y)
    @staticmethod
    def backward(ctx,g):
        x,y=ctx.saved_tensors;return _C.fusedssim_backward(.0001,.0009,x,y,g.contiguous()),None
def photo(x,y):return .8*(x-y).abs().mean()+.2*(1-SSIM.apply(x.contiguous(),y).mean())

class Scene:
    def __init__(self,name='fr1_xyz'):
        meta=json.loads((ROOT/'scenes.json').read_text())[name]
        self.name=name;self.cp=Path(meta['checkpoint']);self.data=Path(meta['dataset'])
        cloud=next((self.cp/'point_cloud').glob('iteration_*'))
        self.source=load_selector(str(self.cp/'selector.pt'),'cuda').eval().requires_grad_(False)
        self.g=GaussianModel(3);self.g.load_ply(str(cloud/'point_cloud.ply'))
        for key in ATTR:getattr(self.g,key).requires_grad_(False)
        self.render_bases=[self.g.get_xyz,self.g.get_scaling.detach(),self.g.get_rotation.detach(),self.g.get_features_dc,self.g.get_features_rest,self.g._opacity]
        self.N=len(self.g.get_xyz);self.M=len(self.source.anchors.features)
        self.owners=self.source.anchors.rows.detach();self.uniform=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')['cached_uniform']
        inactive=getattr(self.source.anchors,'slot_inactive',torch.zeros(self.N,dtype=torch.bool,device='cuda')).flatten()
        self.eligible=~(inactive&(self.g.get_opacity.flatten()<1/255))
        self.K={r:min(int(math.floor(r*self.N)),int(self.eligible.sum())) for r in RATIOS}
        assert min(self.K.values())>0
        self.attrs=torch.cat([(self.g.get_xyz-self.source.anchors.centers[self.owners])/self.source.anchors.offset_scales[self.owners],
            self.g.get_scaling,self.g.get_features_dc.squeeze(1),self.g.get_opacity],1).detach()
        self.frozen_tensors=[getattr(self.g,k) for k in ATTR]+[self.owners,self.source.anchors.centers,self.source.anchors.offset_scales,self.uniform,self.eligible]
        self.frozen_copies=[x.detach().clone() for x in self.frozen_tensors]
        self.bg=torch.zeros(3,device='cuda')
        cameras=sorted(json.loads((self.cp/'cameras.json').read_text()),key=lambda x:x['img_name']);self.intr=cameras[0]
        self.train=[];self.validation=[];self.bank_full=[]
        for j,c in enumerate(cameras):
            pose=np.eye(4);pose[:3,:3]=c['rotation'];pose[:3,3]=c['position']
            full=self.view(c['img_name'],np.linalg.inv(pose));self.bank_full.append(full)
            if j%5==2:self.validation.append(full)
            else:self.train.append(self.view(c['img_name'],np.linalg.inv(pose),.5))
        self.test_paths=meta['test_indices'];self.test=None
        files=[self.cp/'selector.pt',cloud/'point_cloud.ply',cloud/'selector_metadata.ply',self.cp/'cameras.json',self.cp.parent.parent/'CameraTrajectory_TUM.txt']
        self.hashes={str(p):sha(p) for p in files}
        info=dict(name=name,N=self.N,M=self.M,K=self.K,source_sha256=self.hashes,
            train_paths=[x['path'] for x in self.train],validation_paths=[x['path'] for x in self.validation],test_indices=self.test_paths,
            frozen='Master/owners/centers/scales/eligibility/uniform/cameras',candidate_population='all stored Gaussian; frozen slot eligibility',
            train_resolution=[320,240],evaluation_resolution=[640,480])
        path=ROOT/'results'/name/'scene_manifest.json'
        if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(info))
        else:save_json(path,info)
    def view(self,path,w2c,scale=1):
        i=self.intr;w,h=int(i['width']*scale),int(i['height']*scale)
        cam=MiniCam(w,h,focal2fov(i['fx'],i['width']),focal2fov(i['fy'],i['height']),w2c)
        image=cv2.cvtColor(cv2.imread(str(self.data/path)),cv2.COLOR_BGR2RGB)
        if scale!=1:image=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
        return dict(path=path,cam=cam,gt=torch.tensor(image,device='cuda').permute(2,0,1).float().contiguous()/255)
    def load_test(self):
        if self.test is not None:return self.test
        rgb=np.loadtxt(self.data/'rgb.txt',dtype=str,comments='#')
        gt=file_interface.read_tum_trajectory_file(str(self.data/'groundtruth.txt'))
        trajectory=file_interface.read_tum_trajectory_file(str(self.cp.parent.parent/'CameraTrajectory_TUM.txt'))
        _,trajectory=sync.associate_trajectories(gt,trajectory,max_diff=.08)
        bank={v['path'] for v in self.bank_full};self.test=[]
        for index in self.test_paths:
            path=rgb[index,1];assert path not in bank
            j=int(abs(trajectory.timestamps-float(rgb[index,0])).argmin());assert abs(trajectory.timestamps[j]-float(rgb[index,0]))<.08
            self.test.append(self.view(path,np.linalg.inv(trajectory.poses_se3[j])))
        return self.test
    def render(self,ids,res,v,gate=None):
        g=self.g;xyz,scale,rot,dc,rest,opacity=[value[ids] for value in self.render_bases]
        if res is not None:
            xyz=xyz+res[:,:3];scale=scale*res[:,3:6].exp()
            ident=torch.zeros_like(res[:,6:10]);ident[:,0]=1
            rot=_normalize_quaternion(_multiply_quaternion(_normalize_quaternion(ident+res[:,6:10]),rot))
            dc=dc+res[:,11:14].unsqueeze(1);opacity=opacity+res[:,10:11]
        opacity=opacity.sigmoid()
        if gate is not None:opacity=opacity*gate[:,None]
        if not len(ids):return self.bg[:,None,None].expand_as(v['gt']).contiguous(),torch.empty(0,device='cuda'),opacity.flatten()
        image,radii=_make_rasterizer(v['cam'],g.active_sh_degree,self.bg,1.)(means3D=xyz.contiguous(),means2D=torch.zeros_like(xyz),
            dc=dc.contiguous(),shs=rest.contiguous(),colors_precomp=None,opacities=opacity.contiguous(),
            scales=scale.contiguous(),rotations=rot.contiguous(),cov3D_precomp=None)
        return image,radii,opacity.flatten()
    def unchanged(self):return all(sha(p)==h for p,h in self.hashes.items()) and all(getattr(self.g,k).grad is None for k in ATTR) and all(torch.equal(a,b) for a,b in zip(self.frozen_tensors,self.frozen_copies))

def linear_init(layer,seed,scale=None):
    gen=torch.Generator(device='cuda');gen.manual_seed(seed)
    bound=1/math.sqrt(layer.in_features) if scale is None else scale
    with torch.no_grad():
        layer.weight.uniform_(-bound,bound,generator=gen)
        if layer.bias is not None:layer.bias.uniform_(-bound,bound,generator=gen)

class Network(nn.Module):
    def __init__(self,scene,cfg,seed):
        super().__init__();self.cfg=cfg;self.seed=seed;self.scene=scene
        s,r,e=cfg['shared'],cfg['ratio'],cfg['extra'];self.direct=cfg.get('direct',False)
        self.shared=nn.Parameter(torch.zeros(scene.M,s,device='cuda')) if s else None
        self.ratio_tables=nn.ParameterList([nn.Parameter(torch.zeros(scene.M,r,device='cuda')) for _ in TRAIN]) if r else nn.ParameterList()
        self.extra_tables=nn.ParameterList([nn.Parameter(torch.zeros(scene.M,e,device='cuda')) for _ in TRAIN]) if e and not self.direct else nn.ParameterList()
        self.encoder=nn.Linear(s+r+11,32,device='cuda')
        with torch.no_grad():
            base=nn.Linear(11,32,device='cuda');linear_init(base,seed+100)
            self.encoder.weight[:,s+r:].copy_(base.weight);self.encoder.bias.copy_(base.bias)
            for offset,width,key in [(0,s,seed+101),(s,r,seed+102)]:
                if width:
                    gen=torch.Generator(device='cuda');gen.manual_seed(key)
                    self.encoder.weight[:,offset:offset+width].copy_(torch.rand((32,32),generator=gen,device='cuda')[:,:width]*.1-.05)
        self.selector=self.head(32,2,cfg['selector_depth'],seed+200)
        if self.direct:
            self.residual=nn.ParameterList([nn.Parameter(torch.zeros(scene.N,14,device='cuda')) for _ in TRAIN])
            self.compensation=None
        else:
            self.compensation=self.head(32+e,14,cfg['compensation_depth'],seed+300,extra=e)
    @staticmethod
    def head(inputs,outputs,depth,seed,extra=0):
        layers=[];current=inputs
        for j in range(depth):
            layer=nn.Linear(current,32,device='cuda');linear_init(layer,seed+10*j)
            if j==0 and extra:
                common=nn.Linear(32,32,device='cuda');linear_init(common,seed)
                with torch.no_grad():
                    layer.weight[:,:32].copy_(common.weight);layer.bias.copy_(common.bias)
                    gen=torch.Generator(device='cuda');gen.manual_seed(seed+5)
                    layer.weight[:,32:].copy_(torch.rand((32,28),generator=gen,device='cuda')[:,:extra]*.1-.05)
            layers += [layer,nn.ReLU()];current=32
        last=nn.Linear(current,outputs,device='cuda')
        with torch.no_grad():
            last.weight.zero_();last.bias.zero_()
            if depth==0 and extra:
                gen=torch.Generator(device='cuda');gen.manual_seed(seed+5)
                last.weight[:,32:].copy_(torch.rand((outputs,28),generator=gen,device='cuda')[:,:extra]*.1-.05)
        return nn.Sequential(*layers,last)
    @staticmethod
    def query(tables,ratio,ids):
        hi=next((i for i,r in enumerate(TRAIN) if ratio<=r+1e-8),len(TRAIN)-1)
        value=tables[hi][ids]
        if hi and abs(ratio-TRAIN[hi])>1e-8 and ratio<TRAIN[-1]:
            t=(ratio-TRAIN[hi-1])/(TRAIN[hi]-TRAIN[hi-1]);value=(1-t)*tables[hi-1][ids]+t*value
        return value
    def forward(self,ratio):
        sc=self.scene;inputs=[]
        if self.shared is not None:inputs.append(self.shared[sc.owners])
        if len(self.ratio_tables):inputs.append(self.query(self.ratio_tables,ratio,sc.owners))
        inputs += [sc.attrs,torch.full((sc.N,1),math.log(ratio),device='cuda')]
        h=torch.relu(self.encoder(torch.cat(inputs,1)))
        logits=self.selector(h);p=logits.softmax(-1)[:,1]*sc.eligible
        return p,h,logits[:,1]-logits[:,0]
    def compensate(self,h,ids,ratio):
        if self.direct:return self.query(self.residual,ratio,ids)
        values=[h[ids]]
        if len(self.extra_tables):values.append(self.query(self.extra_tables,ratio,self.scene.owners[ids]))
        return self.compensation(torch.cat(values,1))

def schedule(scene,seed,cycles):
    rng=np.random.default_rng(seed);views=[]
    while len(views)<cycles:views.extend(rng.permutation(len(scene.train)).tolist())
    return views[:cycles]

@torch.no_grad()
def evaluate(scene,net,split='validation',full_metrics=False,policies=('same_k',),ratios=RATIOS):
    views=scene.validation if split=='validation' else (scene.load_test() if split=='test' else (scene.train if split=='train' else scene.bank_full))
    if full_metrics:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda()
        lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
    rows=[]
    for ratio in ratios:
        p,h,logits=net(ratio)
        margin=logits.double()-torch.logit(scene.uniform.double());margin[~scene.eligible]=-torch.inf
        order=torch.argsort(margin,descending=True,stable=True)
        for policy in policies:
            ids=order[:scene.K[ratio]].sort().values if policy=='same_k' else torch.nonzero(p>scene.uniform).flatten()
            res=net.compensate(h,ids,ratio)
            assert torch.isfinite(res).all()
            for vi,v in enumerate(views):
                image,radii,alpha=scene.render(ids,res,v)
                pred=image.clamp(0,1);truth=v['gt']
                row=dict(ratio=ratio,policy=policy,split=split,view=vi,path=v['path'],selected=len(ids),
                    visible=int((radii>0).sum()),effective_proxy=int(((radii>0)&(alpha>=1/255)).sum()),
                    p_mean=float(p.mean()),cached_k=int((p>scene.uniform).sum()),psnr=float(-10*torch.log10((pred-truth).square().mean())),photo=float(photo(image,truth)))
                if full_metrics:
                    row.update(ssim=float(ssim(pred[None],truth[None])),lpips=float(lpips(pred[None],truth[None])))
                    ssim.reset();lpips.reset()
                rows.append(row)
    return rows

def train(scene,cfg,seed,steps,directory,checkpoints=None,pilot=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    done=directory/'complete.json'
    if done.exists():return json.loads(done.read_text())
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=Network(scene,cfg,seed);optimizer=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True)
    views=schedule(scene,seed,steps//4);curve=[];validation=[];exposures={r:0 for r in TRAIN};empty={r:0 for r in TRAIN}
    selected_counts=torch.zeros((4,scene.N),device='cuda',dtype=torch.int32)
    start=time.monotonic();checkpoints=checkpoints or [steps];panels=[]
    for step in range(1,steps+1):
        ri=(step-1)%4;ratio=TRAIN[ri];v=scene.train[views[(step-1)//4]]
        optimizer.zero_grad(set_to_none=True)
        p,h,_=net(ratio);ids=torch.nonzero(p.detach()>scene.uniform).flatten()
        res=net.compensate(h,ids,ratio);gate=1+p[ids]-p[ids].detach()
        image,_,_=scene.render(ids,res,v,gate)
        loss_photo=photo(image,v['gt']);budget=(p.sum()-ratio*scene.N).abs()/max(1,ratio*scene.N)
        loss=loss_photo+.01*budget
        assert torch.isfinite(loss),(cfg['id'],seed,step,'loss')
        loss.backward()
        if (step-1)%100<4 or step==steps:
            norms={name:float(par.grad.norm()) for name,par in net.named_parameters() if par.grad is not None}
            assert all(math.isfinite(x) for x in norms.values()),(cfg['id'],seed,step,'gradient')
            curve.append(dict(step=step,ratio=ratio,view=v['path'],photo=float(loss_photo),budget=float(budget),selected=len(ids),
                gradient_norm=float(sum(x*x for x in norms.values())**.5),finite=True))
        optimizer.step();exposures[ratio]+=1
        selected_counts[ri,ids]+=1;empty[ratio]+=int(not len(ids))
        if step in checkpoints:
            for row in evaluate(scene,net):row.update(step=step);validation.append(row)
            if steps>=12000:
                for row in evaluate(scene,net,'train',policies=('cached','same_k'),ratios=TRAIN):row.update(step=step);panels.append(row)
                write_csv(directory/'hard_training_panel.csv',panels)
            write_csv(directory/'training_curve.csv',curve);write_csv(directory/'validation_per_view.csv',validation)
            torch.save(dict(config=cfg,seed=seed,step=step,state_dict=net.state_dict(),optimizer=optimizer.state_dict()),directory/f'step_{step}.pt')
        if step%1000==0:
            save_json(ROOT/'status.json',dict(stage='TRAINING',scene=scene.name,config=cfg['id'],seed=seed,step=step,total=steps))
    clocks={name:int(optimizer.state[p]['step']) if p in optimizer.state else 0 for name,p in net.named_parameters()}
    assert all(x==steps//4 for x in exposures.values())
    assert scene.unchanged()
    np.save(directory/'selected_exposure_counts.npy',selected_counts.cpu().numpy())
    info=dict(config=cfg,seed=seed,steps=steps,ratio_exposures=exposures,empty_steps=empty,adam_clocks=clocks,
        seconds=time.monotonic()-start,parameters=sum(p.numel() for p in net.parameters()),master_frozen=True,
        checkpoint=str(directory/f'step_{steps}.pt'),validation_score=float(np.mean([r['psnr'] for r in validation if r['step']==steps])))
    save_json(done,info);print('TRAINED',cfg['id'],seed,steps,info['validation_score'],info['seconds'],flush=True)
    return info
