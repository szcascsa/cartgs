"""SpatialLocalXYZ上的抽象Anchor Feature；不产生Gaussian私有物理参数。"""
import copy
import math
import torch
from torch import nn


class AnchorField(nn.Module):
    def __init__(self,base,data,config):
        super().__init__();self.base=base;self.data=data;self.config=config
        self.kind=config.get('feature','none');self.fusion=config.get('fusion','mid');self.d=config.get('dim',8)
        self.ablation='normal'
        if self.kind=='none':return
        # Extra initialization cannot perturb the original model initialization/RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(config.get('branch_seed',401));torch.cuda.manual_seed_all(config.get('branch_seed',401))
            if self.kind in ['learned','hybrid','fake','global','multi_only','single_only']:
                self.embedding=nn.Parameter(torch.zeros(1 if self.kind=='global' else data.M,self.d,device='cuda'))
            if self.kind in ['context','hybrid']:
                self.context=nn.Sequential(nn.Linear(data.context.shape[1],32),nn.ReLU(),nn.Linear(32,self.d)).cuda()
                self.context_initial=copy.deepcopy(self.context).requires_grad_(False)
            width=self.d*(2 if self.kind=='hybrid' else 1)
            if self.fusion=='head':
                self.inject=nn.ModuleDict({name:nn.Linear(width,64,bias=False) for name in self.base.heads}).cuda()
                for layer in self.inject.values():nn.init.normal_(layer.weight,0,.02)
            else:
                self.inject=nn.Linear(width*(2 if self.fusion=='film' else 1),128 if self.fusion=='film' else 64,bias=False).cuda()
                nn.init.normal_(self.inject.weight,0,.02)
        if config.get('freeze_base'):self.base.requires_grad_(False)
        if config.get('freeze_feature') and hasattr(self,'embedding'):self.embedding.requires_grad_(False)

    def latent(self,ids,ratio):
        owner=self.data.fake_owners[ids] if self.kind=='fake' else self.data.owners[ids]
        if self.ablation=='shuffle':owner=self.data.shuffle_anchors[owner]
        values=[]
        if hasattr(self,'embedding'):
            e=self.embedding[torch.zeros_like(owner) if self.kind=='global' else owner]
            if self.kind=='multi_only':e=e*(self.data.counts[owner]>1)[:,None]
            if self.kind=='single_only':e=e*(self.data.counts[owner]==1)[:,None]
            values.append(e)
        if hasattr(self,'context'):
            c=self.data.context[owner];values.append(self.context(c)-self.context_initial(c))
        e=torch.cat(values,1) if len(values)>1 else values[0]
        if self.ablation=='zero':e=e*0
        if self.ablation in ['no_single','no_small','no_large']:
            counts=self.data.counts[self.data.owners[ids]]
            keep=counts!=1 if self.ablation=='no_single' else ~((counts>=2)&(counts<=3)) if self.ablation=='no_small' else counts<4
            e=e*keep[:,None]
        return e

    def forward(self,ratio,ids,policy='same_k'):
        n=self.base;code=2*(math.log(ratio)-math.log(.01))/(math.log(.15)-math.log(.01))-1
        g1,b1,g2,b2=n.ratio(self.data.x.new_tensor([[code]])).chunk(4,1)
        h=n.encode(self.data.x[ids]);e=None
        if self.kind!='none':
            e=self.latent(ids,ratio)
            if self.fusion=='early':h=h+self.inject(e)
        h=torch.relu((1+g1)*h+b1)
        if self.kind!='none' and self.fusion=='mid':h=h+self.inject(e)
        h=n.body(h)
        if self.kind!='none' and self.fusion=='film':
            gain,bias=self.inject(torch.cat([e,code*e],1)).chunk(2,1)
            h=torch.relu((1+g2+gain)*h+b2+bias)
        else:h=torch.relu((1+g2)*h+b2)
        parts=[]
        for name,head in n.heads.items():
            if self.kind!='none' and self.fusion=='head':value=head[2](torch.relu(head[0](h)+self.inject[name](e)))
            else:value=head(h)
            if name=='xyz':value=value*self.data.anchor_scale[ids]
            parts.append(value)
        return torch.cat(parts,1)

    def feature_parameters(self):
        return [self.embedding] if hasattr(self,'embedding') and self.embedding.requires_grad else []

    def extra_parameters(self):
        feature={id(p) for p in self.feature_parameters()}
        return [p for name,p in self.named_parameters() if not name.startswith('base.') and p.requires_grad and id(p) not in feature]
