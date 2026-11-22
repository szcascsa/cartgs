"""纯共享神经补偿：确定性属性、邻域token与连续ratio，无身份参数表。"""
import math
import torch
from torch import nn


def mlp(n,w,out):return nn.Sequential(nn.Linear(n,w),nn.ReLU(),nn.Linear(w,out))


class NodeProjection(nn.Module):
    def __init__(self,old):
        super().__init__();width=old.out_features//4
        self.layers=nn.ModuleList([nn.Linear(old.in_features,width) for _ in range(4)])
        with torch.no_grad():
            for j,layer in enumerate(self.layers):layer.weight.copy_(old.weight[j*width:(j+1)*width]);layer.bias.copy_(old.bias[j*width:(j+1)*width])
        self.condition=(0,0,0.)
    def forward(self,h):
        lo,hi,t=self.condition
        v=self.layers[hi](h)
        return v if lo==hi else (1-t)*self.layers[lo](h)+t*v


class ZeroCondition(nn.Module):
    def __init__(self,width):super().__init__();self.width=width
    def forward(self,code):return code.new_zeros(1,4*self.width)


class Features:
    @torch.no_grad()
    def __init__(self,scene,sets):
        self.scene=scene;self.sets=sets;xyz=scene.g.get_xyz
        center=xyz.median(0).values;spread=(xyz-center).square().mean(0).sqrt().clamp_min(.1)
        coords=(xyz-center)/spread
        scale=scene.g.get_scaling;logscale=scale.log()
        q=scene.g.get_rotation;w,x,y,z=q.unbind(1)
        R=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),
            1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
        cov=(R*scale[:,None,:].square())@R.transpose(1,2)
        cov=cov/cov.diagonal(dim1=1,dim2=2).sum(1).clamp_min(1e-12)[:,None,None]
        shape=cov[:,[0,1,2,0,0,1],[0,1,2,1,2,2]]
        base=torch.cat([coords,torch.asinh(scene.attrs[:,:3]),logscale,scene.attrs[:,6:9],scene.g._opacity,shape],1)
        self.mean=base.mean(0);self.std=base.std(0).clamp_min(.1)
        self.own=((base-self.mean)/self.std).clamp(-15,15).detach()  # 19维；只冻结统计，不生成可训练行。
        self.raw=scene.attrs.detach();self.coords=coords.detach();self.xyz=xyz.detach()
        self.anchor_scale=scene.source.anchors.offset_scales[scene.owners].detach()
        self.rotation=R.detach();self.neighbors=[]
        for start in range(0,scene.N,512):self.neighbors.append(torch.cdist(xyz[start:start+512],xyz).topk(32,largest=False).indices)
        self.neighbors=torch.cat(self.neighbors)
        self.contexts={}
        for key,ids in sets.items():
            selected=torch.zeros(scene.N,device='cuda');selected[ids]=1
            ns=self.neighbors[ids];relative=xyz[ns]-xyz[ids,None]
            radius=relative.norm(dim=2).quantile(.5,dim=1).clamp_min(.02)
            rel=relative/radius[:,None,None]
            tokens=torch.cat([self.own[ns],rel,rel.norm(dim=2,keepdim=True),selected[ns,None]],2)
            self.contexts[key]=dict(tokens=tokens,neighbor_ids=ns,relative=relative,selected=selected[ns],radius=radius)
        self.donations={}
        self.moments={}
        self.covariance=cov.detach()*scale.square().sum(1)[:,None,None]
        self.feature_sha_source='immutable Master attributes and deterministic kNN; no learned per-ID input'

    @torch.no_grad()
    def transport_graphs(self):
        if self.donations:return self.donations
        sc=self.scene
        for key,ids in self.sets.items():
            distances,receivers=torch.cdist(self.xyz,self.xyz[ids]).topk(4,largest=False)
            donor_ids=torch.arange(sc.N,device='cuda')[:,None].expand(-1,4).reshape(-1)
            receivers=receivers.reshape(-1);targets=ids[receivers]
            weights=distances.clamp_min(.005).reciprocal().square()
            weights=(weights/weights.sum(1,keepdim=True)).flatten()
            rel=(self.xyz[donor_ids]-self.xyz[targets])/.1
            selected=torch.zeros(sc.N,device='cuda');selected[ids]=1
            tokens=torch.cat([self.own[donor_ids],torch.asinh(rel),rel.norm(dim=1,keepdim=True).log1p(),selected[donor_ids,None]],1)
            self.donations[key]=dict(tokens=tokens,receivers=receivers,weights=weights,donor_ids=donor_ids)
        return self.donations

    @torch.no_grad()
    def moment_features(self):
        if self.moments:return self.moments
        self.transport_graphs();sc=self.scene
        for key,ids in self.sets.items():
            ctx=self.donations[key];rec=ctx['receivers'];donors=ctx['donor_ids'];count=len(ids)
            weights=ctx['weights']*sc.g.get_opacity[donors].flatten()
            norm=torch.zeros(count,device='cuda').index_add(0,rec,weights).clamp_min(1e-8)
            def pool(values):return torch.zeros(count,values.shape[1],device='cuda').index_add(0,rec,values*weights[:,None])/norm[:,None]
            relative=self.xyz[donors]-self.xyz[ids[rec]];mean=pool(relative)
            second=self.covariance[donors]+relative[:,:,None]*relative[:,None,:]
            covariance=pool(second.flatten(1)).reshape(-1,3,3)-mean[:,:,None]*mean[:,None,:]
            old=sc.g.get_scaling[ids].square().sum(1).clamp_min(1e-9)
            shape=covariance[:,[0,1,2,0,0,1],[0,1,2,1,2,2]]/old[:,None]
            shape=torch.asinh(shape)
            dc=pool(sc.attrs[donors,6:9])-sc.attrs[ids,6:9]
            raw=torch.cat([torch.asinh(mean/.1),shape,dc,torch.log1p(norm[:,None])],1) # 13维
            self.moments[key]=raw.detach()
        return self.moments

    def inputs(self,ids,ratio,cfg):
        if cfg.get('raw_only',False):return torch.cat([self.raw[ids],torch.full((len(ids),1),math.log(ratio),device='cuda')],1)
        value=self.own[ids];bands=cfg.get('bands',0)
        if cfg.get('no_world',False) or cfg.get('no_shape',False):
            value=value.clone()
            if cfg.get('no_world',False):value[:,:3]=0
            if cfg.get('no_shape',False):value[:,13:19]=0
        if bands:
            freq=(2**torch.arange(bands,device='cuda'))*math.pi
            angle=self.coords[ids,:,None]*freq
            if cfg.get('no_world',False):angle=torch.zeros_like(angle)
            value=torch.cat([value,angle.sin().flatten(1),angle.cos().flatten(1)],1)
        return value


class SharedNeural(nn.Module):
    def __init__(self,features,cfg,seed):
        super().__init__();self.features=features;self.cfg=cfg;self.kind=cfg.get('kind','self');width=cfg.get('width',64)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        inp=11 if cfg.get('raw_only',False) else 19+6*cfg.get('bands',0)
        self.encode=nn.Linear(inp,width)
        self.body=nn.Linear(width,width)
        self.ratio=mlp(1,32,4*width) if cfg.get('film',True) else nn.Linear(1,width)
        self.context_kind=cfg.get('context','none')
        if self.context_kind=='transport':features.transport_graphs()
        if self.context_kind=='moment':features.moment_features()
        if self.context_kind!='none':
            self.token=mlp(13 if self.context_kind=='moment' else 24,width,width)
            if self.context_kind in ['attention','transport']:
                self.key=nn.Linear(width,width);self.query=nn.Linear(width,width)
                self.weight=nn.Linear(width,1)
            self.fuse=nn.Linear(3*width,width)
        self.knots=cfg.get('knots',False)
        groups=[('xyz',3),('covariance',6),('other',4)] if cfg.get('output')=='affine' else [('xyz',3),('scale',3),('rotation',4),('other',4)]
        self.heads=nn.ModuleDict({label:nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,n*(4 if self.knots else 1)))
            for label,n in groups})
        for head in self.heads.values():nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        if cfg.get('fold_constant',False):
            assert self.knots and cfg.get('film',True)
            with torch.no_grad():
                g1,b1,g2,b2=self.ratio(torch.zeros(1,1)).chunk(4,1)
                for layer,gain,bias in [(self.encode,g1,b1),(self.body,g2,b2)]:
                    layer.weight.mul_((1+gain).flatten()[:,None]);layer.bias.mul_((1+gain).flatten()).add_(bias.flatten())
            self.ratio=ZeroCondition(width)
        if cfg.get('sparse_node_clock',False):
            assert self.knots
            for head in self.heads.values():head[-1]=NodeProjection(head[-1])
        self.to('cuda')

    def forward(self,ratio,ids,policy='same_k'):
        f=self.features;cfg=self.cfg;x=f.inputs(ids,ratio,cfg)
        h=self.encode(x);condition=0. if self.knots else 2*(math.log(ratio)-math.log(.01))/(math.log(.15)-math.log(.01))-1
        code=torch.tensor([[condition]],device='cuda')
        modulation=self.ratio(code)
        if cfg.get('film',True):
            g1,b1,g2,b2=modulation.chunk(4,1);h=torch.relu((1+g1)*h+b1)
        else:h=torch.relu(h+modulation)
        if self.context_kind!='none':
            if self.context_kind=='moment':
                raw_moment=f.moments[ratio,policy]
                if cfg.get('zero_moment',False):raw_moment=torch.zeros_like(raw_moment)
                embedded=self.token(raw_moment)
                h=h+self.fuse(torch.cat([h,embedded,h*embedded],1))
            else:h=self.context_forward(h,ratio,policy)
        h=self.body(h)
        h=torch.relu((1+g2)*h+b2) if cfg.get('film',True) else torch.relu(h)
        if self.knots:
            nodes=[.01,.05,.1,.15];hi=next((j for j,v in enumerate(nodes) if ratio<=v+1e-8),3)
            lo=hi if hi==0 or ratio>=.15 or abs(nodes[hi]-ratio)<1e-8 else hi-1
            weight=0. if lo==hi else (ratio-nodes[lo])/(nodes[hi]-nodes[lo])
            if cfg.get('sparse_node_clock',False):
                for head in self.heads.values():head[-1].condition=(lo,hi,weight)
        parts={name:head(h) for name,head in self.heads.items()}
        if self.knots and not cfg.get('sparse_node_clock',False):
            parts={name:(1-weight)*value.reshape(len(ids),4,-1)[:,lo]+weight*value.reshape(len(ids),4,-1)[:,hi]
                for name,value in parts.items()}
        xyz=parts['xyz']
        if cfg.get('output')=='local':xyz=xyz*f.anchor_scale[ids]
        if cfg.get('output')=='affine':return torch.cat([xyz,parts['covariance'],parts['other']],1)
        return torch.cat([xyz,parts['scale'],parts['rotation'],parts['other']],1)

    def context_forward(self,h,ratio,policy):
            f=self.features;ids=f.sets[ratio,policy]
            ctx=f.donations[ratio,policy] if self.context_kind=='transport' else f.contexts[ratio,policy];tokens=ctx['tokens']
            embedded=self.token(tokens)
            if self.context_kind=='transport':
                rec=ctx['receivers'];score=torch.sigmoid(self.weight(torch.tanh(self.key(embedded)+self.query(h)[rec]))).flatten()*ctx['weights']
                pooled=torch.zeros(len(ids),h.shape[1],device='cuda').index_add(0,rec,embedded*score[:,None])
                sums=torch.zeros(len(ids),device='cuda').index_add(0,rec,score)
                normalized=pooled/sums.clamp_min(1e-6)[:,None]
                # normalized学习内容，log1p(sum)保留覆盖质量，全部来自同一共享网络和确定性邻接。
                h=h+self.fuse(torch.cat([h,normalized,normalized*torch.log1p(sums[:,None])],1))
            else:
                h=self.aggregate_local(h,embedded,ctx)
            return h

    def aggregate_local(self,h,embedded,ctx):
            retained=ctx['selected'].bool();lost=~retained
            def aggregate(mask):
                if self.context_kind=='attention':
                    energy=self.weight(torch.tanh(self.key(embedded)+self.query(h)[:,None])).squeeze(-1)
                    weights=(energy.masked_fill(~mask,-1e4)).softmax(1)*mask
                    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
                else:weights=mask.float()/mask.sum(1,keepdim=True).clamp_min(1)
                return (embedded*weights[:,:,None]).sum(1)
            return h+self.fuse(torch.cat([h,aggregate(retained),aggregate(lost)],1))


def audit_parameters(net):
    records=[]
    for name,p in net.named_parameters():
        # 只能由Linear权重/偏置产生；不通过N/M维数碰巧相同的启发式判断。
        module_path,leaf=name.rsplit('.',1);module=net.get_submodule(module_path)
        assert isinstance(module,nn.Linear) and leaf in ['weight','bias'],(name,type(module))
        records.append(dict(name=name,shape=list(p.shape),count=p.numel(),owner=type(module).__name__,shared=True))
    assert not any(isinstance(m,nn.Embedding) for m in net.modules())
    assert not list(net.buffers()),'数据在Features只读对象中，不注册场景参数表'
    return records


class RatioExperts(nn.Module):
    """ratio节点的共享函数专家；任何参数均不按Gaussian/Anchor身份索引。"""
    def __init__(self,features,cfg,seed):
        super().__init__();self.features=features;self.cfg=cfg
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        inp=19+6*cfg.get('bands',6)
        self.encoder=nn.Sequential(nn.Linear(inp,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.experts=nn.ModuleList([nn.ModuleDict({name:nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,n))
            for name,n in [('xyz',3),('scale',3),('rotation',4),('other',4)]}) for _ in range(4)])
        for expert in self.experts:
            for head in expert.values():nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.to('cuda')
    def forward(self,ratio,ids,policy='same_k'):
        nodes=[.01,.05,.1,.15];hi=next((j for j,v in enumerate(nodes) if ratio<=v+1e-8),3)
        lo=hi if hi==0 or ratio>=.15 or abs(nodes[hi]-ratio)<1e-8 else hi-1
        t=0. if lo==hi else (ratio-nodes[lo])/(nodes[hi]-nodes[lo])
        h=self.encoder(self.features.inputs(ids,ratio,self.cfg))
        def output(j):return torch.cat([head(h) for head in self.experts[j].values()],1)
        value=output(hi)
        return value if lo==hi else (1-t)*output(lo)+t*value


class SinusoidalField(nn.Module):
    """SIREN式坐标/属性隐式场；专用频率初始化，不学习任何个体latent。"""
    def __init__(self,features,cfg,seed):
        super().__init__();self.features=features;self.cfg=cfg;self.omega=30.
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);w=64
        self.first=nn.Linear(20,w);self.middle=nn.Linear(w,w)
        self.heads=nn.ModuleDict({name:nn.ModuleList([nn.Linear(w,w),nn.Linear(w,n*(4 if cfg.get('knots') else 1))])
            for name,n in [('xyz',3),('scale',3),('rotation',4),('other',4)]})
        with torch.no_grad():
            for layer in [self.first,self.middle]+[head[0] for head in self.heads.values()]:
                bound=1/layer.in_features if layer is self.first else math.sqrt(6/layer.in_features)/self.omega
                layer.weight.uniform_(-bound,bound);layer.bias.uniform_(-bound,bound)
            for head in self.heads.values():head[-1].weight.zero_();head[-1].bias.zero_()
        self.to('cuda')
    def forward(self,ratio,ids,policy='same_k'):
        code=0. if self.cfg.get('knots') else 2*(math.log(ratio)-math.log(.01))/(math.log(.15)-math.log(.01))-1
        x=torch.cat([self.features.own[ids],torch.full((len(ids),1),code,device='cuda')],1)
        h=torch.sin(self.omega*self.first(x));h=torch.sin(self.omega*self.middle(h))
        parts=[head[-1](torch.sin(self.omega*head[0](h))) for head in self.heads.values()]
        if self.cfg.get('knots'):
            nodes=[.01,.05,.1,.15];hi=next((j for j,v in enumerate(nodes) if ratio<=v+1e-8),3)
            lo=hi if hi==0 or ratio>=.15 or abs(nodes[hi]-ratio)<1e-8 else hi-1
            t=0. if lo==hi else (ratio-nodes[lo])/(nodes[hi]-nodes[lo])
            parts=[(1-t)*value.reshape(len(ids),4,-1)[:,lo]+t*value.reshape(len(ids),4,-1)[:,hi] for value in parts]
        return torch.cat(parts,1)
