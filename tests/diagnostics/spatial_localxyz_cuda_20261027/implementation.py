"""保留原Parameter对象和顺序：reference / fused / graph三条等价执行路径。"""
import math
import json
from pathlib import Path
import torch
from torch import nn
from kernels import Pack,Film,Output


def validate_network(network):
    assert network.cfg.get('output')=='local' and network.cfg.get('bands')==6 and network.cfg.get('width')==64
    assert network.cfg.get('film',True) and network.cfg.get('context','none')=='none' and not network.cfg.get('knots',False)
    assert all(p.dtype==torch.float32 for p in network.parameters())
    assert not torch.backends.cuda.matmul.allow_tf32


def make_execution(network,backend=None):
    config=json.loads((Path(__file__).parent/'execution_config.json').read_text())
    validate_network(network)
    return Execution(network,config['backends'][backend or config['default_backend']])


class Core(nn.Module):
    def __init__(self,network,fused=False):super().__init__();self.network=network;self.fused=fused
    def forward(self,x,scale,condition):
        n=self.network;g1,b1,g2,b2=n.ratio(condition).chunk(4,1)
        if self.fused:
            h=Film.apply(n.encode(x),g1,b1);h=Film.apply(n.body(h),g2,b2)
        else:
            h=torch.relu((1+g1)*n.encode(x)+b1);h=torch.relu((1+g2)*n.body(h)+b2)
        parts=[head(h) for head in n.heads.values()]
        if self.fused:return Output.apply(*parts,scale)
        return torch.cat([parts[0]*scale,*parts[1:]],1)


class PackedCore(nn.Module):
    def __init__(self,network):super().__init__();self.core=Core(network,True)
    def forward(self,own,coords,anchor_scale,ids,freq,condition):
        return self.core(Pack.apply(own,coords,ids,freq),anchor_scale[ids],condition)


class CachedCore(nn.Module):
    def __init__(self,network):super().__init__();self.core=Core(network,True)
    def forward(self,features55,anchor_scale,ids,condition):
        return self.core(features55[ids],anchor_scale[ids],condition)


class Execution:
    def __init__(self,network,method='reference'):
        self.network=network;self.method=method;self.fused='fused' in method;self.graphs={}
        self.warmup_stream=None
        self.feature_cache=None
        self.core=Core(network,self.fused)
        self.freq=(2**torch.arange(6,device='cuda'))*math.pi
        self.conditions={}
    def condition(self,ratio):
        if ratio not in self.conditions:
            self.conditions[ratio]=torch.tensor([[2*(math.log(ratio)-math.log(.01))/(math.log(.15)-math.log(.01))-1]],device='cuda')
        return self.conditions[ratio]
    def invalidate(self):
        """仅在无未完成Forward/Backward的安全边界调用；释放旧shape/参数storage的Graph。"""
        self.graphs.clear()
        self.feature_cache=None
    def cached_inputs(self):
        f=self.network.features
        if f.own.requires_grad or f.coords.requires_grad:return None
        signature=(f.own.data_ptr(),f.own._version,f.coords.data_ptr(),f.coords._version,tuple(f.own.shape))
        if self.feature_cache is None or signature!=self.feature_cache[0]:
            with torch.no_grad():
                value=f.inputs(torch.arange(len(f.own),device=f.own.device),.01,self.network.cfg)
            self.feature_cache=(signature,value)
        return self.feature_cache[1]
    def pack(self,ids):
        f=self.network.features
        if self.method=='graph_cached55_fused':
            cached=self.cached_inputs()
            if cached is not None:return cached[ids]
        return Pack.apply(f.own,f.coords,ids,self.freq) if self.fused else f.inputs(ids,.01,self.network.cfg)
    def __call__(self,ratio,ids):
        if self.method=='reference':return self.network(ratio,ids)
        if self.method in ['graph_all_fused','graph_shared_fused','graph_cached55_fused'] and len(ids):
            f=self.network.features;cached=self.cached_inputs() if self.method=='graph_cached55_fused' else None
            args=(f.own,f.coords,f.anchor_scale,ids,self.freq,self.condition(ratio)) if cached is None else (cached,f.anchor_scale,ids,self.condition(ratio))
            key=('live' if cached is None else ('cached',cached.data_ptr()),len(ids),len(f.own),f.own.requires_grad,f.coords.requires_grad,f.anchor_scale.requires_grad,
                 f.own.data_ptr(),f.coords.data_ptr(),f.anchor_scale.data_ptr())
            if key not in self.graphs:
                with torch.enable_grad():
                    # 只读人口表同storage直接读；ids/ratio必须是Graph私有写入缓冲，不能别名污染调用方缓存。
                    readonly=3 if cached is None else 2
                    samples=tuple((t.detach() if j<readonly else t.detach().clone()).requires_grad_(t.requires_grad) for j,t in enumerate(args))
                    core=PackedCore(self.network) if cached is None else CachedCore(self.network)
                    if self.method in ['graph_shared_fused','graph_cached55_fused']:
                        from shared_graph_helper import make_shared
                        if self.warmup_stream is None:self.warmup_stream=torch.cuda.Stream()
                        self.graphs[key]=make_shared(core,samples,self.warmup_stream)
                    else:self.graphs[key]=torch.cuda.make_graphed_callables(core,samples,num_warmup_iters=3)
            return self.graphs[key](*args).clone()
        x=self.pack(ids);scale=self.network.features.anchor_scale[ids];condition=self.condition(ratio)
        key=(len(ids),x.requires_grad,scale.requires_grad)
        if self.method.startswith('graph') and len(ids):
            if key not in self.graphs:
                core=Core(self.network,self.fused)
                with torch.enable_grad():
                    samples=tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in (x,scale,condition))
                    self.graphs[key]=torch.cuda.make_graphed_callables(core,samples,num_warmup_iters=3)
            return self.graphs[key](x,scale,condition).clone()
        return self.core(x,scale,condition)
