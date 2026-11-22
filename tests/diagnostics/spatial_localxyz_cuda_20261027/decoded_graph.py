"""原生Gaussian解码与NN共同capture；不改变exp/sigmoid/Quaternion算法或Renderer。"""
import json
from pathlib import Path
import torch
from torch import nn
from baseline_profile import r
from implementation import Execution,CachedCore
from shared_graph_helper import make_shared


def decode(delta,ids,bases):
    xyz,scale,rot,dc,opacity=[v[ids] for v in bases]
    identity=torch.zeros_like(delta[:,6:10]);identity[:,0]=1
    return (xyz+delta[:,:3],scale*delta[:,3:6].exp(),
        r.rt._normalize_quaternion(r.rt._multiply_quaternion(r.rt._normalize_quaternion(identity+delta[:,6:10]),rot)),
        dc+delta[:,11:14].unsqueeze(1),(opacity+delta[:,10:11]).sigmoid())


class DecodedCore(nn.Module):
    def __init__(self,network):super().__init__();self.core=CachedCore(network)
    def forward(self,features55,anchor_scale,ids,condition,*bases):
        delta=self.core(features55,anchor_scale,ids,condition)
        return (delta,*decode(delta,ids,bases))


class DecodedExecution:
    def __init__(self,network,render_bases,method='decoded_graph'):
        assert method in ['decoded_graph','reference','graph_cached55_fused']
        self.method=method;self.network=network;self.base=Execution(network,'reference' if method=='reference' else 'graph_cached55_fused');self.bases=render_bases;self.graphs={};self.stream=torch.cuda.Stream()
    def __call__(self,ratio,ids):
        if self.method!='decoded_graph':
            delta=self.base(ratio,ids);return (delta,*decode(delta,ids,(*self.bases[:4],self.bases[5])))
        features=self.base.cached_inputs();bases=(*self.bases[:4],self.bases[5])
        if features is None or not len(ids):
            delta=self.base(ratio,ids);return (delta,*decode(delta,ids,bases))
        args=(features,self.network.features.anchor_scale,ids,self.base.condition(ratio),*bases)
        key=(len(ids),tuple((tuple(t.shape),t.requires_grad,t.data_ptr()) for j,t in enumerate(args) if j not in [2,3]))
        if key not in self.graphs:
            with torch.enable_grad():
                samples=tuple((t.detach().clone() if j in [2,3] else t.detach()).requires_grad_(t.requires_grad) for j,t in enumerate(args))
                self.graphs[key]=make_shared(DecodedCore(self.network),samples,self.stream)
        return tuple(t.clone() for t in self.graphs[key](*args))
    def render(self,scene,ratio,ids,view):
        return render_decoded(scene,ids,self(ratio,ids),view)
    def invalidate(self):self.graphs.clear();self.base.invalidate()


def render_decoded(scene,ids,values,view):
    delta,xyz,scale,rot,dc,alpha=values
    rest=scene.render_bases[4][ids]
    if not len(ids):return scene.bg[:,None,None].expand_as(view['gt']).contiguous()
    return r.rt._make_rasterizer(view['cam'],scene.g.active_sh_degree,scene.bg,1.)(
        means3D=xyz.contiguous(),means2D=torch.zeros_like(xyz),dc=dc.contiguous(),shs=rest.contiguous(),
        colors_precomp=None,opacities=alpha.contiguous(),scales=scale.contiguous(),rotations=rot.contiguous(),cov3D_precomp=None)[0]


def make_render_execution(network,render_bases,backend=None):
    """性能研究入口；生产不接入。返回delta及原格式解码属性。"""
    from implementation import validate_network
    validate_network(network)
    config=json.loads((Path(__file__).parent/'execution_config.json').read_text())
    return DecodedExecution(network,render_bases,config['render_backends'][backend or config['default_render_backend']])
