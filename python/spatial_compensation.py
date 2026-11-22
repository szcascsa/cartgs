"""SpatialLocalXYZ：共享神经场、固定统计、实时属性；Early16可选Anchor抽象Feature。"""
import math
import torch
from torch import nn


class SpatialCompensation(nn.Module):
    def enable_view_input(self):
        if not self.offset_only or self.ratio_feature_width!=48 or not hasattr(self,'ratio_projection'):
            raise ValueError('View conditioning requires Residual48')
        if hasattr(self,'view_projection'):return
        with torch.random.fork_rng(devices=[]):
            self.view_projection=nn.Linear(8,64,bias=False).to(device=self.encode.weight.device,dtype=self.encode.weight.dtype)
        nn.init.zeros_(self.view_projection.weight)
        self.register_buffer('view_input_schema',torch.tensor(1,dtype=torch.long,device=self.encode.weight.device))

    def decode_scale(self, master_log_scale, residual):
        if hasattr(self, 'softplus_scale_schema'):
            scale = master_log_scale.exp() * torch.nn.functional.softplus(residual + 0.5413248546129181)
            return scale.clamp_min(1.1754943508222875e-38)
        if hasattr(self, 'physical_scale_additive_schema'):
            return master_log_scale.exp() + residual
        return (master_log_scale + residual).exp()

    def __init__(self,early16=False,direct_ratio=False,fourier_bands=6,ratio_feature_width=0,ratio_code=False,ratio_bounds=(.1,.8),shared_anchor128=False,offset_only=False,bounded_xyz=False):
        super().__init__()
        if direct_ratio and not early16:raise ValueError('Direct ratio input requires Early16')
        self.direct_ratio=direct_ratio
        if fourier_bands not in (0,3,6):raise ValueError('Fourier bands must be 0, 3 or 6')
        self.fourier_bands=fourier_bands
        if ratio_feature_width not in ((0,128) if shared_anchor128 else (0,32,48)) or ratio_feature_width and (early16 or direct_ratio or fourier_bands):raise ValueError('Unsupported compensation Feature layout')
        if shared_anchor128 and (ratio_feature_width!=128 or not ratio_code):raise ValueError('Shared Anchor128 requires Feature128 and ratio code')
        if ratio_feature_width==48 and (not offset_only or not ratio_code or shared_anchor128):raise ValueError('Ratio48 requires offset3 and continuous ratio code without a gate')
        if offset_only and not (shared_anchor128 or ratio_feature_width==48):raise ValueError('Offset-only compensation requires Anchor128 or ratio48')
        if bounded_xyz and not offset_only:raise ValueError('Bounded XYZ requires offset-only Anchor128')
        self.offset_only=offset_only
        self.bounded_xyz=bounded_xyz
        self.ratio_feature_width=ratio_feature_width
        self.geometry_head_variant=0
        self.encode=nn.Linear((3 if offset_only else 19)+ratio_feature_width if ratio_feature_width else (56 if direct_ratio else 55),64);self.body=nn.Linear(64,64)
        if ratio_code:
            if ratio_feature_width not in (32,48,128) or not 0<ratio_bounds[0]<ratio_bounds[1]<=1:raise ValueError('Ratio code requires valid bounds')
            self.ratio_encoding_values=tuple(ratio_bounds)
            self.register_buffer('ratio_encoding_bounds',torch.tensor(ratio_bounds,dtype=torch.float64))
            self.ratio_projection=nn.Linear(3,64,bias=False)
            nn.init.zeros_(self.ratio_projection.weight)
        if not direct_ratio and not ratio_feature_width:self.ratio=nn.Sequential(nn.Linear(1,32),nn.ReLU(),nn.Linear(32,256))
        self.heads=nn.ModuleDict({name:nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,width))
            for name,width in [('xyz',3),('scale',3),('rotation',4),('other',4)]})
        for head in self.heads.values():nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.register_buffer('world_center',torch.zeros(3));self.register_buffer('world_spread',torch.ones(3))
        self.register_buffer('mean',torch.zeros(19));self.register_buffer('stddev',torch.ones(19))
        self.register_buffer('normalization_ready',torch.tensor(0,dtype=torch.long))
        if fourier_bands!=6 and not ratio_feature_width:
            mask=torch.zeros(55);mask[:19]=1
            for axis in range(3):
                mask[19+axis*6:19+axis*6+fourier_bands]=1
                mask[37+axis*6:37+axis*6+fourier_bands]=1
            self.register_buffer('fourier_mask',mask)
        if early16:
            self.early_feature=nn.Linear(16,64,bias=False)
            nn.init.normal_(self.early_feature.weight,0,.02)
        if shared_anchor128:
            self.ratio_gate=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,128))
            nn.init.zeros_(self.ratio_gate[-1].weight);nn.init.zeros_(self.ratio_gate[-1].bias)

    @staticmethod
    def physical(xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread,log_scale_input=False):
        w,x,y,z=rotation.unbind(1)
        R=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),
            1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
        if log_scale_input:
            shift=scale.max(1,keepdim=True).values.clamp_min(.5*math.log(1e-12)).detach()
            variance=(2*(scale-shift)).exp()
            cov=(R*variance[:,None,:])@R.transpose(1,2)
            floor=(math.log(1e-12)-2*shift).exp()[:,None,:]
            trace=cov.diagonal(dim1=1,dim2=2).sum(1)[:,None,None]
            cov=cov/torch.where(trace>=floor,trace,floor)
        else:
            cov=(R*scale[:,None,:].square())@R.transpose(1,2)
            cov=cov/cov.diagonal(dim1=1,dim2=2).sum(1).clamp_min(1e-12)[:,None,None]
        shape=cov[:,[0,1,2,0,0,1],[0,1,2,1,2,2]]
        return torch.cat([(xyz-world_center)/world_spread,torch.asinh((xyz-center)/reference),
            scale if log_scale_input else scale.log(),dc.reshape(len(xyz),3),logits,shape],1)

    def input55(self,xyz,scale,rotation,dc,logits,center,reference,log_scale_input=False):
        raw=self.physical(xyz,scale,rotation,dc,logits,center,reference,self.world_center,self.world_spread,log_scale_input)
        coords=(xyz-self.world_center)/self.world_spread
        freq=2**torch.arange(6,device=xyz.device)*math.pi
        angle=coords[:,:,None]*freq
        value=torch.cat([((raw-self.mean)/self.stddev).clamp(-15,15),angle.sin().flatten(1),angle.cos().flatten(1)],1)
        return value*self.fourier_mask if hasattr(self,'fourier_mask') else value

    def configure_geometry_heads(self,mode):
        if mode not in (1,2):raise ValueError('Unknown geometry head variant')
        if mode==1:
            del self.heads['other']
        else:
            del self.heads
            self.geometry_mlp=nn.Sequential(nn.Linear(64,176),nn.ReLU(),nn.Linear(176,10)).to(self.encode.weight.device,self.encode.weight.dtype)
        self.register_buffer('geometry_head_schema',torch.tensor(mode,dtype=torch.long,device=self.encode.weight.device))
        self.geometry_head_variant=mode

    def forward(self,xyz,scale,rotation,dc,logits,center,reference,ratio,log_scale_input=False,anchor_feature=None,view_condition=None):
        if view_condition is not None and (not hasattr(self,'view_projection') or view_condition.shape!=(len(xyz),8)):
            raise ValueError('Invalid Compensation view condition')
        if hasattr(self,'ratio_projection'):
            lo,hi=self.ratio_encoding_values
            t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
            ratio_code=xyz.new_tensor([[t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)]])
        if self.ratio_feature_width:
            if self.offset_only:
                inputs=((torch.asinh((xyz-center)/reference)-self.mean[3:6])/self.stddev[3:6]).clamp(-15,15)
            else:
                raw=self.physical(xyz,scale,rotation,dc,logits,center,reference,self.world_center,self.world_spread,log_scale_input)
                inputs=((raw-self.mean)/self.stddev).clamp(-15,15)
            if anchor_feature is None or anchor_feature.shape!=(len(xyz),self.ratio_feature_width):raise ValueError('Physical19 requires selected compensation features')
            if hasattr(self,'ratio_gate'):anchor_feature=anchor_feature*(1+self.ratio_gate(ratio_code).tanh())
            inputs=torch.cat((inputs,anchor_feature),1)
        else:
            code=xyz.new_tensor([[2*(math.log(ratio)-math.log(.01))/(math.log(.15)-math.log(.01))-1]])
            inputs=self.input55(xyz,scale,rotation,dc,logits,center,reference,log_scale_input)
        if self.direct_ratio:inputs=torch.cat((inputs,code.expand(len(xyz),1)),1)
        encoded=self.encode(inputs)
        if hasattr(self,'ratio_projection'):
            encoded=encoded+self.ratio_projection(ratio_code)
        if hasattr(self,'early_feature'):
            if anchor_feature is None or anchor_feature.shape!=(len(xyz),16):raise ValueError('Early16 requires selected K x 16 Anchor Features')
            encoded=encoded+self.early_feature(anchor_feature)
        if hasattr(self,'view_projection') and (view_condition is not None or torch.is_grad_enabled()):
            encoded=encoded+self.view_projection(view_condition.detach() if view_condition is not None else xyz.new_zeros((1,8)))
        if self.direct_ratio or self.ratio_feature_width:hidden=torch.relu(self.body(torch.relu(encoded)))
        else:
            g1,b1,g2,b2=self.ratio(code).chunk(4,1)
            hidden=torch.relu((1+g1)*encoded+b1)
            hidden=torch.relu((1+g2)*self.body(hidden)+b2)
        mode=self.geometry_head_variant
        if mode==2:
            value=self.geometry_mlp(hidden)
            parts=[value[:,:3],value[:,3:6],value[:,6:10]]
        else:parts=[head(hidden) for head in self.heads.values()]
        if mode:parts.append(hidden.new_zeros((len(hidden),4)))
        if self.bounded_xyz:parts[0]=parts[0]/torch.sqrt(1+parts[0].square().sum(1,keepdim=True)/4)
        parts[0]=parts[0]*reference
        return torch.cat(parts,1)
