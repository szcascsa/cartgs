"""全局Selector的可选视角残差；相机编码与C++ view_residual.h一致。"""
import torch
from torch import nn


@torch.no_grad()
def anchor_view_code(centers,scales,camera_center,view,projection,height,width,near_plane,scene_scale):
    if centers.ndim!=2 or centers.shape[1]!=3 or scales.shape!=(len(centers),1) or min(height,width,near_plane)<=0:
        raise ValueError('Invalid view residual camera/Anchor geometry')
    vector=camera_center.reshape(1,3)-centers
    direction=vector/vector.square().sum(1,keepdim=True).sqrt().clamp_min(1e-6)
    homogeneous=torch.cat((centers,torch.ones_like(centers[:,:1])),1)
    camera,clip=homogeneous@view,homogeneous@projection
    def safe_w(w):
        return torch.where(w.abs()<near_plane,torch.where(w<0,torch.full_like(w,-near_plane),torch.full_like(w,near_plane)),w)
    screen=(clip[:,:2]/safe_w(clip[:,3:4])).clamp(-4,4)
    signs=centers.new_tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                             [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    corners=centers[:,None,:]+scales[:,None,:]*.5*signs
    corners4=torch.cat((corners,torch.ones_like(corners[:,:,:1])),2)
    projected,depth=corners4@projection,(corners4@view)[:,:,2]
    ndc=projected[:,:,:2]/safe_w(projected[:,:,3:4])
    extent=ndc.max(1).values-ndc.min(1).values
    pixels=extent*centers.new_tensor([float(width)/height,1.])*.5
    longest=pixels.max(1).values.clamp(0,1e4)
    longest=torch.where(depth.min(1).values<=near_plane,torch.full_like(longest,1e4),longest)
    longest=torch.where(depth.max(1).values<=near_plane,torch.zeros_like(longest),longest)
    return torch.cat((direction,torch.asinh(camera[:,2]/scene_scale)[:,None],screen,torch.log1p(longest)[:,None]),1)


class SelectorViewResidual(nn.Module):
    def __init__(self,schema=1):
        super().__init__()
        if schema not in (1,2):raise ValueError('Unknown Selector view residual schema')
        self.hidden=nn.Linear(18 if schema==2 else 39,16)
        self.output=nn.Linear(16,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('scene_scale',torch.ones(()))
        self.register_buffer('initialized',torch.zeros((),dtype=torch.long))

    @torch.no_grad()
    def initialize(self,master_xyz):
        if bool(self.initialized):return
        if master_xyz.ndim!=2 or master_xyz.shape[1]!=3 or not len(master_xyz):
            raise ValueError('View normalization needs a nonempty Master')
        self.scene_scale.copy_((master_xyz.max(0).values-master_xyz.min(0).values).max().clamp_min(1e-4))
        self.initialized.fill_(1)

    @property
    def schema(self):return 2 if self.hidden.in_features==18 else 1

    def forward(self,value,view_code):
        if value.ndim!=2 or value.shape[1]!=(11 if self.schema==2 else 32) or view_code.shape!=(len(value),7):
            raise ValueError('View residual input must match checkpoint schema (physical11 or hidden32) and view7')
        return self.output(self.hidden(torch.cat((value.detach(),view_code.detach()),1)).relu())
