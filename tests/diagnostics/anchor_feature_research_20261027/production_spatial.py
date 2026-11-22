"""生产SpatialLocalXYZ：共享神经场、固定统计、实时Gaussian属性，无私有训练表。"""
import math
import torch
from torch import nn


class SpatialCompensation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=nn.Linear(55,64);self.body=nn.Linear(64,64)
        self.ratio=nn.Sequential(nn.Linear(1,32),nn.ReLU(),nn.Linear(32,256))
        self.heads=nn.ModuleDict({name:nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,width))
            for name,width in [('xyz',3),('scale',3),('rotation',4),('other',4)]})
        for head in self.heads.values():nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.register_buffer('world_center',torch.zeros(3));self.register_buffer('world_spread',torch.ones(3))
        self.register_buffer('mean',torch.zeros(19));self.register_buffer('stddev',torch.ones(19))
        self.register_buffer('normalization_ready',torch.tensor(0,dtype=torch.long))

    @staticmethod
    def physical(xyz,scale,rotation,dc,logits,center,reference,world_center,world_spread):
        w,x,y,z=rotation.unbind(1)
        R=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),
            1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
        cov=(R*scale[:,None,:].square())@R.transpose(1,2)
        cov=cov/cov.diagonal(dim1=1,dim2=2).sum(1).clamp_min(1e-12)[:,None,None]
        shape=cov[:,[0,1,2,0,0,1],[0,1,2,1,2,2]]
        return torch.cat([(xyz-world_center)/world_spread,torch.asinh((xyz-center)/reference),
            scale.log(),dc.reshape(len(xyz),3),logits,shape],1)

    def input55(self,xyz,scale,rotation,dc,logits,center,reference):
        raw=self.physical(xyz,scale,rotation,dc,logits,center,reference,self.world_center,self.world_spread)
        coords=(xyz-self.world_center)/self.world_spread
        freq=2**torch.arange(6,device=xyz.device)*math.pi
        angle=coords[:,:,None]*freq
        return torch.cat([((raw-self.mean)/self.stddev).clamp(-15,15),angle.sin().flatten(1),angle.cos().flatten(1)],1)

    def forward(self,xyz,scale,rotation,dc,logits,center,reference,ratio):
        code=xyz.new_tensor([[2*(math.log(ratio)-math.log(.01))/(math.log(.15)-math.log(.01))-1]])
        g1,b1,g2,b2=self.ratio(code).chunk(4,1)
        hidden=torch.relu((1+g1)*self.encode(self.input55(xyz,scale,rotation,dc,logits,center,reference))+b1)
        hidden=torch.relu((1+g2)*self.body(hidden)+b2)
        parts=[head(hidden) for head in self.heads.values()];parts[0]=parts[0]*reference
        return torch.cat(parts,1)
