"""变长Anchor成员集合编码；无Anchor/Gaussian私有参数。"""
import torch
from torch import nn


class AnchorSetEncoder(nn.Module):
    def __init__(self,schema=2):
        super().__init__()
        if schema not in (1,2,3):
            raise ValueError('Unknown Anchor Set Encoder schema')
        self.schema=schema
        if schema!=3:
            self.network=nn.Sequential(nn.Linear(10,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU())

    def forward(self,xyz,scale,dc,opacity,owners,offset_scales,ratio,centers,eligible):
        n,a=len(xyz),len(offset_scales)
        color=dc.squeeze(1) if dc.ndim==3 else dc
        if (xyz.shape!=(n,3) or scale.shape!=xyz.shape or color.shape!=xyz.shape or
                opacity.shape!=(n,1) or owners.shape!=(n,) or owners.dtype!=torch.long or
                offset_scales.shape!=(a,1) or centers.shape!=(a,3) or
                eligible.shape!=(n,) or eligible.dtype!=torch.bool):
            raise ValueError('Anchor set member attributes/owners/eligibility must align')
        if self.schema==3:
            valid=eligible[:,None]
            count=xyz.new_zeros((a,1)).index_add(0,owners,valid.to(xyz.dtype))
            divisor=count.clamp_min(1)
            local=((xyz-centers[owners])/offset_scales[owners]).masked_fill(~valid,0)
            mean=xyz.new_zeros((a,3)).index_add(0,owners,local)/divisor
            d=(local-mean[owners]).masked_fill(~valid,0)
            x,y,z=d.unbind(1)
            moments=torch.stack((x*x,x*y,x*z,y*y,y*z,z*z),1)
            covariance=xyz.new_zeros((a,6)).index_add(0,owners,moments)/divisor
            context=torch.cat((mean,covariance,count/10),1)[owners]
            physical=torch.cat((local,scale,color,opacity),1).masked_fill(~valid,0)
            return torch.cat((physical,ratio.log().expand(n,1),context),1)
        valid=torch.ones_like(eligible) if self.schema==1 else eligible
        count=xyz.new_zeros((a,1)).index_add(0,owners,valid.to(xyz.dtype)[:,None])
        divisor=count.clamp_min(1)
        origin=xyz.new_zeros((a,3)).index_add(0,owners,xyz)/divisor if self.schema==1 else centers
        local=(xyz-origin.index_select(0,owners))/offset_scales.index_select(0,owners)
        inputs=torch.cat((local,scale,color,opacity),1)
        if self.schema==2:
            inputs=inputs.masked_fill(~valid[:,None],0)
        hidden=self.network(inputs)
        if self.schema==2:
            hidden=hidden.masked_fill(~valid[:,None],0)
        mean=xyz.new_zeros((a,16)).index_add(0,owners,hidden)/divisor
        max_input=hidden if self.schema==1 else hidden.masked_fill(~valid[:,None],-float('inf'))
        maximum=torch.zeros_like(mean).scatter_reduce(0,owners[:,None].expand(n,16),max_input,'amax',include_self=False)
        if self.schema==2:
            maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        context=torch.cat((mean,maximum,count/10),1).index_select(0,owners)
        return torch.cat((hidden,context,ratio.log().expand(n,1)),1)
