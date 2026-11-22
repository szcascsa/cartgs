"""复用Selector首层8通道的Anchor上下文与单层残差投影。"""
import torch
from torch import nn


class AnchorContextResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(17,32,bias=False)
        nn.init.zeros_(self.projection.weight)

    def context(self,hidden,owners,eligible,anchors):
        n=len(hidden)
        if (hidden.shape!=(n,32) or owners.shape!=(n,) or owners.dtype!=torch.long or
                eligible.shape!=(n,) or eligible.dtype!=torch.bool):
            raise ValueError('Context residual hidden/owners/eligibility must align')
        count=hidden.new_zeros((anchors,1)).index_add(0,owners,eligible.to(hidden.dtype)[:,None])
        value=hidden[:,:8].masked_fill(~eligible[:,None],0)
        mean=hidden.new_zeros((anchors,8)).index_add(0,owners,value)/count.clamp_min(1)
        maximum=torch.zeros_like(mean).scatter_reduce(0,owners[:,None].expand(n,8),
            value.masked_fill(~eligible[:,None],-float('inf')),'amax',include_self=False)
        maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        return torch.cat((mean,maximum,count/10),1)

    def forward(self,hidden,owners,eligible,anchors):
        return self.projection(self.context(hidden,owners,eligible,anchors))[owners]
