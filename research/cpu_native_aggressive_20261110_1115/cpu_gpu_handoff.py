"""CPU全部准备，pinned连续属性一次上传；有界槽位和CUDA事件保证生命周期。"""
from dataclasses import dataclass
import torch


@dataclass
class PreparedTicket:
    owner: object
    slot: int
    generation: int
    packet: dict
    ids: torch.Tensor
    stats: dict
    released: bool=False

    def wait(self,stream=None):
        if self.released:raise RuntimeError('CPU preparation ticket already released')
        (stream or torch.cuda.current_stream()).wait_event(self.owner.slots[self.slot]['ready'])
        return self.packet

    def finish(self,stream=None):
        if self.released:raise RuntimeError('CPU preparation ticket released twice')
        slot=self.owner.slots[self.slot]
        if slot['generation']!=self.generation:raise RuntimeError('CPU preparation slot generation mismatch')
        stream=stream or torch.cuda.current_stream();stream.wait_event(slot['ready']);slot['done'].record(stream)
        slot['busy']=False;slot['has_done']=True;self.released=True


class CpuGpuHandoff:
    def __init__(self,renderer,slots=3,device='cuda'):
        if slots not in [2,3]:raise ValueError('Use two or three bounded CPU/GPU buffers')
        self.renderer=renderer;self.device=torch.device(device);self.copy_stream=torch.cuda.Stream(device=self.device)
        self.cursor=0;self.slots=[]
        for _ in range(slots):
            self.slots.append(dict(host=torch.empty(renderer.total*renderer.width,dtype=torch.float32,pin_memory=True),
                ids=torch.empty(renderer.total,dtype=torch.long,pin_memory=True),
                gpu=torch.empty(renderer.total*renderer.width,device=self.device),
                ready=torch.cuda.Event(),done=torch.cuda.Event(),has_done=False,busy=False,generation=0))

    def prepare(self,camera,**request):
        index=self.cursor;slot=self.slots[index]
        if slot['busy']:raise RuntimeError('Consume/finish the previous ticket before reusing its slot')
        if slot['has_done']:slot['done'].synchronize()
        flat,ids,stats=self.renderer.prepare(camera,output=slot['host'],id_output=slot['ids'],flat_output=True,**request)
        gpu=slot['gpu'][:flat.numel()]
        with torch.cuda.stream(self.copy_stream):
            gpu.copy_(flat,non_blocking=True);slot['ready'].record(self.copy_stream)
        slot['generation']+=1;slot['busy']=True;self.cursor=(index+1)%len(self.slots)
        # IDs是槽位租约，不拷贝；需要长期保留的调用方显式clone。
        return PreparedTicket(self,index,slot['generation'],self.renderer.packet(gpu,len(ids)),ids,stats)

    def synchronize(self):
        self.copy_stream.synchronize()
        for slot in self.slots:
            if slot['has_done']:slot['done'].synchronize()

    def bytes(self):
        return dict(host=sum(x['host'].numel()*4+x['ids'].numel()*8 for x in self.slots),
            device=sum(x['gpu'].numel()*4 for x in self.slots),slots=len(self.slots))
