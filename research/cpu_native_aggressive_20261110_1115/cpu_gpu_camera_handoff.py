"""属性和本帧全部相机矩阵同包上传，避免把相机H2D成本留在计时外。"""
import copy
import torch
from cpu_gpu_handoff import PreparedTicket


class CpuGpuCameraHandoff:
    def __init__(self,renderer,slots=3,device='cuda'):
        if slots not in [2,3]:raise ValueError('Use two or three bounded CPU/GPU buffers')
        self.renderer=renderer;self.device=torch.device(device);self.copy_stream=torch.cuda.Stream(device=self.device)
        self.cursor=0;self.slots=[];self.capacity=renderer.total*renderer.width
        for _ in range(slots):
            self.slots.append(dict(host=torch.empty(self.capacity+51,dtype=torch.float32,pin_memory=True),
                ids=torch.empty(renderer.total,dtype=torch.long,pin_memory=True),gpu=torch.empty(self.capacity+51,device=self.device),
                ready=torch.cuda.Event(),done=torch.cuda.Event(),has_done=False,busy=False,generation=0))

    def prepare(self,camera,**request):
        index=self.cursor;slot=self.slots[index]
        if slot['busy']:raise RuntimeError('Consume/finish the previous ticket before reusing its slot')
        if slot['has_done']:slot['done'].synchronize()
        flat,ids,stats=self.renderer.prepare(camera,output=slot['host'][:self.capacity],id_output=slot['ids'],flat_output=True,**request)
        n=flat.numel();offset=n
        for name in ['world_view_transform','projection_matrix','full_proj_transform']:
            slot['host'][offset:offset+16].view(4,4).copy_(getattr(camera,name));offset+=16
        slot['host'][offset:offset+3].copy_(camera.camera_center)
        gpu=slot['gpu'][:n+51]
        with torch.cuda.stream(self.copy_stream):
            gpu.copy_(slot['host'][:n+51],non_blocking=True);slot['ready'].record(self.copy_stream)
        gpu_camera=copy.copy(camera);offset=n
        for name in ['world_view_transform','projection_matrix','full_proj_transform']:
            setattr(gpu_camera,name,gpu[offset:offset+16].view(4,4));offset+=16
        gpu_camera.camera_center=gpu[offset:offset+3]
        slot['generation']+=1;slot['busy']=True;self.cursor=(index+1)%len(self.slots)
        ticket=PreparedTicket(self,index,slot['generation'],self.renderer.packet(gpu[:n],len(ids)),ids,stats)
        ticket.camera=gpu_camera;return ticket

    def synchronize(self):
        self.copy_stream.synchronize()
        for slot in self.slots:
            if slot['has_done']:slot['done'].synchronize()

    def bytes(self):
        return dict(host=sum(x['host'].numel()*4+x['ids'].numel()*8 for x in self.slots),
            device=sum(x['gpu'].numel()*4 for x in self.slots),slots=len(self.slots))
