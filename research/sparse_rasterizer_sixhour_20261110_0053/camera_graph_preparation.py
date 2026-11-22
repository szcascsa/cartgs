"""冻结推理相机执行图；保持原BMM、inverse与逐帧错误检查。"""
import torch
from camera_preparation import CameraPreparer


class GraphCameraPreparer(CameraPreparer):
    def __init__(self,camera_class):
        super().__init__(camera_class)
        self.plan=None

    @torch.no_grad()
    def create(self,width,height,fovx,fovy,world_view_transform):
        camera=self.camera_class.__new__(self.camera_class)
        camera.image_width=width;camera.image_height=height;camera.FoVx=fovx;camera.FoVy=fovy
        camera.znear=.01;camera.zfar=100.
        camera.world_view_transform=torch.tensor(world_view_transform,dtype=torch.float32).transpose(0,1).cuda()
        stream=torch.cuda.current_stream()
        key=(width,height,fovx,fovy,.01,100.,stream.device_index,stream.cuda_stream,torch.get_default_dtype())
        if key!=self.key:
            self.projection=self.projection_builder(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).transpose(0,1).cuda()
            self.key=key;self.rebuilds+=1;self.plan=None
        camera.projection_matrix=self.projection
        if self.plan is None:
            self.view=torch.empty_strided((4,4),camera.world_view_transform.stride(),device='cuda',dtype=torch.float32)
            self.view.copy_(camera.world_view_transform)
            self.view.unsqueeze(0).bmm(self.projection.unsqueeze(0))
            torch.linalg.inv_ex(self.view,check_errors=False)
            self.plan=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.plan):
                self.full=self.view.unsqueeze(0).bmm(self.projection.unsqueeze(0)).squeeze(0)
                self.inverse,self.info=torch.linalg.inv_ex(self.view,check_errors=False)
        self.view.copy_(camera.world_view_transform);self.plan.replay()
        torch.ops.aten._linalg_check_errors(self.info,'linalg.inv',is_matrix=True)
        camera.full_proj_transform=self.full.clone()
        camera.camera_center=self.inverse[3,:3].clone()
        return camera
