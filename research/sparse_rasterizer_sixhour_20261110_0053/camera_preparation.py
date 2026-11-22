"""冻结推理的固定内参预计算；每帧仍重新计算位姿、BMM与inverse。"""
import math
import numpy as np
import torch


class CameraPreparer:
    def __init__(self,camera_class):
        self.camera_class=camera_class
        self.projection_builder=camera_class.__init__.__globals__['getProjectionMatrix']
        self.key=None
        self.projection=None
        self.rebuilds=0

    @torch.no_grad()
    def create(self,width,height,fovx,fovy,world_view_transform):
        camera=self.camera_class.__new__(self.camera_class)
        camera.image_width=width;camera.image_height=height;camera.FoVx=fovx;camera.FoVy=fovy
        camera.znear=.01;camera.zfar=100.
        camera.world_view_transform=torch.tensor(world_view_transform,dtype=torch.float32).transpose(0,1).cuda()
        stream=torch.cuda.current_stream()
        key=(width,height,fovx,fovy,camera.znear,camera.zfar,stream.device_index,stream.cuda_stream,torch.get_default_dtype())
        if key!=self.key:
            self.projection=self.projection_builder(znear=camera.znear,zfar=camera.zfar,fovX=fovx,fovY=fovy).transpose(0,1).cuda()
            self.key=key;self.rebuilds+=1
        camera.projection_matrix=self.projection
        camera.full_proj_transform=camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0)).squeeze(0)
        camera.camera_center=camera.world_view_transform.inverse()[3,:3]
        return camera

    def from_source(self,source,frame):
        c=source.scene.calibration
        return self.create(c['width'],c['height'],2*math.atan(c['width']/(2*c['fx'])),
            2*math.atan(c['height']/(2*c['fy'])),np.linalg.inv(np.asarray(source.scene.views[frame]['pose'])))
