"""当前位姿在C++中准备，一次140字节异步上传；不缓存任何旧位姿结果。"""
from prep_study import *


class NativeCameraPreparer:
    def __init__(self, camera_class):
        self.camera_class=camera_class
        self.projection_builder=camera_class.__init__.__globals__['getProjectionMatrix']
        self.ext=dp.deployment_extension();self.key=None

    @torch.no_grad()
    def create_from_pose(self,width,height,fovx,fovy,pose):
        stream=torch.cuda.current_stream();key=(width,height,fovx,fovy,stream.device_index,stream.cuda_stream,torch.get_default_dtype())
        if key!=self.key:
            self.projection=self.projection_builder(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).transpose(0,1).contiguous()
            self.gpu_projection=self.projection.cuda(non_blocking=True);self.key=key
        raw=torch.as_tensor(np.asarray(pose),dtype=torch.float64).contiguous()
        flat=self.ext._preparation_camera(raw,self.projection,self.gpu_projection)
        cam=self.camera_class.__new__(self.camera_class)
        cam.image_width=width;cam.image_height=height;cam.FoVx=fovx;cam.FoVy=fovy;cam.znear=.01;cam.zfar=100.
        cam.world_view_transform=flat[:16].view(4,4);cam.full_proj_transform=flat[16:32].view(4,4);cam.camera_center=flat[32:35]
        cam.projection_matrix=self.gpu_projection
        return cam

    def from_source(self,source,frame):
        c=source.scene.calibration
        return self.create_from_pose(c['width'],c['height'],2*math.atan(c['width']/(2*c['fx'])),
            2*math.atan(c['height']/(2*c['fy'])),source.scene.views[frame]['pose'])
