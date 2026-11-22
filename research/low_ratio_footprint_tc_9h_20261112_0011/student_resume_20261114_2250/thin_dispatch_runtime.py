"""同一182二进制的轻量入口；只复用零元素Tensor、绑定方法与相机内参标量。"""
import math
import torch
from encoded_sign_runtime import EncodedSignRenderer


class ThinDispatchRenderer(EncodedSignRenderer):
    def __init__(self,mode=182):
        super().__init__(mode)
        self._draw=self.native.render
        self._empty=None
        self._fov=None

    def precompile(self,n,camera,background):
        super().precompile(n,camera,background)
        self._empty=background.new_empty(0)

    def render(self,camera,packet,background):
        if torch.is_grad_enabled():raise RuntimeError('Thin dispatch is frozen inference only')
        x=packet['inputs'];means=x['means3D']
        if not len(means):return [background[:,None,None].expand(-1,camera.image_height,camera.image_width).clone(),torch.zeros(0,device=means.device,dtype=torch.int32)]
        if self._empty is None:self._empty=means.new_empty(0)
        empty=self._empty
        key=(camera.FoVx,camera.FoVy)
        if key!=self._fov:
            self._tanx,self._tany=math.tan(key[0]*.5),math.tan(key[1]*.5);self._fov=key
        colors,scales,rotations,covariance,dc,sh=x['colors_precomp'],x['scales'],x['rotations'],x['cov3D_precomp'],x['dc'],x['shs']
        return self._draw(background,means,colors if colors is not None else empty,x['opacities'],
            scales if scales is not None else empty,rotations if rotations is not None else empty,1.,
            covariance if covariance is not None else empty,camera.world_view_transform,camera.full_proj_transform,self._tanx,self._tany,
            camera.image_height,camera.image_width,dc if dc is not None else empty,sh if sh is not None else empty,
            packet['sh_degree'],camera.camera_center,5,256,32,False)
