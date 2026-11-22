"""以训练时相同的左相机校正域评价Stereo；可读取原生OpenCV导出的精确map。"""
import hashlib
from pathlib import Path
import cv2
import numpy as np


class StereoEvaluation:
    def __init__(self, camera_config, width, height, fx, fy, native_maps=None):
        self.width,self.height=int(width),int(height)
        config=Path(camera_config)
        storage=cv2.FileStorage(str(config),cv2.FILE_STORAGE_READ)
        if not storage.isOpened():raise ValueError('Cannot open stereo camera calibration: '+str(config))
        try:
            intrinsics=[];distortion=[]
            for name in ['Camera1','Camera2']:
                values=[storage.getNode(name+'.'+key).real() for key in ['fx','fy','cx','cy']]
                if values[0]<=0 or values[1]<=0:raise ValueError('Missing stereo camera intrinsics')
                intrinsics.append(np.array([[values[0],0,values[2]],[0,values[1],values[3]],[0,0,1]],np.float32).astype(np.float64))
                keys=['k1','k2','p1','p2']
                if not storage.getNode(name+'.k3').empty():keys.append('k3')
                distortion.append(np.array([storage.getNode(name+'.'+key).real() for key in keys],np.float32).astype(np.float64).reshape(-1,1))
            transform=storage.getNode('Stereo.T_c1_c2').mat()
            if transform is None or transform.shape!=(4,4):raise ValueError('Missing Stereo.T_c1_c2')
            transform=transform.astype(np.float32)
            rotation=transform[:3,:3].T.copy();translation=(-rotation@transform[:3,3]).reshape(3,1)
            r1,_,p1,_,_,_,_=cv2.stereoRectify(*[intrinsics[0],distortion[0],intrinsics[1],distortion[1]],
                (self.width,self.height),rotation.astype(np.float64),translation.astype(np.float64),
                flags=cv2.CALIB_ZERO_DISPARITY,alpha=-1,newImageSize=(self.width,self.height))
            self.map1,self.map2=cv2.initUndistortRectifyMap(intrinsics[0],distortion[0],r1,p1[:3,:3],
                (self.width,self.height),cv2.CV_32F)
            expected_fx,expected_fy=float(p1[0,0]),float(p1[1,1])
        finally:storage.release()
        source='reconstructed_opencv_maps'
        if native_maps:
            archive=cv2.FileStorage(str(native_maps),cv2.FILE_STORAGE_READ)
            if not archive.isOpened():raise ValueError('Cannot open native stereo maps')
            try:
                self.map1,self.map2=archive.getNode('map1').mat(),archive.getNode('map2').mat()
                expected_fx,expected_fy=archive.getNode('fx').real(),archive.getNode('fy').real()
            finally:archive.release()
            source='native_training_opencv_maps'
        if self.map1 is None or self.map2 is None or self.map1.shape!=(height,width) or self.map2.shape!=(height,width):
            raise ValueError('Stereo map dimensions disagree with saved model')
        if self.map1.dtype!=np.float32 or self.map2.dtype!=np.float32:
            raise ValueError('Stereo evaluation maps must be FP32')
        if not np.isfinite(self.map1).all() or not np.isfinite(self.map2).all():
            raise ValueError('Stereo evaluation maps contain nonfinite coordinates')
        if abs(expected_fx-fx)>.05 or abs(expected_fy-fy)>.05:
            raise ValueError('Stereo rectified intrinsics disagree with saved model')
        self.metadata=dict(protocol='stereo_left_rectified_training_domain',camera_config=str(config.resolve()),
            calibration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),map_source=source,
            width=width,height=height,fx=fx,fy=fy,padding='entire image; no crop; no mask; native BORDER_CONSTANT',
            map1_sha256=hashlib.sha256(self.map1.tobytes()).hexdigest(),
            map2_sha256=hashlib.sha256(self.map2.tobytes()).hexdigest())

    def transform(self,image):
        if image.shape[:2]!=(self.height,self.width):raise ValueError('Stereo source image dimensions changed')
        return cv2.remap(image,self.map1,self.map2,cv2.INTER_LINEAR)
