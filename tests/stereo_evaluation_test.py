"""Stereo评价校正域：身份标定、原生map逐像素等价、错误相机拒绝。"""
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from stereo_evaluation import StereoEvaluation


class StereoEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.config=self.root/'camera.yaml'
        lines=['%YAML:1.0']
        for name in ['Camera1','Camera2']:
            lines += [f'{name}.{key}: {value}' for key,value in dict(fx=80.,fy=80.,cx=31.5,cy=23.5,k1=0.,k2=0.,p1=0.,p2=0.).items()]
        lines += ['Stereo.T_c1_c2: !!opencv-matrix','   rows: 4','   cols: 4','   dt: f',
                  '   data: [1.,0.,0.,0.1,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]']
        self.config.write_text('\n'.join(lines)+'\n')
        self.image=np.random.default_rng(0).integers(0,256,(48,64,3),dtype=np.uint8)

    def test_identity(self):
        transform=StereoEvaluation(self.config,64,48,80.,80.)
        np.testing.assert_array_equal(transform.transform(self.image),self.image)

    def test_native_exact_maps(self):
        x,y=np.meshgrid(np.arange(64,dtype=np.float32),np.arange(48,dtype=np.float32));x+=.375;y-=.125
        maps=self.root/'native.yml.gz';f=cv2.FileStorage(str(maps),cv2.FILE_STORAGE_WRITE)
        for key,value in [('map1',x),('map2',y),('fx',80.),('fy',80.)]:f.write(key,value)
        f.release();transform=StereoEvaluation(self.config,64,48,80.,80.,maps)
        np.testing.assert_array_equal(transform.transform(self.image),cv2.remap(self.image,x,y,cv2.INTER_LINEAR))
        self.assertEqual(transform.metadata['map_source'],'native_training_opencv_maps')

    def test_wrong_intrinsics(self):
        with self.assertRaisesRegex(ValueError,'intrinsics disagree'):StereoEvaluation(self.config,64,48,81.,80.)

    def test_nonfinite_native_map(self):
        x,y=np.meshgrid(np.arange(64,dtype=np.float32),np.arange(48,dtype=np.float32));x[0,0]=np.nan
        maps=self.root/'bad.yml.gz';f=cv2.FileStorage(str(maps),cv2.FILE_STORAGE_WRITE)
        for key,value in [('map1',x),('map2',y),('fx',80.),('fy',80.)]:f.write(key,value)
        f.release()
        with self.assertRaisesRegex(ValueError,'nonfinite coordinates'):StereoEvaluation(self.config,64,48,80.,80.,maps)

    def test_optional_k3_is_preserved(self):
        with self.config.open('a') as out:out.write('Camera1.k3: 0.05\nCamera2.k3: 0.05\n')
        import unittest.mock
        original=cv2.stereoRectify
        with unittest.mock.patch('stereo_evaluation.cv2.stereoRectify',wraps=original) as operation:
            StereoEvaluation(self.config,64,48,80.,80.)
        self.assertEqual(operation.call_args.args[1].shape,(5,1))
        self.assertEqual(float(operation.call_args.args[1][4,0]),float(np.float32(.05)))

    def test_wrong_image_size(self):
        transform=StereoEvaluation(self.config,64,48,80.,80.)
        with self.assertRaisesRegex(ValueError,'dimensions changed'):transform.transform(self.image[:40])


if __name__=='__main__':unittest.main()
