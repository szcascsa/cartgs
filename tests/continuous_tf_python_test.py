"""验证 Python 评估与在线 TF 的 continuous/cutoff 语义一致。"""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from transform_field import transformed_attributes

class Gaussian:
    def __init__(self,device):
        self.get_xyz=torch.zeros(4,3,device=device)
        self.get_scaling=torch.ones(4,3,device=device)
        self.get_rotation=torch.tensor([[1.,0.,0.,0.]]*4,device=device)

class Field:
    def __init__(self,epsilon):
        self.maturity_compute_epsilon=epsilon
        self.grid=self
        self.rows=0
    def inside_mask(self,xyz): return (xyz.abs()<=2).all(1)
    def __call__(self,xyz,ratio):
        self.rows+=len(xyz)
        rotation=xyz.new_zeros(len(xyz),4); rotation[:,1]=.6
        return torch.full_like(xyz,.3),torch.full_like(xyz,.2),rotation

class ContinuousTF(unittest.TestCase):
    def test_no_hard_gate_on_primary_path(self):
        root=Path(__file__).resolve().parents[1]
        mapper=(root/'src/gaussian_mapper.cpp').read_text(encoding='utf-8')
        primary=mapper.split('GaussianMapper::accumulateLodGradients(',1)[1].split('bool GaussianMapper::isStopped()',1)[0]
        for text in [primary,(root/'src/gi_teacher.cpp').read_text(),(root/'src/transform_field.cpp').read_text(),
                     (root/'src/historical_gradient_render.cpp').read_text(),(root/'python/transform_field.py').read_text()]:
            for gate in ['getSelectorMatureMask','mature_only','compatibility_mature_threshold','mature_count','protected_count']:
                self.assertNotIn(gate,text)
        self.assertIn('first_valid_maturity_cycle_iteration',mapper)

    def test_cpu_and_cuda(self):
        self.assertTrue(torch.cuda.is_available(), '验收必须包含 CUDA')
        for device in ['cpu','cuda']:
            g=Gaussian(device)
            m=torch.tensor([0.,.5,1.,.049],device=device)
            base=[g.get_xyz.clone(),g.get_scaling.clone(),g.get_rotation.clone()]
            field=Field(0)
            attrs=transformed_attributes(field,g,torch.ones(4,device=device),m,.1)
            self.assertEqual(field.rows,4)
            self.assertTrue(torch.equal(attrs['xyz'][0],g.get_xyz[0]))
            self.assertTrue(torch.equal(attrs['rotation'][0],g.get_rotation[0]))
            self.assertTrue(torch.allclose(attrs['xyz'][1],attrs['xyz'][2]*.5))
            self.assertTrue(torch.allclose(attrs['scaling'][1].log(),attrs['scaling'][2].log()*.5))
            self.assertGreater(attrs['rotation'][1,1].item(),0)
            self.assertLess(attrs['rotation'][1,1].item(),attrs['rotation'][2,1].item())
            cutoff=Field(.05)
            result=transformed_attributes(cutoff,g,torch.ones(4,device=device),m,.1)
            self.assertEqual(cutoff.rows,2)
            self.assertTrue(torch.equal(result['xyz'][[0,3]],g.get_xyz[[0,3]]))
            self.assertTrue(torch.equal(result['rotation'][[0,3]],g.get_rotation[[0,3]]))
            for before,after in zip(base,[g.get_xyz,g.get_scaling,g.get_rotation]):
                self.assertTrue(torch.equal(before,after))
            skipped=Field(.05)
            self.assertIsNone(transformed_attributes(skipped,g,torch.ones(4,device=device),torch.zeros_like(m),.1))
            self.assertEqual(skipped.rows,0)

if __name__=='__main__': unittest.main()
