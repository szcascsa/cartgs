"""验证 compact Selector/TF 的评估语义，以及 entry=0 的数值/梯度 parity。"""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from elastic_entry import elastic_maturity,elastic_selector_probability
from elastic_budget import mix_selector_probability,DEFAULTS
from selector import build_selector_mask
from transform_field import transformed_attributes
from continuous_tf_python_test import Gaussian,Field

class ElasticEntry(unittest.TestCase):
    def test_cpu_cuda(self):
        self.assertTrue(torch.cuda.is_available())
        for device in ['cpu','cuda']:
            m=torch.tensor([.2,.3,.65,1.],device=device,requires_grad=True)
            scores=torch.full((4,),.8,device=device,requires_grad=True)
            seen=[]
            def forward(rows):
                seen.extend(rows.tolist());return scores[rows]
            p=elastic_selector_probability(m,m,.1,.3,forward)
            self.assertEqual(seen,[1,2,3])
            self.assertTrue(torch.allclose(p,torch.tensor([.1,.1,.45,.8],device=device)))
            p.sum().backward()
            self.assertIsNone(m.grad)
            self.assertTrue(torch.allclose(scores.grad,torch.tensor([0.,0.,.5,1.],device=device)))
            old=scores.detach().clone().requires_grad_(True)
            new=scores.detach().clone().requires_grad_(True)
            expected=mix_selector_probability(old,m,.1)
            actual=elastic_selector_probability(m,m,.1,0,lambda rows:new)
            self.assertTrue(torch.equal(expected,actual))
            expected.sum().backward();actual.sum().backward()
            self.assertTrue(torch.equal(old.grad,new.grad))
            g=Gaussian(device);metadata=dict(maturity=m.detach(),drift_ema=torch.zeros_like(m),
                cached_uniform=torch.zeros_like(m),mapping_iteration=10)
            class Spy:
                rows=0
                def __call__(self,xyz,rotation,scaling,ratio,tau):
                    self.rows+=len(xyz);return torch.full((len(xyz),),.8,device=xyz.device)
            model=Spy()
            mask,stats=build_selector_mask(model,g,metadata,.1,dict(DEFAULTS,entry_maturity=.3))
            self.assertEqual(model.rows,3);self.assertEqual(stats['selector_candidate_count'],3)
            self.assertEqual(mask[0].item(),1) # Neutral low row remains in full admission.
            field=Field(0);field.entry_maturity=.3
            attrs=transformed_attributes(field,g,mask,m.detach(),.1)
            self.assertEqual(field.rows,3)
            self.assertTrue(torch.equal(attrs['xyz'][0],g.get_xyz[0]))
            self.assertTrue(torch.equal(attrs['rotation'][1],g.get_rotation[1]))
            self.assertTrue(torch.allclose(attrs['xyz'][2],attrs['xyz'][3]*.5))
            self.assertTrue(torch.allclose(attrs['scaling'][2].log(),attrs['scaling'][3].log()*.5))

if __name__=='__main__':unittest.main()
