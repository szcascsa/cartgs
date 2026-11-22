from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from plyfile import PlyData, PlyElement
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elastic_budget import (DEFAULTS, classify_protection, mix_selector_probability,
                            sample_selector_st, ratio_loss, protection_statistics)
from selector import build_selector_mask, load_selector_metadata

class ProtectionTest(unittest.TestCase):
    def metadata(self, maturity):
        n = len(maturity)
        return dict(mapping_iteration=1200, maturity=torch.tensor(maturity),
                    drift_ema=torch.zeros(n), cached_uniform=torch.arange(n).float()/max(1,n))

    def test_handoff_gradient_and_hard_st(self):
        probability = torch.tensor([.8,.8,.8], requires_grad=True)
        maturity = torch.tensor([0.,.5,1.], requires_grad=True)
        mixed = mix_selector_probability(probability,maturity,.1)
        self.assertTrue(torch.allclose(mixed,torch.tensor([.1,.45,.8])))
        mask = sample_selector_st(mixed,torch.tensor([.05,.5,.5]))
        self.assertEqual(mask.tolist(),[1.,0.,1.])
        mask.sum().backward()
        self.assertTrue(torch.equal(probability.grad,maturity.detach()))
        self.assertIsNone(maturity.grad)

    def test_expected_budget_all_rows(self):
        self.assertAlmostEqual(ratio_loss(torch.tensor([.6,.8]),.1).item(),1.2,places=6)
        for ratio in (.01,.05,.10,.15):
            neutral = mix_selector_probability(torch.rand(1000),torch.zeros(1000),ratio)
            self.assertAlmostEqual(neutral.sum().item(),ratio*1000,places=4)
            self.assertLess(ratio_loss(neutral,ratio).item(),1e-5)

    def test_compatibility_and_disabled(self):
        metadata=self.metadata([0.,.5,.79,.8,1.])
        c=classify_protection(metadata,DEFAULTS)
        self.assertEqual(c["mature"].tolist(),[False,False,False,True,True])
        self.assertEqual(c["mature"].dtype,torch.bool)
        self.assertTrue(classify_protection(metadata,DEFAULTS,False)["maturity"].eq(1).all())

    def test_eval_all_rows_uniform_temperature_and_stats(self):
        class Gaussians:
            get_xyz=torch.zeros(1000,3)
            get_rotation=torch.zeros(1000,4)
            get_scaling=torch.ones(1000,3)
        class Model:
            def __call__(self,xyz,rotation,scaling,ratio,tau):
                self.ratio=ratio.clone(); self.tau=tau; self.rows=xyz.shape[0]
                return torch.full((xyz.shape[0],),.9)
        metadata=self.metadata([0.]*500+[1.]*500)
        model=Model()
        mask,stats=build_selector_mask(model,Gaussians(),metadata,.1,DEFAULTS,temperature=.7)
        self.assertEqual(model.rows,1000)
        self.assertTrue(model.ratio.eq(.1).all()); self.assertEqual(model.tau,.7)
        expected=(metadata["cached_uniform"] < torch.tensor([.1]*500+[.9]*500)).float()
        self.assertTrue(torch.equal(mask,expected))
        self.assertAlmostEqual(stats["expected_active"],500,places=3)
        self.assertEqual(stats["target_active"],100)
        self.assertEqual(stats["actual_active"],int(mask.sum()))
        # Compatibility threshold cannot change the Selector population.
        other,_=build_selector_mask(model,Gaussians(),metadata,.1,
                                   dict(DEFAULTS,compatibility_mature_threshold=.2),temperature=.7)
        self.assertTrue(mask.equal(other))
        self.assertFalse(any(k in stats for k in ("protected_gaussians","protection_budget_target","warmup_gaussians")))

    def test_distribution_and_empty(self):
        metadata=self.metadata([0.,.2,.4,.6,.8,1.])
        c=classify_protection(metadata,DEFAULTS)
        stats=protection_statistics(metadata,c)
        self.assertEqual([stats["maturity_bucket_"+str(i)] for i in range(5)],[1,1,1,1,2])
        empty=self.metadata([])
        self.assertEqual(protection_statistics(empty,classify_protection(empty,DEFAULTS))["maturity_mean"],0)

    def test_checkpoint_schema(self):
        data=np.zeros(3,dtype=[("maturity","f4"),("drift_ema","f4"),("cached_uniform","f4")])
        data["maturity"]=[0,.5,1]; data["drift_ema"]=[0,.1,.2]; data["cached_uniform"]=[.1,.2,.3]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"metadata.ply"
            PlyData([PlyElement.describe(data,"vertex")],
                    comments=["protection_schema continuous_cycle_geometric",
                              "protection_mapping_iteration 42"]).write(path)
            restored=load_selector_metadata(path,"cpu")
            self.assertEqual(restored["maturity"].tolist(),[0.,.5,1.])
            self.assertEqual(restored["mapping_iteration"],42)
            PlyData([PlyElement.describe(data,"vertex")],
                    comments=["protection_schema evidence_aware"]).write(path)
            with self.assertRaises(RuntimeError): load_selector_metadata(path,"cpu")

    def test_invalid_config_and_ratio(self):
        with self.assertRaises(ValueError):
            mix_selector_probability(torch.ones(1),torch.zeros(1),float("nan"))
        with self.assertRaises(ValueError):
            classify_protection(self.metadata([0.]),dict(DEFAULTS,beta_slow=.6))

if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
