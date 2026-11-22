"""新实现加载上一轮真实 10 维权重：原生结果 exact 不变，Python/导出属性兼容。"""
import sys
from pathlib import Path
import subprocess
import torch

root=Path('/home/prmi/zwj/shared-selector-appearance-20261021');repo=root/'source';out=root/'validation'
old=Path('/home/prmi/zwj/shared-selector-compensation-20261021')
cp=old/'candidate_seed2026/10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
subprocess.run([str(repo/'bin/anchor_checkpoint_probe'),str(cp/'selector.pt'),str(cloud/'point_cloud.ply'),
    str(cloud/'selector_metadata.ply'),str(out/'v1_native.pt'),'explicit_blocks'],check=True)
before=torch.jit.load(str(old/'validation/shared_compensation_native.pt'),map_location='cuda').state_dict()
after=torch.jit.load(str(out/'v1_native.pt'),map_location='cuda').state_dict()
for key in before:
    if key!='native_forward_ms':assert torch.equal(before[key],after[key]),key
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,build_selector_mask,compensated_attributes
from gaussian_model import GaussianModel
model=load_selector(str(cp/'selector.pt'),'cuda');g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
assert model.compensation[-1].out_features==10
config=dict(drift_temperature=.1,beta_fast=.6,beta_slow=.15,compatibility_mature_threshold=.8,entry_maturity=0)
for i,r in enumerate([.01,.05,.1,.15]):
    mask,_=build_selector_mask(model,g,metadata,r,config,1.,False);ids=torch.nonzero(mask).flatten()
    assert torch.equal(mask.bool(),before['mask_'+str(i)])
    result=compensated_attributes(model,g,mask,r)
    assert set(result)=={'xyz','scaling','rotation'}
    for name,native in [('xyz','xyz'),('scaling','scale'),('rotation','rotation')]:
        assert torch.allclose(result[name][ids],before['compensated_'+native+'_'+str(i)],atol=2e-6,rtol=2e-5)
print('PASS: real v1 checkpoint native outputs EXACT unchanged, Python selected rows and geometry compatible')
