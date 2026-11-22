"""同一fr1地图和原SpatialLocalXYZ终点；仅生成只读原始属性、权重与选择集合。"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/prmi/zwj/spatial-localxyz-cuda-20261027')
from baseline_profile import fixture,r


class Carrier(torch.nn.Module):
    def forward(self):return self.input_0


torch.backends.cuda.matmul.allow_tf32=False
sc,f,net,payload=fixture();module=Carrier()
owners=sc.source.anchors.rows
values=[sc.g.get_xyz,sc.g.get_scaling,sc.g.get_rotation,sc.g.get_features_dc,sc.g._opacity,
    sc.source.anchors.centers[owners],sc.source.anchors.offset_scales[owners]]
for j,value in enumerate(values):module.register_buffer('input_'+str(j),value.detach())
for name,value in net.named_parameters():module.register_buffer('param__'+name.replace('.','__'),value.detach())
with torch.no_grad():
    for j,ratio in enumerate(r.rt.RATIOS):
        ids=f.sets[ratio,'same_k'];module.register_buffer('ids_'+str(j),ids)
        module.register_buffer('expected_'+str(j),net(ratio,ids))
torch.jit.script(module).save(str(ROOT/'real_fixture.pt'))
(ROOT/'fixture_manifest.json').write_text(json.dumps(dict(N=sc.N,M=sc.M,source_checkpoint='SpatialLocalXYZ/5100/step_12000.pt',selection=f.manifest,scope='live field K-row physical input recomputation and active Gaussian input gradients; no Renderer in field microbenchmark'),indent=2))
print('REAL_FIXTURE_READY',sc.N,sc.M,flush=True)
