"""提取真实布局与原生重复中Adam放大误差的定位证据。"""
import json
from pathlib import Path
import torch

ROOT=Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
class TensorArchive(torch.nn.Module):
    def forward(self,x: torch.Tensor)->torch.Tensor:
        return x
source=Path('/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply/selector.pt')
checkpoint=torch.jit.load(str(source),map_location='cpu')
archive=TensorArchive()
for key,value in checkpoint.named_buffers():
    archive.register_buffer(key.replace('.','_'),value.detach())
torch.jit.script(archive).save(str(ROOT/'anchor_input_buffers.pt'))
a=torch.jit.load(str(ROOT/'native1/lod_0.050000_parity.pt'),map_location='cpu').state_dict()
b=torch.jit.load(str(ROOT/'native2/lod_0.050000_parity.pt'),map_location='cpu').state_dict()
counts=torch.bincount(a['owner'])
result=dict(N=len(a['owner']),A=len(counts),members={str(i):int((counts==i).sum()) for i in range(1,11)},
            levels={str(i):int((a['anchor_levels']==i).sum()) for i in range(3)},
            missing_master_gradients=[i for i in range(6) if 'gradient_'+str(i) not in a],
            unique_gaussian_uids=len(torch.unique(a['gaussian_uid'])))
result['frozen_head_last_weight_max']=float(checkpoint.state_dict()['compensation.2.weight'].abs().max())
result['frozen_head_last_bias_max']=float(checkpoint.state_dict()['compensation.2.bias'].abs().max())
key='parameter_after_4';diff=(a[key]-b[key]).abs();flat=int(diff.argmax());row,column=flat//3,flat%3
result['adam_amplification']=dict(parameter='log_scale',row=row,column=column,
    max_abs=float(diff.max()),not_allclose_elements=int((~torch.isclose(a[key],b[key],atol=2e-5,rtol=2e-4)).sum()),
    values=[float(x[key][row,column]) for x in [a,b]],
    gradients=[float(x['gradient_4'][row,column]) for x in [a,b]],
    before_m=[float(x['before_4_m'][row,column]) for x in [a,b]],
    before_v=[float(x['before_4_v'][row,column]) for x in [a,b]],
    gradient_max_abs=float((a['gradient_4']-b['gradient_4']).abs().max()))
(ROOT/'fixture_details.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
