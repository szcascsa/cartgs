"""验证原生Spatial checkpoint与生产Python实现；不依赖研究网络解码器。"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--validation',default='validation');a=parser.parse_args()
sys.path.insert(0,'/home/prmi/zwj/mycartgs/python')
from spatial_compensation import SpatialCompensation
torch.backends.cuda.matmul.allow_tf32=False
state=torch.jit.load(str(ROOT/a.validation/'model.pt')).state_dict();model=SpatialCompensation().cuda();model.load_state_dict(state)
fixture=torch.jit.load(str(ROOT/a.validation/'fixture.pt')).state_dict();values=[fixture['input_'+str(j)] for j in range(7)]
with torch.no_grad():actual=model(*values,.075)
diff=(actual-fixture['expected']).abs();assert torch.all(diff<=1e-6+1e-5*fixture['expected'].abs())
summary=dict(pass_=True,max_abs=float(diff.max()),rms=float(diff.square().mean().sqrt()),parameters=sum(p.numel() for p in model.parameters()))
(ROOT/(a.validation+'_python_parity.json')).write_text(json.dumps(summary,indent=2));print(summary,flush=True)
