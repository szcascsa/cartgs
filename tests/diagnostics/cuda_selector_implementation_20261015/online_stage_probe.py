"""定位真实 checkpoint near-tie 失败的首个数值分歧；不改被测实现。"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData
import torch
from torch.utils.cpp_extension import load

dest = Path(__file__).resolve().parent
root = dest.parents[2]
sys.path.insert(0, str(root/'python'))
from selector import load_selector

extension = load(name='cartgs_selector_cuda_stage_probe',
    sources=[str(dest/'online_stage_probe.cpp'), str(root/'src/selector_cuda.cu')],
    extra_include_paths=[str(root)], extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--fmad=false', '-gencode=arch=compute_86,code=sm_86'],
    extra_ldflags=['-lcublas', '-lcublasLt'], verbose=False)
checkpoint = next((root/'results/selector_online_ab_final_20261015/desk/legacy/repeat2').glob('*_shutdown/ply'))
cloud = next((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
vertex = PlyData.read(cloud)['vertex']
def tensor(names):
    return torch.from_numpy(np.stack([vertex[n] for n in names], axis=1).astype(np.float32)).cuda()
xyz = tensor(['x','y','z'])
rotation = torch.nn.functional.normalize(tensor([f'rot_{i}' for i in range(4)]),dim=1)
scale = tensor([f'scale_{i}' for i in range(3)]).exp()
model = load_selector(str(checkpoint/'selector.pt'),'cuda')
rows = []
def compare(mode, ratio, stage, a, b):
    e = (a-b).abs()
    maximum = float(e.max())
    rows.append(dict(mode=mode,ratio=ratio,stage=stage,max_abs=maximum,rms=float(e.square().mean().sqrt()),
        p99=float(e.quantile(.99)),unequal=int((a!=b).sum()),
        worst_global_gaussian_id=int(e.reshape(e.shape[0],-1).max(1).values.argmax()) if maximum else -1))
with torch.no_grad():
    for mode in ('row_ratio','scalar_ratio'):
        for ratio in (.01,.05,.1,.15):
            inputs = [xyz,rotation,scale,torch.full((len(vertex) if mode=='row_ratio' else 1,1),ratio,device='cuda')]
            legacy = []
            for branch, x in zip((model.pos_emd,model.rotation_emd,model.scale_emd,model.time_emd),inputs):
                h = branch[1](branch[0](x));legacy.append(h)
                legacy.append(branch[3](branch[2](h)))
            packed = torch.cat([legacy[i] if legacy[i].shape[0]==len(vertex) else legacy[i].expand(len(vertex),32)
                                for i in (1,3,5,7)],dim=1)
            legacy.append(model.soft_net(packed))
            cuda = extension.stages(inputs,list(model.parameters()))
            for stage in range(9):compare(mode,ratio,f'stage_{stage}',legacy[stage],cuda[stage])
            for transpose in (False,True):
                direct=extension.ratio_first_gemm(inputs[3],model.time_emd[0].weight,model.time_emd[0].bias,transpose)
                compare(mode,ratio,'ratio_first_op_t' if transpose else 'ratio_first_op_n',legacy[6],direct)
            for stage in (0,2,4,6):
                # A Torch Linear applied to the same first-layer output isolates
                # second GEMM differences from propagated first-layer errors.
                branch = (model.pos_emd,model.rotation_emd,model.scale_emd,model.time_emd)[stage//2]
                reference = branch[3](branch[2](cuda[stage]))
                compare(mode,ratio,f'same_input_second_{stage//2}',reference,cuda[stage+1])
            compare(mode,ratio,'same_input_final',model.soft_net(cuda[9]),cuda[8])
            a,b = (v.softmax(-1)[:,1] for v in (legacy[8],cuda[8]))
            compare(mode,ratio,'probability',a,b)
            noise = -legacy[8]
            hard_a = (legacy[8]+noise).softmax(-1).argmax(-1)
            hard_b = (cuda[8]+noise).softmax(-1).argmax(-1)
            compare(mode,ratio,'fixed_tie_hard',hard_a.float(),hard_b.float())
with (dest/'online_stage_probe.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary=dict(N=len(vertex),checkpoint=str(checkpoint),allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    first_differences={f'{mode}/{ratio}':next((r for r in rows if r['mode']==mode and r['ratio']==ratio and r['unequal']),None)
                       for mode in ('row_ratio','scalar_ratio') for ratio in (.01,.05,.1,.15)})
(dest/'online_stage_probe.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
