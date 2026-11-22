"""只读 checkpoint/环境，导出原始 TF 的真实 representative input。"""
import hashlib, json, os, struct, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData

repo = Path('/home/prmi/zwj/mycartgs')
dest = Path(__file__).resolve().parent
sys.path.insert(0, str(repo / 'python'))
from selector import load_selector, load_selector_metadata, build_selector_mask
from elastic_budget import classify_protection
checkpoint = repo / 'results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply'
state = json.loads((checkpoint / 'transform_state.json').read_text())
cloud = next((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
v = PlyData.read(str(cloud))['vertex']
def columns(names):
    return torch.tensor(np.stack([v[n] for n in names], axis=1).copy(), device='cuda')
class Cloud:
    get_xyz = columns(['x', 'y', 'z'])
    get_scaling = columns(['scale_0','scale_1','scale_2']).exp()
    raw_q = columns(['rot_0','rot_1','rot_2','rot_3'])
    get_rotation = raw_q / raw_q.norm(dim=1, keepdim=True).clamp_min(1e-12)
g = Cloud()
meta = load_selector_metadata(str(cloud.parent / 'selector_metadata.ply'), 'cuda')
config = dict(age_ready=100, optimization_ready=80, unique_keyframe_ready=4,
              stale_age=1000, max_budget_fraction=0.35)
model = load_selector(str(checkpoint / 'selector.pt'), 'cuda')
mature = classify_protection(meta, config, True)['mature']
lo, hi = [torch.tensor(state[k],device='cuda') for k in ('aabb_min','aabb_max')]
inside = ((g.get_xyz >= lo) & (g.get_xyz <= hi)).all(1)
counts = []
with torch.no_grad():
    for ratio in (0.01,0.05,0.10,0.15):
        mask, stats = build_selector_mask(model,g,meta,ratio,config,protection_enabled=True)
        chosen = (mask > 0) & mature & inside
        counts.append(dict(ratio=ratio, selected=int((mask>0).sum()),
                           tf_N=int(chosen.sum()), stats=stats))
        if ratio == 0.05:
            data = torch.cat((g.get_xyz[chosen],g.get_scaling[chosen],g.get_rotation[chosen]),1).cpu().numpy()
            (dest/'input.bin').write_bytes(struct.pack('<q',len(data))+data.astype('<f4').tobytes())
            np.save(dest/'input.npy',data)
spec=[state[k] for k in ('spatial_resolution','ratio_resolution','feature_dim','hidden_dim')]
spec += [len(state['multires'])]+state['multires']+state['aabb_min']+state['aabb_max']
(dest/'spec.txt').write_text(' '.join(map(str,spec)))
(dest/'checkpoint_state.json').write_text(json.dumps(state,indent=2))
(dest/'checkpoint_selector.yaml').write_bytes((checkpoint/'selector_config.yaml').read_bytes())
def cmd(s):
    p=subprocess.run(s,shell=True,cwd=repo,capture_output=True,text=True)
    return dict(exit=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
files=['src/transform_field.cpp','src/hexplane_field.cpp','include/transform_field.h',
       'include/hexplane_field.h','include/aabb.h','include/quaternion_utils.h',
       'src/gaussian_mapper.cpp','src/change_distill.cpp','src/change_management.cpp',
       'src/constrained_adam.cpp','src/gradient_constraint.cpp','src/drift_stability.cpp',
       'python/transform_field.py','cfg/selector/fast.yaml','cfg/selector/selector_config.yaml','CMakeLists.txt']
payload=dict(checkpoint=str(checkpoint),state=state,gaussians=len(v),mature=int(mature.sum()),
    inside=int(inside.sum()),counts=counts,
    counts_kind='deterministic checkpoint inference; not temporal mapping quantiles',
    source_hashes={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},
    git={k:cmd('git '+k) for k in ['branch --show-current','rev-parse HEAD','rev-parse --show-toplevel','status --short -- .']},
    torch_version=torch.__version__,torch_cuda=torch.version.cuda,python=sys.version,
    gpu=torch.cuda.get_device_name(),cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    env={k:os.environ.get(k) for k in ('CUDA_LAUNCH_BLOCKING','CUDA_VISIBLE_DEVICES')},
    nvidia=cmd('nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem --format=csv'),
    nvcc=cmd('/usr/local/cuda-11.8/bin/nvcc --version'),gcc=cmd('gcc --version'),
    nsys=cmd('command -v nsys'),cmake=cmd("grep -E 'Torch_DIR|CMAKE_CUDA_ARCHITECTURES|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER' build-cuda118-sm86-gcc9/CMakeCache.txt"))
(dest/'inventory.json').write_text(json.dumps(payload,indent=2,default=str))
print(json.dumps({k:payload[k] for k in ['checkpoint','gaussians','mature','inside','counts','torch_version']},indent=2,default=str))
