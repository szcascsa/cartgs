"""冻结真实 checkpoint 输入、源码校验和与运行环境；不修改模型。"""
import hashlib, json, os, re, struct, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData

repo = Path('/home/prmi/zwj/mycartgs')
dest = Path(__file__).resolve().parent
sys.path.insert(0, str(repo/'python'))
from selector import load_selector, load_selector_metadata, build_selector_mask
from elastic_budget import classify_protection
checkpoint = repo/'results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply'
cloud = next((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
vertex = PlyData.read(cloud)['vertex']
fields = vertex.data.dtype.names
sh = sorted([n for n in fields if n.startswith('f_rest_')],key=lambda n:int(n[7:]))
names = ['x','y','z']+['rot_'+str(i) for i in range(4)]+['scale_'+str(i) for i in range(3)]+['opacity']+['f_dc_'+str(i) for i in range(3)]+sh
data = np.stack([vertex[n] for n in names],1).astype('<f4')
(dest/'input.bin').write_bytes(struct.pack('<qq',*data.shape)+data.tobytes())
meta = load_selector_metadata(cloud.parent/'selector_metadata.ply','cuda')
config = dict(age_ready=100,optimization_ready=80,unique_keyframe_ready=4,stale_age=1000,max_budget_fraction=0.35)
meta_data = torch.stack([meta['birth_iter'].float(),meta['optimized_count'].float(),meta['unique_keyframe_count'].float(),meta['cached_uniform']],1).cpu().numpy().astype('<f4')
(dest/'metadata.bin').write_bytes(struct.pack('<qq',*meta_data.shape)+meta_data.tobytes())
(dest/'spec.txt').write_text(str(meta['mapping_iteration']))
model = load_selector(checkpoint/'selector.pt','cuda')
raw = torch.tensor(data,device='cuda')
class Cloud: pass
g = Cloud(); g.get_xyz=raw[:,:3]; g.get_rotation=torch.nn.functional.normalize(raw[:,3:7],dim=1);g.get_scaling=raw[:,7:10].exp()
counts=[]
with torch.no_grad():
    for r in (.01,.05,.10,.15):
        mask,stats=build_selector_mask(model,g,meta,r,config)
        counts.append(stats)
        if r==.05:
            ids=torch.nonzero(classify_protection(meta,config)['mature']).flatten()[:257]
            prob=model(g.get_xyz[ids],g.get_rotation[ids],g.get_scaling[ids],torch.full((ids.numel(),1),r,device='cuda'))
            (dest/'python_probability.bin').write_bytes(prob.cpu().numpy().astype('<f4').tobytes())
(dest/'parameter_shapes.json').write_text(json.dumps([dict(name=n,shape=list(p.shape),stride=list(p.stride()),dtype=str(p.dtype),device=str(p.device),numel=p.numel()) for n,p in model.named_parameters()],indent=2))
def command(c):
    p=subprocess.run(c,shell=True,cwd=repo,text=True,capture_output=True)
    return dict(exit=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
files=list((repo/'include').glob('*.h'))+list((repo/'src').glob('*'))+list((repo/'python').glob('*.py'))+list((repo/'improvements/selector').glob('*'))+[repo/'CMakeLists.txt']+list((repo/'cfg/selector').glob('*.yaml'))
hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
(dest/'source_hashes_before.json').write_text(json.dumps(hashes,indent=2))
payload=dict(checkpoint=str(checkpoint),cloud=str(cloud),N=len(data),M=int(classify_protection(meta,config)['mature'].sum()),counts=counts,
    parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),python=sys.version,
    cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,config=config,
    git={c:command('git '+c) for c in ['branch --show-current','rev-parse HEAD','rev-parse --show-toplevel','status --short -- .']},
    environment={c:command(c) for c in ['nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,clocks.sm,clocks.mem --format=csv','/usr/local/cuda-11.8/bin/nvcc --version','gcc --version','/usr/local/cuda-11.8/bin/nsys --version',"grep -E 'Torch_DIR|CMAKE_CUDA_ARCHITECTURES|CMAKE_CXX_COMPILER' build-cuda118-sm86-gcc9/CMakeCache.txt"]},
    env_vars={k:os.environ.get(k) for k in ['CUDA_LAUNCH_BLOCKING','CUDA_VISIBLE_DEVICES']})
(dest/'inventory.json').write_text(json.dumps(payload,indent=2))
(dest/'checkpoint_selector.yaml').write_bytes((checkpoint/'selector_config.yaml').read_bytes())
print(json.dumps({k:payload[k] for k in ['N','M','parameters','counts']},indent=2),flush=True)

# 当前 evidence-aware 日志与历史旧实现分别统计，禁止混为同一分布。
rows=[]
for root in ['protection_recheck_20261012','sparse_fine_20261013','ultrafast_rotation_20261013']:
    base=repo/'results'/root
    if not base.exists(): continue
    for log in base.rglob('*.log'):
        with log.open(errors='replace') as stream:
            for line_no,line in enumerate(stream,1):
                if '[LOD Compact]' in line:
                    values={k:float(v) for k,v in re.findall(r'(\w+)=([-+0-9.eE]+)',line)}
                    values.update(path=str(log.relative_to(repo)),line=line_no)
                    rows.append(values)
quantiles={}
for r in (.01,.05,.1,.15):
    subset=[v for v in rows if abs(v.get('ratio',0)-r)<1e-5]
    quantiles[str(r)]={k:dict(zip(['min','median','p90','p95','max'],map(float,np.quantile([v[k] for v in subset if k in v],[0,.5,.9,.95,1])))) for k in ['active_gaussians','protection_actual','mature_actual','mature_budget_target','tf','target_total_gaussians'] if any(k in v for v in subset)}
(dest/'workload_rows.json').write_text(json.dumps(rows,indent=2))
(dest/'workload_quantiles.json').write_text(json.dumps(dict(rows=len(rows),kind='logged temporal values; each log path and line retained; missing network population is not inferred',ratios=quantiles),indent=2))
print('workload rows',len(rows),flush=True)
