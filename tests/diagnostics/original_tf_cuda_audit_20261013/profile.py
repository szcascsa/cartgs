"""原始 C++ operator 顺序的 Python 诊断镜像；必须先通过 native parity。"""
import contextlib, csv, gc, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

dest=Path(__file__).resolve().parent
repo=dest.parents[2]
sys.path.insert(0,str(repo/'python'))
from transform_field import load_transform_field
inventory=json.loads((dest/'inventory.json').read_text())
checkpoint=inventory['checkpoint']
torch.set_num_threads(1)
model=load_transform_field(checkpoint,'cuda')
data=torch.tensor(np.load(dest/'input.npy'),device='cuda')
timings=None
profiling=False
@contextlib.contextmanager
def scope(name):
    ctx=record_function(name) if profiling else contextlib.nullcontext()
    with ctx:
        if timings is not None:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        yield
        if timings is not None:
            b.record();timings[name].append((a,b))
def grid(x,ratio):
    with scope('input_preparation'):
        r=x.new_full((len(x),1),ratio)
        c=torch.cat((2.0*(x-model.grid.aabb_min)/(model.grid.aabb_max-model.grid.aabb_min)-1.0,r),1)
    features=[]
    for level in range(len(model.grid.multires)):
        product=None
        for name,(u,v) in zip(model.grid.plane_names,model.grid.pairs):
            with scope('plane_query'):
                coords=torch.stack((c.select(1,u),c.select(1,v)),1).view(1,1,len(x),2)
                value=F.grid_sample(getattr(model.grid,f'L{level}_{name}'),coords,
                    mode='bilinear',padding_mode='border',align_corners=True).view(model.grid.feature_dim,len(x)).transpose(0,1)
            if product is None:product=value
            else:
                with scope('feature_fusion'):product=product*value
        features.append(product)
    with scope('feature_fusion'):return torch.cat(features,1)
def forward(x,s,q,ratio=0.05):
    f=grid(x,ratio)
    with scope('mlp_forward'):
        h=model.feature_out(f)
        dx=model.position_out(F.relu(model.position_hidden(F.relu(h))))
        ds=model.scale_out(F.relu(model.scale_hidden(F.relu(h))))
        dq=model.rotation_out(F.relu(model.rotation_hidden(F.relu(h))))
    with scope('output_transform'):
        norm=lambda v:v/torch.sqrt((v*v).sum(1,keepdim=True)).clamp_min(1e-12)
        identity=torch.zeros_like(dq);identity[:,0]=1
        a=norm(identity+dq)
        w=a[:,:1]*q[:,:1]-(a[:,1:]*q[:,1:]).sum(1,keepdim=True)
        v=a[:,:1]*q[:,1:]+q[:,:1]*a[:,1:]+torch.cross(a[:,1:],q[:,1:],dim=1)
        return x+dx,s*torch.exp(ds),norm(torch.cat((w,v),1))
def regularization():
    losses=[None,None,None]
    for level in range(len(model.grid.multires)):
        for name in model.grid.plane_names:
            p=getattr(model.grid,f'L{level}_{name}')
            d=p[:,:,1:,:]-p[:,:,:-1,:]
            smooth=(d[:,:,1:,:]-d[:,:,:-1,:]).square().mean()
            j=1 if 'e' in name else 0
            losses[j]=smooth if losses[j] is None else losses[j]+smooth
            if j==1:
                v=(1.0-p).abs().mean()
                losses[2]=v if losses[2] is None else losses[2]+v
    return 0.0002*losses[0]+0.001*losses[1]+0.0001*losses[2]
def inputs(n):
    d=data[torch.arange(n,device='cuda')%len(data)]
    return [d[:,a:b].contiguous().detach().requires_grad_() for a,b in ((0,3),(3,6),(6,10))]

# Verify diagnostic mirror against unchanged native production outputs/gradients.
x,s,q=inputs(min(len(data),257))
y=forward(x,s,q)
sum(v.square().mean() for v in y).backward()
ref=torch.jit.load(str(dest/'native_parity.pt'),map_location='cuda').state_dict()
parity={}
def compare(name,a,b):
    error=(a-b).abs()
    parity[name]=dict(max_abs=float(error.max()),mean_abs=float(error.mean()),
                      passed=bool(torch.allclose(a,b,atol=2e-6,rtol=2e-4)))
for k,a in zip(('xyz','scaling','rotation'),y):compare(k,a,ref[k])
for k,a in zip(('xyz_grad','scale_grad','rotation_grad'),(x.grad,s.grad,q.grad)):compare(k,a,ref[k])
for i,(name,p) in enumerate(model.named_parameters()):compare(name+'.grad',p.grad,ref['grad_'+str(i)])
(dest/'mirror_parity.json').write_text(json.dumps(parity,indent=2))
assert all(v['passed'] for v in parity.values()),'native mirror parity failed'
model.zero_grad(set_to_none=True)
shapes=[dict(name=k,shape=list(p.shape),stride=list(p.stride()),contiguous=p.is_contiguous(),
             dtype=str(p.dtype),numel=p.numel()) for k,p in model.named_parameters()]
(dest/'parameter_shapes.json').write_text(json.dumps(shapes,indent=2))
print('native mirror parity passed',flush=True)

# Stage attribution includes event instrumentation; native CSV is authoritative total.
rows=[]
for n in (() if '--profile-only' in sys.argv else (1000,5000,10000,20000,50000)):
    x,s,q=inputs(n)
    for _ in range(100):forward(x,s,q)
    torch.cuda.synchronize()
    samples=defaultdict(list)
    for _ in range(500):
        timings=defaultdict(list)
        y=forward(x,s,q)
        torch.cuda.synchronize()
        for name,pairs in timings.items():samples[name].append(sum(a.elapsed_time(b) for a,b in pairs))
        timings=None
    for name,v in samples.items():
        rows.append(dict(N=n,stage=name,mean_ms=float(np.mean(v)),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95))))
    print('stage',n,'complete',flush=True)
if rows:
    with (dest/'stage_benchmark.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
del y,ref
model.zero_grad(set_to_none=True)
gc.collect();torch.cuda.empty_cache()

# Profile typical real TF population, with all base attribute gradients enabled.
n=len(data); x,s,q=inputs(n)
named=list(model.named_parameters())
adam={name:[torch.zeros_like(p),torch.zeros_like(p),0] for name,p in named}
def adam_step(preview=False):
    proposed=[]
    with torch.no_grad():
        for name,p in named:
            if p.grad is None:continue
            m,v,t=adam[name]; t+=1
            if preview:m=m.clone();v=v.clone()
            m.mul_(0.9).add_(p.grad,alpha=0.1)
            v.mul_(0.999).addcmul_(p.grad,p.grad,value=0.001)
            denom=(v.sqrt()/math.sqrt(1.0-0.999**t)).add_(1e-15)
            lr=0.0016 if name.startswith('grid.') else 0.00016
            if preview:
                value=p.detach().clone().addcdiv_(m,denom,value=-lr/(1.0-0.9**t))
                delta=value-p.detach();candidate=p.detach().add(delta,alpha=1.0)
                proposed.append((name,p,candidate,m,v,t))
            else:
                p.addcdiv_(m,denom,value=-lr/(1.0-0.9**t));adam[name]=[m,v,t]
        for name,p,value,m,v,t in proposed:p.copy_(value);adam[name]=[m,v,t]
def iteration(use_reg,preview):
    with scope('zero_grad'):
        model.zero_grad(set_to_none=True)
        for v in (x,s,q):v.grad=None
    with scope('forward'):y=forward(x,s,q)
    with scope('loss'):
        loss=(y[0]-x.detach()).mean()+(y[1]/s.detach()).mean()+y[2].mean()
    if use_reg:
        with scope('regularization'):loss=loss+regularization()
    with scope('backward'):loss.backward()
    with scope('optimizer'):adam_step(preview)
summaries={}
for mode,use_reg,preview in (('core',False,False),('regularized',True,False),('preview',True,True)):
    for _ in range(100):iteration(use_reg,preview)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    base_alloc=torch.cuda.memory_allocated()
    profiling=True
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
        for _ in range(3):iteration(use_reg,preview);prof.step()
        torch.cuda.synchronize()
    profiling=False
    trace=dest/f'profile_{mode}.json';prof.export_chrome_trace(str(trace))
    events=json.loads(trace.read_text())['traceEvents']
    rf=[e for e in events if e.get('cat')=='user_annotation' and e.get('name') in ('forward','backward','optimizer','zero_grad','loss','regularization')]
    cpus={str(e.get('args',{}).get('External id')):e for e in events if e.get('cat')=='cpu_op'}
    runtimes={e.get('args',{}).get('correlation'):e for e in events if e.get('cat')=='cuda_runtime' and 'Launch' in e.get('name','')}
    categories=Counter();gpu_time=Counter();kernels=Counter();launches=Counter();sync=Counter();copies=Counter()
    def category(e):
        ts=e.get('ts',0)
        spans=[v for v in rf if v['ts']<=ts<=v['ts']+v['dur']]
        return min(spans,key=lambda v:v['dur'])['name'] if spans else 'outside'
    for e in events:
        if e.get('cat')=='kernel':
            cpu=runtimes.get(e.get('args',{}).get('correlation'),{})
            k=category(cpu);categories[k]+=1;gpu_time[k]+=e.get('dur',0);kernels[e['name']]+=e.get('dur',0)
        if e.get('cat')=='cuda_runtime':
            if 'Launch' in e['name']:launches[category(e)]+=1
            if 'Synchronize' in e['name']:sync[e['name']]+=1
            if 'Memcpy' in e['name']:copies[e['name']]+=1
    ops=[]
    for e in prof.key_averages(group_by_input_shape=True):
        ops.append(dict(name=e.key,count=e.count,shapes=e.input_shapes,
            self_cpu_us=e.self_cpu_time_total,cuda_us=e.cuda_time_total,
            self_cuda_us=e.self_cuda_time_total,cpu_memory=e.cpu_memory_usage,cuda_memory=e.cuda_memory_usage))
    ops.sort(key=lambda v:v['self_cuda_us'],reverse=True)
    summaries[mode]=dict(N=n,iterations=3,kernels=dict(categories),kernel_us=dict(gpu_time),
        runtime_launches=dict(launches),synchronizations=dict(sync),memcpy=dict(copies),
        top20_kernels=kernels.most_common(20),top20_ops=ops[:20],
        aten_calls=sum(o['count'] for o in ops if o['name'].startswith('aten::')),
        aten_positive_allocation_bytes=sum(max(o['cuda_memory'],0) for o in ops if o['name'].startswith('aten::empty')),
        resident_at_start=base_alloc,peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        small_kernel_count=sum(e.get('dur',0)<10 for e in events if e.get('cat')=='kernel'),
        total_kernel_count=sum(e.get('cat')=='kernel' for e in events),
        operators=ops)
    print('profile',mode,categories,flush=True)
(dest/'profiler_summary.json').write_text(json.dumps(summaries,indent=2))
