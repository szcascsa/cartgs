"""四类验收分开记录：正常数值/真实选择为 gate，all-tie 仅诊断。"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from plyfile import PlyData
import torch
from torch.utils.cpp_extension import load

args=argparse.ArgumentParser()
args.add_argument('mode',choices=['numerical','selection'])
args.add_argument('--strategy',type=int,default=0)
args.add_argument('--tag',default='')
args=args.parse_args()
dest=Path(__file__).resolve().parent
root=dest.parents[2]
prefix=args.mode+('_'+args.tag if args.tag else '')
sys.path.insert(0,str(root/'python'))
from selector import load_selector,load_selector_metadata
from elastic_budget import classify_protection
from elastic_entry import elastic_selector_probability

torch.set_num_threads(1)
torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32=False
extension=load(name='cartgs_selector_revised_parity',sources=[str(dest/'parity_bindings.cpp'),
    str(root/'src/selector_cuda.cpp'),str(root/'src/selector_cuda.cu')],extra_include_paths=[str(root)],
    extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-gencode=arch=compute_86,code=sm_86'],
    extra_ldflags=['-lcublas','-lcublasLt'],verbose=False)
tree=ast.parse((root/'python/run.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_selector_config')
namespace=dict(os=os,cv2=cv2)
exec(compile(ast.Module(body=[function],type_ignores=[]),'run.py:load_selector_config','exec'),namespace)
online=root/'results/selector_online_ab_final_20261015'
runs=json.loads((online/'status.json').read_text())['runs']
records=[]
failures=[]

def record(row):
    row=dict(case=case,**row)
    records.append(row)
    if row.get('passed') is False:failures.append(row)

def errors(a,b):
    same=torch.equal(a,b)
    if same:return dict(exact=True,max_abs=0.,rms=0.,p99=0.,worst_element=-1,relative_l2=0.,cosine=1.)
    da,db=a.detach().double().flatten(),b.detach().double().flatten()
    delta=da-db;error=delta.abs();na=da.norm();nb=db.norm()
    cosine=float(torch.dot(da,db)/(na*nb)) if na and nb else float(na==nb)
    return dict(exact=False,max_abs=float(error.max()),rms=float(delta.square().mean().sqrt()),
        p99=float(error.quantile(.99)),worst_element=int(error.argmax()),
        relative_l2=float(delta.norm()/na.clamp_min(1e-12)),cosine=cosine)

def numerical(stage,a,b,gradient=False):
    metric=errors(a,b)
    passed=(metric['relative_l2']<=1e-4 and metric['cosine']>=.99999) if gradient else (
        metric['max_abs']<=5e-7 and metric['rms']<=5e-8)
    record(dict(kind='gradient' if gradient else 'forward',stage=stage,**metric,passed=passed))

def margin_stats(x):
    if not x.numel():return None
    return dict(min=float(x.min()),mean=float(x.mean()),p01=float(x.quantile(.01)),
        p50=float(x.quantile(.5)),p99=float(x.quantile(.99)),max=float(x.max()))

def selection(kind,a,b,margin,probability_error,seed=None):
    mismatch=a!=b;changed=int(mismatch.sum());n=a.numel();count_a=int(a.sum());count_b=int(b.sum())
    threshold=max(1e-6,4*probability_error)
    within=int((mismatch&(margin<=threshold)).sum())
    ambiguity={str(t):dict(population=int((margin<=t).sum()),mismatches_inside=int((mismatch&(margin<=t)).sum()))
               for t in (1e-7,5e-7,1e-6)}
    row=dict(kind=kind,seed=seed,N=n,legacy_selected=count_a,cuda_selected=count_b,
        selected_count_relative_difference=abs(count_b-count_a)/max(count_a,1),
        hard_disagreement=changed,hard_disagreement_ratio=changed/max(n,1),
        ambiguity_threshold=threshold,disagreement_ambiguity_fraction=within/changed if changed else 1.,
        threshold_sensitivity=ambiguity,legacy_margin=margin_stats(margin),mismatch_margin=margin_stats(margin[mismatch]))
    row['passed']=row['selected_count_relative_difference']<=.0005 and row['hard_disagreement_ratio']<=.0005 and row['disagreement_ambiguity_fraction']>=.99
    record(row)

def legacy_stages(model,inputs):
    stages=[]
    for branch,x in zip((model.pos_emd,model.rotation_emd,model.scale_emd,model.time_emd),inputs):
        h=branch[1](branch[0](x));stages.extend([h,branch[3](branch[2](h))])
    n=inputs[0].shape[0]
    packed=torch.cat([stages[i].expand(n,32) for i in (1,3,5,7)],dim=1)
    return stages+[model.soft_net(packed)]

for run in runs:
    checkpoint=next((online/run['name']).glob('*_shutdown/ply'))
    cloud=next((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
    vertex=PlyData.read(cloud)['vertex']
    def tensor(names):
        return torch.from_numpy(np.stack([vertex[n] for n in names],axis=1).astype(np.float32)).cuda()
    xyz=tensor(['x','y','z']).requires_grad_()
    rotation=torch.nn.functional.normalize(tensor([f'rot_{i}' for i in range(4)]),dim=1).detach().requires_grad_()
    scale=tensor([f'scale_{i}' for i in range(3)]).exp().detach().requires_grad_()
    model=load_selector(str(checkpoint/'selector.pt'),'cuda')
    parameters=list(model.parameters())
    metadata=load_selector_metadata(cloud.parent/'selector_metadata.ply','cuda')
    config=namespace['load_selector_config'](str(checkpoint/'selector_config.yaml'))
    classified=classify_protection(metadata,config,bool(config['protection_enabled']))
    tau=config['temperature'];entry=config.get('entry_maturity',0.)
    for ratio in (.01,.05,.10,.15):
        for layout in ('scalar','row'):
            case=f"{run['name']}/ratio{ratio}/{layout}"
            target=torch.full((1 if layout=='scalar' else len(vertex),1),ratio,device='cuda',requires_grad=True)
            inputs=[xyz,rotation,scale,target]
            if args.mode=='numerical':
                state=torch.cuda.get_rng_state()
                legacy=legacy_stages(model,inputs)
                with torch.no_grad():
                    repeat=legacy_stages(model,inputs)
                    cuda=extension.stages(inputs,parameters)
                    for i in range(9):
                        numerical(f'activation_{i}',legacy[i],cuda[i])
                        record(dict(kind='legacy_repeat',stage=f'activation_{i}',**errors(legacy[i],repeat[i])))
                    numerical('probability',(legacy[8]/tau).softmax(-1),(cuda[8]/tau).softmax(-1))
                output=extension.network(inputs,parameters,args.strategy)
                dy=torch.linspace(-1,1,len(vertex)*2,device='cuda').reshape(-1,2)
                ga=torch.autograd.grad(legacy[8],inputs+parameters,dy)
                gb=torch.autograd.grad(output,inputs+parameters,dy)
                for i,(a,b) in enumerate(zip(ga,gb)):numerical(f'input_{i}' if i<4 else f'parameter_{i-4}',a,b,True)
                record(dict(kind='rng',stage='network_consumption',passed=torch.equal(state,torch.cuda.get_rng_state())))
                del legacy,repeat,cuda,output,ga,gb
            else:
                with torch.no_grad():
                    legacy=legacy_stages(model,inputs)
                    cuda=extension.stages(inputs,parameters)
                    za,zb=legacy[8],cuda[8]
                    pa,pb=(za/tau).softmax(-1)[:,1],(zb/tau).softmax(-1)[:,1]
                    pe=errors(pa,pb)['max_abs']
                    def mixed(probability):
                        def forward(ids):
                            if ids is None:return probability
                            x=[v.index_select(0,ids) for v in inputs[:3]]+[target if layout=='scalar' else target.index_select(0,ids)]
                            return (extension.network(x,parameters,args.strategy)/tau).softmax(-1)[:,1]
                        return elastic_selector_probability(metadata['maturity'],classified['maturity'],ratio,entry,forward)
                    # Current fixtures have entry=0; the saved uniform is reused.
                    assert entry==0,'Use explicit legacy/cuda subset callbacks before testing a nonzero-entry fixture'
                    ma,mb=mixed(pa),mixed(pb);u=metadata['cached_uniform']
                    selection('cached_uniform',u<ma,u<mb,(ma-u).abs(),errors(ma,mb)['max_abs'])
                    for seed in (0,1,2,42,123,2026):
                        generator=torch.Generator(device='cuda').manual_seed(seed)
                        initial=generator.get_state()
                        ga=-torch.empty_like(za).exponential_(generator=generator).log()
                        after_a=generator.get_state();next_a=torch.rand(32,device='cuda',generator=generator)
                        generator.set_state(initial)
                        gb=-torch.empty_like(zb).exponential_(generator=generator).log()
                        after_b=generator.get_state();next_b=torch.rand(32,device='cuda',generator=generator)
                        a,b=(za+ga)/tau,(zb+gb)/tau
                        selection('normal_gumbel_fixed_noise',a.softmax(-1).argmax(-1).bool(),b.softmax(-1).argmax(-1).bool(),
                                  ((za[:,1]+ga[:,1])-(za[:,0]+ga[:,0])).abs(),pe,seed)
                        record(dict(kind='rng',stage='legacy_generator',seed=seed,passed=torch.equal(ga,gb) and torch.equal(after_a,after_b) and torch.equal(next_a,next_b)))
                    noise=-za
                    hard_a=(za+noise).softmax(-1).argmax(-1);hard_b=(zb+noise).softmax(-1).argmax(-1)
                    def ordered(t):
                        bits=t.contiguous().view(torch.int32).to(torch.int64)
                        return torch.where(bits<0,-2147483648-bits,bits)
                    record(dict(kind='adversarial_diagnostic_only',logits_error=errors(za,zb),probability_error=errors(pa,pb),
                        hard_flips=int((hard_a!=hard_b).sum()),hard_flip_ratio=float((hard_a!=hard_b).float().mean()),
                        max_logit_ulp=int((ordered(za)-ordered(zb)).abs().max()),
                        first_differing_layer=next((i for i in range(9) if not torch.equal(legacy[i],cuda[i])),None)))
    (dest/f'{prefix}_parity.json').write_text(json.dumps(dict(production_accepted=False,records=records,failures=failures),ensure_ascii=False,indent=2)+'\n')
    print(run['name']+' '+args.mode+' complete; gate failures='+str(len(failures)),flush=True)
summary=dict(mode=args.mode,cases=len(runs)*4*2,records=len(records),failures=len(failures),
    strategy=args.strategy,source_sha256=hashlib.sha256((root/'src/selector_cuda.cu').read_bytes()).hexdigest(),tf32=False)
(dest/f'{prefix}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(dest/f'{prefix}.exit').write_text(str(int(bool(failures)))+'\n')
print(json.dumps(summary),flush=True)
raise SystemExit(bool(failures))
