"""同 checkpoint 的三方直接计算验收；不调用 renderer 或 SLAM。"""
import ast
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from plyfile import PlyData
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

dest = Path(__file__).resolve().parent
root = Path('/home/prmi/zwj/selector-cuda-20261015')
sys.path.insert(0, str(root/'python'))
from selector import load_selector, load_selector_metadata
from elastic_budget import classify_protection, sample_selector_st, ratio_loss
from elastic_entry import elastic_selector_probability

torch.set_num_threads(1)
torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32 = False
sources = [dest/'bindings.cpp']+[root/'src'/f for f in ['selector_cuda.cpp','selector_cuda.cu',
    'constrained_adam.cpp','transform_field_cuda_optimizer.cpp','transform_field_cuda_optimizer.cu',
    'protection_profiler.cpp','gi_teacher.cpp']]
extension = load(name='cartgs_selector_backward_gate_6', sources=list(map(str,sources)), extra_include_paths=[str(root)],
    extra_cflags=['-O3'], extra_cuda_cflags=['-O3','--fmad=false','-gencode=arch=compute_86,code=sm_86'],
    extra_ldflags=['-lcublas','-lcublasLt'], verbose=False)
tree = ast.parse((root/'python/run.py').read_text())
function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_selector_config')
namespace = dict(os=os, cv2=cv2)
exec(compile(ast.Module(body=[function],type_ignores=[]),'run.py:load_selector_config','exec'), namespace)
models = json.loads((dest/'checkpoints.json').read_text())
summary = dict(scope='Selector direct boundary only; no renderer/SLAM quality gate', tf32=False,
    checkpoints=[], categories={}, failures=[], source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
stream = (dest/'tensor_metrics.jsonl').open('w')
selections = (dest/'selection.jsonl').open('w')
training_file = (dest/'training.csv').open('w',newline='')
training = csv.DictWriter(training_file,fieldnames=['checkpoint','step','ratio','layout','sampling','frame_slot',
    'loss_legacy','loss_repeat','loss_cuda','loss_relative','logits_max_abs','probability_max_abs','selected_count_legacy',
    'selected_count_cuda','hard_mismatch_ratio','parameter_relative_l2','gradient_relative_l2','gradient_cosine',
    'adam_m_relative_l2','adam_v_relative_l2','steps_exact','passed'])
training.writeheader()
case = ''


def metric(a,b,rows=None):
    if a is None or b is None:
        return dict(defined=a is not None, other_defined=b is not None, exact=a is None and b is None,
            max_abs=0.,mean_abs=0.,rms=0.,relative_l2=0.,cosine=1.,p99=0.,worst_index=-1,worst_gaussian_id=-1,
            nonzero_reference=0,nonzero_other=0,finite=True,reference_rms=0.)
    same = torch.equal(a,b)
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    count_a = int(torch.count_nonzero(a))
    count_b = count_a if same else int(torch.count_nonzero(b))
    if same:
        return dict(defined=True,other_defined=True,exact=True,max_abs=0.,mean_abs=0.,rms=0.,relative_l2=0.,cosine=1.,
            p99=0.,worst_index=-1,worst_gaussian_id=-1,nonzero_reference=count_a,nonzero_other=count_b,finite=finite,reference_rms=None)
    x,y = a.detach().double().flatten(),b.detach().double().flatten()
    error=(x-y).abs();nx=float(x.norm());ny=float(y.norm());worst=int(error.argmax()) if error.numel() else -1
    gaussian=-1
    if rows is not None and a.ndim and a.shape[0]==rows.numel() and worst>=0:
        gaussian=int(rows[worst//(a.numel()//a.shape[0])])
    return dict(defined=True,other_defined=True,exact=False,max_abs=float(error.max()),mean_abs=float(error.mean()),
        rms=float(error.square().mean().sqrt()),p99=float(error.quantile(.99)),relative_l2=float(error.norm())/max(nx,1e-30),
        cosine=max(-1.,min(1.,float(torch.dot(x,y))/(nx*ny))) if nx and ny else float(nx==ny),
        worst_index=worst,worst_gaussian_id=gaussian,nonzero_reference=count_a,nonzero_other=count_b,finite=finite,
        reference_rms=float(x.square().mean().sqrt()))


def accepts(value,baseline,kind):
    if not value['finite'] or value['defined']!=value['other_defined']:
        return False
    if value['exact']:
        return True
    if kind=='gradient' and value['reference_rms']<=1e-7:
        return value['max_abs']<=max(1e-5,2*baseline['max_abs']) and value['rms']<=max(1e-7,2*baseline['rms'])
    relative = 1e-6 if kind in ['forward','optimizer'] else 1e-5
    cosine = .999999 if kind=='forward' else .99999
    return (value['relative_l2']<=max(relative,2*baseline['relative_l2']) and
        1-value['cosine']<=max(1-cosine,2*(1-baseline['cosine'])) and
        (kind!='forward' or value['max_abs']<=max(1e-6,2*baseline['max_abs'])))


def compare(category,name,values,kind='forward',rows=None,gate=True):
    baseline=metric(values[0],values[1],rows);value=metric(values[0],values[2],rows)
    passed=accepts(value,baseline,kind) if gate else True
    for label,m in [('legacy_repeat',baseline),('cuda',value)]:
        stream.write(json.dumps(dict(case=case,category=category,tensor=name,comparison=label,**m,
            passed=passed if label=='cuda' else None,diagnostic_only=not gate))+'\n')
    aggregate=summary['categories'].setdefault(category,dict(checks=0,failures=0,max_abs=0.,max_relative_l2=0.,
        max_rms=0.,minimum_cosine=1.,legacy_repeat_max_abs=0.,legacy_repeat_max_relative_l2=0.,all_exact=True))
    aggregate['checks']+=1;aggregate['failures']+=not passed
    aggregate['max_abs']=max(aggregate['max_abs'],value['max_abs'])
    aggregate['max_relative_l2']=max(aggregate['max_relative_l2'],value['relative_l2'])
    aggregate['max_rms']=max(aggregate['max_rms'],value['rms'])
    aggregate['minimum_cosine']=min(aggregate['minimum_cosine'],value['cosine'])
    aggregate['legacy_repeat_max_abs']=max(aggregate['legacy_repeat_max_abs'],baseline['max_abs'])
    aggregate['legacy_repeat_max_relative_l2']=max(aggregate['legacy_repeat_max_relative_l2'],baseline['relative_l2'])
    aggregate['all_exact'] &= value['exact']
    if not passed:summary['failures'].append(dict(case=case,category=category,tensor=name,metric=value,baseline=baseline))
    return value,passed


def statistics(x):
    if not x.numel():return None
    return dict(min=float(x.min()),p01=float(x.quantile(.01)),p50=float(x.quantile(.5)),p99=float(x.quantile(.99)),max=float(x.max()))


def selection(kind,hard,margin,rows,seed=None):
    a,b,c=[v.detach().bool() for v in hard]
    mismatch=a!=c;repeat=a!=b;count=int(mismatch.sum());na=int(a.sum());nc=int(c.sum())
    disagreement=count/max(a.numel(),1);count_error=abs(na-nc)/max(na,1)
    inside=int((mismatch & (margin<=1e-6)).sum())
    passed=disagreement<=1e-4 and count_error<=1e-4 and (not count or inside/count>=.99)
    row=dict(case=case,kind=kind,seed=seed,N=a.numel(),legacy_selected=na,cuda_selected=nc,
        hard_disagreement=count,hard_mismatch_ratio=disagreement,selected_count_relative_difference=count_error,
        selected_ids_exact=torch.equal(rows[a],rows[c]),legacy_repeat_disagreement=int(repeat.sum()),
        mismatch_near_boundary_fraction=inside/count if count else None,margin_distribution=statistics(margin),
        mismatch_margin_distribution=statistics(margin[mismatch]),
        mismatches=[dict(global_id=int(i),margin=float(m)) for i,m in zip(rows[mismatch].cpu(),margin[mismatch].cpu())],passed=passed)
    selections.write(json.dumps(row)+'\n')
    if not passed:summary['failures'].append(row)
    return row


def legacy_stages(model,inputs):
    stages=[]
    for layer,x in zip([model.pos_emd,model.rotation_emd,model.scale_emd,model.time_emd],inputs):
        h=layer[1](layer[0](x));stages.extend([h,layer[3](layer[2](h))])
    return stages+[model.soft_net(torch.cat([stages[i].expand(inputs[0].shape[0],32) for i in (1,3,5,7)],dim=1))]


def raw_clone(raw):
    return [x.detach().clone().requires_grad_() for x in raw]


def prepare(raw,ratio,layout):
    return [raw[0],F.normalize(raw[1],dim=1),raw[2].exp(),
            torch.full((1 if layout=='scalar' else raw[0].shape[0],1),ratio,device='cuda',requires_grad=True)]


def targets(inputs,raw,model):
    return inputs+[raw[1],raw[2],raw[3]]+list(model.parameters())


def mixed(z,inputs,model,backend,metadata,classification,ratio,entry,tau):
    def forward(ids):
        if ids is None:return (z/tau).softmax(-1)[:,1]
        x=[v.index_select(0,ids) for v in inputs[:3]]+[inputs[3] if inputs[3].shape[0]==1 else inputs[3].index_select(0,ids)]
        output=extension.network(x,list(model.parameters())) if backend=='cuda' else legacy_stages(model,x)[8]
        return (output/tau).softmax(-1)[:,1]
    return elastic_selector_probability(metadata['maturity'],classification['maturity'],ratio,entry,forward)


def gumbel_st(z,noise,tau):
    soft=((z+noise)/tau).softmax(-1)
    hard=F.one_hot(soft.argmax(-1),2).to(soft.dtype)
    return soft,hard-soft.detach()+soft


def guard_st(z,uniform,tau):
    logit=z[:,1]-z[:,0];u=uniform.clamp(1e-7,1-1e-7)
    soft=((logit-(u.log()-(1-u).log()))/tau).sigmoid()
    hard=(uniform<logit.sigmoid()).to(soft.dtype)
    return hard-soft.detach()+soft


def flat(items):
    return torch.cat([v.detach().flatten() for v in items if v is not None])


def save():
    summary['passed']=not summary['failures']
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    stream.flush();selections.flush();training_file.flush()


def parity(base,raw,metadata,config,entry,tau):
    global case
    n=raw[0].shape[0];ids=torch.arange(n,device='cuda')
    classification=classify_protection(metadata,config,bool(config['protection_enabled']))
    generator=torch.Generator(device='cuda').manual_seed(3401)
    dy=torch.randn(n,2,device='cuda',generator=generator)
    dp=torch.randn(n,device='cuda',generator=generator)
    ds=torch.randn(n,device='cuda',generator=generator)
    photo=torch.randn(n,1,device='cuda',generator=generator)
    teacher_probability=torch.rand(n,device='cuda',generator=generator)
    importance=torch.rand(n,device='cuda',generator=generator)
    observed=torch.rand(n,device='cuda',generator=generator)>.2
    clones=[copy.deepcopy(base) for _ in range(3)]
    for m in clones:
        assert len(list(m.parameters()))==18
        assert all(torch.equal(a,b) for a,b in zip(base.parameters(),m.parameters()))
    for ratio in (.01,.05,.10,.15):
        gi_label,gi_weight=extension.gi_targets(importance,observed,metadata['maturity'],ratio,entry)
        for layout in ['scalar','row']:
            case=f'{checkpoint_name}/ratio{ratio}/{layout}'
            raw_sets=[raw_clone(raw) for _ in range(3)];inputs=[prepare(r,ratio,layout) for r in raw_sets]
            for i,name in enumerate(['xyz','activated_quaternion','activated_scale','ratio']):
                compare('input_preparation',name,[x[i] for x in inputs],rows=ids)
            state=torch.cuda.get_rng_state()
            stages=[legacy_stages(clones[i],inputs[i]) if i<2 else extension.stages(inputs[i],list(clones[i].parameters())) for i in range(3)]
            with torch.no_grad():
                for i,name in enumerate(['position_l1','position_l2','rotation_l1','rotation_l2','scale_l1','scale_l2','ratio_l1','ratio_l2','logits']):
                    compare('network_forward',name,[s[i] for s in stages],rows=ids)
            z=[stages[0][8],stages[1][8],extension.network(inputs[2],list(clones[2].parameters()))]
            compare('rng','network_consumption',[state,state,torch.cuda.get_rng_state()])
            p=[(v/tau).softmax(-1)[:,1] for v in z]
            for tag,values in [('keep_probability',p),('logits',z)]:compare('probability' if tag=='keep_probability' else 'custom_function_forward',tag,values,rows=ids)
            actual=[mixed(z[i],inputs[i],clones[i],'cuda' if i==2 else 'legacy',metadata,classification,ratio,entry,tau) for i in range(3)]
            compare('production_probability','elastic_mix',actual,rows=ids)
            cached=[sample_selector_st(v,metadata['cached_uniform']) for v in actual]
            compare('cached_st','forward',cached,rows=ids,gate=False)
            selection('cached_uniform',cached,(actual[0]-metadata['cached_uniform']).abs(),ids)
            noises={}
            for seed in [0,1,2,42,123,2026]:
                noise=-torch.empty_like(z[0]).exponential_(generator=torch.Generator(device='cuda').manual_seed(seed)).log()
                noises[seed]=noise
                samples=[gumbel_st(v,noise,tau) for v in z]
                compare('gumbel_soft',str(seed),[s[0] for s in samples],rows=ids)
                compare('gumbel_st',str(seed),[s[1] for s in samples],rows=ids,gate=False)
                selection('normal_gumbel',[s[1][:,1] for s in samples],
                    ((z[0][:,1]+noise[:,1])-(z[0][:,0]+noise[:,0])).abs(),ids,seed)
            # Intentionally all-tie: diagnostic only, never appended to failures.
            with torch.no_grad():
                a,b=z[0],z[2];noise=-a
                bits=lambda t:torch.where(t.contiguous().view(torch.int32).long()<0,
                    -2147483648-t.contiguous().view(torch.int32).long(),t.contiguous().view(torch.int32).long())
                selections.write(json.dumps(dict(case=case,kind='all_tie_diagnostic_only',
                    flips=int(((a+noise).argmax(-1)!=(b+noise).argmax(-1)).sum()),
                    logits_error=metric(a,b,ids),max_ulp=int((bits(a)-bits(b)).abs().max()),
                    first_differing_stage=next((i for i in range(9) if not torch.equal(stages[0][i],stages[2][i])),None)))+'\n')
            st=[gumbel_st(v,noises[42],tau)[1][:,1] for v in z]
            guard=[guard_st(v,metadata['cached_uniform'],tau) for v in z]
            losses={key:[] for key in ['ratio','gi','photo_opacity','combined','guard_uniform','replay','cd','dsd']}
            for i in range(3):
                selected=torch.nonzero(cached[i].detach()>0).flatten()
                opacity=raw_sets[i][3].sigmoid()
                photo_loss=(opacity.index_select(0,selected)*cached[i].index_select(0,selected).unsqueeze(1)*photo.index_select(0,selected)).sum()
                r=ratio_loss(actual[i],ratio);gi=extension.gi_loss(cached[i],gi_label,gi_weight,False)
                losses['ratio'].append(r);losses['gi'].append(gi);losses['photo_opacity'].append(photo_loss)
                losses['combined'].append(photo_loss+.01*r+gi)
                losses['guard_uniform'].append((guard[i]*ds).sum())
                losses['replay'].append(photo_loss+.01*r+gi)
                losses['cd'].append((p[i]-teacher_probability).abs().mean())
                losses['dsd'].append(F.binary_cross_entropy_with_logits((z[i][:,1]-z[i][:,0])/tau,teacher_probability))
            tests=[('logits_vjp',z,dy),('probability_vjp',p,dp),('gumbel_st_vjp',st,ds)]
            tests += [(name,value,None) for name,value in losses.items()]
            for name,outputs,upstream in tests:
                if upstream is None:compare('loss',name,outputs)
                grads=[torch.autograd.grad(outputs[i],targets(inputs[i],raw_sets[i],clones[i]),upstream,
                    retain_graph=True,allow_unused=True) for i in range(3)]
                for j in range(len(grads[0])):
                    target_names=['xyz','activated_quaternion','activated_scale','ratio','raw_quaternion','raw_log_scale','raw_opacity']+list(dict(clones[0].named_parameters()))
                    compare(name,target_names[j],[g[j] for g in grads],'gradient',ids if j<7 else None)
                if name in ['guard_uniform','replay','cd','dsd']:
                    proposals=[[p.detach().clone().requires_grad_() for p in m.parameters()] for m in clones]
                    for i in range(3):extension.Adam(proposals[i],i==2).step([g.contiguous() if g is not None else None for g in grads[i][7:]],config['learning_rate'])
                    for j in range(18):compare('special_path_update',name+'/'+str(j),[v[j] for v in proposals],'optimizer')
            print(case+' direct complete; failures='+str(len(summary['failures'])),flush=True)
            save()


def train(base,raw,metadata,config,entry,tau):
    global case
    case=checkpoint_name+'/training'
    nets=[copy.deepcopy(base) for _ in range(3)];raw_sets=[raw_clone(raw) for _ in range(3)]
    parameters=[list(m.parameters()) for m in nets]
    optimizers=[extension.Adam(parameters[i],i==2) for i in range(3)]
    classification=classify_protection(metadata,config,bool(config['protection_enabled']))
    n=raw[0].shape[0];rows=torch.arange(n,device='cuda')
    generator=torch.Generator(device='cuda').manual_seed(20261015)
    external=[torch.randn(n,1,device='cuda',generator=generator)*.001 for _ in range(3)]
    importance=torch.rand(n,device='cuda',generator=generator);observed=torch.rand(n,device='cuda',generator=generator)>.2
    teachers={r:extension.gi_targets(importance,observed,metadata['maturity'],r,entry) for r in [.01,.05,.1,.15]}
    noises=[-torch.empty(n,2,device='cuda').exponential_(generator=torch.Generator(device='cuda').manual_seed(seed)).log() for seed in [0,1,2,42,123,2026]]
    mismatches=[];maxima=dict(parameter=0.,adam_m=0.,adam_v=0.,loss=0.)
    for step in range(1,1001):
        ratio=[.01,.05,.1,.15][(step-1)%4];layout='scalar' if step%2 else 'row';slot=(step-1)%3
        inputs=[prepare(v,ratio,layout) for v in raw_sets]
        z=[legacy_stages(nets[i],inputs[i])[8] if i<2 else extension.network(inputs[i],parameters[i]) for i in range(3)]
        p=[(v/tau).softmax(-1)[:,1] for v in z]
        actual=[mixed(z[i],inputs[i],nets[i],'cuda' if i==2 else 'legacy',metadata,classification,ratio,entry,tau) for i in range(3)]
        sampling='cached_uniform' if step%2 else 'gumbel'
        hard=[sample_selector_st(v,metadata['cached_uniform']) for v in actual] if step%2 else [gumbel_st(v,noises[(step-1)%6],tau)[1][:,1] for v in z]
        losses=[];grads=[]
        for i in range(3):
            selected=torch.nonzero(hard[i].detach()>0).flatten()
            photo=(raw_sets[i][3].sigmoid().index_select(0,selected)*hard[i].index_select(0,selected).unsqueeze(1)*external[slot].index_select(0,selected)).sum()
            label,weight=teachers[ratio]
            losses.append(photo+.01*ratio_loss(actual[i],ratio)+extension.gi_loss(hard[i],label,weight,False))
            grads.append(torch.autograd.grad(losses[i],targets(inputs[i],raw_sets[i],nets[i]),allow_unused=True))
        # Each step records a whole-network comparison; per-tensor details at milestones.
        zm,zok=compare('training_logits',str(step),z,rows=rows)
        pm,pok=compare('training_probability',str(step),p,rows=rows)
        lm,lok=compare('training_loss',str(step),losses,'training')
        gm,gok=compare('training_gradient',str(step),[flat(g[7:]) for g in grads],'gradient')
        if step in [1,10,100,1000]:
            for j in range(len(grads[0])):compare('training_milestone_gradient',f'{step}/{j}',[g[j] for g in grads],'gradient',rows if j<7 else None)
        for i in range(3):optimizers[i].step([g.contiguous() if g is not None else None for g in grads[i][7:]],config['learning_rate'])
        states=[o.state() for o in optimizers]
        wm,wok=compare('training_parameter',str(step),[flat(p) for p in parameters],'training')
        mm,mok=compare('training_adam_m',str(step),[flat(s[0]) for s in states],'training')
        vm,vok=compare('training_adam_v',str(step),[flat(s[1]) for s in states],'training')
        step_exact=states[0][2]==states[1][2]==states[2][2]
        mismatch=float(((hard[0].detach()>0)!=(hard[2].detach()>0)).float().mean());mismatches.append(mismatch)
        if mismatch:
            margin=(actual[0]-metadata['cached_uniform']).abs() if step%2 else ((z[0][:,1]+noises[(step-1)%6][:,1])-(z[0][:,0]+noises[(step-1)%6][:,0])).abs()
            selection('training_'+str(step),hard,margin,rows)
        passed=all([zok,pok,lok,gok,wok,mok,vok,step_exact])
        if not step_exact:summary['failures'].append(dict(case=case,step=step,kind='adam_step'))
        training.writerow(dict(checkpoint=checkpoint_name,step=step,ratio=ratio,layout=layout,sampling=sampling,frame_slot=slot,
            loss_legacy=float(losses[0]),loss_repeat=float(losses[1]),loss_cuda=float(losses[2]),loss_relative=lm['relative_l2'],
            logits_max_abs=zm['max_abs'],probability_max_abs=pm['max_abs'],selected_count_legacy=int((hard[0].detach()>0).sum()),
            selected_count_cuda=int((hard[2].detach()>0).sum()),hard_mismatch_ratio=mismatch,parameter_relative_l2=wm['relative_l2'],
            gradient_relative_l2=gm['relative_l2'],gradient_cosine=gm['cosine'],adam_m_relative_l2=mm['relative_l2'],adam_v_relative_l2=vm['relative_l2'],steps_exact=step_exact,passed=passed))
        for key,value in [('parameter',wm),('adam_m',mm),('adam_v',vm),('loss',lm)]:maxima[key]=max(maxima[key],value['relative_l2'])
        if step%100==0:
            print(case+f' step={step} param_rel={wm["relative_l2"]} mismatch={mismatch} failures={len(summary["failures"])}',flush=True);save()
    average=float(np.mean(mismatches))
    if average>1e-4:summary['failures'].append(dict(case=case,average_hard_mismatch=average))
    return dict(steps=1000,average_hard_mismatch=average,p99_hard_mismatch=float(np.quantile(mismatches,.99)),max_relative_l2=maxima)


def optimizer_test(base):
    global case
    for mode in ['direct','preview','direct_options','preview_options']:
        case='optimizer/'+mode
        parameters=[[p.detach().clone().requires_grad_() for p in base.parameters()] for _ in range(3)]
        options=dict(eps=1e-8,beta1=.8,beta2=.99,decay=.02) if 'options' in mode else {}
        optimizers=[extension.Adam(parameters[i],i==2,**options) for i in range(3)]
        generator=torch.Generator(device='cuda').manual_seed(125)
        for step in range(1,1001):
            lr=.001*(.999**step) if step%17 else 0.
            grads=[None if (step+j)%7==0 else torch.zeros_like(p) if step%11==0 else torch.randn(p.shape,device='cuda',generator=generator)*.01 for j,p in enumerate(parameters[0])]
            if mode.startswith('preview'):
                before=[[p.detach().clone() for p in group] for group in parameters];old_states=[o.state() for o in optimizers]
                old_states=[([m.clone() for m in s[0]],[v.clone() for v in s[1]],list(s[2])) for s in old_states]
                candidate=[o.preview(grads,lr,.2 if step%2 else 1.,.25 if step%3==0 else 1.,step%5!=0) for o in optimizers]
                for j in range(len(candidate[0])):compare('optimizer_candidate',f'{step}/{j}',[c[j] for c in candidate],'optimizer')
                if step%5==0:
                    for i in range(3):
                        now=optimizers[i].state()
                        assert all(torch.equal(a,b) for a,b in zip(before[i],parameters[i]))
                        assert now[2]==old_states[i][2] and all(torch.equal(a,b) for a,b in zip(now[0]+now[1],old_states[i][0]+old_states[i][1]))
            else:
                for o in optimizers:o.step(grads,lr)
            states=[o.state() for o in optimizers]
            for j in range(18):
                for key,value in [('parameter',[p[j] for p in parameters]),('m',[s[0][j] for s in states]),('v',[s[1][j] for s in states])]:compare('optimizer',f'{step}/{j}/{key}',value,'optimizer')
            assert states[0][2]==states[1][2]==states[2][2]
        print(case+' 1000 steps complete',flush=True);save()


try:
    for item in models:
        checkpoint_name=item['name'];path=Path(item['path']);path=path if path.is_absolute() else root/path
        cloud,=(path/'point_cloud').glob('iteration_*/point_cloud.ply')
        vertex=PlyData.read(cloud)['vertex']
        def tensor(names):return torch.from_numpy(np.stack([vertex[n] for n in names],axis=1).astype(np.float32)).cuda()
        raw=[tensor(['x','y','z']),tensor([f'rot_{i}' for i in range(4)]),tensor([f'scale_{i}' for i in range(3)]),tensor(['opacity'])]
        metadata=load_selector_metadata(cloud.parent/'selector_metadata.ply','cuda')
        config=namespace['load_selector_config'](str(path/'selector_config.yaml'))
        base=load_selector(str(path/'selector.pt'),'cuda')
        storage=cv2.FileStorage(str(path/'selector_config.yaml'),cv2.FILE_STORAGE_READ)
        node=storage.getNode('Selector.learning_rate');config['learning_rate']=node.real() if not node.empty() else .001
        storage.release()
        entry=config.get('entry_maturity',0.);tau=config['temperature']
        record=dict(name=checkpoint_name,path=str(path),N=len(vertex),mapping_iteration=metadata['mapping_iteration'],entry=entry,tau=tau,
            checkpoint_sha256=hashlib.sha256((path/'selector.pt').read_bytes()).hexdigest(),
            gaussian_sha256=hashlib.sha256(cloud.read_bytes()).hexdigest(),metadata_sha256=hashlib.sha256((cloud.parent/'selector_metadata.ply').read_bytes()).hexdigest(),
            load_count=1,parameter_tensors=len(list(base.parameters())),parameters=sum(p.numel() for p in base.parameters()))
        summary['checkpoints'].append(record);print('checkpoint '+json.dumps(record),flush=True)
        parity(base,raw,metadata,config,entry,tau)
        record['training']=train(base,raw,metadata,config,entry,tau)
        if checkpoint_name=='room_current':optimizer_test(base)
        save()
    summary['complete']=True
    save()
except Exception as error:
    summary['error']=repr(error);save();raise
finally:
    stream.close();selections.close();training_file.close()
raise SystemExit(bool(summary['failures']))
