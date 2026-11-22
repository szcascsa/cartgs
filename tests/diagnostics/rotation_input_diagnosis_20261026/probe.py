"""Rotation输入的几何表示、最终梯度及固定集合交叉渲染诊断。"""
import importlib.util
import json
import math
from pathlib import Path
import torch
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/rotation-input-diagnosis-20261026')
SOURCE=Path('/home/prmi/zwj/s00-rotation-encoder-20261026')
spec=importlib.util.spec_from_file_location('quaternion_experiment',SOURCE/'run.py')
qrun=importlib.util.module_from_spec(spec)
spec.loader.exec_module(qrun)
rt.ROOT=ROOT


def rotation_matrix(q):
    w,x,y,z=torch.nn.functional.normalize(q,dim=1).unbind(1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)


def representation(q,scale,kind):
    R=rotation_matrix(q)
    if kind=='rot6':
        return R[:,:,:2].transpose(1,2).reshape(-1,6)
    assert kind=='cov6'
    variance=scale.square()
    variance=variance/variance.sum(1,keepdim=True)
    cov=(R*variance[:,None,:])@R.transpose(1,2)
    return torch.stack([cov[:,0,0],cov[:,1,1],cov[:,2,2],
        math.sqrt(2)*cov[:,0,1],math.sqrt(2)*cov[:,0,2],math.sqrt(2)*cov[:,1,2]],1)


def load(kind,seed,scene):
    net=qrun.Network(scene,qrun.config(kind),seed)
    path=SOURCE/'results/fr1_xyz'/qrun.PROTOCOL['configs'][kind]/str(seed)/'step_12000.pt'
    net.load_state_dict(torch.load(path,map_location='cuda')['state_dict'])
    return net.eval()


def ids_for(scene,net,ratio):
    p,h,logits=net(ratio)
    margin=logits.double()-torch.logit(scene.uniform.double())
    margin[~scene.eligible]=-torch.inf
    ids=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values
    return ids,p,h


@torch.no_grad()
def input_probe(scene):
    q=qrun.previous.canonical_rotation(scene.g.get_rotation.detach())
    s=scene.g.get_scaling.detach()
    raw=scene.attrs
    def stats(v):
        return dict(min=v.min(0).values.cpu().tolist(),max=v.max(0).values.cpu().tolist(),
            mean=v.mean(0).cpu().tolist(),std=v.std(0).cpu().tolist())
    anisotropy=s.max(1).values/s.min(1).values
    nearest=[]
    for start in range(0,len(q),512):
        dot=q[start:start+512]@q.T
        index=torch.arange(len(dot),device='cuda')
        dot[index,start+index]=0
        nearest.append(dot.abs().argmax(1))
    near=torch.cat(nearest)
    cos=(q*q[near]).sum(1)
    angle=2*torch.acos(cos.abs().clamp(0,1))*180/math.pi
    distance=(q-q[near]).norm(dim=1)
    boundary=[]
    gen=torch.Generator(device='cuda').manual_seed(92311)
    for eps in [1e-4,1e-3,1e-2]:
        signs=[]
        for _ in range(16):
            axis=torch.randn(q.shape[0],3,device='cuda',generator=gen)
            axis=torch.nn.functional.normalize(axis,dim=1)
            dq=torch.cat([torch.full((len(q),1),math.cos(eps/2),device='cuda'),axis*math.sin(eps/2)],1)
            moved=rt._multiply_quaternion(dq,q)
            moved=qrun.previous.canonical_rotation(moved)
            signs.append((q*moved).sum(1)<0)
        flips=torch.stack(signs)
        boundary.append(dict(perturbation_radians=eps,trial_count=16,
            antipodal_jump_fraction=float(flips.float().mean()),any_jump_point_fraction=float(flips.any(0).float().mean())))
    rot6=representation(q,s,'rot6');cov6=representation(q,s,'cov6')
    q180=torch.zeros_like(q);q180[:,1]=1
    # 右乘局部X轴180度：同一Scale的Gaussian协方差保持不变。
    same_ellipsoid=rt._multiply_quaternion(q,q180)
    covariance_error=float((cov6-representation(same_ellipsoid,s,'cov6')).abs().max())
    results=dict(N=scene.N,raw10=stats(raw),quaternion4=stats(q),rot6=stats(rot6),cov6=stats(cov6),
        max_quaternion_norm_error=float((q.norm(dim=1)-1).abs().max()),
        nearly_isotropic_fraction_1p05=float((anisotropy<1.05).float().mean()),
        nearly_isotropic_fraction_1p2=float((anisotropy<1.2).float().mean()),
        anisotropy_quantiles=torch.quantile(anisotropy,torch.tensor([.1,.5,.9,.99],device='cuda')).cpu().tolist(),
        nearest_rotation_neighbor=dict(angle_degree_p50=float(angle.median()),
            angle_degree_p90=float(angle.quantile(.9)),canonical_q_distance_gt1_fraction=float((distance>1).float().mean()),
            angle_lt5_and_distance_gt1_fraction=float(((angle<5)&(distance>1)).float().mean())),
        small_rotation_boundary_probe=boundary,
        equivalent_covariance_max_error=covariance_error,
        equivalent_covariance_quaternion_distance=float((q-qrun.previous.canonical_rotation(same_ellipsoid)).norm(dim=1).mean()),
        equivalent_covariance_rot6_distance=float((rot6-representation(same_ellipsoid,s,'rot6')).norm(dim=1).mean()))
    assert covariance_error<2e-6
    assert torch.equal(rot6,representation(-q,s,'rot6'))
    assert torch.equal(cov6,representation(-q,s,'cov6'))
    rt.save_json(ROOT/'input_statistics.json',results)
    return results


def gradient_probe(scene):
    rows=[]
    for seed in [3100,3101]:
        for kind in ['baseline','rotation_all']:
            net=load(kind,seed,scene)
            for ratio in rt.TRAIN:
                for vi in [0,len(scene.train)//3,2*len(scene.train)//3,len(scene.train)-1]:
                    p,h,_=net(ratio)
                    ids=torch.nonzero(p.detach()>scene.uniform).flatten()
                    delta=net.compensate(h,ids,ratio)
                    gate=1+p[ids]-p[ids].detach()
                    v=scene.train[vi]
                    image,_,_=scene.render(ids,delta,v,gate)
                    photo=rt.photo(image,v['gt'])
                    comp_image,_,_=scene.render(ids,delta,v)
                    comp=rt.photo(comp_image,v['gt'])
                    sel_image,_,_=scene.render(ids,delta.detach(),v,gate)
                    sel=rt.photo(sel_image,v['gt'])
                    budget=.01*(p.sum()-ratio*scene.N).abs()/(ratio*scene.N)
                    named=[('encoder',net.encoder.weight),('latent',net.extra_tables[rt.TRAIN.index(ratio)])]
                    gradients={}
                    for loss_name,loss in [('compensation',comp),('selection',sel),('budget',budget),('total',photo+budget)]:
                        gs=torch.autograd.grad(loss,[par for _,par in named],allow_unused=True,retain_graph=True)
                        gradients[loss_name]={name:(g if g is not None else torch.zeros_like(par))
                            for (name,par),g in zip(named,gs)}
                    for name,_ in named:
                        parts={k:v[name] for k,v in gradients.items()}
                        residual=parts['total']-parts['selection']-parts['compensation']-parts['budget']
                        scopes=[('all',slice(None))]+([('rotation_columns',(slice(None),slice(11,None)))]
                            if kind=='rotation_all' and name=='encoder' else [])
                        for scope,index in scopes:
                            values={key:value[index].flatten() for key,value in parts.items()}
                            row=dict(seed=seed,kind=kind,ratio=ratio,view=vi,parameter=name,scope=scope,
                                selected=len(ids),photo=float(photo),budget=float(budget),
                                gradient_additivity_relative_error=float(residual.norm()/(parts['total'].norm()+1e-12)))
                            for key,val in values.items():row[key+'_grad_norm']=float(val.norm())
                            for a,b in [('selection','compensation'),('budget','compensation'),('budget','selection')]:
                                row[a+'_'+b+'_cosine']=float(torch.dot(values[a],values[b])/(values[a].norm()*values[b].norm()+1e-20))
                            rows.append(row)
            del net
            torch.cuda.empty_cache()
    rt.write_csv(ROOT/'gradient_snapshot.csv',rows)


@torch.no_grad()
def crossover(scene):
    views=[('bank_train',v) for v in scene.bank_full if v['path'] in {x['path'] for x in scene.train}]
    views += [('test',v) for v in scene.load_test()]
    rows=[]
    for seed in [3100,3101]:
        nets={kind:load(kind,seed,scene) for kind in ['baseline','rotation_all']}
        for ratio in rt.RATIOS:
            states={kind:ids_for(scene,net,ratio) for kind,net in nets.items()}
            for selection in nets:
                ids=states[selection][0]
                for compensation in ['baseline','rotation_all','none']:
                    delta=None if compensation=='none' else nets[compensation].compensate(states[compensation][2],ids,ratio)
                    for split,v in views:
                        image,_,_=scene.render(ids,delta,v)
                        rows.append(dict(seed=seed,ratio=ratio,selection=selection,compensation=compensation,
                            split=split,path=v['path'],selected=len(ids),
                            psnr=float(-10*torch.log10((image.clamp(0,1)-v['gt']).square().mean())),
                            photo=float(rt.photo(image,v['gt']))))
        del nets
        torch.cuda.empty_cache()
    rt.write_csv(ROOT/'crossover_per_view.csv',rows)


if __name__=='__main__':
    scene=rt.Scene('fr1_xyz')
    input_probe(scene)
    gradient_probe(scene)
    crossover(scene)
    assert scene.unchanged()
    rt.save_json(ROOT/'probe_complete.json',dict(master_frozen=True,source='two previous final12000 checkpoints',
        gradient_scope='final checkpoint snapshot on4 preregistered Bank views per ratio; not full training history',
        crossover_scope='off-support module recombination, not retraining or additive causal attribution'))
