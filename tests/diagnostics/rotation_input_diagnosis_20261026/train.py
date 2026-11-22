"""只替换新增Rotation输入编码，保留四Head与原训练协议。"""
import gc
import json
import math
from pathlib import Path
import torch
from torch import nn
from probe import ROOT,SOURCE,qrun,rt,representation,rotation_matrix

PROTOCOL=json.loads((ROOT/'representation_protocol.json').read_text())


def config(kind):
    cfg=qrun.config('rotation_all')
    cfg.update(id=PROTOCOL['configs'][kind],rotation_representation=kind)
    return cfg


class Network(qrun.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        self.rotation_features=representation(scene.g.get_rotation.detach(),scene.g.get_scaling.detach(),
            cfg['rotation_representation']).detach()
        encoder=nn.Linear(17,32,device='cuda')
        with torch.no_grad():
            encoder.weight[:,:11].copy_(self.encoder.weight[:,:11])
            encoder.weight[:,11:].zero_()
            encoder.bias.copy_(self.encoder.bias)
        self.encoder=encoder
        for branch in self.compensation.branches.values():
            old=branch[0]
            first=nn.Linear(65,32,device='cuda')
            with torch.no_grad():
                first.weight[:,:11].copy_(old.weight[:,:11])
                first.weight[:,11:17].zero_()
                first.weight[:,17:].copy_(old.weight[:,15:])
                first.bias.copy_(old.bias)
            branch[0]=first

    def raw(self,ratio,ids=None):
        attrs=self.scene.attrs if ids is None else self.scene.attrs[ids]
        features=self.rotation_features if ids is None else self.rotation_features[ids]
        return torch.cat([attrs,torch.full((len(attrs),1),math.log(ratio),device=attrs.device),features],1)


rt.Network=Network


@torch.no_grad()
def covariance_renderer_gate(scene):
    ids=torch.arange(min(512,scene.N),device='cuda')
    q=scene.g.get_rotation.detach()[ids];scale=scene.g.get_scaling.detach()[ids]
    R=rotation_matrix(q);cov=(R*scale.square()[:,None,:])@R.transpose(1,2)
    packed=torch.stack([cov[:,0,0],cov[:,0,1],cov[:,0,2],cov[:,1,1],cov[:,1,2],cov[:,2,2]],1)
    v=scene.train[0]
    image,_,_=scene.render(ids,None,v)
    xyz,_,_,dc,rest,opacity=[value[ids] for value in scene.render_bases]
    direct,_=rt._make_rasterizer(v['cam'],scene.g.active_sh_degree,scene.bg,1.)(means3D=xyz.contiguous(),
        means2D=torch.zeros_like(xyz),dc=dc.contiguous(),shs=rest.contiguous(),colors_precomp=None,
        opacities=opacity.sigmoid().contiguous(),scales=None,rotations=None,cov3D_precomp=packed.contiguous())
    error=float((image-direct).abs().max())
    assert torch.allclose(image,direct,atol=2e-4,rtol=2e-4),error
    rt.save_json(ROOT/'covariance_renderer_gate.json',dict(rows=len(ids),max_image_error=error,
        covariance_convention='R diag(scale^2) R^T, packed xx xy xz yy yz zz',passed=True))


def gate(scene,kind,seed):
    net=Network(scene,config(kind),seed)
    base=qrun.Network(scene,qrun.config('baseline'),seed)
    max_hidden_error=0.
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=base(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert torch.equal(p,p0)
        assert torch.allclose(h,h0,atol=2e-6,rtol=2e-6)
        assert torch.equal(net.compensate(h,ids,ratio),base.compensate(h0,ids,ratio))
        max_hidden_error=max(max_hidden_error,float((h-h0).abs().max()))
    # 零输出初始化的首步不要求隐藏层梯度非零，另用临时投影验证通路。
    for branch in net.compensation.branches.values():
        with torch.no_grad():branch[-1].weight.fill_(.01)
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten()
    net.compensate(h,ids,.05).square().mean().backward()
    assert net.encoder.weight.grad[:,11:].norm()>0
    assert all(branch[0].weight.grad[:,11:17].norm()>0 for branch in net.compensation.branches.values())
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    rt.save_json(ROOT/f'correctness_{kind}_{seed}.json',dict(kind=kind,seed=seed,
        initial_probability_exact=True,initial_compensation_exact=True,max_hidden_error=max_hidden_error,
        input_width=17,head_input_width=65,rotation_path_gradient=True,inactive_ratio_gradient_none=True,
        gradient_probe='temporary nonzero output projection, not actual step1',
        total_parameters=sum(p.numel() for p in net.parameters()),master_frozen=scene.unchanged()))
    del net,base
    gc.collect();torch.cuda.empty_cache()


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((SOURCE/'results/fr1_xyz/scene_manifest.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==PROTOCOL['runtime_sha256']
    assert rt.sha(SOURCE/'run.py')==PROTOCOL['source_run_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary']))==PROTOCOL['renderer_sha256']
    covariance_renderer_gate(scene)
    rt.np.savez_compressed(ROOT/'input_encodings.npz',**{k:representation(scene.g.get_rotation.detach(),scene.g.get_scaling.detach(),k).cpu().numpy() for k in ['rot6','cov6']})
    for kind,seed in PROTOCOL['jobs']:gate(scene,kind,seed)
    runs=[]
    for kind,seed in PROTOCOL['jobs']:
        cfg=config(kind);directory=ROOT/'results/fr1_xyz'/cfg['id']/str(seed)
        info=rt.train(scene,cfg,seed,12000,directory,checkpoints=[4000,8000,12000])
        net=Network(scene,cfg,seed)
        net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict'])
        for split in ['bank','validation','test']:
            target=directory/(split+'_full.csv')
            if not target.exists():
                rows=rt.evaluate(scene,net.eval(),split,full_metrics=True,policies=('same_k','cached'))
                for row in rows:row.update(config=cfg['id'],seed=seed)
                rt.write_csv(target,rows)
        with torch.no_grad():
            rt.save_json(directory/'input_updates.json',dict(encoder_feature_weight_norm=float(net.encoder.weight[:,11:].norm()),
                head_feature_weight_norms={name:float(branch[0].weight[:,11:17].norm()) for name,branch in net.compensation.branches.items()}))
        assert scene.unchanged()
        runs.append(info);rt.save_json(ROOT/'runs.json',runs)
        print('COMPLETE',kind,seed,flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='COMPLETE',models=4))


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'train.exit').write_text('1');raise
    (ROOT/'train.exit').write_text('0')
