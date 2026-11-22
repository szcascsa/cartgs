"""候选合法性、真实photo梯度、输入长尾和Teacher可学习性检查。"""
import argparse
import copy
import json
from pathlib import Path
import torch
import run as r
from neural import audit_parameters
from teacher_probe import physical_target


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='');args=p.parse_args()
    scene=r.rt.Scene('fr1_xyz');f=r.Fixture(scene);rows=[];grads=[]
    if args.checkpoint:
        payload=torch.load(args.checkpoint,map_location='cuda');configs=[payload['config']];seed=payload['seed']
    else:configs=json.loads((r.ROOT/'protocol.json').read_text())['configs'];seed=5100
    input_stats=dict(N=scene.N,M=scene.M,width=f.features.own.shape[1],
        original_offset_max=float(scene.attrs[:,:3].abs().max()),processed_feature_max=float(f.features.own.abs().max()),
        source_map=scene.hashes,private_input_parameters=False)
    r.rt.save_json(r.ROOT/'input_statistics.json',input_stats)
    zero=torch.zeros(2,14,device='cuda',requires_grad=True);physical=physical_target(zero)
    physical[:,6:9].sum().backward();assert zero.grad[:,7:10].abs().min()>0
    r.rt.save_json(r.ROOT/'teacher_tangent_gate.json',dict(zero_rotation_gradient=zero.grad.cpu().tolist(),pass_=True))
    for cfg in configs:
        if cfg['kind']=='reference59':continue
        net=r.make(f,cfg,seed)
        if args.checkpoint:net.load_state_dict(payload['state_dict'])
        audit=audit_parameters(net);ids=f.sets[.05,'same_k']
        output=net(.05,ids);loss=r.rt.photo(scene.render(ids,output,scene.train[0])[0],scene.train[0]['gt']);loss.backward()
        norms={name:float(p.grad.norm()) for name,p in net.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        assert any(value>0 for key,value in norms.items() if 'heads' in key)
        rows.append(dict(config=cfg['id'],parameters=sum(p.numel() for p in net.parameters()),
            only_shared_linear_params=True,no_embeddings=True,no_trainable_scene_buffers=True,nonzero_real_photo_head_gradient=True))
        # 同一只读特征/上下文可在另一数据对象重建，输出不依赖对象身份或Gaussian ID参数。
        replica=r.make(f,cfg,seed);replica.load_state_dict(net.state_dict())
        assert torch.equal(replica(.05,ids),output.detach())
        grads.append(dict(config=cfg['id'],gradients=norms,parameter_list=audit))
    assert scene.unchanged();r.rt.save_json(r.ROOT/'neural_correctness.json',rows);r.rt.save_json(r.ROOT/'gradient_probe.json',grads)
    print('NEURAL_GATES_PASS',len(rows),flush=True)


if __name__=='__main__':main()
