"""固定预算E8/E50单变量对照；旧8维与R3结果只作匹配参考。"""
import gc
import hashlib
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-e50-compensation-20261026')
OLD=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
BaseNetwork=rt.Network
rt.ROOT=ROOT

class Network(BaseNetwork):
    @staticmethod
    def head(inputs,outputs,depth,seed,extra=0):
        if extra!=50:return BaseNetwork.head(inputs,outputs,depth,seed,extra)
        assert inputs==82 and outputs==14 and depth==2
        first=nn.Linear(82,32,device='cuda');common=nn.Linear(32,32,device='cuda')
        rt.linear_init(common,seed)
        generator=torch.Generator(device='cuda');generator.manual_seed(seed+5)
        old_columns=torch.rand((32,28),generator=generator,device='cuda')*.1-.05
        new_columns=torch.rand((32,22),generator=generator,device='cuda')*.1-.05
        with torch.no_grad():
            first.weight[:,:32].copy_(common.weight);first.bias.copy_(common.bias)
            first.weight[:,32:].copy_(torch.cat([old_columns,new_columns],1))
        second=nn.Linear(32,32,device='cuda');rt.linear_init(second,seed+10)
        last=nn.Linear(32,14,device='cuda')
        with torch.no_grad():last.weight.zero_();last.bias.zero_()
        return nn.Sequential(first,nn.ReLU(),second,nn.ReLU(),last)

rt.Network=Network
SCENE=None

def init_scene(name):
    global SCENE
    SCENE=rt.Scene(name)

def evaluate(info,split):
    directory=Path(info['checkpoint']).parent;output=directory/(split+'_full.csv')
    if output.exists():return
    net=Network(SCENE,info['config'],info['seed'])
    payload=torch.load(info['checkpoint'],map_location='cuda');net.load_state_dict(payload['state_dict'])
    rows=rt.evaluate(SCENE,net.eval(),split,full_metrics=True,policies=('same_k','cached'))
    for row in rows:row.update(config=info['config']['id'],seed=info['seed'])
    temp=output.with_suffix('.tmp');rt.write_csv(temp,rows);temp.replace(output)
    del net,payload;gc.collect()

def worker(seed):
    cfg=PROTOCOL['candidate'];directory=ROOT/'results'/SCENE.name/cfg['id']/str(seed)
    info=rt.train(SCENE,cfg,seed,12000,directory,checkpoints=[4000,8000,12000])
    evaluate(info,'validation');evaluate(info,'test')
    assert SCENE.unchanged()
    return dict(scene=SCENE.name,**info)

def gate(name):
    scene=rt.Scene(name)
    old_manifest=json.loads((OLD/'results'/name/'scene_manifest.json').read_text())
    current=json.loads((ROOT/'results'/name/'scene_manifest.json').read_text())
    assert old_manifest==current,'Source map/frame/population changed'
    phase=PROTOCOL['baseline_phases'][name]
    for seed in PROTOCOL['seeds'][name]:
        baseline=OLD/'results'/name/phase/'S00R00E08D12'/str(seed)/'complete.json'
        info=json.loads(baseline.read_text());assert info['steps']==12000 and info['master_frozen']
        assert all(v==3000 for v in info['ratio_exposures'].values())
    seed=PROTOCOL['seeds'][name][0];eight=dict(PROTOCOL['candidate'],extra=8,id='S00R00E08D12')
    a=Network(scene,eight,seed);b=Network(scene,PROTOCOL['candidate'],seed)
    for key,value in a.state_dict().items():
        if key.startswith('extra_tables.'):assert torch.equal(value,b.state_dict()[key][:,:8])
        elif key=='compensation.0.weight':
            assert torch.equal(value[:,:32],b.state_dict()[key][:,:32])
            assert torch.equal(value[:,32:],b.state_dict()[key][:,32:40])
        else:assert torch.equal(value,b.state_dict()[key]),key
    with torch.no_grad():
        for ratio in rt.RATIOS:
            pa,ha,_=a(ratio);pb,hb,_=b(ratio)
            assert torch.equal(pa,pb) and torch.equal(ha,hb)
            ids=torch.nonzero(pa>scene.uniform).flatten()
            assert b.compensate(hb,ids,ratio).shape==(len(ids),14)
            assert torch.count_nonzero(b.compensate(hb,ids,ratio))==0
    assert scene.unchanged()
    rt.save_json(ROOT/(name+'_gate.json'),dict(source_manifest_equal=True,baseline_steps_equal=True,
        common_initial_weights_equal=True,first8_latent_projection_columns_equal=True,initial_probabilities_equal=True,
        initial_compensation_zero=True,N=scene.N,M=scene.M,K=scene.K))
    del a,b,scene;gc.collect();torch.cuda.empty_cache()

def main():
    old=json.loads((OLD/'delivery_verification.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==old['prototype_sha256']['runtime.py'],'Runtime protocol drift'
    assert rt.sha(Path(old['renderer_binary']))==old['renderer_sha256'],'Renderer binary drift'
    all_runs=[]
    for name,seeds in PROTOCOL['seeds'].items():
        gate(name)
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=init_scene,initargs=(name,)) as pool:
            futures=[pool.submit(worker,seed) for seed in seeds]
            for future in as_completed(futures):
                info=future.result();all_runs.append(info)
                rt.save_json(ROOT/'runs.json',all_runs)
                status=dict(stage='E50',scene=name,seed=info['seed'],completed=len(all_runs),total=16)
                rt.save_json(ROOT/'status.json',status);print('COMPLETED',status,flush=True)
    rt.save_json(ROOT/'status.json',dict(stage='TRAINING_COMPLETE',completed=16,total=16))

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'run.exit').write_text('1');raise
    (ROOT/'run.exit').write_text('0')
