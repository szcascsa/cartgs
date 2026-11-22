"""相同N/K、交错CUDA Events；仅冻结输入PyTorch原型成本，不冒充生产Mapping。"""
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import torch
from af_runtime import ROOT,Study,RATIOS,csv,save,old


def samples(fn,n=25,prepare=None):
    for _ in range(5):
        if prepare is not None:prepare()
        fn()
    torch.cuda.synchronize();pairs=[];start=time.perf_counter()
    for _ in range(n):
        if prepare is not None:prepare()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();pairs.append((a,b))
    host=(time.perf_counter()-start)*1000/n;torch.cuda.synchronize()
    return [a.elapsed_time(b) for a,b in pairs],host


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    study=Study(a.scene);models={};info={}
    for name in ['Baseline','Head8','AnchorFiLM8','FrozenBaseHead8','FrozenBaseContextHead8']:
        tier='frozen_mechanism' if name=='FrozenBaseContextHead8' else ('mechanism_continue' if a.scene=='fr1_xyz' else 'frozen_mechanism') if name=='FrozenBaseHead8' else 'confirm_continue'
        directory=ROOT/'results'/tier/a.scene/name/'7200';meta=json.loads((directory/'complete.json').read_text())
        payload=torch.load(meta['checkpoint'],map_location='cuda');net=study.make(meta['config'],7200);net.load_state_dict(payload['state_dict'])
        optimizer=study.optimizer(net,meta['config'],'continue');optimizer.load_state_dict(payload['optimizer'])
        models[name]=(net,optimizer,copy.deepcopy(payload['state_dict']),copy.deepcopy(payload['optimizer']))
        parameters=sum(x.numel() for x in net.parameters() if x.requires_grad);base=sum(x.numel() for x in net.base.parameters() if x.requires_grad);total=sum(x.numel() for x in net.parameters())
        info[name]=dict(trainable=parameters,base_trainable=base,total_weights=total,extra_trainable=parameters-base,parameter_bytes=total*4,adam_moment_bytes=parameters*8,gradient_bytes=parameters*4,base_frozen=name.startswith('FrozenBase'))
    rows=[]
    for block in range(3):
        for ratio in RATIOS:
            ids=study.fixture.sets[ratio,'same_k'];vjp=torch.ones(len(ids),14,device='cuda')/max(len(ids),1)
            for name in (list(models) if block%2==0 else list(models)[::-1]):
                net,opt,state,optimstate=models[name];net.load_state_dict(state);opt.load_state_dict(copy.deepcopy(optimstate))
                for stage in ['gather55','feature_gather','forward','backward_only','forward_backward','render_photo_adam']:
                    if stage=='feature_gather' and name=='Baseline':continue
                    prepared={}
                    def prepare():
                        opt.zero_grad(set_to_none=True);prepared['delta']=net(ratio,ids)
                    def fn():
                        if stage=='gather55':return study.x[ids]
                        if stage=='feature_gather':
                            return net.embedding[study.owners[ids]] if hasattr(net,'embedding') else study.context[study.owners[ids]]
                        if stage=='forward':
                            with torch.no_grad():return net(ratio,ids)
                        if stage=='backward_only':return (prepared.pop('delta')*vjp).sum().backward()
                        opt.zero_grad(set_to_none=True);delta=net(ratio,ids)
                        if stage=='forward_backward':return (delta*vjp).sum().backward()
                        image,_,_=study.scene.render(ids,delta,study.scene.train[0]);loss=old.rt.photo(image,study.scene.train[0]['gt']);loss.backward();opt.step()
                    for other,opt_other,_,_ in models.values():opt_other.zero_grad(set_to_none=True)
                    start_allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();vals,host=samples(fn,prepare=prepare if stage=='backward_only' else None)
                    for j,value in enumerate(vals):rows.append(dict(scene=a.scene,N=study.scene.N,M=study.M,ratio=ratio,K=len(ids),model=name,stage=stage,block=block,sample=j,gpu_ms=value,host_enqueue_ms=host,start_allocated=start_allocated,peak_allocated=torch.cuda.max_memory_allocated(),peak_extra=torch.cuda.max_memory_allocated()-start_allocated))
                    net.load_state_dict(state);opt.load_state_dict(copy.deepcopy(optimstate))
            csv(ROOT/f'performance_{a.scene}.csv',rows)
    save(ROOT/f'performance_{a.scene}.json',dict(parameters=info,scope='cached55 frozen input PyTorch prototype; no live physical55, no production CUDA fusion or online SLAM; shared server GPU; extra gather row count K',samples=25,blocks=3))
    print('PERFORMANCE_COMPLETE',a.scene,flush=True)
if __name__=='__main__':main()
