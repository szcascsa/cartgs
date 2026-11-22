"""非零生产同函数、初始函数、真实photo梯度和UID/Adam迁移检查。"""
import json
import torch
from af_runtime import ROOT,Study,CONFIG,old,save


def remap_embedding(table,old_uids,new_uids,optimizer):
    lookup={int(uid):j for j,uid in enumerate(old_uids.cpu().tolist())}
    source=torch.tensor([lookup.get(int(uid),-1) for uid in new_uids.cpu().tolist()],device=table.device)
    def gather(value):
        padded=torch.cat([value.detach(),value.new_zeros(1,value.shape[1])]);return padded[torch.where(source>=0,source,len(table))]
    new=torch.nn.Parameter(gather(table));state=optimizer.state.pop(table,{})
    for group in optimizer.param_groups:group['params']=[new if p is table else p for p in group['params']]
    for name in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
        if name in state:state[name]=gather(state[name])
    optimizer.state[new]=state
    return new


def main():
    torch.backends.cuda.matmul.allow_tf32=False;study=Study('fr1_xyz');rows=[]
    baseline=study.make(dict(feature='none'),7200)
    original=old.make(study.fixture,CONFIG,5100);original.load_state_dict(study.payload['state_dict'])
    for ratio in old.rt.RATIOS:
        ids=study.fixture.sets[ratio,'same_k'];a=baseline(ratio,ids);b=original(ratio,ids)
        assert torch.equal(a,b);rows.append(dict(test='original_output',ratio=ratio,max_abs=float((a-b).abs().max())))
    configs=[dict(id='Early8',feature='learned',fusion='early',dim=8),dict(id='Mid8',feature='learned',fusion='mid',dim=8),
        dict(id='AnchorFiLM8',feature='learned',fusion='film',dim=8),dict(id='Head8',feature='learned',fusion='head',dim=8),
        dict(id='ContextFiLM8',feature='context',fusion='film',dim=8),dict(id='HybridFiLM8',feature='hybrid',fusion='film',dim=8)]
    for cfg in configs:
        net=study.make(cfg,7200);ids=study.fixture.sets[.05,'same_k'];a=net(.05,ids);b=baseline(.05,ids)
        assert torch.equal(a,b),(cfg['id'],float((a-b).abs().max()))
        image,_,_=study.scene.render(ids,a,study.scene.train[0]);loss=old.rt.photo(image,study.scene.train[0]['gt']);loss.backward()
        norms={name:float(p.grad.norm()) for name,p in net.named_parameters() if p.grad is not None and not name.startswith('base.')}
        if hasattr(net,'embedding'):assert net.embedding.grad.norm()>0,cfg
        if hasattr(net,'context'):assert sum(v for k,v in norms.items() if k.startswith('context.'))>0,cfg
        rows.append(dict(test=cfg['id'],ratio=.05,max_abs=0.,gradient_norms=norms))
    table=torch.nn.Parameter(torch.randn(3,8,device='cuda'));optimizer=torch.optim.Adam([table],lr=.001)
    table.square().sum().backward();optimizer.step();oldtable=table.detach().clone();moment=optimizer.state[table]['exp_avg'].clone()
    new=remap_embedding(table,torch.tensor([91,13,27]),torch.tensor([27,102,91]),optimizer)
    assert torch.equal(new[0],oldtable[2]) and torch.equal(new[2],oldtable[0]) and new[1].count_nonzero()==0
    assert torch.equal(optimizer.state[new]['exp_avg'][0],moment[2]) and optimizer.state[new]['exp_avg'][1].count_nonzero()==0
    assert int(optimizer.state[new]['step'])==1 and optimizer.param_groups[0]['params'][0] is new
    save(ROOT/'correctness_gates.json',dict(pass_=True,rows=rows,lifecycle='stable UID reorder/prune/append preserves values and moments; new rows zero; existing tensor Adam clock retained',map_frozen=True))
    print('ALL_GATES_PASS',len(rows),flush=True)
if __name__=='__main__':main()
