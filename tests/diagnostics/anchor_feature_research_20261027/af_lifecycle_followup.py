"""UID重排/新增后的真实Adam更新，排除只搬状态但叶子不可继续训练。"""
import io,json
import torch
from af_gates import remap_embedding
from af_runtime import ROOT
torch.manual_seed(17);table=torch.nn.Parameter(torch.randn(3,8,device='cuda'));opt=torch.optim.Adam([table],lr=.001,eps=1e-15,fused=True)
table.square().sum().backward();opt.step();new=remap_embedding(table,torch.tensor([91,13,27]),torch.tensor([27,102,91]),opt)
assert new.is_leaf and new.grad is None and opt.state[new]['exp_avg'][1].count_nonzero()==0
before=new.detach().clone();opt.zero_grad(set_to_none=True);(new-1).square().sum().backward();opt.step()
assert new.grad.shape==(3,8) and int(opt.state[new]['step'])==2 and (new[1]-before[1]).abs().sum()>0
buffer=io.BytesIO();torch.save(dict(features=new.detach(),uids=torch.tensor([27,102,91]),optimizer=opt.state_dict()),buffer);buffer.seek(0);payload=torch.load(buffer)
restored=torch.nn.Parameter(payload['features'].clone());second=torch.optim.Adam([restored],lr=.001,eps=1e-15,fused=True);second.load_state_dict(payload['optimizer']);second.zero_grad(set_to_none=True);(restored-1).square().sum().backward();second.step()
assert int(second.state[restored]['step'])==3 and torch.equal(payload['uids'],torch.tensor([27,102,91]))
record=dict(passed=True,scope='isolated stable UID helper + post-remap Adam + checkpoint parameter/state restore; no production event integration',uid_order=[27,102,91],clock_after_remap_step=2,clock_after_restore_step=3)
(ROOT/'lifecycle_followup.json').write_text(json.dumps(record,indent=2));print('LIFECYCLE_FOLLOWUP_PASS',flush=True)
