"""Bank发现驱动的优化器控制：同权重、同帧序，仅清空原base Adam历史。"""
import argparse,json,gc,types
from pathlib import Path
import torch
from af_runtime import ROOT,Study,train,save

p=argparse.ArgumentParser();p.add_argument('--queue',required=True);a=p.parse_args();q=json.loads(Path(a.queue).read_text())
torch.backends.cuda.matmul.allow_tf32=False;study=Study(q['scene']);original=study.optimizer
def fresh(self,net,cfg,route):return original(net,cfg,'scratch')
study.optimizer=types.MethodType(fresh,study);runs=[]
for job in q['jobs']:
    runs.append(train(study,job,q['tier']));save(ROOT/(Path(a.queue).stem+'_results.json'),runs);gc.collect();torch.cuda.empty_cache()
save(ROOT/(Path(a.queue).stem+'_complete.json'),dict(runs=len(runs),difference='same trained base values and step0; base Adam moments and step reset, only diagnostic, not matched optimizer architecture comparison'))
