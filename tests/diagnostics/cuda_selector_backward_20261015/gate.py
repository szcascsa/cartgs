"""复用 direct parity oracle；只增加候选 dispatch、输出目录和阶段选择。"""
import hashlib
import json
import os
from pathlib import Path
import sys

D = Path(__file__).resolve().parent
root = D.parents[2]
oracle = root/'tests/diagnostics/cuda_selector_direct_parity_20261015'
strategy = int(sys.argv[1])
mode = sys.argv[2]
assert mode in ['isolated', 'training', 'all']
tag = os.environ.get('SELECTOR_DIAGNOSTIC_TAG', '')
out = D/f'candidate_{strategy}_{mode}{"_"+tag if tag else ""}'
out.mkdir(exist_ok=False)
binding = (oracle/'bindings.cpp').read_text()
binding = binding.replace('selector_cuda::Candidate::CublasStrict);',
    f'selector_cuda::Candidate::CublasStrict,static_cast<selector_cuda::BackwardStrategy>({strategy}));',1)
(out/'bindings.cpp').write_text(binding)
(out/'checkpoints.json').write_bytes((oracle/'checkpoints.json').read_bytes())
s = (oracle/'direct.py').read_text()
s = s.replace("root = dest.parents[2]", f'root = Path({str(root)!r})')
s = s.replace("name='cartgs_selector_direct_parity'", f"name='cartgs_selector_backward_gate_{strategy}'")
if mode == 'isolated':
    s = s.replace("record['training']=train(base,raw,metadata,config,entry,tau)", "record['training']=None")
    s = s.replace("if checkpoint_name=='room_current':optimizer_test(base)", 'if False:optimizer_test(base)')
elif mode == 'training':
    s = s.replace('        parity(base,raw,metadata,config,entry,tau)', '        # isolated gate already completed')
    s = s.replace("if checkpoint_name=='room_current':optimizer_test(base)", 'if False:optimizer_test(base)')
(out/'executed.py').write_text(s)
(out/'provenance.json').write_text(json.dumps(dict(strategy=strategy,mode=mode,
    oracle_sha256=hashlib.sha256((oracle/'direct.py').read_bytes()).hexdigest(),
    adaptation='只改变 backward strategy、独立输出目录、isolated/training 调度；计算/容差不变'),ensure_ascii=False,indent=2)+'\n')
os.execv(sys.executable,[sys.executable,str(out/'executed.py')])
