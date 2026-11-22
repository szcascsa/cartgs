"""仅替换已安装Torch helper的一行预热Stream；不修改site-packages或Autograd定义。"""
import hashlib
import inspect
import json
from pathlib import Path
import torch
import torch.cuda.graphs


def make_shared(module,samples,stream):
    source=inspect.getsource(torch.cuda.make_graphed_callables)
    old='with torch.cuda.stream(torch.cuda.Stream()):'
    assert source.count(old)==1,'Installed Torch helper changed; use verified reference helper'
    replacement=source.replace(old,'with torch.cuda.stream(_spatial_warmup_stream):')
    root=Path(__file__).parent
    target=root/'torch_graph_helper_shared_upstream.py';target.write_text(replacement)
    (root/'shared_helper_provenance.json').write_text(json.dumps(dict(torch=torch.__version__,
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),modified_sha256=hashlib.sha256(replacement.encode()).hexdigest(),
        change='one line: reuse caller-owned warmup stream; unchanged three warmups, forward/backward capture and autograd'),indent=2))
    namespace=dict(torch.cuda.graphs.__dict__);namespace['_spatial_warmup_stream']=stream
    exec(compile(replacement,str(target),'exec'),namespace)
    return namespace['make_graphed_callables'](module,samples,num_warmup_iters=3)
