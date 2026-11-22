"""逐节点定位编译布局变化导致的差异，不把小误差直接视为通过。"""
from common import *
import gc
torch.set_num_threads(1);torch.set_num_interop_threads(1)
with torch.no_grad():
    source=load_scene('Replica__office2__rgbd');cam=camera(source['records'][len(source['records'])//2]);rows=[]
    for variant in ['baseline','control','histogram','six','combined']:
        engine=native(source,variant,diagnose=True);packet,ids,info=request(engine,cam,'local0.3');diag=engine.native.diagnostics()
        data={name:value.clone() for name,value in packet['inputs'].items() if isinstance(value,torch.Tensor)}
        data.update(scores=diag[2].clone(),residual=diag[3].clone(),ids=ids.clone())
        if variant=='baseline':truth=data
        for name,value in data.items():
            expected=truth[name];assert value.shape==expected.shape
            error=(value-expected).abs();record=dict(variant=variant,node=name,same=bool(torch.equal(value,expected)),different=int((value!=expected).sum()),max_abs=float(error.max()) if value.numel() else 0.)
            print(record,flush=True);rows.append(record)
        del engine;gc.collect()
    write_csv(HERE/'reports/PROBE_NODES.csv',rows)
