"""质量评价全部完成、GPU 空闲后运行；匹配 N 的独立 Selector forward 计时。"""
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/"python"))
from selector import load_selector
from gaussian_model import GaussianModel
r=Path(sys.argv[1]);v=r/"validation"
assert (v/"audit.exit").read_text()=="0"
plan=json.loads((v/"evaluation_plan.json").read_text());n=min(plan["populations"]+plan.get("historical_populations",[]))
sources=[r]
if plan.get("historical_populations"):
    protocol=json.loads((repo/"tests/anchor_selector/xyz_local_protocol.json").read_text())
    sources.append(Path(protocol["historical_reference"]))
results=[]
with torch.no_grad():
    # Alternate the first arm across seeds, fixed order independent of metrics.
    for source,seed,arms in [(source,seed,arms) for source in sources for seed,arms in [(2026,("fixed","adaptive")),(2027,("adaptive","fixed"))]]:
        for arm in arms:
            name=arm+"_seed"+str(seed);p=source/name/"10000_shutdown/ply"
            model=load_selector(str(p/"selector.pt"),"cuda");g=GaussianModel(3)
            g.load_ply(str(p/"point_cloud/iteration_10000/point_cloud.ply"))
            ids=torch.arange(n,device="cuda")
            xyz=g.get_xyz[:n];scale=g.get_scaling[:n];dc=g.get_features[:n,:1];opacity=g.get_opacity[:n]
            ratio=torch.tensor([[.1]],device="cuda")
            def forward():return model(xyz,None,scale,ratio,source_rows=ids,sh_dc=dc,opacity=opacity)
            times=[];peaks=[]
            for repeat in range(7):
                for _ in range(50):forward()
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(500):forward()
                end.record();end.synchronize()
                times.append(start.elapsed_time(end)/500);peaks.append(torch.cuda.max_memory_allocated()-base)
            results.append(dict(arm=arm,seed=seed,source=str(source),input_width=model.local_emd[0].weight.shape[1],matched_n=n,ratio=.1,ms_repeats=times,median_ms=float(np.median(times)),p10_ms=float(np.quantile(times,.1)),p90_ms=float(np.quantile(times,.9)),temporary_peak_bytes=max(peaks),feature_count=len(model.anchors.features)))
            del model,g,ids,xyz,scale,dc,opacity,ratio
            gc.collect();torch.cuda.empty_cache()
(v/"matched_n_forward.json").write_text(json.dumps(dict(protocol="no concurrent evaluation/training; preactivated attributes; source rows; 7x500 calls after 50 warmups; CUDA events; alternating paired order",results=results),indent=2))
print(json.dumps(results,indent=2))
