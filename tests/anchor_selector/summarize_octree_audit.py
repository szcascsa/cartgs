"""只汇总锁定协议结果，不选 seed/checkpoint，不据评价结果调参。"""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
r=Path(sys.argv[1]);v=r/"validation"
rows=json.loads((v/"quality_summary.json").read_text())
jobs=json.loads((v/"training_jobs.json").read_text())
assert len(rows)==36 and len(jobs)==4
plan=json.loads((v/"evaluation_plan.json").read_text())
def save(name,data):
    with (v/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
quality=[];views=[]
for row in rows:
    directory=Path(row["output"])
    item={k:row[k] for k in ("arm","seed","policy","level")}
    item.update(active_gaussians=row["stats"]["active_gaussians"],**row["metrics"])
    quality.append(item)
    per={m:np.loadtxt(directory/(m+".txt")).reshape(-1) for m in ("psnr","ssim","lpips")}
    assert all(len(a)==plan["frames"] for a in per.values())
    for i,frame in enumerate(json.loads((v/"common_frames.json").read_text())["gt_indices"]):
        views.append(dict(arm=row["arm"],seed=row["seed"],policy=row["policy"],level=row["level"],frame=frame,**{m:float(a[i]) for m,a in per.items()}))
save("quality_summary.csv",quality);save("quality_per_view.csv",views)
paired=[]
for seed in (2026,2027):
    for policy in ("cached_bernoulli","same_k"):
        for level in (["Full"] if policy=="cached_bernoulli" else [])+["LOD_1","LOD_5","LOD_10","LOD_15"]:
            arms={a:next(x for x in rows if x["arm"]==a and x["seed"]==seed and x["policy"]==policy and x["level"]==level) for a in ("fixed","adaptive")}
            for metric in ("psnr","ssim","lpips"):
                values={a:np.loadtxt(Path(x["output"])/(metric+".txt")) for a,x in arms.items()}
                delta=values["adaptive"]-values["fixed"]
                paired.append(dict(seed=seed,policy=policy,level=level,metric=metric,mean=float(delta.mean()),median=float(np.median(delta)),p10=float(np.quantile(delta,.1)),p90=float(np.quantile(delta,.9)),improved_view_fraction=float((delta*(1 if metric!="lpips" else -1)>0).mean())))
save("paired_statistics.csv",paired)
runtime=[];anchor=[]
for job in jobs:
    output=Path(job["output"]);name=output.name
    checkpoint=output/"10000_shutdown/ply"
    audit=json.loads((v/(name+"_checkpoint_audit.json")).read_text())
    mapping=np.load(v/(name+"_mapping.npz"))
    state=json.loads((checkpoint/"ultrafast_rotation_state.json").read_text())
    text=(output/"training.log").read_text()
    maintenance=[float(s) for s in re.findall(r"\[Anchor Octree\].*?refinement_ms=([0-9.e+-]+)",text)]
    split_count=sum(map(int,re.findall(r"\[Anchor Octree\] splits=(\d+)",text)))
    gpu=(output/"GpuPeakUsageMB.txt").read_text().strip()
    record=dict(arm=job["arm"],seed=job["seed"],wall_seconds=job["wall_seconds"],gpu_peak_file=gpu,
        full_renders=state["full_render_count"],lod_renders=state["lod_render_count"],online_end_mapping=state["online_end_mapping_iteration"],tail_steps=state["tail_training_steps"],
        native_forward_ms=audit["native_forward_ms_median"],python_forward_ms=audit["forward_ms_median"],feature_count=audit["feature_count"],feature_bytes=audit["feature_bytes"],
        feature_plus_one_adam_moments_bytes=audit["feature_plus_two_adam_moments_bytes"],layout_buffer_bytes=audit["layout_buffer_bytes"],forward_temporary_peak_bytes=audit["forward_temporary_peak_bytes"],
        refinement_calls=len(maintenance),split_count=split_count,refinement_host_total_ms=sum(maintenance),refinement_host_p95_ms=float(np.quantile(maintenance,.95)) if maintenance else 0)
    runtime.append(record)
    anchor.append(dict(arm=job["arm"],seed=job["seed"],population=audit["population"],features=audit["feature_count"],
        **{"nodes_L"+str(l):audit["node_count_by_level"][str(l)] for l in range(3)},
        **{"occupied_leaves_L"+str(l):audit["occupied_leaf_count_by_level"][str(l)] for l in range(3)},
        retained_empty_nodes=audit["retained_empty_nodes"],max_leaf_occupancy=audit["occupancy_quantiles"][-1],l2_over_capacity_cells=audit["l2_over_capacity_count"],l2_over_capacity_gaussians=audit["l2_over_capacity_gaussians"]))
    anchor[-1].update(internal_nodes=int(mapping["branches"].sum()),retained_empty_leaf_nodes=int(((mapping["occupancy"]==0)&(~mapping["branches"])).sum()))
save("runtime.csv",runtime);save("anchor_statistics.csv",anchor)
protocol=Path(sys.argv[2]) if len(sys.argv)>2 else r/"source/tests/anchor_selector/octree_protocol.json"
result=dict(protocol=json.loads(protocol.read_text()),plan=plan,quality=quality,paired=paired,runtime=runtime,anchors=anchor,
            correctness=json.loads((v/"correctness.json").read_text()))
(v/"audit_summary.json").write_text(json.dumps(result,indent=2))
print(json.dumps(dict(anchors=anchor,runtime=runtime),indent=2))
