"""与旧27维/root-offset结果对照；必要时只重评旧模型的统一K。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
r=Path(sys.argv[1]);repo=r/"source";v=r/"validation"
protocol=json.loads((repo/"tests/anchor_selector/xyz_local_protocol.json").read_text())
old=Path(protocol["historical_reference"])
plan=json.loads((v/"evaluation_plan.json").read_text())
old_plan=json.loads((old/"validation/evaluation_plan.json").read_text())
frames=json.loads((v/"common_frames.json").read_text())["gt_indices"]
old_frames=json.loads((old/"validation/common_frames.json").read_text())["gt_indices"]
old_index={frame:i for i,frame in enumerate(old_frames)}
indices=np.array([old_index[f] for f in frames]);assert len(indices)==len(frames)
old_rows=json.loads((old/"validation/quality_summary.json").read_text())
new_rows=json.loads((v/"quality_summary.json").read_text());historical=[];hashes={}
for row in old_rows:
    directory=Path(row["output"]);new=dict(row)
    if row["policy"]=="same_k" and plan["common_k"]!=old_plan["common_k"]:
        result=old/(row["arm"]+"_seed"+str(row["seed"]));checkpoint=result/"10000_shutdown/ply"
        directory=r/"historical_same_k"/result.name/row["level"];directory.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,repo/"tests/anchor_selector/si_evaluation_entry.py",result,
            "/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz","--output_path",directory,
            "--evaluation-frame-set",v/"common_frames.json","--no-save-images","--selector",checkpoint/"selector.pt",
            "--selector-ratio",str(row["ratio"]),"--selector-config",checkpoint/"selector_config.yaml"]
        for p in [checkpoint/"selector.pt",checkpoint/"transform_field.pt",checkpoint/"point_cloud/iteration_10000/point_cloud.ply"]:
            hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        with (directory/"evaluation.log").open("w") as f:
            subprocess.run(list(map(str,command)),env=dict(os.environ,OMP_NUM_THREADS="2",ANCHOR_SI_EVAL_POLICY="same_k",ANCHOR_SI_EVAL_PLAN=str(v/"evaluation_plan.json")),stdout=f,stderr=subprocess.STDOUT,check=True)
        new["stats"]=json.loads((directory/"selector_stats.json").read_text())
        new["view_indices"]=list(range(len(frames)))
    else:new["view_indices"]=indices.tolist()
    new["output"]=str(directory)
    new["metrics"]={m:float(np.loadtxt(directory/(m+".txt"))[new["view_indices"]].mean()) for m in ("psnr","ssim","lpips")}
    historical.append(new)
    (v/"historical_quality_summary.json").write_text(json.dumps(historical,indent=2))
    print("historical",row["arm"],row["seed"],row["policy"],row["level"],flush=True)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
comparisons=[]
for row in new_rows:
    for old_arm in (["adaptive","fixed"] if row["arm"]=="adaptive" else ["fixed"]):
        prior=next(x for x in historical if x["arm"]==old_arm and x["seed"]==row["seed"] and x["policy"]==row["policy"] and x["level"]==row["level"])
        for metric in ("psnr","ssim","lpips"):
            a=np.loadtxt(Path(row["output"])/(metric+".txt"));b=np.loadtxt(Path(prior["output"])/(metric+".txt"))[prior["view_indices"]]
            delta=a-b
            comparisons.append(dict(comparison="30_"+row["arm"]+" - 27_"+old_arm,seed=row["seed"],policy=row["policy"],level=row["level"],metric=metric,
                new_count=row["stats"]["active_gaussians"],old_count=prior["stats"]["active_gaussians"],mean=float(delta.mean()),median=float(np.median(delta)),p10=float(np.quantile(delta,.1)),p90=float(np.quantile(delta,.9)),improved_view_fraction=float((delta*(1 if metric!="lpips" else -1)>0).mean())))
with (v/"comparison_with_previous.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
(v/"historical_comparison.exit").write_text("0")
