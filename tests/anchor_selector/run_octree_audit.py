"""固定协议四次在线训练，先锁定共同视图/K，再进行质量评价。"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from plyfile import PlyData
from evo.core import sync
from evo.tools import file_interface

repo, dataset = map(Path, sys.argv[1:3])
root=repo.parent
protocol_path=Path(sys.argv[3]) if len(sys.argv)>3 else repo/"tests/anchor_selector/octree_protocol.json"
protocol=json.loads(protocol_path.read_text())
global_xyz=protocol.get("global_xyz",False)
env=dict(os.environ,OMP_NUM_THREADS="2",QT_QPA_PLATFORM="offscreen",MPLBACKEND="Agg")
base=Path(protocol["base_config"]).read_text()
assert "Selector.anchor_gaussian_attributes: 1" in base
assert "GlobalGI.mode: \"disabled\"" in base or "GlobalGI.mode: disabled" in base
assert "anchor_adaptive" not in base
if global_xyz:base+="\nSelector.anchor_global_xyz: 1\n"
jobs=[]
for seed in protocol["seeds"]:
    for arm in protocol["arms"]:
        output=root/(arm+"_seed"+str(seed));output.mkdir(exist_ok=False)
        config=output/"input_selector.yaml"
        config.write_text(base+"\nSelector.anchor_adaptive: "+str(int(arm=="adaptive"))+"\n")
        command=[repo/"bin/tum_rgbd",repo/"third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt",
                 repo/"cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml",
                 repo/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",dataset,
                 repo/"cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt",output,"no_viewer","--selector-config",config]
        start=time.monotonic()
        with (output/"training.log").open("w") as log:
            run=subprocess.run(list(map(str,command)),cwd=repo,env=dict(env,CARTGS_AUDIT_SEED=str(seed)),stdout=log,stderr=subprocess.STDOUT)
        job=dict(seed=seed,arm=arm,output=str(output),exit_code=run.returncode,wall_seconds=time.monotonic()-start)
        jobs.append(job);(root/"validation/training_jobs.json").write_text(json.dumps(jobs,indent=2))
        assert run.returncode==0,job
        text=(output/"training.log").read_text()
        assert ("gaussian_attributes=1 input_dim="+str(30 if global_xyz else 27)) in text and "SI/GI=disabled gi_loss_weight=0" in text
        assert "10000 iterations completed." in text
        assert not any(float(v)!=0 for v in re.findall(r"gi_supervised_count=([0-9.e+-]+)",text))
        print("trained",job,flush=True)

rgb=np.loadtxt(dataset/"rgb.txt",dtype=str,comments="#")[:,0].astype(float)
covered=[];populations=[];hashes={}
for job in jobs:
    output=Path(job["output"]);checkpoint=output/"10000_shutdown/ply"
    cloud=checkpoint/"point_cloud/iteration_10000"
    populations.append(len(PlyData.read(cloud/"point_cloud.ply")["vertex"]))
    gt=file_interface.read_tum_trajectory_file(str(dataset/"groundtruth.txt"))
    estimated=file_interface.read_tum_trajectory_file(str(output/"CameraTrajectory_TUM.txt"))
    _,assoc=sync.associate_trajectories(gt,estimated,max_diff=.08)
    ids=[int(np.argmin(abs(rgb-t))) for t in assoc.timestamps if abs(rgb-t).min()<.08]
    assert len(ids)==len(set(ids));covered.append(set(ids))
    for p in [checkpoint/"selector.pt",checkpoint/"transform_field.pt",cloud/"point_cloud.ply",cloud/"selector_metadata.ply"]:
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
frames=sorted(set.intersection(*covered));assert len(frames)>700
historical=protocol.get("historical_reference")
reference_populations=[]
if historical:
    prior=Path(historical)/"validation"
    frames=sorted(set(frames)&set(json.loads((prior/"common_frames.json").read_text())["gt_indices"]))
    reference_populations=json.loads((prior/"evaluation_plan.json").read_text())["populations"]
    assert len(frames)>700
frame_path=root/"validation/common_frames.json"
frame_path.write_text(json.dumps(dict(gt_indices=frames,policy="all common associated GT views, no quality filtering"),indent=2))
plan=dict(common_k={str(r):int(np.floor(min(populations+reference_populations)*r)) for r in protocol["ratios"]},populations=populations,historical_populations=reference_populations,frames=len(frames),model_sha256=hashes)
plan_path=root/"validation/evaluation_plan.json";plan_path.write_text(json.dumps(plan,indent=2))
print("LOCKED EVALUATION PLAN",plan["common_k"],len(frames),flush=True)
summary=[]
for job in jobs:
    output=Path(job["output"]);checkpoint=output/"10000_shutdown/ply"
    with (root/"validation"/(output.name+"_native.log")).open("w") as log:
        subprocess.run([str(repo/"bin/anchor_checkpoint_probe"),str(checkpoint/"selector.pt"),
            str(checkpoint/"point_cloud/iteration_10000/point_cloud.ply"),str(checkpoint/"point_cloud/iteration_10000/selector_metadata.ply"),
            str(root/"validation"/(output.name+"_probe.pt")),"global_xyz" if global_xyz else "attributes"],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(repo/"tests/anchor_selector/octree_checkpoint_audit.py"),str(root),output.name],env=env,check=True)
    for policy in ("cached_bernoulli","same_k"):
        arms=([("Full",None)] if policy=="cached_bernoulli" else [])+[("LOD_"+str(round(r*100)),r) for r in protocol["ratios"]]
        for level,ratio in arms:
            directory=root/"quality"/output.name/policy/level;directory.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,repo/"tests/anchor_selector/si_evaluation_entry.py",output,dataset,"--output_path",directory,
                     "--evaluation-frame-set",frame_path,"--no-save-images"]
            if ratio is not None:command += ["--selector",checkpoint/"selector.pt","--selector-ratio",str(ratio),"--selector-config",checkpoint/"selector_config.yaml"]
            with (directory/"evaluation.log").open("w") as log:
                subprocess.run(list(map(str,command)),env=dict(env,ANCHOR_SI_EVAL_POLICY=policy,ANCHOR_SI_EVAL_PLAN=str(plan_path)),stdout=log,stderr=subprocess.STDOUT,check=True)
            assert json.loads((directory/"evaluation_frames.json").read_text())["gt_indices"]==frames
            stats=json.loads((directory/"selector_stats.json").read_text())
            metrics={m:float(np.loadtxt(directory/(m+".txt")).mean()) for m in ("psnr","ssim","lpips")}
            summary.append(dict(arm=job["arm"],seed=job["seed"],policy=policy,level=level,ratio=ratio,output=str(directory),stats=stats,metrics=metrics))
            (root/"validation/quality_summary.json").write_text(json.dumps(summary,indent=2))
            print(job["arm"],job["seed"],policy,level,stats["active_gaussians"],metrics,flush=True)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
(root/"validation/audit.exit").write_text("0")
