"""一次无SI属性输入训练；复用既有792帧和baseline，不挑checkpoint。"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

repo, dataset = map(Path, sys.argv[1:3])
root = repo.parent
protocol = json.loads((repo/"tests/anchor_selector/attribute_protocol.json").read_text())
reference = Path(protocol["reference"])
env = dict(os.environ, OMP_NUM_THREADS="2", QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg",
           CARTGS_AUDIT_SEED=str(protocol["seed"]))
output = root/"attribute_no_si_seed2026"
output.mkdir(exist_ok=False)
base = reference/"runs/no_si_seed2026/input_selector.yaml"
config = base.read_text()
assert "anchor_gaussian_attributes:" not in config
config += "\nSelector.anchor_gaussian_attributes: 1\n"
(output/"input_selector.yaml").write_text(config)
command = [repo/"bin/tum_rgbd", repo/"third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt",
           repo/"cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml",
           repo/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml", dataset,
           repo/"cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt", output, "no_viewer",
           "--selector-config", output/"input_selector.yaml"]
start = time.monotonic()
with (output/"training.log").open("w") as log:
    run = subprocess.run(list(map(str, command)), cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
job = dict(seed=protocol["seed"], exit_code=run.returncode, wall_seconds=time.monotonic()-start,
           output=str(output), command=list(map(str, command)))
(root/"validation/training_job.json").write_text(json.dumps(job, indent=2))
assert run.returncode == 0, job
log = (output/"training.log").read_text()
assert "gaussian_attributes=1 input_dim=27" in log and "SI/GI=disabled gi_loss_weight=0" in log
assert "10000 iterations completed." in log
assert not any(float(value) != 0 for value in re.findall(r"gi_supervised_count=([0-9.e+-]+)", log))
print("training completed", job["wall_seconds"], flush=True)
checkpoint = output/"10000_shutdown/ply"
cloud = checkpoint/"point_cloud/iteration_10000"
with (root/"validation/checkpoint_native.log").open("w") as stream:
    subprocess.run([str(repo/"bin/anchor_checkpoint_probe"), str(checkpoint/"selector.pt"),
        str(cloud/"point_cloud.ply"), str(cloud/"selector_metadata.ply"),
        str(root/"validation/checkpoint_probe.pt"), "attributes"], env=env, stdout=stream,
        stderr=subprocess.STDOUT, check=True)
sys.path.insert(0, str(repo/"python"))
import torch
from gaussian_model import GaussianModel
from selector import load_selector, load_selector_metadata, build_selector_mask
from elastic_budget import DEFAULTS
model = load_selector(str(checkpoint/"selector.pt"), "cuda")
gaussians = GaussianModel(3); gaussians.load_ply(str(cloud/"point_cloud.ply"))
metadata = load_selector_metadata(str(cloud/"selector_metadata.ply"), "cuda")
probe = torch.jit.load(str(root/"validation/checkpoint_probe.pt"), map_location="cuda").state_dict()
assert model.gaussian_attributes and int(model.anchor_layout_version) == 3
assert (metadata["maturity"] == 1).all() and (metadata["drift_ema"] == 0).all()
assert all(torch.isfinite(value).all() for value in model.state_dict().values())
parity = []
for i, ratio in enumerate((.01,.05,.1,.15)):
    with torch.no_grad():
        p = model(gaussians.get_xyz, None, gaussians.get_scaling, torch.tensor([[ratio]],device="cuda"),
                  sh_dc=gaussians.get_features[:,:1],opacity=gaussians.get_opacity)
        error = float((p-probe["probability_"+str(i)]).abs().max())
        mask, stats = build_selector_mask(model,gaussians,metadata,ratio,DEFAULTS,protection_enabled=False)
        assert error < 1e-6 and torch.equal(mask.bool(),probe["mask_"+str(i)]), (ratio,error)
        parity.append(dict(ratio=ratio,max_probability_error=error,mask_exact=True,active=int(mask.sum())))
(root/"validation/checkpoint_parity.json").write_text(json.dumps(parity,indent=2))
del model, gaussians, metadata, probe
torch.cuda.empty_cache()
frames = reference/"validation/common_frames.json"
assert len(json.loads(frames.read_text())["gt_indices"]) == 792
source_files = [checkpoint/"selector.pt", checkpoint/"transform_field.pt", cloud/"point_cloud.ply", cloud/"selector_metadata.ply"]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
summary = []
for policy in ("cached_bernoulli", "same_k"):
    arms = [("Full", None)] if policy == "cached_bernoulli" else []
    arms += [("LOD_"+str(int(100*r)),r) for r in (.01,.05,.1,.15)]
    for arm, ratio in arms:
        directory = root/"quality"/policy/arm
        directory.mkdir(parents=True,exist_ok=False)
        command = [sys.executable,repo/"tests/anchor_selector/si_evaluation_entry.py",output,dataset,
            "--output_path",directory,"--evaluation-frame-set",frames,"--no-save-images"]
        if ratio is not None:
            command += ["--selector",checkpoint/"selector.pt","--selector-ratio",str(ratio),
                        "--selector-config",checkpoint/"selector_config.yaml"]
        with (directory/"evaluation.log").open("w") as stream:
            subprocess.run(list(map(str,command)),env=dict(env,ANCHOR_SI_EVAL_POLICY=policy,
                ANCHOR_SI_EVAL_PLAN=str(reference/"validation/evaluation_plan.json")),stdout=stream,
                stderr=subprocess.STDOUT,check=True)
        assert json.loads((directory/"evaluation_frames.json").read_text())["gt_indices"] == json.loads(frames.read_text())["gt_indices"]
        stats = json.loads((directory/"selector_stats.json").read_text())
        metrics = {m:float(np.loadtxt(directory/(m+".txt")).mean()) for m in ("psnr","ssim","lpips")}
        summary.append(dict(policy=policy,level=arm,ratio=ratio,stats=stats,metrics=metrics,output=str(directory)))
        (root/"validation/quality_summary.json").write_text(json.dumps(summary,indent=2))
        print(policy,arm,stats["active_gaussians"],metrics,flush=True)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
(root/"validation/model_sha256.json").write_text(json.dumps(hashes,indent=2))
(root/"validation/audit.exit").write_text("0")
