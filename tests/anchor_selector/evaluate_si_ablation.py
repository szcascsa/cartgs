"""四个最终模型使用全部共同 GT 图像评估；不调参、不挑 checkpoint。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from plyfile import PlyData
from evo.core import sync
from evo.tools import file_interface

repo, dataset = map(Path, sys.argv[1:3])
root = repo.parent
jobs = json.loads((root/"validation/training_jobs.json").read_text())
assert len(jobs) == 4 and all(job["exit_code"] == 0 for job in jobs)
rgb = np.loadtxt(dataset/"rgb.txt", dtype=str, comments="#")[:, 0].astype(float)
populations, covered, sources = [], [], {}
for job in jobs:
    result = Path(job["output"])
    checkpoint = result/"10000_shutdown/ply"
    population = checkpoint/"point_cloud/iteration_10000/point_cloud.ply"
    populations.append(len(PlyData.read(population)["vertex"]))
    reference = file_interface.read_tum_trajectory_file(str(dataset/"groundtruth.txt"))
    estimated = file_interface.read_tum_trajectory_file(str(result/"CameraTrajectory_TUM.txt"))
    _, associated = sync.associate_trajectories(reference, estimated, max_diff=.08)
    ids = [int(np.argmin(abs(rgb-t))) for t in associated.timestamps if abs(rgb-t).min() < .08]
    assert len(ids) == len(set(ids))
    covered.append(set(ids))
    for path in (population, checkpoint/"selector.pt", checkpoint/"selector_config.yaml", checkpoint/"transform_field.pt"):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
frames = sorted(set.intersection(*covered))
assert len(frames) > 700, len(frames)
frame_path = root/"validation/common_frames.json"
frame_path.write_text(json.dumps(dict(gt_indices=frames, policy="all common associated GT indices; no quality filtering"), indent=2))
ratios = [.01, .05, .1, .15]
plan = dict(frames=len(frames), populations=populations, reference_population=min(populations),
            common_k={str(r): int(np.floor(min(populations)*r)) for r in ratios}, source_sha256=sources)
plan_path = root/"validation/evaluation_plan.json"
plan_path.write_text(json.dumps(plan, indent=2))
print("evaluation plan", len(frames), "frames; common K", plan["common_k"], flush=True)
summary = []
for job in jobs:
    result = Path(job["output"])
    checkpoint = result/"10000_shutdown/ply"
    for policy in ("cached_bernoulli", "same_k"):
        arms = [("Full", None)] if policy == "cached_bernoulli" else []
        arms += [("LOD_"+str(int(100*r)), r) for r in ratios]
        for arm, ratio in arms:
            output = root/"quality"/result.name/policy/arm
            output.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, repo/"tests/anchor_selector/si_evaluation_entry.py", result, dataset,
                       "--output_path", output, "--evaluation-frame-set", frame_path, "--no-save-images"]
            if ratio is not None:
                command += ["--selector", checkpoint/"selector.pt", "--selector-ratio", str(ratio),
                            "--selector-config", checkpoint/"selector_config.yaml"]
            with (output/"evaluation.log").open("w") as log:
                run = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg",
                             ANCHOR_SI_EVAL_POLICY=policy, ANCHOR_SI_EVAL_PLAN=str(plan_path)))
            assert run.returncode == 0, (result.name, policy, arm, run.returncode)
            assert json.loads((output/"evaluation_frames.json").read_text())["gt_indices"] == frames
            stats = json.loads((output/"selector_stats.json").read_text())
            metrics = {metric: float(np.loadtxt(output/(metric+".txt")).mean()) for metric in ("psnr", "ssim", "lpips")}
            summary.append(dict(arm=job["arm"], seed=job["seed"], policy=policy, ratio=ratio,
                                level=arm, output=str(output), stats=stats, metrics=metrics))
            (root/"validation/quality_summary.json").write_text(json.dumps(summary, indent=2))
            print(result.name, policy, arm, stats["active_gaussians"], metrics, flush=True)
for path, digest in sources.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
(root/"validation/evaluation.exit").write_text("0")
