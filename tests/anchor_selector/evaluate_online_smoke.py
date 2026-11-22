"""固定同一组时间均匀视图，仅验证现有 Full/LoD evaluator 接入，不选模型或参数。"""
import json
from pathlib import Path
import os
import subprocess
import sys
import numpy as np

repo, dataset = map(Path, sys.argv[1:3])
legacy = len(sys.argv) > 3 and sys.argv[3] == "legacy"
root = repo.parent; result = root/("online-fr1_xyz-legacy" if legacy else "online-fr1_xyz")
checkpoint = max(result.glob("*_shutdown"), key=lambda p:int(p.name.split("_")[0]))/"ply"
rgb = np.loadtxt(dataset/"rgb.txt", dtype=str, comments="#")[:,0].astype(float)
poses = np.loadtxt(result/"CameraTrajectory_TUM.txt")[:,0]
truth = np.loadtxt(dataset/"groundtruth.txt", comments="#")[:,0]
available = sorted(set(int(np.argmin(abs(rgb-t))) for t in poses if abs(truth-t).min() < .08))
indices = [available[i] for i in np.linspace(int(.1*len(available)),int(.9*len(available))-1,8,dtype=int)]
frames = root/"validation/evaluation_frames.json"
if not legacy:
    frames.write_text(json.dumps(dict(gt_indices=indices, policy="eight uniform temporal indices; no quality filtering"),indent=2))
else:
    assert frames.is_file(), "Legacy must use the already frozen Anchor evaluation views"
summary = []
arms = [("Full",None)] if legacy else [("Full",None),("LOD_1",.01),("LOD_5",.05),("LOD_10",.1),("LOD_15",.15)]
for arm, ratio in arms:
    output = root/("quality_legacy" if legacy else "quality_smoke")/arm
    output.mkdir(parents=True,exist_ok=True)
    # evo's installed user config requests TkAgg. Override only this process;
    # never modify the user's global evo config or production metric code.
    launcher = ("from evo.tools import settings; settings.SETTINGS.plot_backend='Agg'; "
                "import sys,runpy; sys.path.insert(0,"+repr(str(repo/"python"))+"); "
                "sys.argv=["+repr(str(repo/"python/run.py"))+"]+sys.argv[1:]; "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [sys.executable,"-c",launcher,result,dataset,"--output_path",output,
               "--evaluation-frame-set",frames,"--no-save-images"]
    if ratio is not None:
        command += ["--selector",checkpoint/"selector.pt","--selector-ratio",str(ratio),
                    "--selector-config",checkpoint/"selector_config.yaml"]
    with (output/"evaluation.log").open("w") as log:
        run = subprocess.run([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg"))
    if run.returncode:
        print(arm,"FAILED",run.returncode,flush=True); raise SystemExit(run.returncode)
    metrics = dict(line.strip().split(":",1) for line in (output/"eval.txt").read_text().splitlines())
    stats = json.loads((output/"selector_stats.json").read_text())
    summary.append(dict(arm=arm,ratio=ratio,metrics=metrics,stats=stats))
    (root/"validation"/("legacy_full_quality.json" if legacy else "quality_smoke.json")).write_text(json.dumps(summary,indent=2))
    print(arm,metrics,flush=True)
