"""使用冻结 Legacy-SI checkpoint 和既有共同视图补评四档 LoD，不训练。"""
import json
import os
from pathlib import Path
import subprocess
import sys

repo, dataset = map(Path, sys.argv[1:3])
root = repo.parent
result = root/"online-fr1_xyz-legacy"
checkpoint = result/"5874_shutdown/ply"
frames = root/"validation/evaluation_frames.json"
assert frames.is_file(), "必须复用已冻结的共同评价视图"
summary = []
for arm, ratio in (("LOD_1", .01), ("LOD_5", .05), ("LOD_10", .1), ("LOD_15", .15)):
    output = root/"quality_legacy"/arm
    output.mkdir(parents=True, exist_ok=True)
    launcher = ("from evo.tools import settings; settings.SETTINGS.plot_backend='Agg'; "
                "import sys,runpy; sys.path.insert(0,"+repr(str(repo/"python"))+"); "
                "sys.argv=["+repr(str(repo/"python/run.py"))+"]+sys.argv[1:]; "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [sys.executable, "-c", launcher, result, dataset, "--output_path", output,
               "--evaluation-frame-set", frames, "--no-save-images",
               "--selector", checkpoint/"selector.pt", "--selector-ratio", str(ratio),
               "--selector-config", checkpoint/"selector_config.yaml"]
    with (output/"evaluation.log").open("w") as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg"), check=True)
    assert json.loads((output/"evaluation_frames.json").read_text())["gt_indices"] == json.loads(frames.read_text())["gt_indices"]
    metrics = dict(line.strip().split(":", 1) for line in (output/"eval.txt").read_text().splitlines())
    stats = json.loads((output/"selector_stats.json").read_text())
    summary.append(dict(arm=arm, ratio=ratio, metrics=metrics, stats=stats))
    (root/"validation/legacy_lod_quality.json").write_text(json.dumps(summary, indent=2))
    print(arm, metrics, stats["active_gaussians"], flush=True)
