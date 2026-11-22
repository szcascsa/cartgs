"""完整 shutdown 模型评估：Full 与四个 LoD ratio；不训练、不抽帧。"""
import json
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

diagnostics = Path(__file__).resolve().parent
root = diagnostics.parents[2]
python = "/home/prmi/anaconda3/envs/zwj/bin/python"
parser = argparse.ArgumentParser()
parser.add_argument("--protocol", choices=("original", "desk_start1000"), default="original")
args = parser.parse_args()
prefix = "guard_off_" if args.protocol == "original" else "guard_off_desk_start1000_"
assert (diagnostics / (prefix + "online.exit")).read_text().strip() == "0"
mapping = json.loads((diagnostics / (prefix + "online_runs.json")).read_text())
environment = os.environ.copy()
environment["LD_LIBRARY_PATH"] = str(root / "third_party/libtorch-cu118/libtorch/lib") + ":/usr/local/cuda-11.8/lib64:" + environment.get("LD_LIBRARY_PATH", "")
manifest = {"protocol": "all matched trajectory frames; no training; per-image metrics in batches of 8", "runs": []}

def save():
    (diagnostics / (prefix + "online_evaluations.json")).write_text(json.dumps(manifest, indent=2))

try:
    for run in mapping["runs"]:
        source = Path(run["output"])
        checkpoints = list(source.glob("*_shutdown"))
        assert len(checkpoints) == 1, checkpoints
        checkpoint = checkpoints[0]
        for ratio in (None, .01, .05, .10, .15):
            label = "full" if ratio is None else f"lod_{int(ratio*100):02d}"
            output = source / "evaluation" / label
            assert not output.exists(), f"Refuse to overwrite: {output}"
            output.mkdir(parents=True)
            command = [python, str(diagnostics / "evaluate_headless_batched.py"),
                       str(source), run["dataset"], "--output_path", str(output),
                       "--shutdown-model", checkpoint.name]
            if ratio is not None:
                command += ["--selector", str(checkpoint / "ply/selector.pt"),
                            "--selector-ratio", str(ratio), "--selector-config",
                            str(source / "selector_config.yaml")]
            item = dict(scene=run["scene"], backend=run["backend"], ratio=ratio,
                        label=label, output=str(output), command=command,
                        status="running", start=time.time())
            manifest["runs"].append(item)
            save()
            with (output / "evaluation.log").open("w") as log:
                result = subprocess.run(command, cwd=root, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            item.update(exit_code=result.returncode, wall_seconds=time.time()-item["start"])
            assert result.returncode == 0, item
            metrics = {}
            counts = []
            for metric in ("psnr", "ssim", "lpips"):
                values = np.atleast_1d(np.loadtxt(output / (metric + ".txt")))
                assert values.size and np.isfinite(values).all(), (item, metric)
                metrics[metric] = float(values.mean())
                counts.append(values.size)
            assert len(set(counts)) == 1, counts
            assert (output / "metrics_traj.txt").is_file()
            item.update(status="complete", evaluated_frames=counts[0], metrics=metrics)
            stats = output / "selector_stats.json"
            if stats.is_file():
                item["selector_stats"] = json.loads(stats.read_text())
            # 仅删除本次评估生成的图像；模型与逐帧指标全部保留。
            for name in ("image", "gt"):
                images = output / name
                if images.is_dir():
                    shutil.rmtree(images)
            save()
            print(run["scene"], run["backend"], label, metrics, flush=True)
    (diagnostics / (prefix + "evaluation.exit")).write_text("0\n")
except BaseException:
    (diagnostics / (prefix + "evaluation.exit")).write_text("1\n")
    raise
