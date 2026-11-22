"""整理原始在线日志与指标，显式保留采样范围和缺失项。"""
import csv
import argparse
import json
import re
from pathlib import Path

diagnostics = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--protocol", choices=("original", "desk_start1000"), default="original")
args = parser.parse_args()
prefix = "guard_off_" if args.protocol == "original" else "guard_off_desk_start1000_"
manifest = json.loads((diagnostics / (prefix + "online_runs.json")).read_text())
memory = list(csv.DictReader((diagnostics / (prefix + "online_memory_samples.csv")).open()))
evaluation = diagnostics / (prefix + "online_evaluations.json")
evaluations = json.loads(evaluation.read_text())["runs"] if evaluation.exists() else []
summary = []
for run in manifest["runs"]:
    directory = Path(run["output"])
    log = (directory / "mapping.log").read_text(errors="replace")
    reports = re.findall(r"^\[Mapping PERF FINAL\] (.*)$", log, re.M)
    counters = re.findall(r"^\[CUDA TF ONLINE\] (.*)$", log, re.M)
    if not reports or not counters:
        continue
    item = {key: run[key] for key in ("scene", "backend", "frames", "wall_seconds", "output")}
    item["mapping_profile"] = dict(re.findall(r"(\w+)=([^ ]+)", reports[-1]))
    item["counters"] = dict(re.findall(r"(\w+)=([^ ]+)", counters[-1]))
    compact = re.findall(r"^\[LOD Compact\] (.*)$", log, re.M)
    item["periodic_lod_observations"] = [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in compact]
    item["periodic_lod_note"] = "every 1000 iterations; not a per-step transformed-N mean"
    states = list(directory.glob("*_shutdown/ply/transform_state.json"))
    assert len(states) == 1, states
    item["transform_state"] = json.loads(states[0].read_text())
    assert item["transform_state"]["tail_completed_steps"] == 0
    assert item["transform_state"]["post_slam_mappings"] == 0
    samples = [row for row in memory if row["scene"] == run["scene"] and row["backend"] == run["backend"]]
    item["gpu_memory"] = dict(method="nvidia-smi process memory, 1s sampling; includes SLAM",
                              samples=len(samples),
                              observed_peak_mib=max((int(row["used_gpu_memory_mib"]) for row in samples), default=None),
                              first_timestamp=samples[0]["timestamp"] if samples else None,
                              last_timestamp=samples[-1]["timestamp"] if samples else None)
    item["quality"] = [entry for entry in evaluations if entry["scene"] == run["scene"] and entry["backend"] == run["backend"]]
    trajectory = directory / "evaluation/full/metrics_traj.txt"
    if trajectory.exists():
        text = trajectory.read_text()
        rmse = re.findall(r"^\s*rmse\s+([0-9.eE+-]+)", text, re.M)
        assert len(rmse) == 2, text
        item["trajectory"] = dict(translation_ape_rmse_m=float(rmse[0]),
                                  rotation_ape_rmse_unitless=float(rmse[1]),
                                  alignment="SE(3) Umeyama; no scale alignment")
    summary.append(item)
(diagnostics / (prefix + "online_summary.json")).write_text(json.dumps(summary, indent=2))
print(json.dumps([{key: item[key] for key in ("scene", "backend", "counters", "mapping_profile", "gpu_memory")} for item in summary], indent=2))
