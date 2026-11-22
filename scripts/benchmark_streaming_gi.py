"""Run repeated native TUM room benchmarks with unchanged training settings."""
import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path

PAIRS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
SOURCES = (
    "src/gaussian_mapper.cpp", "src/global_gi.cpp", "include/global_gi.h",
    "include/global_gi_bank.h", "src/gi_teacher.cpp", "include/gi_teacher.h",
    "include/global_gi_snapshot_cache.h", "include/mapping_profiler.h", "CMakeLists.txt",
    "include/ultrafast_training_timer.h", "src/gaussian_renderer.cpp",
    "cuda_rasterizer/rasterize_points.cu", "lib/libcuda_rasterizer.so",
    "third_party/ORB-SLAM3/src/KeyFrame.cc", "third_party/ORB-SLAM3/src/MapPoint.cc",
    "third_party/ORB-SLAM3/src/Map.cc", "third_party/ORB-SLAM3/src/Tracking.cc",
    "third_party/ORB-SLAM3/src/Atlas.cc", "third_party/ORB-SLAM3/src/GIAssociationState.cc",
    "third_party/ORB-SLAM3/include/GIAssociationState.h", "third_party/ORB-SLAM3/include/KeyFrame.h",
    "third_party/ORB-SLAM3/include/Map.h", "third_party/ORB-SLAM3/include/MapPoint.h",
    "third_party/ORB-SLAM3/include/Atlas.h", "third_party/ORB-SLAM3/CMakeLists.txt",
    "cfg/selector/fast.yaml", "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
    "cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_room.yaml", "bin/tum_rgbd",
    "lib/libgaussian_mapper.so", "third_party/ORB-SLAM3/lib/libORB_SLAM3.so",
)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def summarize(directory, wall_seconds, returncode):
    result = {"process_wall_seconds": wall_seconds, "returncode": returncode}
    for line in (directory / "run.log").read_text(errors="replace").splitlines():
        if "[GlobalGI]" in line or "[UltrafastRotation" in line or "[Mapping PERF FINAL]" in line:
            pairs = dict(PAIRS.findall(line))
            if pairs.get("event") in ("final", "save", "training_end") or "[Mapping PERF FINAL]" in line:
                for key, value in pairs.items():
                    try:
                        result[key] = float(value)
                    except ValueError:
                        pass
    keys = ("global_gi_bank_total_ms", "global_gi_incremental_total_ms",
            "global_gi_topology_cumulative_ms", "global_gi_full_refresh_total_ms")
    steps = result.get("ultrafast_total_normal_steps", result.get("online_end_rotation_step", result.get("iteration")))
    if steps and all(key in result for key in keys):
        result["gi_maintenance_ms_per_step"] = sum(result[key] for key in keys) / steps
        if "gi_teacher_host_ms_per_step" in result:
            result["gi_including_teacher_ms_per_step"] = (result["gi_maintenance_ms_per_step"]
                + result["gi_teacher_host_ms_per_step"] + result.get("gi_loss_host_ms_per_step", 0))
    tracking = list((directory / "model").rglob("TrackingTime.txt"))
    if tracking:
        values = [1000.0 * float(line.split()[0]) for line in tracking[0].read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
        result.update(tracking_frames=len(values), tracking_mean_ms=statistics.mean(values),
                      tracking_p95_ms=percentile(values, 0.95))
    gpu = []
    for line in (directory / "gpu.csv").read_text().splitlines():
        fields = line.split(",")
        if len(fields) == 3:
            try:
                gpu.append(float(fields[1]))
            except ValueError:
                pass
    if gpu:
        result.update(gpu_mean_percent=statistics.mean(gpu), gpu_p95_percent=percentile(gpu, 0.95))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    project = args.project.resolve()
    batch = args.root / args.label
    batch.mkdir(parents=True, exist_ok=False)
    hashes = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
              for name in SOURCES if (project / name).is_file()}
    manifest = {"head": args.head, "hashes": hashes, "dataset": "/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_room"}
    associations = project / "cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_room.txt"
    manifest["input_frames"] = sum(bool(line.strip()) and not line.lstrip().startswith("#")
                                  for line in associations.read_text().splitlines())
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for index in range(args.runs):
        directory = batch / f"run_{index + 1}"
        directory.mkdir()
        command = [str(project / "bin/tum_rgbd"),
                   str(project / "third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt"),
                   "./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_room.yaml",
                   "./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml", manifest["dataset"],
                   "./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_room.txt",
                   str(directory / "model"), "no_viewer", "--selector-config", "./cfg/selector/fast.yaml"]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (directory / "gpu.csv").open("w") as gpu, (directory / "run.log").open("w") as log:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits", "-l", "1"], stdout=gpu)
            try:
                start = time.monotonic()
                run = subprocess.run(["/usr/bin/time", "-v", "-o", str(directory / "time.txt")] + command,
                                     cwd=project, stdout=log, stderr=subprocess.STDOUT)
                elapsed = time.monotonic() - start
            finally:
                monitor.terminate()
                monitor.wait()
        (directory / "exit_code").write_text(str(run.returncode) + "\n")
        result = summarize(directory, elapsed, run.returncode)
        results.append(result)
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (batch / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"label": args.label, "run": index + 1, "result": result}), flush=True)


if __name__ == "__main__":
    main()
