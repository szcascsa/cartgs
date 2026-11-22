"""采样在线进程的 nvidia-smi memory；不是 Torch allocator peak。"""
import csv
import argparse
import json
import subprocess
import time
from pathlib import Path

diagnostics = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--protocol", choices=("original", "desk_start1000"), default="original")
args = parser.parse_args()
prefix = "guard_off_" if args.protocol == "original" else "guard_off_desk_start1000_"
target = diagnostics / (prefix + "online_memory_samples.csv")
new = not target.exists()
with target.open("a", newline="") as stream:
    out = csv.writer(stream)
    if new:
        out.writerow(["timestamp", "scene", "backend", "pid", "used_gpu_memory_mib"])
    while not (diagnostics / (prefix + "online.exit")).exists():
        source = diagnostics / (prefix + "online_runs.json")
        if not source.exists():
            time.sleep(.1)
            continue
        runs = json.loads(source.read_text())["runs"]
        current = runs[-1]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"], text=True)
        for line in processes.splitlines():
            pid, name, memory = [field.strip() for field in line.split(",", 2)]
            if name.endswith("tum_rgbd_cuda_tf_online"):
                out.writerow([time.time(), current["scene"], current["backend"], pid, memory])
                stream.flush()
        time.sleep(1)
