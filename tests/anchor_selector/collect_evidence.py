"""汇总既有验证结果和源码 SHA，不启动训练或调整实验参数。"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
output = root/"validation"
read = lambda path: json.loads(path.read_text())
new = read(output/"quality_smoke.json")
legacy = read(output/"legacy_full_quality.json")
frames = read(root/"quality_smoke/Full/evaluation_frames.json")["gt_indices"]
assert frames == read(root/"quality_legacy/Full/evaluation_frames.json")["gt_indices"]
paired = {}
for metric, filename, sign in (("psnr","psnr.txt",1),("ssim","ssim.txt",1),("lpips","lpips.txt",-1)):
    delta = np.loadtxt(root/"quality_smoke/Full"/filename)-np.loadtxt(root/"quality_legacy/Full"/filename)
    paired[metric] = dict(mean=float(delta.mean()),median=float(np.median(delta)),
                          p10=float(np.percentile(delta,10)),p90=float(np.percentile(delta,90)),
                          improved_view_fraction=float((sign*delta>0).mean()),
                          worse_view_fraction=float((sign*delta<0).mean()),views=len(delta))
(output/"paired_full_statistics.json").write_text(json.dumps(paired,indent=2))
with (output/"quality_summary.csv").open("w",newline="") as file:
    writer = csv.DictWriter(file,fieldnames=["method","arm","ratio","gaussians","views","psnr","ssim","lpips"])
    writer.writeheader()
    for method, arms in (("Anchor",new),("Legacy-SI",legacy)):
        for arm in arms:
            writer.writerow(dict(method=method,arm=arm["arm"],ratio=arm["ratio"],
                                 gaussians=arm["stats"]["active_gaussians"],views=len(frames),
                                 **{k:float(arm["metrics"][k]) for k in ("psnr","ssim","lpips")}))
runtime = {}
for method, folder, job in (("Anchor","online-fr1_xyz","online_smoke.json"),
                            ("Legacy-SI","online-fr1_xyz-legacy","legacy_smoke.json")):
    memory = (root/folder/"GpuPeakUsageMB.txt").read_text()
    runtime[method] = dict(job=read(root/job), allocator_memory_mb=dict(
        re.findall(r"Peak (\w+) \(MB\): ([\d.]+)",memory)))
(output/"runtime.json").write_text(json.dumps(runtime,indent=2))
manifest = read(output/"source_manifest.json")
mismatches = [path for path, digest in manifest["files"].items()
              if hashlib.sha256((root/"source"/path).read_bytes()).hexdigest() != digest]
source = dict(head=manifest["head"],file_count=len(manifest["files"]),exact=not mismatches,mismatches=mismatches)
(output/"source_verification.json").write_text(json.dumps(source,indent=2))
assert not mismatches, mismatches
summary = dict(architecture="anchor_joint_log_r_v2",network_parameters=1794,feature_width=16,
               correctness=read(output/"correctness.json"),source=source,
               online_checkpoint=read(output/"online_checkpoint.json"),
               paired_full=paired,quality_acceptance="NOT_ESTABLISHED; negative Full smoke regression",
               limitations=["one scene and seed, eight views", "different online optimizer counts and map populations",
                            "cached Bernoulli is not exact K", "no full training resume"])
(output/"audit_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(source=source,paired_full=paired),indent=2))
