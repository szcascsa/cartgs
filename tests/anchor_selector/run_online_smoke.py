"""完成小规模 gates 后跑一条完整 TUM 序列，尾优化限制为 20 次，仅用于接入回归。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo, dataset = (Path(x).resolve() for x in sys.argv[1:3])
legacy = len(sys.argv) > 3 and sys.argv[3] == "legacy"
root = repo.parent
gates = json.loads((root/"validation/correctness.json").read_text())
assert len(gates) == 8 and all(x["passed"] for x in gates.values())
output = root/("online-fr1_xyz-legacy" if legacy else "online-fr1_xyz")
output.mkdir(exist_ok=False)
command = [repo/"bin/tum_rgbd", repo/"third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt",
           repo/"cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml",
           repo/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml", dataset,
           repo/"cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt", output,
           "no_viewer", "--selector-config", root/("legacy_smoke.yaml" if legacy else "online_smoke.yaml")]
record = dict(command=[str(x) for x in command], status="RUNNING", tail_steps=20,
              purpose="functional integration regression, not quality superiority")
path = root/("legacy_smoke.json" if legacy else "online_smoke.json")
path.write_text(json.dumps(record, indent=2))
start = time.monotonic()
with (root/("legacy_smoke.log" if legacy else "online_smoke.log")).open("w") as log:
    result = subprocess.run(record["command"], cwd=repo,
                            env=dict(os.environ, OMP_NUM_THREADS="2", QT_QPA_PLATFORM="offscreen"),
                            stdout=log, stderr=subprocess.STDOUT)
record.update(status="COMPLETED", exit_code=result.returncode, wall_seconds=time.monotonic()-start)
path.write_text(json.dumps(record, indent=2))
raise SystemExit(result.returncode)
